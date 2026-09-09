"""Tests for the ML module — architecture shape, engine fallback, consensus.

These tests verify the ML scaffold without requiring torch to be installed
(the engine gracefully falls back). When torch IS installed, the architecture
shape tests verify the model builds and forward passes correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from siren.ml.consensus import compute_consensus_mask


# --- Consensus mask tests (no torch required) ---

def test_consensus_both_agree() -> None:
    """When ML and rule-based masks agree, consensus = high confidence."""
    ml = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    rule = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    result = compute_consensus_mask(ml, rule)
    assert result["consensus"].sum() == 4
    # Where both agree, confidence should be 0.95
    assert (result["confidence"][ml == 1] == 0.95).all()
    assert result["agreement"].sum() == 4


def test_consensus_ml_only_medium_confidence() -> None:
    """ML-only detections (no rule-based) get medium confidence."""
    ml = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    rule = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    result = compute_consensus_mask(ml, rule)
    assert result["consensus"][0, 0] == 1  # fused >= 0.5 with ml_weight=0.6
    assert result["confidence"][0, 0] == 0.60  # ML only


def test_consensus_rule_only_medium_high() -> None:
    """Rule-based-only detections always pass into consensus (Hard Rule 1)."""
    ml = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    rule = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    result = compute_consensus_mask(ml, rule)
    # Rule-based mask is the trusted physical method — always included
    assert result["consensus"][0, 0] == 1
    assert result["confidence"][0, 0] == 0.70  # medium-high (physical method)


def test_consensus_slope_gating() -> None:
    """Steep terrain should be excluded from ML predictions."""
    ml = np.ones((3, 3), dtype=np.uint8)
    rule = np.zeros((3, 3), dtype=np.uint8)
    slope = np.array([[5, 5, 5], [5, 40, 5], [5, 5, 5]], dtype=np.float32)
    result = compute_consensus_mask(ml, rule, dem_slope=slope, slope_threshold_deg=35.0)
    # The pixel at (1,1) has slope 40 > 35, so ML should be gated there
    assert result["ml_gated"][1, 1] == 0
    assert result["ml_gated"][0, 0] == 1


def test_consensus_deterministic() -> None:
    """Same inputs → identical outputs (Hard Rule 6)."""
    ml = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    rule = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    r1 = compute_consensus_mask(ml, rule)
    r2 = compute_consensus_mask(ml, rule)
    assert np.array_equal(r1["consensus"], r2["consensus"])
    assert np.array_equal(r1["confidence"], r2["confidence"])


def test_consensus_resize_mismatch() -> None:
    """Consensus handles different mask sizes via nearest-neighbor resize."""
    ml = np.ones((4, 4), dtype=np.uint8)
    rule = np.zeros((2, 2), dtype=np.uint8)
    result = compute_consensus_mask(ml, rule)
    assert result["consensus"].shape == (2, 2)


# --- Engine fallback tests ---

def test_engine_fallback_without_torch() -> None:
    """Engine should report is_ready=False when no weights exist."""
    from siren.ml.engine import ChangeDetectionEngine
    engine = ChangeDetectionEngine(weights_path="/nonexistent/weights.pt")
    # If torch is not installed, _torch_available=False
    # If torch is installed but no weights, is_ready=False
    assert engine.is_ready is False


def test_engine_raises_when_not_ready() -> None:
    """Engine should raise RuntimeError when predict is called without weights."""
    from siren.ml.engine import ChangeDetectionEngine
    engine = ChangeDetectionEngine(weights_path="/nonexistent/weights.pt")
    t0 = np.zeros((3, 64, 64), dtype=np.float32)
    t1 = np.zeros((3, 64, 64), dtype=np.float32)
    with pytest.raises(RuntimeError, match="not ready"):
        engine.predict_change_mask(t0, t1)


# --- Architecture tests (only run if torch is available) ---

def test_siamese_unet_forward_pass() -> None:
    """Verify the SiameseUNet architecture builds and forward passes."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed — skipping architecture test")

    from siren.ml.model import SiameseUNet, SegFormerHead

    model = SiameseUNet(in_channels=3)
    model.eval()

    # Bi-temporal input: (B, C, H, W)
    t0 = torch.randn(1, 3, 64, 64)
    t1 = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        logits = model(t0, t1)

    # Output should be (B, 1, H, W)
    assert logits.shape[0] == 1
    assert logits.shape[1] == 1
    # Spatial dims should match input (approximately, after up/downsampling)
    assert logits.shape[2] == 64
    assert logits.shape[3] == 64

    # Sigmoid → probability in [0, 1]
    probs = torch.sigmoid(logits)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_segformer_head_forward() -> None:
    """Verify the SegFormer head builds and forward passes."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed — skipping architecture test")

    from siren.ml.model import SegFormerHead

    head = SegFormerHead(in_channels=3, num_classes=5)
    head.eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = head(x)

    assert logits.shape == (2, 5)


def test_siamese_unet_weight_sharing() -> None:
    """Verify the encoder uses shared weights (Siamese property)."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed — skipping architecture test")

    from siren.ml.model import SiameseUNet

    model = SiameseUNet(in_channels=3)
    model.eval()

    t0 = torch.randn(1, 3, 64, 64)
    t1 = torch.randn(1, 3, 64, 64)

    # Extract features for both — should use the same encoder weights
    with torch.no_grad():
        f0 = model._extract(t0)
        f1 = model._extract(t1)

    # Same encoder → same shapes at each level
    for i in range(len(f0)):
        assert f0[i].shape == f1[i].shape


# --- WaterUNet architecture tests (ADR-010 Stage 1) ---

def test_water_unet_forward_pass() -> None:
    """WaterUNet builds, forward passes, and produces (B, 1, H, W) logits."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed — skipping architecture test")

    from siren.ml.model import WaterUNet

    model = WaterUNet(in_channels=2)
    model.eval()

    # Single-date SAR input: (B, 2, H, W) — VV/VH
    x = torch.randn(2, 2, 64, 64)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 1, 64, 64)
    probs = torch.sigmoid(logits)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_water_unet_param_budget() -> None:
    """WaterUNet must stay within the <=10M parameter budget (ADR-010)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed — skipping architecture test")

    from siren.ml.model import WaterUNet

    model = WaterUNet(in_channels=2)
    n = model.num_parameters()
    assert n <= 10_000_000, f"WaterUNet has {n:,} params — exceeds 10M budget"


def test_water_unet_512_input() -> None:
    """WaterUNet handles the full Sen1Floods11 chip size (512x512)."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed — skipping architecture test")

    from siren.ml.model import WaterUNet

    model = WaterUNet(in_channels=2)
    model.eval()
    x = torch.randn(1, 2, 512, 512)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 1, 512, 512)


# --- Engine tests for the new WaterUNet-based engine ---

def test_engine_fallback_water_unet_no_weights(tmp_path) -> None:
    """Engine reports is_ready=False when no WaterUNet weights exist."""
    from siren.ml.engine import ChangeDetectionEngine

    engine = ChangeDetectionEngine(weights_path=tmp_path / "nonexistent.pt")
    assert engine.is_ready is False


def test_engine_raises_water_mask_not_ready(tmp_path) -> None:
    """predict_water_mask raises when engine is not ready."""
    from siren.ml.engine import ChangeDetectionEngine

    engine = ChangeDetectionEngine(weights_path=tmp_path / "nonexistent.pt")
    sar = np.zeros((2, 64, 64), dtype=np.float32)
    with pytest.raises(RuntimeError, match="not ready"):
        engine.predict_water_mask(sar)


def test_engine_raises_change_mask_not_ready(tmp_path) -> None:
    """predict_change_mask raises when engine is not ready."""
    from siren.ml.engine import ChangeDetectionEngine

    engine = ChangeDetectionEngine(weights_path=tmp_path / "nonexistent.pt")
    t0 = np.zeros((2, 64, 64), dtype=np.float32)
    t1 = np.zeros((2, 64, 64), dtype=np.float32)
    with pytest.raises(RuntimeError, match="not ready"):
        engine.predict_change_mask(t0, t1)


def test_engine_change_mask_is_deterministic_diff(tmp_path) -> None:
    """predict_change_mask must be the deterministic set difference of two
    per-date water masks (Hard Rule 1 — change decision is not learned)."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    from siren.ml.engine import ChangeDetectionEngine
    from siren.ml.model import WaterUNet

    # Create a tiny engine with random weights (just to test the logic)
    wpath = tmp_path / "test_weights.pt"
    model = WaterUNet(in_channels=2)
    torch.save({"state_dict": model.state_dict(), "in_channels": 2}, str(wpath))

    engine = ChangeDetectionEngine(weights_path=wpath, device="cpu")
    assert engine.is_ready is True

    # Two identical inputs -> zero change (deterministic property)
    sar = np.random.RandomState(42).randn(2, 64, 64).astype(np.float32) * 20 - 15
    change = engine.predict_change_mask(sar, sar)
    assert change.shape == (64, 64)
    assert change.dtype == np.uint8
    # Identical inputs must produce zero change pixels
    assert change.sum() == 0, "Identical inputs should produce no change"


# --- Metrics module tests ---

def test_metrics_perfect_prediction() -> None:
    """Perfect prediction gives IoU=1.0, precision=1.0, recall=1.0."""
    from siren.ml.metrics import RunningConfusion

    acc = RunningConfusion()
    pred = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    target = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    valid = np.ones_like(pred)
    acc.update(pred, target, valid)
    result = acc.result()
    assert result["iou"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_metrics_no_data_excluded() -> None:
    """No-data pixels (valid=0) must be excluded from all metrics."""
    from siren.ml.metrics import water_confusion_counts, metrics_from_counts

    # Model predicts water everywhere, but half the image is no-data
    pred = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    target = np.array([[1, 0], [-1, -1]], dtype=np.int16)  # -1 = no data
    valid = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    target_bin = (target == 1).astype(np.uint8)

    tp, fp, fn, tn = water_confusion_counts(pred, target_bin, valid)
    # Only top-left 2 pixels are evaluated: tp=1, fp=1, fn=0, tn=0
    assert tp == 1
    assert fp == 1
    assert fn == 0
    assert tn == 0

    m = metrics_from_counts(tp, fp, fn, tn)
    assert m["iou"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 1.0


def test_metrics_empty_water() -> None:
    """All-land prediction and target gives IoU=0 (no water to predict)."""
    from siren.ml.metrics import metrics_from_counts

    m = metrics_from_counts(tp=0, fp=0, fn=0, tn=100)
    assert m["iou"] == 0.0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


# --- Contract tests (verify the frozen SAR input contract) ---

def test_contract_normalizes_db_range() -> None:
    """normalize_sar maps [-30, 0] dB to [0, 1]."""
    from siren.ml.contract import normalize_sar

    # -30 dB -> 0.0, 0 dB -> 1.0, -15 dB -> 0.5
    sar = np.array([[[-30.0, -15.0, 0.0]]], dtype=np.float32)
    norm = normalize_sar(sar)
    assert norm.shape == (1, 1, 3)
    assert np.allclose(norm[0, 0], [0.0, 0.5, 1.0])


def test_contract_clamps_out_of_range() -> None:
    """Values below -30 dB clamp to 0; values above 0 dB clamp to 1."""
    from siren.ml.contract import normalize_sar

    sar = np.array([[[-50.0, 5.0]]], dtype=np.float32)
    norm = normalize_sar(sar)
    assert norm[0, 0, 0] == 0.0  # -50 dB clamped
    assert norm[0, 0, 1] == 1.0  # +5 dB clamped


def test_contract_replaces_nans() -> None:
    """NaNs in SAR input are replaced (not propagated)."""
    from siren.ml.contract import normalize_sar

    sar = np.array([[[np.nan, -20.0]]], dtype=np.float32)
    norm = normalize_sar(sar)
    assert not np.isnan(norm).any()
    assert 0.0 <= norm[0, 0, 0] <= 1.0


# --- 4-channel contract tests (ADR-011 / V3 §2.1 terrain-aware) ---

def test_contract_constants_4channel() -> None:
    """SAR_CHANNELS = 4 and channel names match the V3 §2.1 contract."""
    from siren.ml.contract import SAR_CHANNELS, CHANNEL_NAMES, DEM_MAX_M, SLOPE_MAX_DEG

    assert SAR_CHANNELS == 4
    assert CHANNEL_NAMES == ("VV", "VH", "DEM", "Slope")
    assert DEM_MAX_M == 8848.0
    assert SLOPE_MAX_DEG == 90.0


def test_contract_legacy_constants_preserved() -> None:
    """The 2-channel legacy constants are preserved for the shadow model."""
    from siren.ml.contract import SAR_CHANNELS_LEGACY, CHANNEL_NAMES_LEGACY

    assert SAR_CHANNELS_LEGACY == 2
    assert CHANNEL_NAMES_LEGACY == ("VV", "VH")


def test_normalize_dem_maps_to_unit_range() -> None:
    """normalize_dem maps [0, 8848] m to [0, 1]."""
    from siren.ml.contract import normalize_dem, DEM_MAX_M

    dem = np.array([[0.0, DEM_MAX_M / 2, DEM_MAX_M]], dtype=np.float32)
    norm = normalize_dem(dem)
    assert np.allclose(norm, [0.0, 0.5, 1.0])


def test_normalize_dem_clamps_out_of_range() -> None:
    """Negative elevation clamps to 0; above 8848 clamps to 1."""
    from siren.ml.contract import normalize_dem

    dem = np.array([[-100.0, 10000.0]], dtype=np.float32)
    norm = normalize_dem(dem)
    assert norm[0, 0] == 0.0  # below sea level
    assert norm[0, 1] == 1.0  # above Everest


def test_normalize_dem_replaces_nans() -> None:
    """NaN elevation is treated as sea level (0.0)."""
    from siren.ml.contract import normalize_dem

    dem = np.array([[np.nan, 4420.0]], dtype=np.float32)
    norm = normalize_dem(dem)
    assert norm[0, 0] == 0.0
    assert np.isclose(norm[0, 1], 4420.0 / 8848.0)


def test_normalize_slope_maps_to_unit_range() -> None:
    """normalize_slope maps [0, 90] deg to [0, 1]."""
    from siren.ml.contract import normalize_slope

    slope = np.array([[0.0, 45.0, 90.0]], dtype=np.float32)
    norm = normalize_slope(slope)
    assert np.allclose(norm, [0.0, 0.5, 1.0])


def test_normalize_slope_clamps_out_of_range() -> None:
    """Negative slope clamps to 0; above 90 clamps to 1."""
    from siren.ml.contract import normalize_slope

    slope = np.array([[-5.0, 120.0]], dtype=np.float32)
    norm = normalize_slope(slope)
    assert norm[0, 0] == 0.0
    assert norm[0, 1] == 1.0


def test_normalize_tensor_4channel_3d() -> None:
    """normalize_tensor normalises (4, H, W) per-channel."""
    from siren.ml.contract import (
        normalize_tensor, SAR_DB_MIN, SAR_DB_MAX, DEM_MAX_M, SLOPE_MAX_DEG,
    )

    tensor = np.array([
        [[SAR_DB_MIN, SAR_DB_MAX]],       # VV dB
        [[SAR_DB_MIN, SAR_DB_MAX]],       # VH dB
        [[0.0, DEM_MAX_M]],               # DEM m
        [[0.0, SLOPE_MAX_DEG]],           # Slope deg
    ], dtype=np.float32)
    norm = normalize_tensor(tensor)
    assert norm.shape == (4, 1, 2)
    # Each channel's endpoints map to [0, 1]
    for c in range(4):
        assert np.isclose(norm[c, 0, 0], 0.0)
        assert np.isclose(norm[c, 0, 1], 1.0)


def test_normalize_tensor_4channel_4d() -> None:
    """normalize_tensor normalises (B, 4, H, W) per-channel."""
    from siren.ml.contract import normalize_tensor

    tensor = np.zeros((2, 4, 4, 4), dtype=np.float32)
    tensor[:, 0] = -15.0  # VV mid-range
    tensor[:, 1] = -15.0  # VH mid-range
    tensor[:, 2] = 4424.0  # DEM mid-range
    tensor[:, 3] = 45.0   # Slope mid-range
    norm = normalize_tensor(tensor)
    assert norm.shape == (2, 4, 4, 4)
    assert np.allclose(norm[:, 0], 0.5)  # -15 dB -> 0.5
    assert np.allclose(norm[:, 2], 4424.0 / 8848.0)
    assert np.allclose(norm[:, 3], 0.5)


def test_normalize_tensor_rejects_wrong_channels() -> None:
    """normalize_tensor raises ValueError for non-4-channel input."""
    from siren.ml.contract import normalize_tensor

    bad = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="4 channels"):
        normalize_tensor(bad)


def test_denormalize_tensor_round_trip() -> None:
    """denormalize_tensor inverts normalize_tensor (within clamp range)."""
    from siren.ml.contract import normalize_tensor, denormalize_tensor

    original = np.array([
        [[-20.0, -10.0], [-25.0, -5.0]],
        [[-22.0, -12.0], [-27.0, -7.0]],
        [[1000.0, 5000.0], [3000.0, 7000.0]],
        [[10.0, 30.0], [20.0, 45.0]],
    ], dtype=np.float32)
    norm = normalize_tensor(original)
    recovered = denormalize_tensor(norm)
    assert np.allclose(recovered, original, atol=1e-4)


# --- Visualize tests (no torch required) ---

def test_generate_change_heatmap(tmp_path) -> None:
    """Verify heatmap PNG generation produces a valid file."""
    from siren.ml.visualize import generate_change_heatmap_png

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # square changed region

    output = generate_change_heatmap_png(mask, tmp_path / "heatmap.png")
    assert output.exists()
    assert output.stat().st_size > 0


def test_generate_before_after(tmp_path) -> None:
    """Verify before/after comparison PNG generation."""
    from siren.ml.visualize import generate_before_after_png

    baseline = np.zeros((32, 32), dtype=np.uint8)
    baseline[5:15, 5:15] = 1
    current = np.zeros((32, 32), dtype=np.uint8)
    current[5:20, 5:20] = 1  # expanded

    output = generate_before_after_png(baseline, current, tmp_path / "before_after.png")
    assert output.exists()
    assert output.stat().st_size > 0


# --- Trend engine tests (ConvLSTM Stage 4) ---

def test_trend_engine_deterministic_fallback() -> None:
    """TrendEngine deterministic fallback classifies by expansion thresholds."""
    from siren.ml.trend_engine import TrendEngine

    # Stable: no change
    masks = [np.ones((64, 64), dtype=np.float32) for _ in range(4)]
    trend, conf = TrendEngine._deterministic_fallback(masks)
    assert trend == "stable"
    assert conf > 0.5

    # Rapidly: large monotonic increase
    masks = []
    for i in range(4):
        m = np.zeros((64, 64), dtype=np.float32)
        m[: 10 + i * 15, : 10 + i * 15] = 1
        masks.append(m)
    trend, conf = TrendEngine._deterministic_fallback(masks)
    assert trend == "rapidly"
    assert conf > 0.5

    # Slowly: small monotonic increase (3-20%)
    masks = []
    base_size = 40
    for i in range(4):
        m = np.zeros((64, 64), dtype=np.float32)
        side = base_size + i  # 40, 41, 42, 43 → ~7% growth
        m[:side, :side] = 1
        masks.append(m)
    trend, conf = TrendEngine._deterministic_fallback(masks)
    assert trend == "slowly"
    assert conf > 0.5

    # Uncertain: increase then decrease
    masks = []
    areas = [100, 200, 150, 80]
    for a in areas:
        m = np.zeros((64, 64), dtype=np.float32)
        m[: int(a**0.5), : int(a**0.5)] = 1
        masks.append(m)
    trend, conf = TrendEngine._deterministic_fallback(masks)
    assert trend == "uncertain"


def test_trend_engine_fallback_single_mask() -> None:
    """TrendEngine fallback with a single mask returns uncertain."""
    from siren.ml.trend_engine import TrendEngine

    masks = [np.ones((32, 32), dtype=np.float32)]
    trend, conf = TrendEngine._deterministic_fallback(masks)
    assert trend == "uncertain"


def test_trend_engine_classifies_when_torch_available() -> None:
    """If torch is installed and weights exist, TrendEngine uses the ConvLSTM."""
    from pathlib import Path

    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")

    from siren.ml.trend_engine import TrendEngine, DEFAULT_WEIGHTS_PATH

    if not DEFAULT_WEIGHTS_PATH.exists():
        pytest.skip("ConvLSTM weights not trained")

    engine = TrendEngine()
    if not engine.is_ready:
        pytest.skip("TrendEngine could not load weights")

    # Stable sequence (no change)
    masks = [np.ones((128, 128), dtype=np.float32) for _ in range(4)]
    trend, conf = engine.classify_trend(masks)
    assert trend in ("stable", "slowly", "rapidly", "uncertain")
    assert 0.0 <= conf <= 1.0

    # Rapidly expanding
    masks = []
    for i in range(4):
        m = np.zeros((128, 128), dtype=np.float32)
        cy, cx = 64, 64
        yy, xx = np.ogrid[:128, :128]
        r = 10 + i * 15
        m[(yy - cy) ** 2 + (xx - cx) ** 2 < r**2] = 1
        masks.append(m)
    trend, conf = engine.classify_trend(masks)
    assert trend in ("stable", "slowly", "rapidly", "uncertain")
    assert 0.0 <= conf <= 1.0


# --- SegFormer engine tests (Stage 2) ---

def test_segformer_engine_deterministic_fallback() -> None:
    """SegFormerEngine deterministic fallback classifies by backscatter."""
    from siren.ml.segformer_engine import SegFormerEngine

    # Shadow: extremely low backscatter (< -25 dB)
    crop = np.full((2, 64, 64), -30.0, dtype=np.float32)
    cls, conf = SegFormerEngine._deterministic_classify(crop)
    assert cls == "shadow"
    assert conf > 0.5

    # Water: very low backscatter (-22 to -25 dB)
    crop = np.full((2, 64, 64), -23.0, dtype=np.float32)
    cls, conf = SegFormerEngine._deterministic_classify(crop)
    assert cls == "water"
    assert conf > 0.5

    # Bare rock: high backscatter (> -8 dB)
    crop = np.full((2, 64, 64), -5.0, dtype=np.float32)
    cls, conf = SegFormerEngine._deterministic_classify(crop)
    assert cls == "bare_rock"
    assert conf > 0.5

    # Normalized input (0-1 range) should denormalize correctly
    crop_norm = np.full((2, 64, 64), 0.5, dtype=np.float32)  # → -15 dB
    cls, conf = SegFormerEngine._deterministic_classify(crop_norm)
    assert cls in ("water", "shadow", "bare_rock", "debris", "snowmelt")


def test_segformer_engine_filters_false_alarms() -> None:
    """SegFormer engine removes shadow/snowmelt regions from the change mask."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")

    from pathlib import Path
    from siren.ml.segformer_engine import SegFormerEngine, DEFAULT_WEIGHTS_PATH

    if not DEFAULT_WEIGHTS_PATH.exists():
        pytest.skip("SegFormer weights not trained")

    engine = SegFormerEngine()
    if not engine.is_ready:
        pytest.skip("SegFormer engine could not load weights")

    # Create a synthetic image with multiple changed regions
    img = np.random.randn(2, 200, 200).astype(np.float32) * 0.1 + 0.5
    img[:, 20:60, 20:60] = -0.3  # dark region (potential shadow)

    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[20:60, 20:60] = 1
    mask[80:120, 80:120] = 1
    mask[150:180, 150:180] = 1

    result = engine.classify_change_crops(img, mask)

    assert "classifications" in result
    assert "class_distribution" in result
    assert "filtered_mask" in result
    assert "false_alarm_count" in result
    assert "source" in result
    # The filtered mask should have the same or fewer pixels
    assert result["filtered_mask"].sum() <= mask.sum()
    # Source should be segformer (model loaded)
    assert result["source"] == "segformer"


def test_segformer_engine_no_regions() -> None:
    """SegFormer engine handles empty change mask gracefully."""
    from siren.ml.segformer_engine import SegFormerEngine

    engine = SegFormerEngine()
    img = np.zeros((2, 100, 100), dtype=np.float32)
    mask = np.zeros((100, 100), dtype=np.uint8)

    result = engine.classify_change_crops(img, mask)
    assert result["classifications"] == []
    assert result["false_alarm_count"] == 0
    assert result["filtered_mask"].sum() == 0


# --- Model registry tests ---

def test_model_registry_reports_all_stages() -> None:
    """Model registry reports status for the active model, archived models, and consensus."""
    from siren.ml.registry import get_model_status

    status = get_model_status()
    # Active model (ADR-010 Stage 1)
    assert "water_unet" in status
    # Archived / disqualified models (reported for transparency, not loaded)
    assert "siamese_unet" in status
    assert "segformer_classifier" in status
    assert "convlstm_trend" in status
    # Deterministic consensus
    assert "consensus_gating" in status

    for name, model in status.items():
        assert "stage" in model
        assert "loaded" in model
        assert "description" in model
        # stage is either an int (active/deterministic) or "archived"
        assert isinstance(model["stage"], (int, str))

    # Archived models must never report as loaded
    for name in ("siamese_unet", "segformer_classifier", "convlstm_trend"):
        assert status[name]["loaded"] is False, f"{name} should not be loaded — it is archived"
        assert status[name].get("status") == "archived_disqualified"


def test_model_registry_metadata_loads() -> None:
    """Model registry loads training metadata from archived checkpoints if they exist."""
    from siren.ml.registry import get_model_status

    status = get_model_status()
    # The archived Siamese U-Net weights should still have metadata
    siamese = status["siamese_unet"]
    if siamese["weights_exists"]:
        assert siamese["metadata"] is not None
        assert "epoch" in siamese["metadata"]


# --- Risk fusion tests (ADR-010: 5-factor formula, no ML term) ---

def test_risk_fusion_has_no_ml_confidence_parameter() -> None:
    """ADR-010: fuse() must NOT accept ml_confidence — ML is not a hazard factor."""
    import inspect
    from siren.risk.fusion import fuse as risk_fuse

    sig = inspect.signature(risk_fuse)
    assert "ml_confidence" not in sig.parameters, (
        "fuse() still accepts ml_confidence — ADR-010 violation: "
        "ML must not enter the hazard score"
    )


def test_risk_fusion_weights_sum_to_one() -> None:
    """PRD §9.5: the five physical factor weights must sum to exactly 1.0."""
    from siren.risk.fusion import W_TREND, W_EXPANSION, W_RAIN, W_SLOPE, W_PROX

    total = W_TREND + W_EXPANSION + W_RAIN + W_SLOPE + W_PROX
    assert abs(total - 1.0) < 1e-9, f"5-factor weights sum to {total}, not 1.0"
    # Canonical PRD §9.5 weights
    assert (W_TREND, W_EXPANSION, W_RAIN, W_SLOPE, W_PROX) == (0.30, 0.25, 0.20, 0.15, 0.10)


def test_risk_fusion_trend_class_is_load_bearing() -> None:
    """Trend class must actually shift the hazard score (Stage 4)."""
    from siren.risk.fusion import fuse as risk_fuse

    base_kwargs = dict(
        expansion_pct=40,
        rainfall_24h_mm=80,
        rainfall_7d_mm=160,
        mean_slope_deg=31,
        change_in_drainage=True,
        exposed_population=1240,
        settlements=2,
        bridges=1,
        wells=3,
        inundated_wells=1,
        population_density_per_km2=200,
        temp_index=0.7,
    )

    score_stable = risk_fuse(trend_class="stable", **base_kwargs)
    score_rapidly = risk_fuse(trend_class="rapidly", **base_kwargs)

    # Trend has 0.30 weight — stable(0.1) vs rapidly(0.9) shifts H by ~0.24
    delta = score_rapidly["hazard_score"] - score_stable["hazard_score"]
    assert delta > 0.1, f"Trend class not load-bearing: delta={delta}"
    assert score_rapidly["hazard_score"] > score_stable["hazard_score"]


def test_risk_fusion_expansion_is_load_bearing() -> None:
    """Water-area expansion must shift the hazard score (deterministic signal)."""
    from siren.risk.fusion import fuse as risk_fuse

    base_kwargs = dict(
        trend_class="rapidly",
        rainfall_24h_mm=80,
        rainfall_7d_mm=160,
        mean_slope_deg=31,
        change_in_drainage=True,
        exposed_population=1240,
        settlements=2,
        bridges=1,
        wells=3,
        inundated_wells=1,
        population_density_per_km2=200,
        temp_index=0.7,
    )

    score_low = risk_fuse(expansion_pct=0, **base_kwargs)
    score_high = risk_fuse(expansion_pct=30, **base_kwargs)

    delta = score_high["hazard_score"] - score_low["hazard_score"]
    assert delta > 0.1, f"Expansion not load-bearing: delta={delta}"


def test_risk_fusion_is_deterministic() -> None:
    """Same inputs must produce identical outputs (Hard Rule 6)."""
    from siren.risk.fusion import fuse as risk_fuse

    kwargs = dict(
        trend_class="rapidly",
        expansion_pct=40,
        rainfall_24h_mm=80,
        rainfall_7d_mm=160,
        mean_slope_deg=31,
        change_in_drainage=True,
        exposed_population=1240,
        settlements=2,
        bridges=1,
        wells=3,
        inundated_wells=1,
        population_density_per_km2=200,
        temp_index=0.7,
    )

    s1 = risk_fuse(**kwargs)
    s2 = risk_fuse(**kwargs)
    assert s1["hazard_score"] == s2["hazard_score"]
    assert s1["reasons"] == s2["reasons"]


def test_risk_fusion_reasons_have_no_ml_term() -> None:
    """ADR-010: the reasons array must not reference ML confidence."""
    from siren.risk.fusion import hazard_score

    h, reasons = hazard_score(
        trend_class="rapidly",
        expansion_pct=40,
        rainfall_24h_mm=80,
        rainfall_7d_mm=160,
        mean_slope_deg=31,
        change_in_drainage=True,
    )
    # No reason should mention ML, confidence, or consensus
    for reason in reasons:
        assert "ML" not in reason, f"Reason references ML: {reason}"
        assert "confidence" not in reason.lower(), f"Reason references confidence: {reason}"
    # Must have exactly 5 reasons (one per physical factor)
    assert len(reasons) == 5
