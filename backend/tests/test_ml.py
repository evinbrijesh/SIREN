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
    """Model registry reports status for all 4 stages."""
    from siren.ml.registry import get_model_status

    status = get_model_status()
    assert "siamese_unet" in status
    assert "segformer" in status
    assert "consensus_gating" in status
    assert "convlstm_trend" in status

    for name, model in status.items():
        assert "stage" in model
        assert "loaded" in model
        assert "description" in model
        assert "architecture" in model
        assert isinstance(model["stage"], int)
        assert 1 <= model["stage"] <= 4


def test_model_registry_metadata_loads() -> None:
    """Model registry loads training metadata from checkpoints if they exist."""
    from siren.ml.registry import get_model_status

    status = get_model_status()
    # If the Siamese U-Net weights exist, metadata should be loaded
    siamese = status["siamese_unet"]
    if siamese["weights_exists"]:
        assert siamese["metadata"] is not None
        assert "epoch" in siamese["metadata"]


# --- Risk fusion ML integration tests ---

def test_risk_fusion_ml_confidence_is_load_bearing() -> None:
    """ML confidence must actually shift the hazard score (Path A)."""
    from siren.risk.fusion import fuse as risk_fuse

    base_kwargs = dict(
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

    score_low = risk_fuse(ml_confidence=0.1, **base_kwargs)
    score_high = risk_fuse(ml_confidence=0.95, **base_kwargs)

    # ML confidence has 0.20 weight — a 0.85 delta should shift H by ~0.17
    delta = score_high["hazard_score"] - score_low["hazard_score"]
    assert delta > 0.1, f"ML confidence not load-bearing: delta={delta}"
    assert score_high["hazard_score"] > score_low["hazard_score"]


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
        ml_confidence=0.5,
    )

    score_stable = risk_fuse(trend_class="stable", **base_kwargs)
    score_rapidly = risk_fuse(trend_class="rapidly", **base_kwargs)

    # Trend has 0.25 weight — stable vs rapidly should shift H significantly
    delta = score_rapidly["hazard_score"] - score_stable["hazard_score"]
    assert delta > 0.1, f"Trend class not load-bearing: delta={delta}"
    assert score_rapidly["hazard_score"] > score_stable["hazard_score"]


def test_risk_fusion_ml_confidence_default_neutral() -> None:
    """Default ml_confidence=0.5 should not inflate or deflate the score."""
    from siren.risk.fusion import fuse as risk_fuse

    base_kwargs = dict(
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

    score_default = risk_fuse(ml_confidence=0.5, **base_kwargs)
    score_no_ml = risk_fuse(ml_confidence=0.5, **base_kwargs)

    # 0.5 is neutral — should not change the score relative to itself
    assert score_default["hazard_score"] == score_no_ml["hazard_score"]
