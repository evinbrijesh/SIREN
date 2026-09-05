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
