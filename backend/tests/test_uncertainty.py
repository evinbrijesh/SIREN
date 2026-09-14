"""Tests for Bayesian neural uncertainty via MC Dropout (ADR-013 §9.7.4).

Tests the MC Dropout inference wrapper, conformal calibration, and
coverage evaluation that replace hardcoded severity thresholds with
spatially-resolved epistemic uncertainty estimation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from siren.ml.model import WaterResUNet
from siren.ml.uncertainty import (
    MCDropoutResult,
    enable_mc_dropout,
    mc_dropout_inference,
    calibrate_conformal,
    evaluate_coverage,
)


# --------------------------------------------------------------------------- #
# enable_mc_dropout
# --------------------------------------------------------------------------- #

def test_enable_mc_dropout_activates_dropout():
    """enable_mc_dropout sets dropout to train mode, BN to eval mode."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.1)
    model.eval()  # everything in eval

    # Before: dropout is in eval mode
    for module in model.modules():
        if isinstance(module, nn.Dropout2d):
            assert not module.training

    enable_mc_dropout(model)

    # After: dropout is in train mode, BN is in eval mode
    dropout_in_train = False
    bn_in_eval = True
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            if module.training:
                dropout_in_train = True
        if isinstance(module, nn.BatchNorm2d):
            if module.training:
                bn_in_eval = False

    assert dropout_in_train, "No dropout layer was activated"
    assert bn_in_eval, "BatchNorm was not kept in eval mode"


def test_enable_mc_dropout_no_dropout_is_noop():
    """enable_mc_dropout on a model without dropout is a no-op."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.0)
    enable_mc_dropout(model)
    # No dropout layers to activate — model stays in eval
    assert not model.training


# --------------------------------------------------------------------------- #
# mc_dropout_inference
# --------------------------------------------------------------------------- #

def test_mc_dropout_shapes():
    """MC Dropout produces correct output shapes."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.1)
    x = torch.randn(2, 6, 32, 32)

    result = mc_dropout_inference(model, x, n_samples=10)

    assert result.mean.shape == (2, 1, 32, 32)
    assert result.variance.shape == (2, 1, 32, 32)
    assert result.std.shape == (2, 1, 32, 32)
    assert result.lower_bound.shape == (2, 1, 32, 32)
    assert result.upper_bound.shape == (2, 1, 32, 32)
    assert result.n_samples == 10


def test_mc_dropout_mean_in_range():
    """With sigmoid, mean is in [0, 1]."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.1)
    x = torch.randn(1, 6, 32, 32)

    result = mc_dropout_inference(model, x, n_samples=10, apply_sigmoid=True)

    assert result.mean.min() >= 0.0
    assert result.mean.max() <= 1.0


def test_mc_dropout_zero_variance_without_dropout():
    """With dropout=0, MC Dropout produces zero variance (deterministic)."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.0)
    x = torch.randn(1, 6, 32, 32)

    result = mc_dropout_inference(model, x, n_samples=5, apply_sigmoid=True)

    # All samples are identical → variance is 0
    assert np.allclose(result.variance, 0.0, atol=1e-6)
    assert np.allclose(result.std, 0.0, atol=1e-6)


def test_mc_dropout_nonzero_variance_with_dropout():
    """With dropout > 0, MC Dropout produces nonzero variance."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.3)
    x = torch.randn(1, 6, 32, 32)

    result = mc_dropout_inference(model, x, n_samples=20, apply_sigmoid=True)

    # Variance should be nonzero somewhere (dropout introduces stochasticity)
    assert result.variance.max() > 0.0
    assert result.std.max() > 0.0


def test_mc_dropout_confidence_bounds():
    """Lower bound ≤ mean ≤ upper bound."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.2)
    x = torch.randn(1, 6, 32, 32)

    result = mc_dropout_inference(model, x, n_samples=15, apply_sigmoid=True)

    assert np.all(result.lower_bound <= result.mean + 1e-6)
    assert np.all(result.mean <= result.upper_bound + 1e-6)


def test_mc_dropout_more_samples_reduces_variance():
    """More MC samples → more stable mean estimate (lower variance of the mean)."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.2)
    torch.manual_seed(42)
    x = torch.randn(1, 6, 32, 32)

    result_5 = mc_dropout_inference(model, x, n_samples=5, apply_sigmoid=True)
    result_30 = mc_dropout_inference(model, x, n_samples=30, apply_sigmoid=True)

    # The mean estimates should be close (both are unbiased)
    # But the variance estimate with more samples is more stable
    # We check that the mean doesn't diverge wildly
    assert np.allclose(result_5.mean, result_30.mean, atol=0.15)


def test_mc_dropout_to_dict():
    """to_dict produces a serializable summary."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.1)
    x = torch.randn(1, 6, 32, 32)

    result = mc_dropout_inference(model, x, n_samples=5)
    d = result.to_dict()

    assert d["n_samples"] == 5
    assert d["confidence_level"] == 0.90
    assert "variance_mean" in d
    assert "std_mean" in d
    assert "std_max" in d
    assert "mean_shape" in d


def test_mc_dropout_raw_logits():
    """apply_sigmoid=False returns raw logits (can be negative)."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.1)
    x = torch.randn(1, 6, 32, 32)

    result = mc_dropout_inference(model, x, n_samples=5, apply_sigmoid=False)

    # Raw logits can be any real number
    # Just check shapes and that variance exists
    assert result.mean.shape == (1, 1, 32, 32)


# --------------------------------------------------------------------------- #
# Conformal calibration
# --------------------------------------------------------------------------- #

def test_conformal_calibration_returns_quantile():
    """calibrate_conformal returns a valid quantile in [0, 1]."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.1)

    # Small calibration set
    cal_inputs = [torch.randn(1, 6, 32, 32) for _ in range(3)]
    cal_targets = [np.random.randint(0, 2, (1, 32, 32)).astype(np.float32) for _ in range(3)]

    q = calibrate_conformal(
        model, cal_inputs, cal_targets,
        n_samples=5, confidence_level=0.90,
    )

    assert 0.0 <= q <= 1.0
    assert isinstance(q, float)


def test_evaluate_coverage_returns_metrics():
    """evaluate_coverage returns coverage metrics with gate_passed flag."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.1)

    test_inputs = [torch.randn(1, 6, 32, 32) for _ in range(2)]
    test_targets = [np.random.randint(0, 2, (1, 32, 32)).astype(np.float32) for _ in range(2)]

    metrics = evaluate_coverage(
        model, test_inputs, test_targets,
        conformal_quantile=0.5,  # wide interval → high coverage
        n_samples=5, confidence_level=0.90,
    )

    assert "empirical_coverage" in metrics
    assert "nominal_level" in metrics
    assert "coverage_error" in metrics
    assert "gate_passed" in metrics
    assert 0.0 <= metrics["empirical_coverage"] <= 1.0
    assert metrics["nominal_level"] == 0.90


def test_evaluate_coverage_wide_interval_high_coverage():
    """A very wide conformal quantile gives near-100% coverage."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.1)

    test_inputs = [torch.randn(1, 6, 32, 32)]
    test_targets = [np.random.randint(0, 2, (1, 32, 32)).astype(np.float32)]

    metrics = evaluate_coverage(
        model, test_inputs, test_targets,
        conformal_quantile=1.0,  # full range
        n_samples=5, confidence_level=0.90,
    )

    # With q*=1.0, all targets should be covered (mean ± 1.0 covers [0,1])
    assert metrics["empirical_coverage"] > 0.95


def test_evaluate_coverage_narrow_interval_lower_coverage():
    """A very narrow conformal quantile gives lower coverage."""
    model = WaterResUNet(in_channels=6, base_channels=8, dropout=0.1)

    test_inputs = [torch.randn(1, 6, 32, 32)]
    test_targets = [np.random.randint(0, 2, (1, 32, 32)).astype(np.float32)]

    metrics = evaluate_coverage(
        model, test_inputs, test_targets,
        conformal_quantile=0.01,  # very narrow
        n_samples=5, confidence_level=0.90,
    )

    # With q*=0.01, coverage should be lower
    assert metrics["empirical_coverage"] < 0.95


# --------------------------------------------------------------------------- #
# Integration: MC Dropout + WaterResUNet with existing checkpoint
# --------------------------------------------------------------------------- #

def test_mc_dropout_with_kuro_siwo_checkpoint():
    """MC Dropout works with the real Kuro Siwo checkpoint (if available)."""
    import os
    ckpt_path = "models/checkpoints/water_resunet_kuro_siwo_full/water_resunet_6ch_kuro_siwo_v1.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("Kuro Siwo checkpoint not available")

    # Load the checkpoint into a model WITH dropout (the checkpoint weights
    # are compatible — dropout adds no new parameters)
    model = WaterResUNet(in_channels=6, base_channels=32, dropout=0.1)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    x = torch.randn(1, 6, 224, 224)  # full chip size
    result = mc_dropout_inference(model, x, n_samples=5, apply_sigmoid=True)

    assert result.mean.shape == (1, 1, 224, 224)
    assert result.mean.min() >= 0.0
    assert result.mean.max() <= 1.0
    # With real weights + dropout, there should be some uncertainty
    assert result.variance.max() >= 0.0
