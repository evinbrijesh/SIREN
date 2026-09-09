"""Tests for WaterResUNet, L_gravity loss, and DANN (V3 §2.3, §2.6, §2.7).

These tests verify the 4-channel architecture, the physics-informed gravity
penalty, and the domain-adversarial training components. All tests use
small synthetic tensors (32x32) for speed — no real data required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# WaterResUNet architecture tests (V3 §2.3)
# ---------------------------------------------------------------------------

def test_water_resunet_forward_pass_4channel():
    """WaterResUNet accepts 4-channel input and produces (B, 1, H, W) logits."""
    from siren.ml.model import WaterResUNet
    from siren.ml.contract import SAR_CHANNELS

    model = WaterResUNet(in_channels=SAR_CHANNELS, base_channels=16)
    x = torch.randn(2, SAR_CHANNELS, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


def test_water_resunet_forward_with_features():
    """forward_with_features returns (logits, bottleneck) for DANN."""
    from siren.ml.model import WaterResUNet
    from siren.ml.contract import SAR_CHANNELS

    model = WaterResUNet(in_channels=SAR_CHANNELS, base_channels=16)
    x = torch.randn(1, SAR_CHANNELS, 32, 32)
    logits, bottleneck = model.forward_with_features(x)
    assert logits.shape == (1, 1, 32, 32)
    # Bottleneck: base_channels * 16 = 256 channels at H/16 = 2x2
    assert bottleneck.shape == (1, 256, 2, 2)


def test_water_resunet_param_budget():
    """WaterResUNet stays within the <=10M parameter budget (V3 §2.3)."""
    from siren.ml.model import WaterResUNet
    from siren.ml.contract import SAR_CHANNELS

    model = WaterResUNet(in_channels=SAR_CHANNELS, base_channels=32)
    n_params = model.num_parameters()
    assert n_params <= 10_000_000, f"WaterResUNet has {n_params} params (>10M)"


def test_water_resunet_output_is_logits():
    """Output is raw logits (not sigmoid) — negative and positive values present."""
    from siren.ml.model import WaterResUNet
    from siren.ml.contract import SAR_CHANNELS

    model = WaterResUNet(in_channels=SAR_CHANNELS, base_channels=16)
    x = torch.randn(1, SAR_CHANNELS, 32, 32)
    out = model(x)
    assert out.requires_grad
    # Logits can be any real number (not bounded [0, 1])
    assert out.min() < 0 or out.max() > 1


def test_water_resunet_residual_block_skip_connection():
    """ResidualBlock preserves spatial dimensions and adds skip connection."""
    from siren.ml.model import ResidualBlock

    block = ResidualBlock(16, 32)
    x = torch.randn(1, 16, 8, 8)
    out = block(x)
    assert out.shape == (1, 32, 8, 8)


def test_water_resunet_deterministic_in_eval_mode():
    """WaterResUNet is deterministic in eval mode (BatchNorm fixed)."""
    from siren.ml.model import WaterResUNet
    from siren.ml.contract import SAR_CHANNELS

    model = WaterResUNet(in_channels=SAR_CHANNELS, base_channels=16)
    model.eval()
    x = torch.randn(1, SAR_CHANNELS, 32, 32)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# L_gravity loss tests (V3 §2.6)
# ---------------------------------------------------------------------------

def test_gravity_penalty_flat_water_is_low():
    """Water on flat terrain (constant DEM) has near-zero gravity penalty."""
    from siren.ml.losses import gravity_penalty

    logits = torch.zeros(1, 1, 16, 16)  # sigmoid(0) = 0.5 → water everywhere
    dem = torch.full((1, 1, 16, 16), 1000.0)  # flat
    penalty = gravity_penalty(logits, dem)
    assert penalty.item() < 1e-4  # near-zero variance


def test_gravity_penalty_ridge_water_is_high():
    """Water spread across varying elevations has a high gravity penalty."""
    from siren.ml.losses import gravity_penalty

    logits = torch.zeros(1, 1, 16, 16)  # water everywhere
    # DEM with a ridge: elevation varies from 1000 to 5000
    dem = torch.linspace(1000, 5000, 16).reshape(1, 1, 1, 16).expand(1, 1, 16, 16)
    penalty = gravity_penalty(logits, dem)
    assert penalty.item() > 0.005  # high variance (scaled by 0.5 coverage)


def test_gravity_penalty_no_water_is_zero():
    """When the model predicts no water, the gravity penalty is ~0."""
    from siren.ml.losses import gravity_penalty

    logits = torch.full((1, 1, 16, 16), -10.0)  # sigmoid(-10) ≈ 0 → no water
    dem = torch.linspace(1000, 5000, 16).reshape(1, 1, 1, 16).expand(1, 1, 16, 16)
    penalty = gravity_penalty(logits, dem)
    assert penalty.item() < 1e-4


def test_water_loss_combines_dice_bce_gravity():
    """WaterLoss returns a dict with total, dice, bce, gravity components."""
    from siren.ml.losses import WaterLoss

    loss_fn = WaterLoss(lambda_gravity=0.1)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    dem = torch.rand(2, 1, 16, 16) * 5000

    result = loss_fn(logits, target, dem)
    assert "total" in result
    assert "dice" in result
    assert "bce" in result
    assert "gravity" in result
    assert result["total"].requires_grad


def test_water_loss_without_dem_skips_gravity():
    """WaterLoss with dem=None produces zero gravity penalty."""
    from siren.ml.losses import WaterLoss

    loss_fn = WaterLoss(lambda_gravity=0.1)
    logits = torch.randn(1, 1, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (1, 1, 16, 16)).float()

    result = loss_fn(logits, target, dem=None)
    assert result["gravity"].item() == 0.0


def test_water_loss_gradients_flow():
    """WaterLoss total produces gradients on the logits."""
    from siren.ml.losses import WaterLoss

    loss_fn = WaterLoss(lambda_gravity=0.1)
    logits = torch.randn(1, 1, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (1, 1, 16, 16)).float()
    dem = torch.rand(1, 1, 16, 16) * 5000

    result = loss_fn(logits, target, dem)
    result["total"].backward()
    assert logits.grad is not None
    assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))


# ---------------------------------------------------------------------------
# DANN tests (V3 §2.7)
# ---------------------------------------------------------------------------

def test_gradient_reversal_forward_is_identity():
    """GRL forward pass is identity (no change to input)."""
    from siren.ml.dann import gradient_reversal

    x = torch.randn(4, 8)
    out = gradient_reversal(x, lambda_=1.0)
    assert torch.allclose(out, x)


def test_gradient_reversal_negates_gradient():
    """GRL backward pass negates and scales the gradient."""
    from siren.ml.dann import gradient_reversal

    x = torch.randn(4, 8, requires_grad=True)
    out = gradient_reversal(x, lambda_=1.0)
    out.sum().backward()
    # Gradient should be -1 (negated, scaled by lambda=1)
    assert torch.allclose(x.grad, -torch.ones_like(x.grad))


def test_gradient_reversal_lambda_scales_gradient():
    """GRL lambda scales the reversed gradient."""
    from siren.ml.dann import gradient_reversal

    x = torch.randn(4, 8, requires_grad=True)
    out = gradient_reversal(x, lambda_=0.5)
    out.sum().backward()
    assert torch.allclose(x.grad, -0.5 * torch.ones_like(x.grad))


def test_domain_discriminator_forward():
    """DomainDiscriminator produces (B, 1) logits from pooled features."""
    from siren.ml.dann import DomainDiscriminator

    disc = DomainDiscriminator(in_features=64, hidden_dim=32)
    features = torch.randn(4, 64)
    out = disc(features)
    assert out.shape == (4, 1)


def test_dann_wrapper_forward_returns_seg_and_domain():
    """DANNWrapper returns (seg_logits, domain_logits)."""
    from siren.ml.model import WaterResUNet
    from siren.ml.dann import DANNWrapper
    from siren.ml.contract import SAR_CHANNELS

    segmenter = WaterResUNet(in_channels=SAR_CHANNELS, base_channels=16)
    # Bottleneck channels = 16 * 16 = 256
    dann = DANNWrapper(segmenter, in_features=256, lambda_domain=1.0)
    x = torch.randn(2, SAR_CHANNELS, 32, 32)
    seg_logits, domain_logits = dann(x)
    assert seg_logits.shape == (2, 1, 32, 32)
    assert domain_logits.shape == (2, 1)


def test_dann_wrapper_gradients_flow_through_reversal():
    """DANN gradients flow through the GRL to the encoder (reversed)."""
    from siren.ml.model import WaterResUNet
    from siren.ml.dann import DANNWrapper, domain_loss
    from siren.ml.contract import SAR_CHANNELS

    segmenter = WaterResUNet(in_channels=SAR_CHANNELS, base_channels=16)
    dann = DANNWrapper(segmenter, in_features=256, lambda_domain=1.0)
    x = torch.randn(2, SAR_CHANNELS, 32, 32)
    seg_logits, domain_logits = dann(x)
    labels = torch.tensor([[0.0], [1.0]])
    loss = domain_loss(domain_logits, labels)
    loss.backward()
    # The encoder should have gradients (reversed through GRL)
    assert segmenter.enc1.conv1.weight.grad is not None


def test_domain_loss_computes_bce():
    """domain_loss computes binary cross-entropy on domain logits."""
    from siren.ml.dann import domain_loss

    logits = torch.tensor([[0.5], [-0.5]], requires_grad=True)
    labels = torch.tensor([[1.0], [0.0]])
    loss = domain_loss(logits, labels)
    assert loss.item() > 0
    assert loss.requires_grad
