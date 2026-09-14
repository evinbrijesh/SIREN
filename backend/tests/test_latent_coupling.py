"""Tests for latent spatial conditioning (ADR-013 §9.7.3).

Tests the LatentCoupler, build_fno_input, and SegmentationFNOCoupler
that connect the WaterResUNet bottleneck to the FNO input, replacing
the scalar V_breach injection with a continuous latent conditioning.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from siren.geo.hydro_surrogate import FNO2D
from siren.ml.latent_coupling import (
    LatentCoupler,
    SegmentationFNOCoupler,
    build_fno_input,
)
from siren.ml.model import WaterResUNet


# --------------------------------------------------------------------------- #
# LatentCoupler
# --------------------------------------------------------------------------- #

def test_latent_coupler_shapes():
    """LatentCoupler projects bottleneck (B, C, H/16, W/16) → (B, d, H_t, W_t)."""
    coupler = LatentCoupler(bottleneck_channels=512, d_latent=16, target_size=(64, 64))
    bottleneck = torch.randn(2, 512, 14, 14)  # 224/16 = 14
    out = coupler(bottleneck)
    assert out.shape == (2, 16, 64, 64)


def test_latent_coupler_no_target_size():
    """Without target_size, output matches bottleneck spatial size."""
    coupler = LatentCoupler(bottleneck_channels=512, d_latent=8)
    bottleneck = torch.randn(1, 512, 14, 14)
    out = coupler(bottleneck)
    assert out.shape == (1, 8, 14, 14)


def test_latent_coupler_override_target_size():
    """forward(target_size=...) overrides the constructor default."""
    coupler = LatentCoupler(bottleneck_channels=512, d_latent=16, target_size=(64, 64))
    bottleneck = torch.randn(1, 512, 14, 14)
    out = coupler(bottleneck, target_size=(128, 128))
    assert out.shape == (1, 16, 128, 128)


def test_latent_coupler_parameter_count():
    """LatentCoupler has a small parameter count (1x1 conv only)."""
    coupler = LatentCoupler(bottleneck_channels=512, d_latent=16)
    # 1x1 conv: 512 * 16 + 16 (bias) = 8208
    assert coupler.num_parameters() == 512 * 16 + 16


def test_latent_coupler_gradient_flow():
    """Gradient flows through the coupler (differentiable)."""
    coupler = LatentCoupler(bottleneck_channels=512, d_latent=16, target_size=(64, 64))
    bottleneck = torch.randn(1, 512, 14, 14, requires_grad=True)
    out = coupler(bottleneck)
    loss = out.sum()
    loss.backward()
    assert bottleneck.grad is not None
    assert bottleneck.grad.shape == bottleneck.shape


# --------------------------------------------------------------------------- #
# build_fno_input
# --------------------------------------------------------------------------- #

def test_build_fno_input_scalar_only():
    """Scalar-only path: DEM + V_breach → (B, 2, H, W)."""
    dem = torch.randn(2, 1, 64, 64)
    v_breach = torch.tensor([1e7, 2e7])
    inp = build_fno_input(dem, v_breach, latent=None, use_latent=False)
    assert inp.shape == (2, 2, 64, 64)
    # Channel 0 = DEM
    assert torch.allclose(inp[:, 0:1], dem)
    # Channel 1 = V_breach broadcast
    assert torch.allclose(inp[0, 1], torch.full((64, 64), 1e7))
    assert torch.allclose(inp[1, 1], torch.full((64, 64), 2e7))


def test_build_fno_input_latent_only():
    """Latent-only path: DEM + z_lake → (B, 1+d, H, W) (no scalar)."""
    dem = torch.randn(2, 1, 64, 64)
    latent = torch.randn(2, 16, 64, 64)
    inp = build_fno_input(dem, v_breach_scalar=None, latent=latent, use_latent=True)
    assert inp.shape == (2, 17, 64, 64)
    assert torch.allclose(inp[:, 0:1], dem)
    assert torch.allclose(inp[:, 1:], latent)


def test_build_fno_input_latent_plus_scalar():
    """Latent + scalar: DEM + V_breach + z_lake → (B, 2+d, H, W)."""
    dem = torch.randn(2, 1, 64, 64)
    v_breach = torch.tensor([1e7, 2e7])
    latent = torch.randn(2, 16, 64, 64)
    inp = build_fno_input(dem, v_breach, latent, use_latent=True)
    assert inp.shape == (2, 18, 64, 64)
    # Channel 0 = DEM, 1 = V_breach, 2..18 = latent
    assert torch.allclose(inp[:, 0:1], dem)
    # Channel 1 = V_breach broadcast as constant grid
    assert torch.allclose(inp[0, 1, 0, 0], torch.tensor(1e7))
    assert torch.allclose(inp[1, 1, 0, 0], torch.tensor(2e7))


def test_build_fno_input_latent_resized():
    """Latent is resized to match DEM grid if shapes differ."""
    dem = torch.randn(1, 1, 64, 64)
    latent = torch.randn(1, 8, 14, 14)  # different spatial size
    inp = build_fno_input(dem, None, latent, use_latent=True)
    assert inp.shape == (1, 9, 64, 64)


def test_build_fno_input_dem_3d():
    """3D DEM (B, H, W) is expanded to (B, 1, H, W)."""
    dem = torch.randn(2, 64, 64)
    v_breach = torch.tensor([1e7, 2e7])
    inp = build_fno_input(dem, v_breach, None, use_latent=False)
    assert inp.shape == (2, 2, 64, 64)


# --------------------------------------------------------------------------- #
# FNO2D with variable in_channels
# --------------------------------------------------------------------------- #

def test_fno2d_scalar_only_backward_compat():
    """FNO2D with in_channels=2 (default) works as the original scalar path."""
    fno = FNO2D(modes=8, width=16, n_points=3, n_layers=2, in_channels=2)
    x = torch.randn(2, 2, 32, 32)
    out = fno(x)
    assert out["h_water"].shape == (2, 1, 32, 32)
    assert out["t_arrival"].shape == (2, 3, 32, 32)
    assert (out["h_water"] >= 0).all()
    assert (out["t_arrival"] >= 0).all()


def test_fno2d_latent_conditioned():
    """FNO2D with in_channels=2+d_latent accepts latent-conditioned input."""
    d_latent = 16
    fno = FNO2D(modes=8, width=16, n_points=3, n_layers=2, in_channels=2 + d_latent)
    x = torch.randn(2, 2 + d_latent, 32, 32)
    out = fno(x)
    assert out["h_water"].shape == (2, 1, 32, 32)
    assert out["t_arrival"].shape == (2, 3, 32, 32)


def test_fno2d_wrong_channels_raises():
    """FNO2D raises if input channels don't match in_channels."""
    fno = FNO2D(in_channels=2)
    x = torch.randn(1, 18, 32, 32)  # wrong channel count
    with pytest.raises(AssertionError, match="Input must have 2 channels"):
        fno(x)


def test_fno2d_latent_more_parameters():
    """Latent-conditioned FNO has more parameters than scalar-only (wider input_proj)."""
    fno_scalar = FNO2D(modes=8, width=16, n_layers=2, in_channels=2)
    fno_latent = FNO2D(modes=8, width=16, n_layers=2, in_channels=18)
    assert fno_latent.num_parameters() > fno_scalar.num_parameters()


def test_fno2d_existing_checkpoint_loads():
    """Existing scalar-only FNO checkpoint loads with in_channels=2 (backward compat)."""
    import os
    ckpt_path = "models/checkpoints/fno_hydro_surrogate_v1.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("FNO checkpoint not available")
    fno = FNO2D(modes=16, width=32, n_points=3, n_layers=4, in_channels=2)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        fno.load_state_dict(checkpoint["state_dict"])
    else:
        fno.load_state_dict(checkpoint)
    # Verify it runs
    x = torch.randn(1, 2, 64, 64)
    out = fno(x)
    assert out["h_water"].shape == (1, 1, 64, 64)


# --------------------------------------------------------------------------- #
# SegmentationFNOCoupler (end-to-end)
# --------------------------------------------------------------------------- #

def test_segmentation_fno_coupler_end_to_end():
    """End-to-end: SAR → segmentation → latent → FNO produces all outputs."""
    seg = WaterResUNet(in_channels=6, base_channels=8)  # small for testing
    d_latent = 8
    coupler = LatentCoupler(
        bottleneck_channels=8 * 16,  # base_channels * 16
        d_latent=d_latent,
        target_size=(32, 32),
    )
    fno = FNO2D(modes=8, width=16, n_points=3, n_layers=2, in_channels=1 + d_latent)
    model = SegmentationFNOCoupler(seg, fno, coupler, d_latent=d_latent)

    sar = torch.randn(1, 6, 64, 64)  # 6-channel SAR input
    dem = torch.randn(1, 1, 32, 32)  # FNO grid

    out = model(sar, dem, v_breach_scalar=None, use_latent=True)

    assert "segmentation" in out
    assert "h_water" in out
    assert "t_arrival" in out
    assert "latent" in out
    assert out["segmentation"].shape == (1, 1, 64, 64)
    assert out["h_water"].shape == (1, 1, 32, 32)
    assert out["t_arrival"].shape == (1, 3, 32, 32)
    assert out["latent"].shape == (1, d_latent, 32, 32)


def test_segmentation_fno_coupler_scalar_fallback():
    """Scalar fallback path: use_latent=False reverts to scalar-only FNO."""
    seg = WaterResUNet(in_channels=6, base_channels=8)
    d_latent = 8
    coupler = LatentCoupler(bottleneck_channels=8 * 16, d_latent=d_latent)
    fno = FNO2D(modes=8, width=16, n_points=3, n_layers=2, in_channels=2)
    model = SegmentationFNOCoupler(seg, fno, coupler, d_latent=d_latent)

    sar = torch.randn(1, 6, 64, 64)
    dem = torch.randn(1, 1, 32, 32)
    v_breach = torch.tensor([1e7])

    out = model(sar, dem, v_breach_scalar=v_breach, use_latent=False)

    assert out["latent"] is None
    assert out["h_water"].shape == (1, 1, 32, 32)
    assert out["t_arrival"].shape == (1, 3, 32, 32)


def test_segmentation_fno_coupler_gradient_flow():
    """Gradient flows from FNO loss back to segmentation encoder."""
    seg = WaterResUNet(in_channels=6, base_channels=8)
    d_latent = 4
    coupler = LatentCoupler(bottleneck_channels=8 * 16, d_latent=d_latent, target_size=(32, 32))
    fno = FNO2D(modes=8, width=16, n_points=3, n_layers=2, in_channels=1 + d_latent)
    model = SegmentationFNOCoupler(seg, fno, coupler, d_latent=d_latent)

    sar = torch.randn(1, 6, 64, 64, requires_grad=True)
    dem = torch.randn(1, 1, 32, 32)

    out = model(sar, dem, use_latent=True)
    # Loss from FNO output
    loss = out["h_water"].sum() + out["t_arrival"].sum()
    loss.backward()

    # Gradient must flow back to the SAR input through the segmentation encoder
    assert sar.grad is not None
    assert sar.grad.shape == sar.shape
    # And to the segmentation model parameters
    seg_params_with_grad = sum(
        1 for p in seg.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert seg_params_with_grad > 0, "No gradient reached segmentation parameters"
    # And to the coupler parameters
    coupler_params_with_grad = sum(
        1 for p in coupler.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert coupler_params_with_grad > 0, "No gradient reached coupler parameters"
