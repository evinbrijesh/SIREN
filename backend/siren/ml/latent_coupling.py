"""Latent spatial conditioning: segmentation bottleneck → FNO input (ADR-013 §9.7.3).

Replaces the scalar V_breach injection into the FNO with a continuous latent
spatial conditioning from the segmentation network's bottleneck. This creates
a differentiable path from raw radar bytes to downstream flood dynamics,
preserving the spatial structure of the lake basin (shoreline steepness,
moraine outlet width, basin geometry) that a 1D scalar discards.

Contract:
    WaterResUNet bottleneck: (B, C_bottleneck, H/16, W/16)
    → LatentCoupler projects + upsamples
    → (B, d_latent, H_fno, W_fno) concatenated with DEM channel
    → FNO2D input: (B, 1 + d_latent, H_fno, W_fno)  [DEM + projected z_lake]

The scalar V_breach path is retained as a labeled fallback (d_latent=0 reverts
to the original 2-channel FNO contract).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LatentCoupler(nn.Module):
    """Project segmentation bottleneck to FNO input grid (ADR-013 §9.7.3).

    Takes the bottleneck embedding z_lake from WaterResUNet
    (B, C_bottleneck, H/16, W/16) and produces a spatial latent conditioning
    (B, d_latent, H_target, W_target) for the FNO input.

    The projection is a 1×1 convolution (channel mixing) followed by bilinear
    upsampling to the FNO grid size. This preserves the spatial structure of
    the lake basin while reducing the channel dimension to the FNO's latent
    width.

    Args:
        bottleneck_channels: number of channels in the segmentation bottleneck
            (WaterResUNet: base_channels * 16 = 512 for base_channels=32).
        d_latent: dimension of the latent conditioning vector (number of
            channels fed to the FNO alongside the DEM). 0 reverts to the
            scalar-only FNO contract (no latent conditioning).
        target_size: (H_target, W_target) spatial size of the FNO input grid.
            If None, the coupler outputs at the bottleneck resolution and the
            FNO is responsible for resizing.
    """

    def __init__(
        self,
        bottleneck_channels: int = 512,
        d_latent: int = 16,
        target_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.d_latent = d_latent
        self.target_size = target_size

        # 1×1 conv to project bottleneck channels → d_latent channels
        self.projection = nn.Conv2d(
            bottleneck_channels, d_latent, kernel_size=1, bias=True,
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Project and upsample the segmentation bottleneck.

        Args:
            bottleneck: (B, C_bottleneck, H_b, W_b) from WaterResUNet._encode().
            target_size: override (H_target, W_target) for the FNO grid.
                If None, uses self.target_size or the bottleneck size.

        Returns:
            (B, d_latent, H_target, W_target) projected latent conditioning.
        """
        # Project channels: (B, C_bottleneck, H_b, W_b) → (B, d_latent, H_b, W_b)
        z = self.projection(bottleneck)

        # Determine target size
        size = target_size or self.target_size
        if size is not None:
            # Bilinear upsample to FNO grid
            z = F.interpolate(
                z, size=size, mode="bilinear", align_corners=False,
            )
        return z

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_fno_input(
    dem: torch.Tensor,
    v_breach_scalar: torch.Tensor | None,
    latent: torch.Tensor | None,
    use_latent: bool = True,
) -> torch.Tensor:
    """Assemble the FNO input tensor from DEM + scalar/latent conditioning.

    This is the bridge between the segmentation pipeline and the FNO. It
    constructs the input tensor that FNO2D.forward() expects:

    - Latent path (use_latent=True, latent provided):
        (B, 1 + d_latent, H, W) — channel 0 = DEM, channels 1..d_latent = z_lake
    - Scalar fallback (use_latent=False or latent is None):
        (B, 2, H, W) — channel 0 = DEM, channel 1 = V_breach (broadcast)
    - Latent + scalar (both provided, use_latent=True):
        (B, 2 + d_latent, H, W) — DEM + V_breach + z_lake

    The scalar V_breach is always included as a fallback channel when
    available, so the FNO can learn to use it alongside the latent. When
    the latent is unavailable (e.g., segmentation model not loaded), the
    FNO reverts to the scalar-only 2-channel contract.

    Args:
        dem: (B, 1, H, W) or (B, H, W) normalized DEM grid.
        v_breach_scalar: (B,) or (B, 1) scalar V_breach value, or None.
        latent: (B, d_latent, H, W) projected latent conditioning, or None.
        use_latent: whether to include the latent channels.

    Returns:
        (B, C, H, W) FNO input tensor where C depends on the path.
    """
    # Ensure DEM is (B, 1, H, W)
    if dem.dim() == 3:
        dem = dem.unsqueeze(1)
    elif dem.dim() == 4 and dem.shape[1] != 1:
        dem = dem[:, :1]  # take first channel if multi-channel DEM

    channels = [dem]
    B, _, H, W = dem.shape

    # Add scalar V_breach as a constant grid (fallback channel)
    if v_breach_scalar is not None:
        if v_breach_scalar.dim() == 1:
            v_breach_scalar = v_breach_scalar.view(-1, 1, 1, 1)
        elif v_breach_scalar.dim() == 2:
            v_breach_scalar = v_breach_scalar.view(-1, 1, 1, 1)
        v_grid = v_breach_scalar.expand(-1, 1, H, W)
        channels.append(v_grid)

    # Add latent conditioning
    if use_latent and latent is not None:
        # Ensure latent matches the DEM grid size
        if latent.shape[-2:] != (H, W):
            latent = F.interpolate(
                latent, size=(H, W), mode="bilinear", align_corners=False,
            )
        channels.append(latent)

    return torch.cat(channels, dim=1)


class SegmentationFNOCoupler(nn.Module):
    """End-to-end coupler: WaterResUNet bottleneck → latent-conditioned FNO.

    This module wraps the segmentation model + latent coupler + FNO into a
    single differentiable pipeline. It enables gradient flow from the FNO
    loss back to the segmentation encoder, which is the key architectural
    change of ADR-013 §9.7.3.

    Args:
        segmentation_model: WaterResUNet (or compatible) with forward_with_features().
        fno_model: FNO2D (or compatible) with forward().
        latent_coupler: LatentCoupler projecting the bottleneck to d_latent.
        d_latent: latent dimension (must match the FNO's expected input).
    """

    def __init__(
        self,
        segmentation_model: nn.Module,
        fno_model: nn.Module,
        latent_coupler: LatentCoupler,
        d_latent: int = 16,
    ) -> None:
        super().__init__()
        self.segmentation_model = segmentation_model
        self.fno_model = fno_model
        self.latent_coupler = latent_coupler
        self.d_latent = d_latent

    def forward(
        self,
        sar_input: torch.Tensor,
        dem: torch.Tensor,
        v_breach_scalar: torch.Tensor | None = None,
        fno_grid_size: tuple[int, int] | None = None,
        use_latent: bool = True,
    ) -> dict[str, torch.Tensor]:
        """End-to-end forward: SAR → segmentation → latent → FNO.

        Args:
            sar_input: (B, C_sar, H_sar, W_sar) input to the segmentation model.
            dem: (B, 1, H_fno, W_fno) normalized DEM for the FNO grid.
            v_breach_scalar: (B,) optional scalar V_breach for fallback channel.
            fno_grid_size: (H_fno, W_fno) target FNO grid size.
            use_latent: if True, use latent conditioning; if False, scalar-only.

        Returns:
            Dict with:
                'segmentation': (B, 1, H_sar, W_sar) water mask logits
                'h_water': (B, 1, H_fno, W_fno) predicted water depth
                't_arrival': (B, n_points, H_fno, W_fno) arrival times
                'latent': (B, d_latent, H_fno, W_fno) projected latent (or None)
        """
        # 1. Segmentation forward with bottleneck extraction
        logits, bottleneck = self.segmentation_model.forward_with_features(sar_input)

        # 2. Latent conditioning (or None for scalar fallback)
        latent = None
        if use_latent:
            target_size = fno_grid_size or (dem.shape[-2], dem.shape[-1])
            latent = self.latent_coupler(bottleneck, target_size=target_size)

        # 3. Assemble FNO input
        fno_input = build_fno_input(dem, v_breach_scalar, latent, use_latent=use_latent)

        # 4. FNO forward
        fno_output = self.fno_model(fno_input)

        return {
            "segmentation": logits,
            "h_water": fno_output["h_water"],
            "t_arrival": fno_output["t_arrival"],
            "latent": latent,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
