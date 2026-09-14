"""Multi-modal SAR + optical cross-attention fusion (ADR-013 §9.7.2).

Replaces single-modality SAR segmentation (capped at ~0.62 IoU by terrain
layover and speckle) with a cross-attention transformer that fuses dual-pol
Sentinel-1 SAR (all-weather) with cloud-masked Sentinel-2 multispectral
imagery (NDWI, MNDWI bands).

The cross-attention layer lets SAR features query optical features when
cloud cover is low, and smoothly rely on SAR priors when clouds are
detected. A cloud-gated attention mask handles variable optical
availability — when the cloud mask indicates clouds, the optical features
are downweighted automatically.

Target: push segmentation IoU beyond 0.80 across steep terrains.

Gate: event-held-out IoU > 0.75 AND precision ≥ 0.85 on real paired
SAR+optical data. Until this gate passes, the SAR-only 6-channel model
(ADR-011.1 gate-passed, IoU 0.62) remains the primary segmenter.

Contract:
    Input:  SAR (B, 6, H, W) + optical (B, 3, H, W) [NDWI, MNDWI, cloud_mask]
            optical can be None (cloud-blocked → SAR-only fallback)
    Output: (B, 1, H, W) water probability logits
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CloudGatedCrossAttention(nn.Module):
    """Cross-attention with cloud-gated masking (ADR-013 §9.7.2).

    SAR features (queries) attend to optical features (keys/values).
    The cloud mask modulates attention weights: clouded optical pixels
    are downweighted, forcing the model to rely on SAR priors when
    optical data is unavailable.

    Args:
        d_model: feature dimension (channel count).
        n_heads: number of attention heads.
        dropout: attention dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_proj = nn.Conv2d(d_model, d_model, 1)
        self.k_proj = nn.Conv2d(d_model, d_model, 1)
        self.v_proj = nn.Conv2d(d_model, d_model, 1)
        self.out_proj = nn.Conv2d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        sar_features: torch.Tensor,
        optical_features: torch.Tensor | None,
        cloud_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Cross-attention from SAR (queries) to optical (keys/values).

        Args:
            sar_features: (B, C, H, W) SAR encoder features (queries).
            optical_features: (B, C, H, W) optical encoder features (keys/values),
                or None if optical is unavailable (cloud-blocked).
            cloud_mask: (B, 1, H, W) cloud mask in [0, 1] where 1 = cloud.
                Used to downweight optical features in clouded regions.
                None if no cloud information.

        Returns:
            (B, C, H, W) fused features. If optical is None, returns SAR
            features unchanged (SAR-only fallback).
        """
        if optical_features is None:
            # No optical data — SAR-only fallback
            return sar_features

        B, C, H, W = sar_features.shape

        # Project queries (SAR), keys (optical), values (optical)
        q = self.q_proj(sar_features)    # (B, C, H, W)
        k = self.k_proj(optical_features)
        v = self.v_proj(optical_features)

        # Reshape for multi-head attention: (B, n_heads, head_dim, H*W)
        q = q.view(B, self.n_heads, self.head_dim, H * W)
        k = k.view(B, self.n_heads, self.head_dim, H * W)
        v = v.view(B, self.n_heads, self.head_dim, H * W)

        # Attention scores: (B, n_heads, H*W, H*W)
        scores = torch.einsum("bhdi,bhdj->bhij", q, k) / (self.head_dim ** 0.5)

        # Cloud-gated masking: downweight optical features in clouded regions
        if cloud_mask is not None:
            # cloud_mask: (B, 1, H, W) → (B, H*W)
            cloud_flat = cloud_mask.squeeze(1).view(B, 1, H * W)  # (B, 1, H*W)
            # Apply mask to keys: clouded optical pixels get -inf attention score
            # This forces the model to ignore optical features where clouds exist
            mask = cloud_flat.expand(-1, self.n_heads, -1)  # (B, n_heads, H*W)
            # Broadcast mask to (B, n_heads, H*W, H*W) — mask the key dimension
            mask = mask.unsqueeze(2).expand(-1, -1, H * W, -1)  # (B, n_heads, H*W, H*W)
            scores = scores.masked_fill(mask > 0.5, float("-inf"))

        # Softmax over key dimension (handle all-masked rows → return zeros)
        # If all keys are masked (fully clouded), softmax produces NaN.
        # Replace NaN rows with uniform attention (equivalent to no optical info).
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)  # (B, n_heads, head_dim, H*W)
        out = out.reshape(B, C, H, W)

        out = self.out_proj(out)
        return out + sar_features  # residual connection


class MultiModalFusionNet(nn.Module):
    """Multi-modal SAR + optical fusion network (ADR-013 §9.7.2).

    A dual-encoder U-Net with cross-attention fusion. The SAR encoder
    processes 6-channel SAR input; the optical encoder processes 3-channel
    optical input (NDWI, MNDWI, cloud mask). Cross-attention at the
    bottleneck fuses the two modalities, with cloud-gated masking.

    When optical data is unavailable (cloud-blocked), the model falls
    back to SAR-only (the optical encoder is skipped and cross-attention
    returns the SAR features unchanged).

    Input:
        sar: (B, 6, H, W) — VV_post, VH_post, VV_pre, VH_pre, ΔVV, ΔVH
        optical: (B, 3, H, W) or None — NDWI, MNDWI, cloud_mask
    Output:
        (B, 1, H, W) — water probability logits

    Args:
        sar_channels: number of SAR input channels (default 6).
        optical_channels: number of optical input channels (default 3).
        base_channels: base width of the encoders (default 32).
        n_heads: number of attention heads in cross-attention.
        dropout: dropout rate for MC Dropout uncertainty.
    """

    def __init__(
        self,
        sar_channels: int = 6,
        optical_channels: int = 3,
        base_channels: int = 32,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        c = base_channels

        # SAR encoder (reuses ResidualBlock from model.py)
        from siren.ml.model import ResidualBlock
        self.sar_enc1 = ResidualBlock(sar_channels, c, dropout=dropout)
        self.sar_enc2 = ResidualBlock(c, c * 2, dropout=dropout)
        self.sar_enc3 = ResidualBlock(c * 2, c * 4, dropout=dropout)
        self.sar_enc4 = ResidualBlock(c * 4, c * 8, dropout=dropout)
        self.sar_bottleneck = ResidualBlock(c * 8, c * 16, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

        # Optical encoder (lighter — fewer channels)
        opt_c = c // 2  # half width for optical
        self.opt_enc1 = ResidualBlock(optical_channels, opt_c, dropout=dropout)
        self.opt_enc2 = ResidualBlock(opt_c, opt_c * 2, dropout=dropout)
        self.opt_enc3 = ResidualBlock(opt_c * 2, opt_c * 4, dropout=dropout)
        self.opt_enc4 = ResidualBlock(opt_c * 4, opt_c * 8, dropout=dropout)
        self.opt_bottleneck = ResidualBlock(opt_c * 8, opt_c * 16, dropout=dropout)
        self.opt_pool = nn.MaxPool2d(2)

        # Project optical bottleneck to SAR bottleneck dimension for cross-attention
        self.opt_to_sar_proj = nn.Conv2d(opt_c * 16, c * 16, 1)

        # Cross-attention fusion at the bottleneck
        self.cross_attention = CloudGatedCrossAttention(
            d_model=c * 16, n_heads=n_heads, dropout=dropout,
        )

        # Decoder (same as WaterResUNet)
        from siren.ml.model import DoubleConv
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c * 8 + c * 8, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 4 + c * 4, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 2 + c * 2, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c + c, c)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional optical fusion.

        Args:
            sar: (B, 6, H, W) SAR input.
            optical: (B, 3, H, W) optical input (NDWI, MNDWI, cloud_mask),
                or None if cloud-blocked.

        Returns:
            (B, 1, H, W) water probability logits.
        """
        # SAR encoder
        s1 = self.sar_enc1(sar)
        s2 = self.sar_enc2(self.pool(s1))
        s3 = self.sar_enc3(self.pool(s2))
        s4 = self.sar_enc4(self.pool(s3))
        s_b = self.sar_bottleneck(self.pool(s4))

        # Cross-attention fusion
        if optical is not None:
            # Optical encoder
            o1 = self.opt_enc1(optical)
            o2 = self.opt_enc2(self.opt_pool(o1))
            o3 = self.opt_enc3(self.opt_pool(o2))
            o4 = self.opt_enc4(self.opt_pool(o3))
            o_b = self.opt_bottleneck(self.opt_pool(o4))

            # Project optical bottleneck to SAR dimension
            o_b_proj = self.opt_to_sar_proj(o_b)

            # Extract cloud mask (channel 2 of optical input)
            cloud_mask = optical[:, 2:3]  # (B, 1, H, W) at full resolution
            # Downsample cloud mask to bottleneck resolution
            cloud_mask_b = F.interpolate(
                cloud_mask, size=s_b.shape[-2:], mode="bilinear", align_corners=False,
            )

            # Cross-attention: SAR queries optical
            fused_b = self.cross_attention(s_b, o_b_proj, cloud_mask_b)
        else:
            # SAR-only fallback
            fused_b = s_b

        # Decoder
        d4 = self.up4(fused_b)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        return self.head(d1)

    def forward_with_features(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, bottleneck_features) for latent coupling.

        The bottleneck features are the fused SAR+optical representation
        that can be fed to the FNO via the LatentCoupler (ADR-013 §9.7.3).
        """
        # SAR encoder
        s1 = self.sar_enc1(sar)
        s2 = self.sar_enc2(self.pool(s1))
        s3 = self.sar_enc3(self.pool(s2))
        s4 = self.sar_enc4(self.pool(s3))
        s_b = self.sar_bottleneck(self.pool(s4))

        # Cross-attention fusion
        if optical is not None:
            o1 = self.opt_enc1(optical)
            o2 = self.opt_enc2(self.opt_pool(o1))
            o3 = self.opt_enc3(self.opt_pool(o2))
            o4 = self.opt_enc4(self.opt_pool(o3))
            o_b = self.opt_bottleneck(self.opt_pool(o4))
            o_b_proj = self.opt_to_sar_proj(o_b)
            cloud_mask = optical[:, 2:3]
            cloud_mask_b = F.interpolate(
                cloud_mask, size=s_b.shape[-2:], mode="bilinear", align_corners=False,
            )
            fused_b = self.cross_attention(s_b, o_b_proj, cloud_mask_b)
        else:
            fused_b = s_b

        # Decoder
        d4 = self.up4(fused_b)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        logits = self.head(d1)
        return logits, fused_b

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
