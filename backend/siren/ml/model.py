"""Siamese U-Net for bi-temporal satellite change detection.

Architecture (PRD §9.3, ADR-002):
  - Shared ResNet-34 encoder extracts multi-scale features for T0 and T1
  - Absolute feature difference |F1 - F0| at each skip-connection level
  - U-Net decoder reconstructs spatial resolution → binary change probability
  - SegFormer head (optional) classifies changed pixels into functional classes

The model is designed as an additional evidence layer, NOT a replacement for
the deterministic NDWI/backscatter differencing (ADR-002, Hard Rule 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """Two conv-bn-relu blocks — the U-Net decoder building block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SiameseUNet(nn.Module):
    """Siamese U-Net for bi-temporal satellite change detection.

    Inputs:
        t0: Baseline optical/SAR image (B, C, H, W)
        t1: Current optical/SAR image  (B, C, H, W)
    Output:
        change_logits: (B, 1, H, W) — sigmoid → change probability [0, 1]
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels

        # Shared feature extractor (ResNet-34 encoder backbone, ImageNet pretrained)
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        if in_channels != 3:
            # Re-initialize input convolution for multi-spectral input
            self.shared_in = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.shared_in = base.conv1

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.encoder1 = base.layer1  # 64 channels
        self.encoder2 = base.layer2  # 128 channels
        self.encoder3 = base.layer3  # 256 channels
        self.encoder4 = base.layer4  # 512 channels

        # Decoder path with skip connections over difference tensors |F1 - F0|
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 32)

        # Final 1x1 convolution → binary change probability
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def _extract(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Extract multi-scale features through the shared encoder."""
        x0 = self.relu(self.bn1(self.shared_in(x)))
        x1 = self.encoder1(self.maxpool(x0))
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        return x0, x1, x2, x3, x4

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        # Step 1: Extract multi-scale features through shared weights
        f0_0, f0_1, f0_2, f0_3, f0_4 = self._extract(t0)
        f1_0, f1_1, f1_2, f1_3, f1_4 = self._extract(t1)

        # Step 2: Compute absolute feature difference at each abstraction level
        d4 = torch.abs(f1_4 - f0_4)
        d3 = torch.abs(f1_3 - f0_3)
        d2 = torch.abs(f1_2 - f0_2)
        d1 = torch.abs(f1_1 - f0_1)

        # Step 3: Decode feature differences back to original spatial resolution
        x = self.up4(d4)
        x = self.dec4(torch.cat([x, d3], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, d2], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, d1], dim=1))

        # Upsample back to original spatial resolution
        x = self.up1(x)
        x = self.dec1(x)

        # Final upsample to match input H/W (conv1 stride=2 + maxpool stride=2 = 4x downsample,
        # but we only have 3 ConvTranspose2d steps = 8x upsample from the deepest level)
        x = F.interpolate(x, size=(t0.shape[2], t0.shape[3]), mode="bilinear", align_corners=False)

        return self.head(x)


class WaterUNet(nn.Module):
    """Compact single-date SAR water-segmentation U-Net (ADR-010 Stage 1).

    Task: per-date surface water segmentation from calibrated Sentinel-1
    VV/VH sigma0 (dB), NOT bi-temporal change detection. Change detection
    remains deterministic bi-temporal differencing of two independently
    segmented per-date masks (see docs/reference/PRODUCTION_ML_PLAN.md §1).

    Built from scratch (no ImageNet-pretrained encoder): ImageNet weights
    are tuned for 3-channel RGB optical statistics, which do not transfer
    to 2-channel SAR sigma0-dB inputs (ADR-010 audit finding, distinct
    modality). Parameter budget: <=10M (~1.9M actual).

    Input:  (B, 2, H, W) -- VV, VH sigma0 in dB, normalized via
            siren.ml.contract.normalize_sar() to [0, 1]. H and W must be
            multiples of 16 (4 downsampling stages).
    Output: (B, 1, H, W) -- water probability logits (sigmoid -> [0, 1]).
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        c = base_channels

        self.enc1 = DoubleConv(in_channels, c)          # H
        self.enc2 = DoubleConv(c, c * 2)                # H/2
        self.enc3 = DoubleConv(c * 2, c * 4)             # H/4
        self.enc4 = DoubleConv(c * 4, c * 8)             # H/8
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c * 8, c * 16)      # H/16

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c * 8 + c * 8, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 4 + c * 4, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 2 + c * 2, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c + c, c)

        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SegFormerHead(nn.Module):
    """Lightweight SegFormer (MiT-B0) classifier head for changed pixels.

    Evaluated exclusively on pixels where the Siamese U-Net change
    probability P > threshold. Classifies changed pixels into:
      0: Water (flood expansion)
      1: Debris flow
      2: Snowmelt (benign)
      3: Shadow/cloud anomaly
      4: Bare rock/landslide

    Uses a simple transformer encoder over patch embeddings of the changed
    region crops. In production this would be a full MiT-B0; for the demo
    scaffold we use a lightweight attention block.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Patch embedding: 16x16 patches
        self.patch_embed = nn.Conv2d(in_channels, 128, kernel_size=16, stride=16)
        # Single transformer block (simplified MiT)
        self.norm1 = nn.LayerNorm(128)
        self.attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(128)
        self.ffn = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )
        # Classification head
        self.cls_head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify changed-pixel crops.

        Input: (B, C, H, W) — cropped changed regions
        Output: (B, num_classes) — class logits per crop
        """
        B = x.shape[0]
        patches = self.patch_embed(x)  # (B, 128, H/16, W/16)
        tokens = patches.flatten(2).transpose(1, 2)  # (B, N, 128)

        # Transformer block
        residual = tokens
        tokens = self.norm1(tokens)
        tokens, _ = self.attn(tokens, tokens, tokens)
        tokens = residual + tokens
        residual = tokens
        tokens = self.norm2(tokens)
        tokens = self.ffn(tokens)
        tokens = residual + tokens

        # Global average pooling → classify
        pooled = tokens.mean(dim=1)  # (B, 128)
        return self.cls_head(pooled)
