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
