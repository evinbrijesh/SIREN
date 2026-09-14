"""Neural bathymetry inversion (ADR-013 §9.7.1).

Replaces the Huggel et al. (2002) empirical area-volume power law with a
neural network that reconstructs submerged lake bed topography z_bed(x, y)
from surrounding sub-aerial moraine DEM contours and lake boundary shape
features.

Glacial lake beds are carved by ice dynamics. A small U-Net trained on
glacial bed inversion datasets (Millan et al. 2022 consensus ice-thickness,
Farinotti et al. 2019 ITMIX 2) predicts the submerged bed elevation map
ẑ_bed(x, y). The breach volume is then computed by hypsometric integration
over the neural bed prediction:

    V_breach = Σ max(0, z_surface − ẑ_bed(x, y)) · pixel_area

This replaces the scalar Huggel formula V = 0.104 · A^1.421 with a spatially-
resolved neural prediction that preserves lake basin geometry.

Training data requirement: Millan/Farinotti consensus ice-thickness estimates
or surveyed lake bathymetry. Until the gate (< 15% MAPE on held-out lakes)
passes, the Huggel formula remains the primary path in breach_volume.py.

Contract:
    Input:  (B, 2, H, W) — channel 0 = DEM (lake region masked), channel 1 = lake mask
    Output: (B, 1, H, W) — predicted bed elevation ẑ_bed(x, y)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BathymetryUNet(nn.Module):
    """Small U-Net for glacial lake bed elevation prediction (ADR-013 §9.7.1).

    Predicts the submerged lake bed topography z_bed(x, y) from the
    surrounding moraine DEM and lake boundary. The architecture is a compact
    U-Net (3–5M params) with 2 downsampling stages, suitable for the small
    spatial extent of glacial lakes (~1–5 km).

    Input:  (B, 2, H, W) — channel 0 = DEM (lake region set to 0 or water
                            surface elevation), channel 1 = lake boundary mask
    Output: (B, 1, H, W) — predicted bed elevation ẑ_bed(x, y)

    Args:
        in_channels: number of input channels (2 = DEM + lake mask).
        base_channels: base width of the encoder (default 32).
        n_down: number of downsampling stages (default 3).
        dropout: dropout rate for MC Dropout uncertainty (default 0.0).
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        n_down: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        c = base_channels

        # Encoder
        self.enc1 = _ConvBlock(in_channels, c, dropout=dropout)
        self.enc2 = _ConvBlock(c, c * 2, dropout=dropout)
        self.enc3 = _ConvBlock(c * 2, c * 4, dropout=dropout) if n_down >= 3 else None
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        bottleneck_channels = c * 4 if n_down >= 3 else c * 2
        self.bottleneck = _ConvBlock(bottleneck_channels, bottleneck_channels * 2, dropout=dropout)

        # Decoder
        if n_down >= 3:
            self.up3 = nn.ConvTranspose2d(bottleneck_channels * 2, bottleneck_channels, 2, stride=2)
            self.dec3 = _ConvBlock(bottleneck_channels * 2, bottleneck_channels)
            self.up2 = nn.ConvTranspose2d(bottleneck_channels, c * 2, 2, stride=2)
            self.dec2 = _ConvBlock(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = _ConvBlock(c * 2, c)
        else:
            self.up2 = nn.ConvTranspose2d(bottleneck_channels * 2, c * 2, 2, stride=2)
            self.dec2 = _ConvBlock(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = _ConvBlock(c * 2, c)
            self.dec3 = None
            self.up3 = None

        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict bed elevation.

        Args:
            x: (B, 2, H, W) — DEM (lake masked) + lake boundary mask.

        Returns:
            (B, 1, H, W) — predicted bed elevation ẑ_bed(x, y).
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        if self.enc3 is not None:
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.up3(b)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
        else:
            b = self.bottleneck(self.pool(e2))
            d2 = self.up2(b)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _ConvBlock(nn.Module):
    """Double conv block with optional dropout (for BathymetryUNet)."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


@dataclass
class NeuralBathymetryResult:
    """Output of neural bathymetry inversion.

    Attributes:
        z_bed: (H, W) predicted bed elevation grid.
        z_surface: water surface elevation (from shoreline rim).
        v_breach_m3: breach volume from hypsometric integration over z_bed.
        lake_area_m2: lake surface area.
        pixel_area_m2: area of one pixel in m².
        provenance: estimation provenance tag.
        method: "neural_bathymetry".
        model_version: version of the bathymetry model used.
    """

    z_bed: np.ndarray
    z_surface: float
    v_breach_m3: float
    lake_area_m2: float
    pixel_area_m2: float
    provenance: str = "neural_bathymetry_v1"
    method: str = "neural_bathymetry"
    model_version: str = "bathymetry_unet_v0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "z_surface": round(self.z_surface, 2),
            "v_breach_m3": round(self.v_breach_m3, 1),
            "lake_area_m2": round(self.lake_area_m2, 1),
            "pixel_area_m2": self.pixel_area_m2,
            "provenance": self.provenance,
            "method": self.method,
            "model_version": self.model_version,
        }


def predict_bed_elevation(
    model: BathymetryUNet,
    dem: np.ndarray,
    lake_mask: np.ndarray,
    z_surface: float,
    pixel_area_m2: float,
    device: torch.device | str = "cpu",
) -> NeuralBathymetryResult:
    """Run neural bathymetry inversion and compute breach volume.

    Args:
        model: trained BathymetryUNet.
        dem: (H, W) DEM grid (lake region will be masked to 0 internally).
        lake_mask: (H, W) binary lake mask.
        z_surface: water surface elevation (shoreline rim elevation).
        pixel_area_m2: area of one pixel in m².
        device: torch device for inference.

    Returns:
        NeuralBathymetryResult with predicted bed elevation and breach volume.
    """
    model.eval()
    dem_arr = np.asarray(dem, dtype=np.float32)
    mask_arr = np.asarray(lake_mask, dtype=np.float32)

    # Prepare input: DEM with lake region masked to 0, + lake mask channel
    dem_masked = dem_arr.copy()
    dem_masked[mask_arr > 0.5] = 0.0  # mask the lake region

    # Normalize DEM to [0, 1] using the shoreline elevation as reference
    dem_min = float(dem_arr[mask_arr < 0.5].min()) if (mask_arr < 0.5).any() else 0.0
    dem_max = float(dem_arr.max())
    if dem_max > dem_min:
        dem_norm = (dem_masked - dem_min) / (dem_max - dem_min)
    else:
        dem_norm = np.zeros_like(dem_masked)

    # Stack inputs: (1, 2, H, W)
    x = np.stack([dem_norm, mask_arr], axis=0)[np.newaxis]  # (1, 2, H, W)
    x_tensor = torch.from_numpy(x).to(device)

    with torch.no_grad():
        z_bed_norm = model(x_tensor).cpu().numpy()[0, 0]  # (H, W)

    # Denormalize: the model predicts bed elevation relative to the DEM range
    z_bed = z_bed_norm * (dem_max - dem_min) + dem_min

    # Only use bed elevation inside the lake
    z_bed_lake = z_bed.copy()
    z_bed_lake[mask_arr < 0.5] = z_surface  # outside lake, set to surface

    # Hypsometric integration: V = Σ max(0, z_surface - z_bed) * pixel_area
    depths = np.maximum(0.0, z_surface - z_bed_lake)
    lake_pixels = mask_arr > 0.5
    v_breach = float(depths[lake_pixels].sum() * pixel_area_m2)
    lake_area = float(lake_pixels.sum() * pixel_area_m2)

    return NeuralBathymetryResult(
        z_bed=z_bed,
        z_surface=float(z_surface),
        v_breach_m3=v_breach,
        lake_area_m2=lake_area,
        pixel_area_m2=pixel_area_m2,
    )


def generate_synthetic_bathymetry_data(
    n_samples: int = 100,
    grid_size: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic bathymetry data for architecture testing.

    Creates synthetic DEM + lake mask + bed elevation triples for testing
    the BathymetryUNet architecture. The synthetic beds follow a parabolic
    basin shape (typical of glacially-carved lakes) with random parameters.

    This is for ARCHITECTURE TESTING ONLY — real training requires Millan/
    Farinotti consensus ice-thickness data (PRD §9.7.1).

    Args:
        n_samples: number of synthetic samples.
        grid_size: spatial grid size.
        seed: random seed for reproducibility.

    Returns:
        Tuple of (dems, lake_masks, bed_elevations) each (N, H, W).
    """
    rng = np.random.default_rng(seed)
    dems = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    lake_masks = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    beds = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)

    for i in range(n_samples):
        # Random lake center and radius
        cy = rng.integers(grid_size // 4, 3 * grid_size // 4)
        cx = rng.integers(grid_size // 4, 3 * grid_size // 4)
        radius = rng.integers(grid_size // 6, grid_size // 3)

        # Lake mask: circular
        yy, xx = np.ogrid[:grid_size, :grid_size]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        lake_mask = (dist < radius).astype(np.float32)

        # Shoreline elevation (random)
        z_rim = float(rng.uniform(4000, 6000))

        # DEM: flat terrain at z_rim, with a parabolic basin under the lake
        dem = np.full((grid_size, grid_size), z_rim, dtype=np.float32)

        # Bed elevation: parabolic basin (z_bed = z_rim - max_depth * (1 - (r/R)^2))
        max_depth = float(rng.uniform(20, 100))  # 20-100 m deep
        r_norm = dist / radius
        bed_depth = max_depth * np.maximum(0, 1 - r_norm ** 2)
        bed = z_rim - bed_depth

        # Add terrain noise outside the lake
        noise = rng.normal(0, 5, (grid_size, grid_size)).astype(np.float32)
        dem[lake_mask < 0.5] += noise[lake_mask < 0.5]

        dems[i] = dem
        lake_masks[i] = lake_mask
        beds[i] = bed

    return dems, lake_masks, beds
