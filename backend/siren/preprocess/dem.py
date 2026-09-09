"""DEM utilities — slope derivation and terrain channel preparation (V3 §2.2).

This module is the canonical home for DEM-derived terrain channels used by
the 4-channel tensor contract (ADR-011 / V3 §2.1). The core slope/normal
computation lives in :mod:`siren.preprocess.rtc` (implemented in Sprint 1
Step 6 for Radiometric Terrain Correction); this module re-exports those
primitives so the ML dataset pipeline has a single DEM entry point and
adds convenience functions for reading a DEM raster + computing the
co-registered slope channel.

Sprint 2 tasks served:
    - V3 §2.2: ``slope_degrees()`` reusable function (re-exported from rtc)
    - V3 §2.2: DEM + slope co-registration convenience for chip grids
    - V3 §2.1: channel 2 (DEM) + channel 3 (Slope) preparation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Re-export the core terrain primitives from rtc.py (Sprint 1 Step 6).
# These are the canonical implementations; this module is the ML-facing
# entry point so the dataset pipeline does not import from rtc directly.
from siren.preprocess.rtc import (  # noqa: F401
    DEFAULT_PIXEL_SIZE_M,
    PixelSize,
    slope_degrees,
    surface_normal,
)

logger = __import__("logging").getLogger(__name__)


def load_dem(dem_path: str | Path) -> tuple[np.ndarray, float]:
    """Read a DEM GeoTIFF and return (elevation_m, pixel_size_m).

    Args:
        dem_path: Path to a DEM raster (e.g. SRTM 30 m or Copernicus GLO-30).

    Returns:
        (elevation, pixel_size_m) where elevation is a 2D float32 array in
        metres and pixel_size_m is the pixel spacing in metres (derived
        from the raster's transform; assumes square pixels).
    """
    import rasterio

    with rasterio.open(str(dem_path)) as src:
        dem = src.read(1).astype(np.float32)
        # Pixel size from the affine transform (dx = a, dy = -e for N-up rasters)
        dx = abs(src.transform.a)
        dy = abs(src.transform.e)
        pixel_size_m = float((dx + dy) / 2.0) if dx > 0 and dy > 0 else DEFAULT_PIXEL_SIZE_M
    return dem, pixel_size_m


def compute_slope_channel(
    dem_path: str | Path,
    target_shape: tuple[int, int] | None = None,
    pixel_size_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a DEM and compute the slope-in-degrees channel for the 4-channel tensor.

    Args:
        dem_path: Path to the DEM raster.
        target_shape: Optional (H, W) to resample the DEM to (e.g. a Sen1Floods11
            chip grid). If None, the DEM is used at native resolution.
        pixel_size_m: Optional pixel size override. If None, derived from the
            raster transform (or the resampled grid if target_shape is given).

    Returns:
        (dem_m, slope_deg) — two 2D float32 arrays at the target resolution:
            - dem_m: elevation in metres [0, 8848]
            - slope_deg: slope angle in degrees [0, 90]
    """
    dem, dem_px = load_dem(dem_path)

    if target_shape is not None and dem.shape != target_shape:
        from scipy.ndimage import zoom
        zh = target_shape[0] / dem.shape[0]
        zw = target_shape[1] / dem.shape[1]
        dem = zoom(dem, (zh, zw), order=1).astype(np.float32)
        # Adjust pixel size for the resampled grid
        if pixel_size_m is None:
            pixel_size_m = dem_px / max(zh, zw)

    if pixel_size_m is None:
        pixel_size_m = dem_px

    slope = slope_degrees(dem, pixel_size_m)
    return dem, slope


def stack_terrain_channels(
    dem_m: np.ndarray,
    slope_deg: np.ndarray,
) -> np.ndarray:
    """Stack DEM + Slope into the (2, H, W) terrain portion of the 4-channel tensor.

    This produces channels 2-3 of the 4-channel contract. The caller is
    responsible for normalisation via :func:`siren.ml.contract.normalize_tensor`
    after concatenating with the SAR channels.

    Args:
        dem_m: 2D float32 array of elevation in metres.
        slope_deg: 2D float32 array of slope in degrees.

    Returns:
        float32 array of shape (2, H, W) — [DEM_m, Slope_deg] (unnormalized).
    """
    if dem_m.shape != slope_deg.shape:
        raise ValueError(
            f"dem_m and slope_deg must have the same shape, got "
            f"{dem_m.shape} vs {slope_deg.shape}"
        )
    return np.stack([dem_m.astype(np.float32), slope_deg.astype(np.float32)], axis=0)
