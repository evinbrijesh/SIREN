"""Optical NDWI water detection (OpenCode-owned, Phase 2).

Computes the NDWI water mask from a Sentinel-2 L2A scene, clipped to the AOI.
NDWI = (Green - NIR) / (Green + NIR), using B03 (green) and B08 (NIR).

This is the OPTICAL path (used when skies are clear). The monsoon observation
is cloud-blocked and routes to SAR (ADR-003); the S2 scene provides the clean
post-monsoon baseline water extent.

CRITICAL: the S2 tile is 10980x10980 (~120M px). Always clip to the AOI window
before computing NDWI to keep memory bounded.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import from_bounds

# AOI in UTM 45N (matches data/assets/dudh_koshi_aoi.geojson)
AOI_BOUNDS_UTM = (465476, 3058430, 500000, 3095036)  # left, bottom, right, top
# NDWI threshold for "water" (isolates water from shadowed terrain)
WATER_THRESHOLD = 0.2


def read_s2_band(s2_zip: str, granule: str, band: str, resolution: str = "R10m") -> tuple[np.ndarray, dict]:
    """Read a single S2 band clipped to the AOI window.

    Returns (array, profile). band is e.g. 'B03' or 'B08'.

    The band filename is <tile_id>_<datetime>_<band>_<res>.jp2, where
    tile_id + datetime is the granule's product prefix (e.g.
    'T45RVL_20251122T045131'). The granule path is passed in full.
    """
    # Derive the product prefix from the granule path: the last segment is
    # 'L2A_T45RVL_A006338_20251122T045408'; the band filename uses the
    # product-level tile prefix 'T45RVL_20251122T045131' (from the SAFE name).
    safe_name = s2_zip.rsplit("/", 1)[-1].replace(".SAFE.zip", "")
    # safe_name = 'S2C_MSIL2A_20251122T045131_N0511_R076_T45RVL_20251122T083010'
    # split: [S2C, MSIL2A, 20251122T045131, N0511, R076, T45RVL, 20251122T083010]
    parts = safe_name.split("_")
    tile_prefix = f"{parts[5]}_{parts[2]}"  # T45RVL_20251122T045131

    inner = (
        f"{s2_zip}/{granule}/IMG_DATA/{resolution}/"
        f"{tile_prefix}_{band}_{resolution[1:]}.jp2"  # dir 'R10m', file suffix '10m'
    )
    with rasterio.open(f"/vsizip/{inner}") as src:
        win = from_bounds(*AOI_BOUNDS_UTM, transform=src.transform)
        return src.read(1, window=win), src.profile


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDWI from green and NIR reflectance (0-10000 scaled)."""
    g = green.astype(np.float32) / 10000.0
    n = nir.astype(np.float32) / 10000.0
    return (g - n) / (g + n + 1e-10)


def water_mask(ndwi: np.ndarray, threshold: float = WATER_THRESHOLD) -> np.ndarray:
    """Threshold NDWI into a boolean water mask."""
    return ndwi > threshold


def baseline_water_mask(s2_zip: str, granule: str, threshold: float = WATER_THRESHOLD) -> tuple[np.ndarray, dict]:
    """Compute the baseline water mask from an S2 scene.

    Returns (water_mask_bool, profile). The profile carries the AOI-clipped
    transform for writing the mask to a GeoTIFF.
    """
    green, profile = read_s2_band(s2_zip, granule, "B03")
    nir, _ = read_s2_band(s2_zip, granule, "B08")
    nd = ndwi(green, nir)
    return water_mask(nd, threshold), profile