"""Cloud mask from a multi-band optical raster (PRD §6.2).

Pure function: a pixel is classified as cloud when it is bright (reflectance
> bright_threshold) in ALL bands. Returns a boolean mask and the cloud
fraction in [0, 1].
"""

from __future__ import annotations

import numpy as np
import rasterio


def cloud_mask(
    optical_raster_path: str,
    bright_threshold: float = 0.5,
) -> tuple[object, float]:
    """Compute a cloud mask from an optical raster.

    Detects clouds as pixels that are bright in ALL bands
    (value > bright_threshold). Works on multi-band optical scenes
    (e.g. 2-band GREEN+NIR).

    Args:
        optical_raster_path: path to optical GeoTIFF (multi-band).
        bright_threshold: reflectance threshold for cloud detection.

    Returns:
        (mask, cloud_fraction) where mask is a numpy boolean array
        (True = cloud) and cloud_fraction is the fraction of valid pixels
        classified as cloud (0..1).
    """
    with rasterio.open(optical_raster_path) as src:
        bands = src.read().astype(np.float32)  # (count, h, w)
        nodata = src.nodata

    count, h, w = bands.shape
    bright = bands > bright_threshold  # (count, h, w)
    cloud_candidate = np.all(bright, axis=0)  # (h, w)

    if nodata is not None:
        valid = np.all(bands != nodata, axis=0)
    else:
        valid = np.ones((h, w), dtype=bool)

    mask = cloud_candidate & valid
    denom = int(valid.sum())
    cloud_fraction = float(mask.sum() / denom) if denom > 0 else 0.0
    return mask, cloud_fraction
