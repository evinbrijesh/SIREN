"""Sentinel-2 L2A optical preprocessing for multi-modal fusion (ADR-013 §9.7.2).

Extracts and computes the optical features needed by MultiModalFusionNet:
  - NDWI  = (B03 - B08) / (B03 + B08)   [Green - NIR]
  - MNDWI = (B03 - B11) / (B03 + B11)   [Green - SWIR1]
  - Cloud mask from SCL (classes 8, 9, 10 = cloud/cirrus, 3 = cloud shadow)

All bands are co-registered to the SAR/DEM grid so the optical features
can be stacked with the 6-channel SAR input for cross-attention fusion.

Contract:
    Input:  S2 L2A SAFE zip path + reference grid (CRS, transform, shape)
    Output: (3, H, W) array — channel 0 = NDWI, 1 = MNDWI, 2 = cloud_mask
            all at the reference grid resolution.

The SCL cloud mask uses the L2A Scene Classification Layer:
    Class 0  = No data
    Class 1  = Saturated/defective
    Class 2  = Dark area pixels
    Class 3  = Cloud shadows      → cloud_mask = 1
    Class 4  = Vegetation
    Class 5  = Bare soils
    Class 6  = Water
    Class 7  = Unclassified
    Class 8  = Cloud medium prob   → cloud_mask = 1
    Class 9  = Cloud high prob     → cloud_mask = 1
    Class 10 = Cirrus              → cloud_mask = 1
    Class 11 = Snow/ice
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject as _warp_reproject
from rasterio.warp import calculate_default_transform

logger = logging.getLogger(__name__)

# SCL classes that indicate cloud or cloud shadow
CLOUD_SCL_CLASSES = {3, 8, 9, 10}  # cloud shadow, cloud medium, cloud high, cirrus


def _find_band_path(safe_zip: zipfile.ZipFile, band_name: str, resolution: str = "10m") -> str | None:
    """Find the path to a specific band inside an S2 L2A SAFE zip.

    Args:
        safe_zip: open ZipFile for the S2 SAFE archive.
        band_name: band identifier (e.g., "B03", "B08", "B11", "SCL").
        resolution: target resolution ("10m", "20m", "60m").

    Returns:
        The path within the zip, or None if not found.
    """
    for name in safe_zip.namelist():
        if "IMG_DATA" in name and name.endswith(".jp2"):
            parts = name.split("/")
            if band_name in parts[-1] and resolution in parts[-1]:
                return name
    return None


def _read_band_from_zip(
    safe_zip: zipfile.ZipFile, band_path: str,
) -> tuple[np.ndarray, dict]:
    """Read a single band from a SAFE zip as a numpy array.

    Returns the array and a metadata dict with CRS, transform, bounds.
    """
    with safe_zip.open(band_path) as f:
        data = rasterio.open(f)
        array = data.read(1)
        meta = {
            "crs": data.crs,
            "transform": data.transform,
            "bounds": data.bounds,
            "width": data.width,
            "height": data.height,
            "nodata": data.nodata,
        }
    return array, meta


def extract_optical_features(
    safe_zip_path: str | Path,
    target_crs: str | None = None,
    target_bounds: tuple[float, float, float, float] | None = None,
    target_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Extract NDWI, MNDWI, and cloud mask from an S2 L2A SAFE archive.

    Reads B03 (Green), B08 (NIR), B11 (SWIR1), and SCL from the SAFE zip,
    computes the water indices and cloud mask, and optionally reprojects
    to a target grid.

    Args:
        safe_zip_path: path to the S2 L2A SAFE zip file.
        target_crs: target CRS for reprojection (e.g., "EPSG:32645").
            If None, uses the native CRS of B03.
        target_bounds: (left, bottom, right, top) in target_crs coordinates.
            If None, uses the native bounds.
        target_shape: (height, width) of the output grid.
            If None, uses the native resolution.

    Returns:
        Dict with:
            'ndwi': (H, W) Normalized Difference Water Index [-1, 1]
            'mndwi': (H, W) Modified NDWI [-1, 1]
            'cloud_mask': (H, W) binary mask (1 = cloud/shadow, 0 = clear)
            'meta': dict with CRS, transform, bounds of the output
    """
    safe_zip_path = Path(safe_zip_path)
    if not safe_zip_path.exists():
        raise FileNotFoundError(f"S2 SAFE zip not found: {safe_zip_path}")

    with zipfile.ZipFile(str(safe_zip_path), "r") as safe_zip:
        # Find band paths
        b03_path = _find_band_path(safe_zip, "B03", "10m")
        b08_path = _find_band_path(safe_zip, "B08", "10m")
        b11_path = _find_band_path(safe_zip, "B11", "20m")
        scl_path = _find_band_path(safe_zip, "SCL", "20m")

        if not all([b03_path, b08_path, b11_path, scl_path]):
            missing = [
                name for name, path in [
                    ("B03", b03_path), ("B08", b08_path),
                    ("B11", b11_path), ("SCL", scl_path),
                ] if path is None
            ]
            raise ValueError(f"Missing bands in SAFE archive: {missing}")

        logger.info("Reading bands from %s", safe_zip_path.name)
        b03, meta_b03 = _read_band_from_zip(safe_zip, b03_path)
        b08, meta_b08 = _read_band_from_zip(safe_zip, b08_path)
        b11_20, meta_b11 = _read_band_from_zip(safe_zip, b11_path)
        scl_20, meta_scl = _read_band_from_zip(safe_zip, scl_path)

    # Compute NDWI = (B03 - B08) / (B03 + B08)
    # B03 and B08 are both 10m, same grid
    ndwi = np.where(
        (b03 + b08) != 0,
        (b03.astype(np.float32) - b08.astype(np.float32)) / (b03.astype(np.float32) + b08.astype(np.float32) + 1e-10),
        0.0,
    )

    # Compute MNDWI = (B03 - B11) / (B03 + B11)
    # B11 is 20m — need to resample to 10m to match B03
    b11_10 = np.zeros_like(b03, dtype=np.float32)
    _warp_reproject(
        source=b11_20.astype(np.float32),
        destination=b11_10,
        src_transform=meta_b11["transform"],
        src_crs=meta_b11["crs"],
        dst_transform=meta_b03["transform"],
        dst_crs=meta_b03["crs"],
        resampling=Resampling.bilinear,
    )
    mndwi = np.where(
        (b03 + b11_10) != 0,
        (b03.astype(np.float32) - b11_10) / (b03.astype(np.float32) + b11_10 + 1e-10),
        0.0,
    )

    # Parse SCL to binary cloud mask
    # SCL is 20m — resample to 10m using nearest-neighbor (it's a classification)
    scl_10 = np.zeros_like(b03, dtype=np.uint8)
    _warp_reproject(
        source=scl_20.astype(np.uint8),
        destination=scl_10,
        src_transform=meta_scl["transform"],
        src_crs=meta_scl["crs"],
        dst_transform=meta_b03["transform"],
        dst_crs=meta_b03["crs"],
        resampling=Resampling.nearest,
    )
    cloud_mask = np.isin(scl_10, list(CLOUD_SCL_CLASSES)).astype(np.float32)

    # Reproject to target grid if specified
    if target_crs is not None and target_bounds is not None and target_shape is not None:
        target_height, target_width = target_shape
        target_transform = from_bounds(*target_bounds, width=target_width, height=target_height)

        ndwi_out = np.zeros((target_height, target_width), dtype=np.float32)
        mndwi_out = np.zeros((target_height, target_width), dtype=np.float32)
        cloud_out = np.zeros((target_height, target_width), dtype=np.float32)

        for src, dst, resamp in [
            (ndwi, ndwi_out, Resampling.bilinear),
            (mndwi, mndwi_out, Resampling.bilinear),
            (cloud_mask, cloud_out, Resampling.nearest),
        ]:
            _warp_reproject(
                source=src,
                destination=dst,
                src_transform=meta_b03["transform"],
                src_crs=meta_b03["crs"],
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=resamp,
            )

        ndwi, mndwi, cloud_mask = ndwi_out, mndwi_out, cloud_out
        meta = {
            "crs": target_crs,
            "transform": target_transform,
            "bounds": target_bounds,
            "width": target_width,
            "height": target_height,
        }
    else:
        meta = meta_b03

    cloud_fraction = float(cloud_mask.mean())
    logger.info(
        "Optical features extracted: NDWI range [%.3f, %.3f], MNDWI range [%.3f, %.3f], "
        "cloud fraction: %.1f%%",
        ndwi.min(), ndwi.max(), mndwi.min(), mndwi.max(), cloud_fraction * 100,
    )

    return {
        "ndwi": ndwi,
        "mndwi": mndwi,
        "cloud_mask": cloud_mask,
        "cloud_fraction": cloud_fraction,
        "meta": meta,
    }


def build_optical_input(
    safe_zip_path: str | Path,
    reference_raster_path: str | Path,
) -> np.ndarray:
    """Build the 3-channel optical input for MultiModalFusionNet.

    Extracts NDWI, MNDWI, and cloud mask from the S2 SAFE archive and
    co-registers them to the reference raster's grid (typically the SAR
    or DEM grid).

    Args:
        safe_zip_path: path to the S2 L2A SAFE zip.
        reference_raster_path: path to the reference raster (SAR or DEM)
            whose grid will be used for co-registration.

    Returns:
        (3, H, W) float32 array:
            channel 0 = NDWI
            channel 1 = MNDWI
            channel 2 = cloud_mask (1 = cloud/shadow, 0 = clear)
    """
    with rasterio.open(reference_raster_path) as ref:
        ref_crs = ref.crs
        ref_bounds = ref.bounds
        ref_shape = (ref.height, ref.width)

    features = extract_optical_features(
        safe_zip_path,
        target_crs=str(ref_crs),
        target_bounds=ref_bounds,
        target_shape=ref_shape,
    )

    optical_input = np.stack([
        features["ndwi"],
        features["mndwi"],
        features["cloud_mask"],
    ], axis=0).astype(np.float32)

    logger.info(
        "Optical input built: shape=%s, cloud fraction=%.1f%%",
        optical_input.shape, features["cloud_fraction"] * 100,
    )
    return optical_input
