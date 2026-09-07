"""Sentinel-1 GRD calibration — extract calibrated sigma0 VV/VH in dB from SAFE archives.

This module reads Sentinel-1 IW GRD SAFE ZIP archives, applies the ESA
calibration LUT to convert Digital Numbers (DN) to sigma0 (radar backscatter
cross-section), and converts to decibels.

The output is a 2-channel (VV, VH) float32 array in dB, matching the
WaterUNet input contract (ml/contract.py):
    - Channel 0: VV sigma0 in dB
    - Channel 1: VH sigma0 in dB
    - Value range: typically [-30, 0] dB before clamping

Calibration formula (ESA Sentinel-1 Product Definition):
    sigma0 = (DN^2) / (sigmaNought^2)
    sigma0_dB = 10 * log10(sigma0)

Where sigmaNought is the calibration LUT value at the pixel's (line, pixel)
position, interpolated from the annotation calibration vectors.

The calibrated rasters are cached to data/processed/ as GeoTIFFs so the
pipeline does not re-extract the 1.7GB SAFE archives on every run.
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

# Decimation factor for reading the 25762x16732 GRD scenes (memory bound).
# At decimation=10, the output is ~2576x1673 — sufficient for WaterUNet
# shadow inference and small enough to cache as a GeoTIFF.
DEFAULT_DECIMATION = 10


def _find_measurement_tiff(safe_zip: str, pol: str) -> str:
    """Find the measurement TIFF path for a given polarisation inside a SAFE ZIP."""
    with zipfile.ZipFile(safe_zip) as z:
        for name in z.namelist():
            if "measurement" in name and name.endswith(".tiff") and f"-{pol}-" in name:
                return name
    raise FileNotFoundError(f"No {pol} measurement TIFF found in {safe_zip}")


def _find_calibration_xml(safe_zip: str, pol: str) -> str:
    """Find the calibration annotation XML for a given polarisation."""
    with zipfile.ZipFile(safe_zip) as z:
        for name in z.namelist():
            if "calibration" in name and name.endswith(".xml") and f"-{pol}-" in name:
                return name
    raise FileNotFoundError(f"No {pol} calibration XML found in {safe_zip}")


def _parse_calibration_lut(safe_zip: str, pol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse the sigmaNought calibration LUT from a SAFE archive.

    Returns:
        (lines, pixels, sigma_nought) where:
          lines: 1D array of line indices (shape: N_lines,)
          pixels: 1D array of pixel indices (shape: N_pixels,)
          sigma_nought: 2D array of calibration values (shape: N_lines, N_pixels)
    """
    calib_xml = _find_calibration_xml(safe_zip, pol)
    with zipfile.ZipFile(safe_zip) as z:
        data = z.read(calib_xml)

    root = ET.fromstring(data)

    # Find all calibrationVector entries
    vectors = root.findall(".//{*}calibrationVector")
    if not vectors:
        raise ValueError(f"No calibration vectors found in {calib_xml}")

    lines_list = []
    pixels_list = None
    sigma_list = []

    for v in vectors:
        line_elem = v.find("{*}line")
        pixel_elem = v.find("{*}pixel")
        sigma_elem = v.find("{*}sigmaNought")

        if line_elem is None or pixel_elem is None or sigma_elem is None:
            continue

        line = int(line_elem.text)
        pixels = np.array([float(x) for x in pixel_elem.text.split()])
        sigma = np.array([float(x) for x in sigma_elem.text.split()])

        lines_list.append(line)
        if pixels_list is None:
            pixels_list = pixels
        sigma_list.append(sigma)

    lines = np.array(lines_list, dtype=np.float32)
    pixels = pixels_list.astype(np.float32)
    sigma_nought = np.array(sigma_list, dtype=np.float32)

    return lines, pixels, sigma_nought


def _interpolate_calibration(
    dn_shape: tuple[int, int],
    lines: np.ndarray,
    pixels: np.ndarray,
    sigma_nought: np.ndarray,
    decimation: int,
) -> np.ndarray:
    """Interpolate the calibration LUT to the full (decimated) image grid.

    Returns a 2D sigmaNought array matching the decimated DN image shape.
    """
    h, w = dn_shape

    # Decimated line/pixel coordinates
    dec_lines = np.arange(h) * decimation
    dec_pixels = np.arange(w) * decimation

    # Build interpolator
    interp = RegularGridInterpolator(
        (lines, pixels), sigma_nought,
        method="linear", bounds_error=False, fill_value=float(sigma_nought[0, 0])
    )

    # Create meshgrid and interpolate
    grid_lines, grid_pixels = np.meshgrid(dec_lines, dec_pixels, indexing="ij")
    points = np.stack([grid_lines.ravel(), grid_pixels.ravel()], axis=-1)
    calib_full = interp(points).reshape(h, w)

    return calib_full.astype(np.float32)


def calibrate_s1_to_db(
    safe_zip: str,
    pol: str = "vv",
    decimation: int = DEFAULT_DECIMATION,
) -> np.ndarray:
    """Read and calibrate a Sentinel-1 GRD polarisation band to sigma0 dB.

    Args:
        safe_zip: Path to the Sentinel-1 SAFE ZIP archive.
        pol: Polarisation band ("vv" or "vh").
        decimation: Read decimation factor (memory vs resolution trade-off).

    Returns:
        2D float32 array of sigma0 in dB, shape (H//dec, W//dec).
    """
    inner_tiff = _find_measurement_tiff(safe_zip, pol)

    # Read the measurement band (decimated)
    with rasterio.open(f"/vsizip/{safe_zip}/{inner_tiff}") as src:
        out_h = src.height // decimation
        out_w = src.width // decimation
        dn = src.read(1, out_shape=(out_h, out_w)).astype(np.float32)

    logger.info(f"Read {pol} DN: shape={dn.shape}, range=[{dn.min():.0f}, {dn.max():.0f}]")

    # Parse and interpolate the calibration LUT
    lines, pixels, sigma_nought = _parse_calibration_lut(safe_zip, pol)
    calib = _interpolate_calibration(dn.shape, lines, pixels, sigma_nought, decimation)

    # Calibrate: sigma0 = DN^2 / sigmaNought^2
    # Guard against division by zero
    calib_safe = np.where(calib > 0, calib, 1.0)
    sigma0 = (dn ** 2) / (calib_safe ** 2)

    # Convert to dB: sigma0_dB = 10 * log10(sigma0)
    # Guard against log of zero/negative
    sigma0 = np.where(sigma0 > 0, sigma0, 1e-10)
    sigma0_db = 10.0 * np.log10(sigma0)

    logger.info(f"Calibrated {pol} sigma0 dB: range=[{sigma0_db.min():.1f}, {sigma0_db.max():.1f}]")

    return sigma0_db.astype(np.float32)


def extract_vv_vh_db(
    safe_zip: str,
    decimation: int = DEFAULT_DECIMATION,
) -> np.ndarray:
    """Extract both VV and VH polarisations, calibrated to dB.

    Args:
        safe_zip: Path to the Sentinel-1 SAFE ZIP archive.
        decimation: Read decimation factor.

    Returns:
        float32 array of shape (2, H, W) where:
          channel 0 = VV sigma0 dB
          channel 1 = VH sigma0 dB
    """
    vv_db = calibrate_s1_to_db(safe_zip, pol="vv", decimation=decimation)
    vh_db = calibrate_s1_to_db(safe_zip, pol="vh", decimation=decimation)

    # Ensure both have the same shape (they should from the same SAFE)
    if vv_db.shape != vh_db.shape:
        # Resize VH to match VV (shouldn't happen, but guard)
        from scipy.ndimage import zoom
        zh, zw = vv_db.shape[0] / vh_db.shape[0], vv_db.shape[1] / vh_db.shape[1]
        vh_db = zoom(vh_db, (zh, zw), order=1).astype(np.float32)

    return np.stack([vv_db, vh_db], axis=0)


def extract_and_cache_vv_vh_db(
    safe_zip: str,
    cache_path: Path,
    decimation: int = DEFAULT_DECIMATION,
) -> np.ndarray:
    """Extract calibrated VV/VH dB from a SAFE archive, with disk caching.

    If the cached GeoTIFF exists, it is read directly. Otherwise, the
    SAFE archive is calibrated and the result is written to cache_path
    as a 2-band float32 GeoTIFF.

    Args:
        safe_zip: Path to the Sentinel-1 SAFE ZIP archive.
        cache_path: Path for the cached GeoTIFF.
        decimation: Read decimation factor.

    Returns:
        float32 array of shape (2, H, W) — VV/VH sigma0 in dB.
    """
    cache_path = Path(cache_path)

    if cache_path.exists():
        with rasterio.open(str(cache_path)) as src:
            data = src.read().astype(np.float32)
        logger.info(f"Loaded cached calibrated VV/VH from {cache_path}: shape={data.shape}")
        return data

    # Extract and calibrate
    data = extract_vv_vh_db(safe_zip, decimation=decimation)

    # Write cache GeoTIFF
    # We need geotransform/CRS from the original SAFE for georeferencing
    inner_tiff = _find_measurement_tiff(safe_zip, "vv")
    with rasterio.open(f"/vsizip/{safe_zip}/{inner_tiff}") as src:
        gt = list(src.transform)
        crs = src.crs
        # Scale geotransform for decimation
        gt[1] = gt[1] * decimation
        gt[5] = gt[5] * decimation

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(cache_path), "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=2,
        dtype="float32",
        crs=crs,
        transform=rasterio.transform.Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3]),
    ) as dst:
        dst.write(data[0], 1)
        dst.write(data[1], 2)
        dst.set_band_description(1, "VV sigma0 dB")
        dst.set_band_description(2, "VH sigma0 dB")

    logger.info(f"Cached calibrated VV/VH to {cache_path}: shape={data.shape}")

    return data


def find_safe_for_observation(observation_id: str, raw_dir: Path) -> str | None:
    """Find the Sentinel-1 SAFE ZIP corresponding to a demo observation.

    The demo observations map to specific dates:
      obs-001 → 2026-07-23
      obs-002 → 2026-08-04
      obs-003 → (no real scene yet — returns None)

    Args:
        observation_id: e.g. "obs-001"
        raw_dir: Path to data/raw/

    Returns:
        Path to the SAFE ZIP, or None if no matching scene exists.
    """
    # Date mapping for demo observations
    date_map = {
        "obs-001": "20260723",
        "obs-002": "20260804",
        # obs-003 target date is 2026-08-12; closest available scene is
        # 2026-08-11 (S1D track 12). However, CDSE download requires
        # credentials not available on this machine. obs-003 uses a
        # synthetic scenario mask (PRD §9.2) with provenance labeled
        # "synthetic_scenario" in DEMO_OBSERVATIONS.
        # "obs-003": "20260811",
    }

    date_str = date_map.get(observation_id)
    if date_str is None:
        return None

    # Find SAFE ZIP matching this date
    for f in Path(raw_dir).glob("S1*_*.SAFE.zip"):
        if date_str in f.name:
            return str(f)

    return None
