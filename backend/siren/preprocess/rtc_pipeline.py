"""RTC pipeline integration — σ⁰ extraction + DEM co-registration + γ⁰ (Sprint 1 Step 6).

Ties together:
  * :mod:`siren.preprocess.sar_calibrate` — Sentinel-1 SAFE → σ⁰ dB (VV/VH)
  * :mod:`siren.preprocess.rtc`            — gamma-nought terrain correction
  * rasterio.warp                          — DEM reproject to the σ⁰ grid

The entry point :func:`calibrate_scene_with_rtc` takes a SAFE archive and
a DEM GeoTIFF, produces a terrain-corrected 2-band (VV, VH) γ⁰ dB GeoTIFF
on the σ⁰ grid, and returns provenance metadata for the audit trail.

This is the function the STAC daemon's ``calibrate_scene_task`` calls when
a scene's SAFE archive and a DEM are both available locally. When either
is missing the daemon marks the job ``pending`` (awaiting download / DEM),
preserving the offline-demo contract (Hard Rule 2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject as _warp_reproject

from siren.preprocess.rtc import (
    DEFAULT_PIXEL_SIZE_M,
    gamma_nought_stack,
    look_vector_from_angles,
)
from siren.preprocess.sar_calibrate import (
    DEFAULT_DECIMATION,
    _find_measurement_tiff,
    extract_incidence_and_look,
    extract_vv_vh_db,
)
from siren.preprocess.cog import calibrate_s1_window_to_db

logger = logging.getLogger(__name__)


def _dem_pixel_size_m(transform: rasterio.transform.Affine, crs: rasterio.crs.CRS) -> float:
    """Approximate DEM pixel size in metres from its transform/CRS.

    For projected CRS the transform's (a, e) directly give metres. For
    geographic CRS (e.g. EPSG:4326) the pixel width in metres varies with
    latitude; we approximate using the scene-centre latitude.
    """
    a = abs(transform.a)  # x pixel size in CRS units
    if crs.is_geographic:
        # degrees → metres at the scene-centre latitude
        _, lat, *_ = rasterio.warp.transform_bounds(
            crs, "EPSG:4326", *rasterio.coords.BoundingBox(
                transform.c, transform.f + a * 0,
                transform.c + a, transform.f,
            ).__iter__() if False else (0, 0, 0, 0),
        ) if False else (0.0, 0.0, 0.0, 0.0)
        # Simpler: use the y origin latitude.
        lat0 = transform.f
        m_per_deg = 111_320.0 * np.cos(np.radians(lat0))
        return float(a * m_per_deg)
    return float(a)


def coregister_dem_to_sar(
    safe_zip: str,
    dem_path: str,
    decimation: int = DEFAULT_DECIMATION,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS, tuple[int, int]]:
    """Reproject a DEM onto a Sentinel-1 σ⁰ grid (decimated).

    Returns:
        (dem_array, transform, crs, (out_h, out_w)) where ``dem_array`` is
        a float32 2D array on the decimated σ⁰ grid, ready for
        :func:`siren.preprocess.rtc.gamma_nought`.
    """
    inner_tiff = _find_measurement_tiff(safe_zip, "vv")
    with rasterio.open(f"/vsizip/{safe_zip}/{inner_tiff}") as src:
        out_h = src.height // decimation
        out_w = src.width // decimation
        # Scale the geotransform for decimation.
        gt = list(src.transform)
        gt[1] = gt[1] * decimation
        gt[5] = gt[5] * decimation
        sar_transform = rasterio.transform.Affine(
            gt[1], gt[2], gt[0], gt[4], gt[5], gt[3]
        )
        sar_crs = src.crs

    dem_arr = np.empty((out_h, out_w), dtype=np.float32)
    with rasterio.open(dem_path) as dem_src:
        _warp_reproject(
            source=rasterio.band(dem_src, 1),
            destination=dem_arr,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=sar_transform,
            dst_crs=sar_crs,
            resampling=Resampling.bilinear,
            src_nodata=dem_src.nodata,
            dst_nodata=dem_src.nodata,
        )
    # Replace any nodata / NaN with 0 (flat) so RTC stays finite.
    dem_arr = np.where(np.isfinite(dem_arr), dem_arr, 0.0).astype(np.float32)
    return dem_arr, sar_transform, sar_crs, (out_h, out_w)


def calibrate_scene_with_rtc(
    safe_zip: str,
    dem_path: str,
    out_path: str | Path,
    decimation: int = DEFAULT_DECIMATION,
    incidence_deg: float | None = None,
    look_azimuth_deg: float | None = None,
) -> dict[str, Any]:
    """Calibrate a Sentinel-1 scene to terrain-corrected γ⁰ dB (VV + VH).

    Pipeline:
      1. Extract σ⁰ VV/VH dB from the SAFE archive (decimated).
      2. Co-register the DEM to the σ⁰ grid (bilinear reproject).
      3. Extract incidence angle + look azimuth from the SAFE annotation
         (unless overridden by the caller).
      4. Apply :func:`gamma_nought_stack` → γ⁰ dB.
      5. Write a 2-band float32 GeoTIFF (VV, VH) to ``out_path``.

    Returns a provenance dict with the geometry, decimation, source paths,
    and per-band dB statistics — suitable for the audit log.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. σ⁰ extraction
    sigma_db = extract_vv_vh_db(safe_zip, decimation=decimation)  # (2, H, W)
    logger.info("Extracted σ⁰ dB: shape=%s", sigma_db.shape)

    # 2. DEM co-registration
    dem, sar_transform, sar_crs, (h, w) = coregister_dem_to_sar(
        safe_zip, dem_path, decimation=decimation
    )
    # Guard against shape mismatch from rounding.
    if dem.shape != sigma_db.shape[1:]:
        from scipy.ndimage import zoom
        zh = sigma_db.shape[1] / dem.shape[0]
        zw = sigma_db.shape[2] / dem.shape[1]
        dem = zoom(dem, (zh, zw), order=1).astype(np.float32)

    # 3. Geometry from annotation (or caller override)
    if incidence_deg is None or look_azimuth_deg is None:
        geom = extract_incidence_and_look(safe_zip)
        if incidence_deg is None:
            incidence_deg = geom["incidence_deg"]
        if look_azimuth_deg is None:
            look_azimuth_deg = geom["look_azimuth_deg"]

    look_vector = look_vector_from_angles(incidence_deg, look_azimuth_deg)

    # 4. RTC
    gamma_db = gamma_nought_stack(sigma_db, dem, look_vector, DEFAULT_PIXEL_SIZE_M)

    # 5. Write cache GeoTIFF
    with rasterio.open(
        str(out_path), "w",
        driver="GTiff",
        height=gamma_db.shape[1],
        width=gamma_db.shape[2],
        count=2,
        dtype="float32",
        crs=sar_crs,
        transform=sar_transform,
    ) as dst:
        dst.write(gamma_db[0], 1)
        dst.write(gamma_db[1], 2)
        dst.set_band_description(1, "VV gamma0 dB (RTC)")
        dst.set_band_description(2, "VH gamma0 dB (RTC)")

    logger.info("Wrote RTC γ⁰ dB cache to %s", out_path)

    return {
        "out_path": str(out_path),
        "safe_zip": str(safe_zip),
        "dem_path": str(dem_path),
        "decimation": decimation,
        "incidence_deg": float(incidence_deg),
        "look_azimuth_deg": float(look_azimuth_deg),
        "shape": [int(gamma_db.shape[1]), int(gamma_db.shape[2])],
        "vv_db_min": float(np.nanmin(gamma_db[0])),
        "vv_db_max": float(np.nanmax(gamma_db[0])),
        "vh_db_min": float(np.nanmin(gamma_db[1])),
        "vh_db_max": float(np.nanmax(gamma_db[1])),
        "rtc_applied": True,
    }


def calibrate_scene_with_rtc_windowed(
    safe_zip: str,
    dem_path: str,
    out_path: str | Path,
    bbox: tuple[float, float, float, float],
    bbox_crs: str = "EPSG:4326",
    decimation: int = DEFAULT_DECIMATION,
    incidence_deg: float | None = None,
    look_azimuth_deg: float | None = None,
) -> dict[str, Any]:
    """AOI-windowed RTC: read only the bbox from the SAFE + DEM, then γ⁰.

    This is the Production Roadmap §2.2 path — instead of reading the
    full ~25 000 × 17 000 S1 GRD scene, it reads only the AOI window
    (via GCP-based windowing for the SAFE and bbox-clipped windowing for
    the DEM), cutting memory and I/O to roughly the AOI's share of the
    scene footprint. The output γ⁰ dB GeoTIFF covers only the AOI.

    The DEM is reprojected onto the windowed σ⁰ grid (which has no CRS
    for S1 GRD SAFE — see :func:`calibrate_s1_window_to_db` — so the DEM
    is resampled to the σ⁰ array shape and the RTC math runs in pixel
    space). Georeferencing of the output is applied from the DEM bbox so
    downstream detect/geo steps can place the result.

    Returns the same provenance dict shape as
    :func:`calibrate_scene_with_rtc`, plus ``bbox`` and ``windowed: True``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Windowed σ⁰ extraction (VV + VH).
    vv_db, _, _ = calibrate_s1_window_to_db(
        safe_zip, "vv", bbox, bbox_crs=bbox_crs, decimation=decimation
    )
    vh_db, _, _ = calibrate_s1_window_to_db(
        safe_zip, "vh", bbox, bbox_crs=bbox_crs, decimation=decimation
    )
    sigma_db = np.stack([vv_db, vh_db], axis=0)
    logger.info("Windowed σ⁰ dB: shape=%s", sigma_db.shape)

    # 2. DEM: read the AOI window from the DEM GeoTIFF (georeferenced),
    #    then resample to the σ⁰ grid shape.
    from siren.preprocess.cog import read_aoi_window
    from scipy.ndimage import zoom

    dem_arr, dem_transform, dem_crs = read_aoi_window(
        dem_path, bbox, bbox_crs=bbox_crs, band=1, decimation=1
    )
    zh = sigma_db.shape[1] / dem_arr.shape[0]
    zw = sigma_db.shape[2] / dem_arr.shape[1]
    dem = zoom(dem_arr, (zh, zw), order=1).astype(np.float32)
    dem = np.where(np.isfinite(dem), dem, 0.0)

    # 3. Geometry from annotation (or caller override).
    if incidence_deg is None or look_azimuth_deg is None:
        geom = extract_incidence_and_look(safe_zip)
        if incidence_deg is None:
            incidence_deg = geom["incidence_deg"]
        if look_azimuth_deg is None:
            look_azimuth_deg = geom["look_azimuth_deg"]
    look_vector = look_vector_from_angles(incidence_deg, look_azimuth_deg)

    # 4. RTC.
    gamma_db = gamma_nought_stack(sigma_db, dem, look_vector, DEFAULT_PIXEL_SIZE_M)

    # 5. Write cache GeoTIFF, georeferenced from the DEM window transform
    #    scaled to the output grid.
    out_h, out_w = gamma_db.shape[1], gamma_db.shape[2]
    scale_x = dem_transform.a * (dem_arr.shape[1] / out_w)
    scale_y = dem_transform.e * (dem_arr.shape[0] / out_h)
    out_transform = rasterio.transform.Affine(
        scale_x, dem_transform.b, dem_transform.c,
        dem_transform.d, scale_y, dem_transform.f,
    )
    with rasterio.open(
        str(out_path), "w",
        driver="GTiff",
        height=out_h, width=out_w, count=2,
        dtype="float32", crs=dem_crs, transform=out_transform,
    ) as dst:
        dst.write(gamma_db[0], 1)
        dst.write(gamma_db[1], 2)
        dst.set_band_description(1, "VV gamma0 dB (RTC, windowed)")
        dst.set_band_description(2, "VH gamma0 dB (RTC, windowed)")

    logger.info("Wrote windowed RTC γ⁰ dB cache to %s (%dx%d)", out_path, out_h, out_w)

    return {
        "out_path": str(out_path),
        "safe_zip": str(safe_zip),
        "dem_path": str(dem_path),
        "decimation": decimation,
        "incidence_deg": float(incidence_deg),
        "look_azimuth_deg": float(look_azimuth_deg),
        "shape": [int(out_h), int(out_w)],
        "bbox": list(bbox),
        "vv_db_min": float(np.nanmin(gamma_db[0])),
        "vv_db_max": float(np.nanmax(gamma_db[0])),
        "vh_db_min": float(np.nanmin(gamma_db[1])),
        "vh_db_max": float(np.nanmax(gamma_db[1])),
        "rtc_applied": True,
        "windowed": True,
    }
