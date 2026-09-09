"""Copernicus DEM GLO-30 tile fetcher for Sen1Floods11 chips (Level 2).

Downloads 1°×1° Copernicus DEM GLO-30 tiles from the AWS Open Data bucket
(``copernicus-dem-30m.s3.amazonaws.com``) and merges them into a VRT or
reprojects directly to a chip's grid.

The Copernicus DEM GLO-30 is a global digital surface model at ~30 m
resolution, derived from the TanDEM-X mission. It is the recommended DEM
for the 4-channel terrain-aware WaterResUNet (ADR-011 / V3 §2.1).

Tiles are cached locally under ``data/raw/dem/copernicus_glo30/`` so that
repeated dataset builds do not re-download the same tiles.

Sprint 2 / Level 2: this module provides the real DEM data source that the
previous scaffolding lacked. The local SRTM tile only covers the Nepal
basin and cannot serve chips in Ghana, Nigeria, Pakistan, etc.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# AWS Open Data bucket for Copernicus DEM GLO-30
COPERNICUS_DEM_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

# Local cache directory
DEM_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "dem" / "copernicus_glo30"


def _tile_name(lat_floor: int, lon_floor: int) -> str:
    """Generate the Copernicus DEM tile name for a 1°×1° cell.

    The naming convention is:
        Copernicus_DSM_COG_10_{N|S}XX_YY_{E|W}ZZZ_WW_DEM

    where XX_YY is the latitude (with decimal separator) and ZZZ_WW is the
    longitude, both referring to the **southwest** (lower-left) corner.

    Examples:
        lat=0, lon=6  → N00_00_E006_00
        lat=-1, lon=6 → S01_00_E006_00
        lat=27, lon=86 → N27_00_E086_00
    """
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    lat_abs = abs(lat_floor)
    lon_abs = abs(lon_floor)
    return f"Copernicus_DSM_COG_10_{ns}{lat_abs:02d}_00_{ew}{lon_abs:03d}_00_DEM"


def _tile_url(lat_floor: int, lon_floor: int) -> str:
    """Full HTTPS URL for a Copernicus DEM tile."""
    name = _tile_name(lat_floor, lon_floor)
    return f"{COPERNICUS_DEM_BASE_URL}/{name}/{name}.tif"


def _tile_cache_path(lat_floor: int, lon_floor: int) -> Path:
    """Local cache path for a Copernicus DEM tile."""
    name = _tile_name(lat_floor, lon_floor)
    return DEM_CACHE_DIR / f"{name}.tif"


def download_tile(lat_floor: int, lon_floor: int, timeout: int = 60) -> Path:
    """Download a single Copernicus DEM GLO-30 tile (with caching).

    Args:
        lat_floor: Floor of the latitude (e.g. 27 for 27.5°N).
        lon_floor: Floor of the longitude (e.g. 86 for 86.3°E).
        timeout: Network timeout in seconds.

    Returns:
        Path to the cached tile GeoTIFF.
    """
    cache_path = _tile_cache_path(lat_floor, lon_floor)
    if cache_path.exists():
        return cache_path

    DEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    url = _tile_url(lat_floor, lon_floor)
    logger.info("Downloading Copernicus DEM tile: %s", _tile_name(lat_floor, lon_floor))
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def tiles_for_bounds(
    west: float, south: float, east: float, north: float,
) -> list[tuple[int, int]]:
    """Determine which 1°×1° DEM tiles are needed for a bounding box.

    Args:
        west, south, east, north: Bounding box in EPSG:4326 degrees.

    Returns:
        List of (lat_floor, lon_floor) tile coordinates.
    """
    lat_min = int(np.floor(south))
    lat_max = int(np.floor(north))
    lon_min = int(np.floor(west))
    lon_max = int(np.floor(east))

    tiles = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            tiles.append((lat, lon))
    return tiles


def fetch_dem_for_bounds(
    west: float, south: float, east: float, north: float,
) -> list[Path]:
    """Download all DEM tiles covering a bounding box (with caching).

    Args:
        west, south, east, north: Bounding box in EPSG:4326 degrees.

    Returns:
        List of paths to the downloaded/cached tile GeoTIFFs.
    """
    tile_coords = tiles_for_bounds(west, south, east, north)
    paths = []
    for lat_floor, lon_floor in tile_coords:
        try:
            path = download_tile(lat_floor, lon_floor)
            paths.append(path)
        except Exception as e:
            logger.warning(
                "Failed to download DEM tile %s: %s",
                _tile_name(lat_floor, lon_floor), e,
            )
    return paths


def build_dem_vrt(
    tile_paths: list[Path],
    vrt_path: Path,
) -> Path:
    """Merge multiple DEM tiles into a single VRT for reprojection.

    Args:
        tile_paths: List of DEM tile GeoTIFF paths.
        vrt_path: Output VRT path.

    Returns:
        Path to the VRT file.
    """
    import rasterio
    from rasterio.merge import merge as rio_merge  # noqa: F401
    from rasterio.vrt import WarpedVRT  # noqa: F401

    # Use gdal.BuildVRT via rasterio's bindings
    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    # Write a simple VRT using GDAL's Python bindings
    try:
        from osgeo import gdal
        tile_strs = [str(p) for p in tile_paths]
        vrt_options = gdal.BuildVRTOptions()
        ds = gdal.BuildVRT(str(vrt_path), tile_strs, options=vrt_options)
        if ds is not None:
            ds = None  # flush
            return vrt_path
    except ImportError:
        pass

    # Fallback: use rasterio.merge to create a merged GeoTIFF
    import rasterio
    from rasterio.merge import merge

    src_files = [rasterio.open(str(p)) for p in tile_paths]
    try:
        mosaic, transform = merge(src_files)
        profile = src_files[0].profile.copy()
        profile.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
        })
        with rasterio.open(str(vrt_path), "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for f in src_files:
            f.close()
    return vrt_path


def coregister_dem_to_grid(
    dem_paths: list[Path],
    target_crs: str,
    target_transform,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Reproject merged DEM tiles to a chip's exact grid.

    This replaces the single-file ``coregister_dem_to_chip`` in
    ``dataset.py`` with a multi-tile version that can handle chips
    spanning multiple 1°×1° DEM tiles.

    Args:
        dem_paths: List of DEM tile GeoTIFF paths to merge + reproject.
        target_crs: Target CRS (e.g. 'EPSG:4326').
        target_transform: Target affine transform (rasterio Affine).
        target_shape: (H, W) of the target grid.

    Returns:
        2D float32 array of DEM elevation in metres, co-registered to
        the target grid.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.merge import merge

    if len(dem_paths) == 0:
        # No DEM data available — return zeros (will be flagged as invalid)
        return np.zeros(target_shape, dtype=np.float32)

    if len(dem_paths) == 1:
        dem_src = rasterio.open(str(dem_paths[0]))
        sources = [dem_src]
    else:
        sources = [rasterio.open(str(p)) for p in dem_paths]

    try:
        # Merge tiles into a single mosaic
        if len(sources) > 1:
            mosaic, mosaic_transform = merge(sources)
            dem_data = mosaic[0]  # (H, W)
            dem_crs = sources[0].crs
        else:
            dem_data = sources[0].read(1)
            mosaic_transform = sources[0].transform
            dem_crs = sources[0].crs

        # Reproject to target grid
        dem_out = np.empty(target_shape, dtype=np.float32)
        reproject(
            source=dem_data,
            destination=dem_out,
            src_transform=mosaic_transform,
            src_crs=dem_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
        return dem_out
    finally:
        for s in sources:
            s.close()


def pixel_size_m_from_transform(
    transform,
    crs: str,
    lat: float,
) -> tuple[float, float]:
    """Compute pixel size in metres from a geographic transform.

    For EPSG:4326 (geographic), converts degree pixel spacing to metres
    using the latitude-dependent conversion:
        1° latitude  ≈ 111,320 m
        1° longitude ≈ 111,320 * cos(lat) m

    For projected CRS, returns the transform's native pixel size directly.

    Args:
        transform: rasterio Affine transform.
        crs: CRS string (e.g. 'EPSG:4326').
        lat: Center latitude of the chip (for geographic CRS conversion).

    Returns:
        (dx_east_m, dy_north_m) pixel size in metres.
    """
    dx = abs(transform.a)
    dy = abs(transform.e)

    if "4326" in str(crs) or "4326" in str(crs).upper():
        # Geographic CRS — convert degrees to metres
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat))
        dx_m = dx * m_per_deg_lon
        dy_m = dy * m_per_deg_lat
        return (float(dx_m), float(dy_m))
    else:
        # Projected CRS — already in metres
        return (float(dx), float(dy))
