from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import zipfile

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

AOI_BOUNDS = (86.64887479493386, 27.649368497466508, 87.00026368382275, 27.98047960857762)


def _png_bytes(array: np.ndarray) -> bytes:
    count, height, width = array.shape
    with MemoryFile() as memory:
        with memory.open(driver="PNG", width=width, height=height, count=count, dtype="uint8") as dst:
            dst.write(array.astype(np.uint8))
        return memory.read()


@lru_cache(maxsize=1)
def hillshade_png(project_root: str) -> bytes:
    path = Path(project_root) / "data" / "raw" / "srtm_30m.tif"
    width, height = 720, 678
    destination = np.zeros((height, width), dtype=np.float32)
    transform = from_bounds(*AOI_BOUNDS, width, height)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
        )
    dy, dx = np.gradient(destination)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(45.0)
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    shaded = np.clip((shaded + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return _png_bytes(shaded[np.newaxis, :, :])


@lru_cache(maxsize=1)
def sar_backscatter_png(project_root: str) -> bytes:
    raw = Path(project_root) / "data" / "raw"
    archive = sorted(raw.glob("S1*_20260804*.SAFE.zip"))[0]
    with zipfile.ZipFile(archive) as zipped:
        member = next(name for name in zipped.namelist() if "/measurement/" in name and "-vv-" in name and name.endswith(".tiff"))
    source_path = f"/vsizip/{archive.resolve()}/{member}"
    width, height = 720, 678
    destination = np.zeros((height, width), dtype=np.float32)
    transform = from_bounds(*AOI_BOUNDS, width, height)
    with rasterio.open(source_path) as src:
        gcps, gcp_crs = src.gcps
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            gcps=gcps,
            src_crs=gcp_crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            dst_nodata=0,
        )
    valid = destination[destination > 0]
    if valid.size == 0:
        return _png_bytes(np.zeros((1, height, width), dtype=np.uint8))
    logged = np.log1p(destination)
    low, high = np.percentile(np.log1p(valid), [2, 98])
    normalized = np.clip((logged - low) / max(high - low, 1e-6), 0, 1)
    return _png_bytes((normalized * 255).astype(np.uint8)[np.newaxis, :, :])


@lru_cache(maxsize=3)
def baseline_optical_crop_png(project_root: str, observation_id: str) -> bytes:
    processed = Path(project_root) / "data" / "processed"
    mask_path = processed / f"{observation_id}_expansion_mask.tif"
    baseline_path = processed / "basemap.tif"
    width, height = 400, 400
    with rasterio.open(mask_path) as mask_src:
        bounds = mask_src.bounds
        destination_transform = from_bounds(*bounds, width, height)
        destination_crs = mask_src.crs
    destination = np.zeros((3, height, width), dtype=np.uint8)
    with rasterio.open(baseline_path) as src:
        for band in range(1, 4):
            reproject(
                source=rasterio.band(src, band),
                destination=destination[band - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=destination_transform,
                dst_crs=destination_crs,
                resampling=Resampling.bilinear,
            )
    return _png_bytes(destination)
