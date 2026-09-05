"""Reproject a raster to a target CRS (PRD §6.2).

Pure function: uses rasterio.warp to reproject the input GeoTIFF into the
target CRS, choosing a sensible default output grid. Nearest-neighbour
resampling keeps synthetic fixture values exact and is deterministic.
"""

from __future__ import annotations

import tempfile

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject as _warp_reproject


def reproject(
    raster_path: str,
    target_crs: str,
    output_path: str | None = None,
) -> str:
    """Reproject a raster to a target CRS.

    Args:
        raster_path: path to input GeoTIFF.
        target_crs: e.g. 'EPSG:4326' or 'EPSG:32645'.
        output_path: where to write the reprojected raster. If None, a
            temporary file is created.

    Returns:
        Path to the reprojected raster.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            prefix="siren_reproj_", suffix=".tif", delete=False
        )
        output_path = tmp.name
        tmp.close()

    with rasterio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for b in range(src.count):
                dst_buf = np.empty((height, width), dtype=src.dtypes[b])
                _warp_reproject(
                    source=rasterio.band(src, b + 1),
                    destination=dst_buf,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata,
                )
                dst.write(dst_buf, b + 1)

    return str(output_path)
