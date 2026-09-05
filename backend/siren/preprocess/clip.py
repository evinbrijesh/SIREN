"""Clip a raster to a basin boundary polygon (PRD §6.2).

Pure function: reads an input raster + a GeoJSON boundary, writes (or returns
a path to) a clipped raster that keeps the input CRS/grid but is masked to the
polygon extent. No network, no unseeded randomness.
"""

from __future__ import annotations

import tempfile

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from shapely.ops import unary_union


def clip_to_basin(
    raster_path: str,
    basin_geojson_path: str,
    output_path: str | None = None,
) -> str:
    """Clip a raster to a basin boundary polygon.

    Args:
        raster_path: path to input GeoTIFF.
        basin_geojson_path: path to GeoJSON file containing a Polygon geometry
            (a FeatureCollection / Feature / bare geometry is also accepted;
            non-areal geometries such as points/lines fall back to their
            bounding-box envelope so a clip mask still exists).
        output_path: where to write the clipped raster. If None, a temporary
            file is created and its path returned.

    Returns:
        Path to the clipped raster. Same CRS/grid as input, masked to polygon.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            prefix="siren_clip_", suffix=".tif", delete=False
        )
        output_path = tmp.name
        tmp.close()

    # Raster CRS, used to reproject the boundary if needed.
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    gdf = gpd.read_file(basin_geojson_path)
    if gdf.empty:
        raise ValueError(f"No geometries found in {basin_geojson_path}")
    if gdf.crs is None:
        # GeoJSON is CRS84 / WGS84 by spec; assume the raster's CRS if unset.
        gdf = gdf.set_crs(raster_crs)
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    geoms = [g for g in gdf.geometry if g is not None]
    if not geoms:
        raise ValueError(f"No valid geometries in {basin_geojson_path}")

    union = unary_union(geoms)
    # Points / lines have zero area — use their envelope as the clip polygon.
    if union.is_empty or union.area == 0:
        union = box(*union.bounds)

    with rasterio.open(raster_path) as src:
        nodata = src.nodata if src.nodata is not None else 0
        out, out_transform = mask(
            src,
            [union],
            crop=True,
            filled=True,
            nodata=nodata,
        )
        profile = src.profile.copy()
        profile.update(
            height=out.shape[-2],
            width=out.shape[-1],
            transform=out_transform,
            nodata=nodata,
        )

    with rasterio.open(output_path, "w", **profile) as dst:
        if out.ndim == 2:
            dst.write(out, 1)
        else:
            for i in range(out.shape[0]):
                dst.write(out[i], i + 1)

    return str(output_path)
