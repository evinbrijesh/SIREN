"""Unit tests for siren.preprocess pure functions (PRD §6.2).

Run:  cd backend && pytest tests/test_preprocess.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

from siren.preprocess.clip import clip_to_basin
from siren.preprocess.cloud_mask import cloud_mask
from siren.preprocess.coregister import co_register
from siren.preprocess.reproject import reproject

FIX = Path(__file__).resolve().parent / "fixtures"
BASELINE = FIX / "rasters" / "baseline.tif"
EXPANDED = FIX / "rasters" / "expanded_water.tif"
CLOUDY = FIX / "rasters" / "cloudy_optical.tif"
OSM = FIX / "osm" / "fake_assets.geojson"


def test_clip_to_basin_polygon():
    """Clip baseline.tif with a polygon built from the OSM assets' bbox."""
    gdf = gpd.read_file(OSM)
    minx, miny, maxx, maxy = gdf.total_bounds
    poly = box(minx, miny, maxx, maxy)
    poly_gdf = gpd.GeoDataFrame({"name": ["basin"]}, geometry=[poly], crs=gdf.crs)

    with tempfile.TemporaryDirectory() as d:
        poly_path = str(Path(d) / "basin.geojson")
        poly_gdf.to_file(poly_path, driver="GeoJSON")
        out = clip_to_basin(
            str(BASELINE), poly_path, str(Path(d) / "clipped.tif")
        )
        with rasterio.open(out) as s, rasterio.open(BASELINE) as src:
            assert s.height > 0 and s.width > 0
            assert s.crs == src.crs
            # Clipped extent is a subset of the original 100x100 grid.
            assert s.width <= src.width
            assert s.height <= src.height
            data = s.read(1)
            assert data.size > 0


def test_clip_to_basin_from_points():
    """Points-only GeoJSON falls back to the envelope; still produces a valid clip."""
    with tempfile.TemporaryDirectory() as d:
        out = clip_to_basin(str(BASELINE), str(OSM), str(Path(d) / "clipped.tif"))
        with rasterio.open(out) as s:
            assert s.height > 0 and s.width > 0
            assert s.width <= 100 and s.height <= 100


def test_reproject_identity():
    """Reprojecting EPSG:4326 -> EPSG:4326 is an identity (no error, same data)."""
    with tempfile.TemporaryDirectory() as d:
        out = reproject(str(BASELINE), "EPSG:4326", str(Path(d) / "reproj.tif"))
        with rasterio.open(out) as s, rasterio.open(BASELINE) as src:
            assert s.crs.to_string() == "EPSG:4326"
            assert s.shape == src.shape
            np.testing.assert_allclose(s.read(1), src.read(1))


def test_reproject_to_utm():
    """Reprojecting to a UTM zone yields a valid raster in that CRS."""
    with tempfile.TemporaryDirectory() as d:
        out = reproject(str(BASELINE), "EPSG:32645", str(Path(d) / "utm.tif"))
        with rasterio.open(out) as s:
            assert s.crs.to_string() == "EPSG:32645"
            assert s.height > 0 and s.width > 0


def test_co_register_same_grid():
    """expanded_water.tif shares baseline.tif's grid -> alignment_error ~0.0."""
    with tempfile.TemporaryDirectory() as d:
        out, err = co_register(
            str(BASELINE), str(EXPANDED), str(Path(d) / "aligned.tif")
        )
        assert abs(err) < 1e-9
        with rasterio.open(out) as s, rasterio.open(BASELINE) as ref, \
                rasterio.open(EXPANDED) as mov:
            assert s.shape == ref.shape
            assert s.crs == ref.crs
            np.testing.assert_allclose(np.array(s.transform), np.array(ref.transform))
            # Same grid + nearest resample -> aligned values equal the moving scene.
            np.testing.assert_allclose(s.read(1), mov.read(1))


def test_cloud_mask_fraction():
    """cloudy_optical.tif has a 50x50 cloud region -> cloud_fraction ~0.25."""
    mask, frac = cloud_mask(str(CLOUDY))
    assert mask.dtype == np.bool_
    assert mask.shape == (100, 100)
    assert abs(frac - 0.25) < 0.05
    # Cloud region is exactly the top-left 50x50 block.
    assert mask[:50, :50].all()
    assert not mask[50:, :50].any()
    assert not mask[:, 50:].any()
