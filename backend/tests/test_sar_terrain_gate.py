"""Tests for SAR-grid terrain gating helpers (detect/sar.py).

Covers:
- dem_slope() numeric correctness (metres per PIXEL, not per degree —
  the bug that produced ~1e-5× slopes and silently disabled gating)
- sar_grid_lonlat() GCP geolocation (None on ungeoreferenced caches)
- sar_grid_polygon_mask() polygon rasterisation onto a geolocated grid
- sar_grid_sample() geographic raster resampling onto the SAR grid
"""

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_origin

from siren.detect.sar import (
    dem_slope,
    sar_grid_lonlat,
    sar_grid_polygon_mask,
    sar_grid_sample,
)


def _write_dem(path, dem, transform=None, crs="EPSG:4326", nodata=None):
    transform = transform or from_origin(86.0, 28.0, 0.001, 0.001)
    with rasterio.open(
        str(path), "w", driver="GTiff",
        height=dem.shape[0], width=dem.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(dem[np.newaxis].astype(np.float32))


def _write_sar_with_gcps(path, shape=(20, 30)):
    """Small SAR cache analog: 2 bands + GCPs forming a known affine.

    GCPs map (col, row) -> (86.0 + 0.001*col, 28.0 - 0.001*row), matching
    the DEM fixtures. 20 GCPs (5x4 grid) -> GDAL uses a cubic polynomial,
    which reproduces an affine field exactly.
    """
    h, w = shape
    gcps = [
        GroundControlPoint(row=r, col=c, x=86.0 + 0.001 * c, y=28.0 - 0.001 * r)
        for r in np.linspace(0, h - 1, 4)
        for c in np.linspace(0, w - 1, 5)
    ]
    with rasterio.open(
        str(path), "w", driver="GTiff", height=h, width=w,
        count=2, dtype="float32",
    ) as dst:
        dst.gcps = (gcps, rasterio.crs.CRS.from_epsg(4326))
        dst.write(np.zeros((2, h, w), dtype=np.float32))


class TestDemSlope:
    def test_flat_terrain_zero_slope(self, tmp_path):
        dem = np.full((50, 50), 5000.0, dtype=np.float32)
        p = tmp_path / "dem.tif"
        _write_dem(p, dem)
        slope, ds = dem_slope(str(p))
        ds.close()
        assert np.nanmax(slope) == pytest.approx(0.0, abs=1e-6)

    def test_45_degree_ramp(self, tmp_path):
        # Pixel pitch ~0.001 deg lon ≈ 103.9 m at lat 28; build a ramp
        # rising by exactly that run per pixel -> slope ≈ 45°.
        res_deg = 0.001
        run_m = 111_320.0 * np.cos(np.deg2rad(28.0)) * res_deg
        dem = np.tile(np.arange(50) * run_m, (50, 1)).astype(np.float32)
        p = tmp_path / "dem.tif"
        _write_dem(p, dem, transform=from_origin(86.0, 28.0, res_deg, res_deg))
        slope, ds = dem_slope(str(p))
        ds.close()
        interior = slope[2:-2, 2:-2]
        assert np.nanmean(interior) == pytest.approx(45.0, abs=1.0)

    def test_slope_not_per_degree_bug(self, tmp_path):
        # Regression: dividing per-pixel gradient by m/degree shrinks the
        # slope ~1e5x. A real 30° ramp must NOT read as ~0.
        res_deg = 0.001
        run_m = 110_540.0 * res_deg
        dem = np.tile(
            (np.arange(60) * run_m * np.tan(np.deg2rad(30.0)))[:, np.newaxis],
            (1, 60),
        ).astype(np.float32)
        p = tmp_path / "dem.tif"
        _write_dem(p, dem)
        slope, ds = dem_slope(str(p))
        ds.close()
        assert np.nanmean(slope[5:-5, 5:-5]) == pytest.approx(30.0, abs=1.5)


class TestSarGridLonlat:
    def test_none_without_gcps(self, tmp_path):
        p = tmp_path / "sar.tif"
        with rasterio.open(
            str(p), "w", driver="GTiff", height=10, width=10,
            count=2, dtype="float32",
        ) as dst:
            dst.write(np.zeros((2, 10, 10), dtype=np.float32))
        assert sar_grid_lonlat(str(p)) is None

    def test_affine_field_reproduced(self, tmp_path):
        p = tmp_path / "sar.tif"
        _write_sar_with_gcps(p)
        ll = sar_grid_lonlat(str(p), stride=2)
        assert ll is not None
        lon, lat = ll
        assert lon.shape == (20, 30)
        # GCP polynomial must reproduce the affine: corner + center checks
        assert lon[0, 0] == pytest.approx(86.0, abs=2e-3)
        assert lat[0, 0] == pytest.approx(28.0, abs=2e-3)
        assert lon[19, 29] == pytest.approx(86.029, abs=2e-3)
        assert lat[19, 29] == pytest.approx(27.981, abs=2e-3)


class TestSarGridPolygonMask:
    def test_polygon_hits_expected_pixels(self, tmp_path):
        sar = tmp_path / "sar.tif"
        _write_sar_with_gcps(sar)
        lon, lat = sar_grid_lonlat(str(sar), stride=2)

        geojson = tmp_path / "poly.geojson"
        # Box covering cols 10-19, rows 0-9 of the SAR grid
        geojson.write_text(
            '{"type":"FeatureCollection","features":[{"type":"Feature",'
            '"properties":{},"geometry":{"type":"Polygon","coordinates":'
            '[[[86.0095,27.9905],[86.0205,27.9905],[86.0205,28.0005],'
            '[86.0095,28.0005],[86.0095,27.9905]]]}}]}'
        )
        mask = sar_grid_polygon_mask(str(geojson), lon, lat)
        assert mask.dtype == bool
        assert mask[5, 15]          # inside
        assert not mask[5, 5]       # left of box
        assert not mask[15, 15]     # below box
        assert mask.sum() > 0

    def test_empty_vector_returns_zeros(self, tmp_path):
        sar = tmp_path / "sar.tif"
        _write_sar_with_gcps(sar)
        lon, lat = sar_grid_lonlat(str(sar), stride=2)
        geojson = tmp_path / "empty.geojson"
        geojson.write_text(
            '{"type":"FeatureCollection","features":[{"type":"Feature",'
            '"properties":{},"geometry":{"type":"Polygon","coordinates":'
            '[[[80.0,20.0],[80.1,20.0],[80.1,20.1],[80.0,20.1],'
            '[80.0,20.0]]]}}]}'
        )
        mask = sar_grid_polygon_mask(str(geojson), lon, lat)
        assert not mask.any()


class TestSarGridSample:
    def test_samples_source_raster(self, tmp_path):
        sar = tmp_path / "sar.tif"
        _write_sar_with_gcps(sar)
        lon, lat = sar_grid_lonlat(str(sar), stride=2)

        # Source raster on the same EPSG:4326 frame: ramp = col index
        src_arr = np.tile(np.arange(60, dtype=np.float32), (40, 1))
        src = tmp_path / "src.tif"
        _write_dem(src, src_arr, transform=from_origin(86.0, 28.0, 0.001, 0.001))

        sampled = sar_grid_sample(str(src), lon, lat)
        assert sampled.shape == lon.shape
        # SAR (row=10, col=15) -> (86.0155, 27.9895) -> src row≈10, col≈15
        assert sampled[10, 15] == pytest.approx(15.0, abs=1.5)
        assert sampled[0, 0] == pytest.approx(0.0, abs=1.5)

    def test_outside_extent_gets_fill(self, tmp_path):
        sar = tmp_path / "sar.tif"
        _write_sar_with_gcps(sar)
        lon, lat = sar_grid_lonlat(str(sar), stride=2)
        # Source raster far away (different extent entirely)
        far = tmp_path / "far.tif"
        _write_dem(
            far, np.ones((10, 10), np.float32),
            transform=from_origin(80.0, 20.0, 0.001, 0.001),
        )
        sampled = sar_grid_sample(str(far), lon, lat, fill=-1.0)
        assert (sampled == -1.0).all()
