"""Tests for the Himalayan lake chip dataset builder.

Covers the GCP-geolocated label rasterization and lake-centred chip
extraction — the pieces that must be geographically honest (a wrong
geotransform put a baseline mask 65 km west once already).
"""

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint


def _write_sar_with_gcps(path, shape=(40, 60)):
    """GCPs map (col, row) -> (86.0 + 0.001*col, 28.0 - 0.001*row)."""
    h, w = shape
    gcps = [
        GroundControlPoint(
            row=r, col=c,
            x=86.0 + 0.001 * c, y=28.0 - 0.001 * r,
        )
        for r in range(0, h, 10)
        for c in range(0, w, 10)
    ]
    with rasterio.open(
        str(path), "w", driver="GTiff", height=h, width=w,
        count=2, dtype="float32",
    ) as dst:
        dst.write(np.zeros((2, h, w), dtype=np.float32))
        dst.gcps = (gcps, rasterio.crs.CRS.from_epsg(4326))
    return path


def _lake_geodf(centroid_lon, centroid_lat, half_deg=0.003):
    """Tiny square lake polygon around a centroid."""
    import geopandas as gpd
    from shapely.geometry import box

    geom = box(
        centroid_lon - half_deg, centroid_lat - half_deg,
        centroid_lon + half_deg, centroid_lat + half_deg,
    )
    return gpd.GeoDataFrame(
        {
            "ID": ["GL_TEST"],
            "Latitude": [centroid_lat],
            "Longitude": [centroid_lon],
            "Lake_Elev": [5000],
            "Area": [0.05],
            "geometry": [geom],
        },
        crs="EPSG:4326",
    )


class TestLakeGridPositions:
    def test_centroid_maps_to_expected_pixel(self, tmp_path):
        from siren.ml.himalayan_lake_dataset import lake_grid_positions

        sar = _write_sar_with_gcps(tmp_path / "sar.tif")
        # lon = 86.0 + 0.001*col, lat = 28.0 - 0.001*row
        # centroid at lon 86.030, lat 27.980 -> col 30, row 20
        lakes = _lake_geodf(86.030, 27.980)
        positions, lon_g, lat_g = lake_grid_positions(lakes, str(sar))
        assert len(positions) == 1
        idx, r, c = positions[0]
        assert idx == 0
        assert abs(r - 20) <= 1
        assert abs(c - 30) <= 1
        assert lon_g.shape == (40, 60)

    def test_outside_centroids_excluded(self, tmp_path):
        from siren.ml.himalayan_lake_dataset import lake_grid_positions

        sar = _write_sar_with_gcps(tmp_path / "sar.tif")
        lakes = _lake_geodf(90.0, 35.0)  # far outside the grid
        positions, _, _ = lake_grid_positions(lakes, str(sar))
        assert positions == []

    def test_missing_gcps_raises(self, tmp_path):
        from siren.ml.himalayan_lake_dataset import lake_grid_positions

        p = tmp_path / "nogcp.tif"
        with rasterio.open(
            str(p), "w", driver="GTiff", height=10, width=10,
            count=2, dtype="float32",
        ) as dst:
            dst.write(np.zeros((2, 10, 10), dtype=np.float32))
        with pytest.raises(ValueError, match="no GCPs"):
            lake_grid_positions(_lake_geodf(86.0, 28.0), str(p))


class TestRasterizeLakeLabels:
    def test_polygon_burns_expected_pixels(self, tmp_path):
        from siren.ml.himalayan_lake_dataset import (
            lake_grid_positions,
            rasterize_lake_labels,
        )

        sar = _write_sar_with_gcps(tmp_path / "sar.tif")
        # half_deg=0.003 -> 3 px half-side -> ~7x7 px label around (20, 30)
        lakes = _lake_geodf(86.030, 27.980, half_deg=0.003)
        positions, _, _ = lake_grid_positions(lakes, str(sar))
        label = rasterize_lake_labels(lakes, positions, str(sar))
        assert label.shape == (40, 60)
        assert label[20, 30] == 1          # centroid inside polygon
        assert label[0, 0] == 0            # far corner outside
        assert 9 <= label.sum() <= 81      # ~7x7 expected

    def test_outside_lakes_burn_nothing(self, tmp_path):
        from siren.ml.himalayan_lake_dataset import (
            lake_grid_positions,
            rasterize_lake_labels,
        )

        sar = _write_sar_with_gcps(tmp_path / "sar.tif")
        lakes = _lake_geodf(90.0, 35.0)
        positions, _, _ = lake_grid_positions(lakes, str(sar))
        label = rasterize_lake_labels(lakes, positions, str(sar))
        assert label.sum() == 0


class TestExtractLakeChips:
    def test_chip_centered_on_lake(self, tmp_path):
        from siren.ml.himalayan_lake_dataset import extract_lake_chips

        h, w = 40, 60
        tensor = np.ones((6, h, w), dtype=np.float32)
        label = np.zeros((h, w), dtype=np.uint8)
        label[15:26, 25:36] = 1  # 11x11 lake at (20, 30)
        lakes = _lake_geodf(86.030, 27.980)
        positions = [(0, 20, 30)]
        x, y, chips = extract_lake_chips(
            tensor, label, lakes, positions, chip=20, background_per=0
        )
        assert x.shape == (1, 6, 20, 20)
        assert y.shape == (1, 20, 20)
        assert chips[0].pos_px == 121
        assert chips[0].lake_id == "GL_TEST"
        assert chips[0].kind == "lake"
        # lake fills the centre of the chip
        assert y[0, 10, 10] == 1
        assert y[0, 0, 0] == 0

    def test_edge_lakes_skipped(self):
        from siren.ml.himalayan_lake_dataset import extract_lake_chips

        h, w = 40, 60
        tensor = np.ones((6, h, w), dtype=np.float32)
        label = np.zeros((h, w), dtype=np.uint8)
        lakes = _lake_geodf(86.030, 27.980)
        # centroid at (2, 2) — chip of 20 would run off the edge
        x, y, chips = extract_lake_chips(
            tensor, label, lakes, [(0, 2, 2)], chip=20, background_per=0
        )
        assert len(chips) == 0
        assert x.shape == (0, 6, 20, 20)

    def test_background_chips_have_no_positives(self):
        from siren.ml.himalayan_lake_dataset import extract_lake_chips

        h, w = 40, 60
        tensor = np.ones((6, h, w), dtype=np.float32)
        label = np.zeros((h, w), dtype=np.uint8)
        label[15:26, 25:36] = 1
        lakes = _lake_geodf(86.030, 27.980)
        x, y, chips = extract_lake_chips(
            tensor, label, lakes, [(0, 20, 30)], chip=8,
            background_stride=8, background_per=3,
        )
        bg = [c for c in chips if c.kind == "background"]
        assert len(bg) <= 3
        for c in bg:
            assert c.pos_px == 0
        bg_idx = [chips.index(c) for c in bg]
        for i in bg_idx:
            assert y[i].sum() == 0
