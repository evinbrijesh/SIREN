"""Tests for windowed AOI reading (preprocess/cog.py) — Sprint 1 Step 5.

Uses the synthetic fixture rasters (100×100, EPSG:4326, spanning the
Dudh Koshi bbox) so no network or real SAFE archives are required. The
CDSE S3 URI transform and env builder are pure-string tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from siren.preprocess.cog import (
    CDSE_S3_BUCKET,
    CDSE_S3_ENDPOINT,
    aoi_window,
    cdse_s3_env,
    cdse_s3_uri_from_href,
    gcp_window,
    read_aoi_window,
)

FIX = Path(__file__).resolve().parent / "fixtures"
BASELINE = FIX / "rasters" / "baseline.tif"

# The fixture rasters span this bbox (see make_fixtures.py).
FIXTURE_BBOX = (86.65, 27.65, 87.00, 27.98)


# --------------------------------------------------------------------------- #
# aoi_window
# --------------------------------------------------------------------------- #
def test_aoi_window_inside_dataset():
    """A bbox inside the dataset yields a non-empty window."""
    with rasterio.open(BASELINE) as src:
        win = aoi_window(src, (86.7, 27.7, 86.9, 27.9))
    assert win.width > 0 and win.height > 0
    # The window must be smaller than the full 100×100 grid.
    assert win.width < 100 and win.height < 100


def test_aoi_window_full_extent_matches_whole_grid():
    """The fixture bbox covers the whole 100×100 grid."""
    with rasterio.open(BASELINE) as src:
        win = aoi_window(src, FIXTURE_BBOX)
    assert win.width <= 100 and win.height <= 100
    assert win.width >= 95 and win.height >= 95  # allow rounding


def test_aoi_window_clamps_to_dataset_bounds():
    """A bbox extending past the dataset is clamped, not errored."""
    with rasterio.open(BASELINE) as src:
        win = aoi_window(src, (86.0, 27.0, 88.0, 29.0))
    assert win.width <= 100 and win.height <= 100


def test_aoi_window_non_intersecting_raises():
    """A bbox far outside the dataset raises ValueError."""
    with rasterio.open(BASELINE) as src:
        with pytest.raises(ValueError):
            aoi_window(src, (100.0, 10.0, 101.0, 11.0))


def test_aoi_window_reprojects_bbox_crs():
    """A bbox in UTM is transformed into the dataset's CRS."""
    from rasterio.warp import transform_bounds

    # Derive a UTM bbox from the fixture's geographic bbox so it
    # guaranteed intersects the raster.
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", "EPSG:32645", 86.7, 27.7, 86.9, 27.9
    )
    with rasterio.open(BASELINE) as src:
        win_utm = aoi_window(
            src, (left, bottom, right, top), bbox_crs="EPSG:32645"
        )
        win_geo = aoi_window(src, (86.7, 27.7, 86.9, 27.9))
    # Both should select a comparable interior region.
    assert win_utm.width > 0 and win_utm.height > 0
    assert abs(win_utm.width - win_geo.width) < 30


# --------------------------------------------------------------------------- #
# gcp_window (Sentinel-1 GRD SAFE-style GCP georeferencing)
# --------------------------------------------------------------------------- #
from types import SimpleNamespace
from rasterio.control import GroundControlPoint


def _make_gcps(grid_rows, grid_cols, x0, dx, y0, dy, h, w):
    """Build a grid of GCPs mapping (row, col) → (lon, lat) like S1 GRD."""
    gcps = []
    for r in grid_rows:
        for c in grid_cols:
            x = x0 + (c / w) * dx
            y = y0 - (r / h) * dy  # rows go south → lat decreases
            gcps.append(GroundControlPoint(row=r, col=c, x=x, y=y, z=0.0))
    return gcps


def test_gcp_window_selects_aoi_region():
    """GCPs inside the bbox map to a pixel window covering that region."""
    h, w = 1000, 2000
    # GCP grid every 200 rows / 400 cols; lon 84→88, lat 28→26.
    rows = np.arange(0, h + 1, 200)
    cols = np.arange(0, w + 1, 400)
    gcps = _make_gcps(rows, cols, x0=84.0, dx=4.0, y0=28.0, dy=2.0, h=h, w=w)
    src = SimpleNamespace(height=h, width=w)
    win = gcp_window(gcps, "EPSG:4326", src, (86.6, 27.6, 87.0, 27.9))
    # Window should be a small interior region, not the whole grid.
    assert 0 < win.width < w
    assert 0 < win.height < h
    assert win.row_off >= 0 and win.col_off >= 0


def test_gcp_window_expands_by_one_cell():
    """The window is expanded by ~one GCP cell beyond the matching GCPs."""
    h, w = 1000, 2000
    rows = np.arange(0, h + 1, 200)
    cols = np.arange(0, w + 1, 400)
    gcps = _make_gcps(rows, cols, x0=84.0, dx=4.0, y0=28.0, dy=2.0, h=h, w=w)
    src = SimpleNamespace(height=h, width=w)
    # Tight bbox around a single GCP at (row=400, col=800) → lon=85.6, lat=27.2.
    win = gcp_window(gcps, "EPSG:4326", src, (85.599, 27.199, 85.601, 27.201))
    # Should include that GCP plus one-cell margin each side (~200 rows, 400 cols).
    assert win.height >= 200
    assert win.width >= 400


def test_gcp_window_non_intersecting_raises():
    h, w = 200, 200
    rows = np.arange(0, h + 1, 50)
    cols = np.arange(0, w + 1, 50)
    gcps = _make_gcps(rows, cols, x0=84.0, dx=2.0, y0=28.0, dy=2.0, h=h, w=w)
    src = SimpleNamespace(height=h, width=w)
    with pytest.raises(ValueError):
        gcp_window(gcps, "EPSG:4326", src, (100.0, 10.0, 101.0, 11.0))


def test_gcp_window_clamps_to_raster_bounds():
    """A bbox at the scene edge clamps the window to [0, h]×[0, w]."""
    h, w = 400, 400
    rows = np.arange(0, h + 1, 100)
    cols = np.arange(0, w + 1, 100)
    gcps = _make_gcps(rows, cols, x0=84.0, dx=4.0, y0=28.0, dy=4.0, h=h, w=w)
    src = SimpleNamespace(height=h, width=w)
    # bbox covering the top-left corner.
    win = gcp_window(gcps, "EPSG:4326", src, (84.0, 27.9, 85.0, 28.0))
    assert win.row_off >= 0 and win.col_off >= 0
    assert win.row_off + win.height <= h
    assert win.col_off + win.width <= w


# --------------------------------------------------------------------------- #
# read_aoi_window
# --------------------------------------------------------------------------- #
def test_read_aoi_window_returns_aoi_subset():
    """read_aoi_window reads only the AOI bbox, not the full grid."""
    data, transform, crs = read_aoi_window(
        str(BASELINE), (86.7, 27.7, 86.9, 27.9), decimation=1
    )
    assert data.ndim == 2
    assert data.dtype == np.float32
    # Sub-window of a 100×100 grid → both dims < 100.
    assert data.shape[0] < 100 and data.shape[1] < 100
    assert crs.to_string() == "EPSG:4326"
    # The window origin is the pixel boundary at or just before the bbox
    # left edge (floored to an integer pixel); the right edge is the
    # pixel boundary at or just before the bbox right. Allow one-pixel
    # slack on both sides.
    right_edge = transform.c + data.shape[1] * transform.a
    px = abs(transform.a)
    assert transform.c <= 86.7 + px
    assert right_edge >= 86.9 - px
    assert transform.f >= 27.9 - px  # top edge near bbox top (rows go south)


def test_read_aoi_window_decimation_reduces_shape():
    """Decimation shrinks the output by ~the decimation factor."""
    d1, _, _ = read_aoi_window(str(BASELINE), (86.7, 27.7, 86.9, 27.9), decimation=1)
    d4, _, _ = read_aoi_window(str(BASELINE), (86.7, 27.7, 86.9, 27.9), decimation=4)
    ratio_h = d1.shape[0] / d4.shape[0]
    ratio_w = d1.shape[1] / d4.shape[1]
    assert 3.0 <= ratio_h <= 5.0
    assert 3.0 <= ratio_w <= 5.0


def test_read_aoi_window_values_match_full_read():
    """The windowed read returns the same values as a full read at the same location."""
    data_win, transform_win, _ = read_aoi_window(
        str(BASELINE), (86.7, 27.7, 86.9, 27.9), decimation=1
    )
    with rasterio.open(BASELINE) as src:
        win = aoi_window(src, (86.7, 27.7, 86.9, 27.9))
        data_full = src.read(1, window=win).astype(np.float32)
    np.testing.assert_allclose(data_win, data_full)


def test_read_aoi_window_handles_env_options():
    """Passing env_options does not break a local read."""
    data, _, _ = read_aoi_window(
        str(BASELINE), (86.7, 27.7, 86.9, 27.9),
        env_options={"GDAL_DISABLE_READDIR_ON_OPEN": "TRUE"},
    )
    assert data.shape[0] > 0


# --------------------------------------------------------------------------- #
# cdse_s3_uri_from_href
# --------------------------------------------------------------------------- #
def test_cdse_s3_uri_basic():
    href = "https://eodata.dataspace.copernicus.eu/Sentinel-1/GRD/2026/07/23/x.tif"
    assert cdse_s3_uri_from_href(href) == f"s3://{CDSE_S3_BUCKET}/Sentinel-1/GRD/2026/07/23/x.tif"


def test_cdse_s3_uri_region_segment():
    """A region-specific eodata host (eodata.ams.… ) maps to the same bucket."""
    href = "https://eodata.ams.dataspace.copernicus.eu/Sentinel-2/MSI/y.jp2"
    assert cdse_s3_uri_from_href(href) == f"s3://{CDSE_S3_BUCKET}/Sentinel-2/MSI/y.jp2"


def test_cdse_s3_uri_non_eodata_returned_unchanged():
    """Non-eodata hrefs are returned unchanged (caller falls back to download)."""
    href = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products(x)/$value"
    assert cdse_s3_uri_from_href(href) == href


def test_cdse_s3_uri_empty_string():
    assert cdse_s3_uri_from_href("") == ""
    assert cdse_s3_uri_from_href(None) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# cdse_s3_env
# --------------------------------------------------------------------------- #
def test_cdse_s3_env_defaults(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sk")
    env = cdse_s3_env()
    assert env["AWS_VIRTUAL_HOSTING"] is False
    assert env["AWS_S3_ENDPOINT"] == CDSE_S3_ENDPOINT
    assert env["AWS_ACCESS_KEY_ID"] == "ak"
    assert env["AWS_SECRET_ACCESS_KEY"] == "sk"


def test_cdse_s3_env_explicit_credentials(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    env = cdse_s3_env(access_key_id="AK", secret_access_key="SK")
    assert env["AWS_ACCESS_KEY_ID"] == "AK"
    assert env["AWS_SECRET_ACCESS_KEY"] == "SK"


def test_cdse_s3_env_without_credentials_still_has_endpoint(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    env = cdse_s3_env()
    assert env["AWS_S3_ENDPOINT"] == CDSE_S3_ENDPOINT
    assert env["AWS_VIRTUAL_HOSTING"] is False
    assert "AWS_ACCESS_KEY_ID" not in env
