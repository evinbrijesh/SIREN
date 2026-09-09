"""Tests for geo/hand.py — Height Above Nearest Drainage (V3 §3.2, Phase 3.2).

Tests cover the HAND computation, exposure checking, and policy defaults.
The full pysheds pipeline is tested with a small synthetic DEM GeoTIFF; the
core tracing logic is tested with synthetic arrays directly.

Note: the full pysheds pipeline (compute_hand) requires numba, which has a
compatibility issue on Python 3.14. The compute_hand tests are skipped when
pysheds cannot run; the core logic tests (_trace_hand, is_exposed_by_hand,
hand_exposure_mask, water_stage_for_severity) run on all Python versions.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from siren.geo.hand import (
    compute_hand,
    _trace_hand,
    is_exposed_by_hand,
    hand_exposure_mask,
    water_stage_for_severity,
    DEFAULT_WATER_STAGE_M,
    DEFAULT_CHANNEL_THRESHOLD,
    HANDResult,
)


def _pysheds_available() -> bool:
    """Check if the pysheds pipeline can run (numba compatibility)."""
    try:
        import numpy as np
        if not hasattr(np, "in1d"):
            np.in1d = np.isin
        from pysheds.grid import Grid
        import tempfile, os
        # Create a tiny test raster and try fill_depressions
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            transform = from_origin(86.0, 28.0, 30.0, 30.0)
            with rasterio.open(
                path, "w", driver="GTiff", height=5, width=5, count=1,
                dtype="float32", crs="EPSG:4326", transform=transform,
            ) as dst:
                dst.write(np.full((5, 5), 100.0, dtype=np.float32), 1)
            grid = Grid.from_raster(path)
            dem = grid.read_raster(path)
            grid.fill_depressions(dem=dem)
            return True
        finally:
            os.unlink(path)
    except Exception:
        return False


PYSHEDS_OK = _pysheds_available()
pysheds_required = pytest.mark.skipif(
    not PYSHEDS_OK, reason="pysheds/numba pipeline unavailable on this Python version"
)


def _write_dem(path, data, transform=None):
    """Helper: write a 2D array as a single-band DEM GeoTIFF."""
    h, w = data.shape
    if transform is None:
        transform = from_origin(86.0, 28.0, 30.0, 30.0)
    with rasterio.open(
        str(path), "w",
        driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data.astype(np.float32), 1)


# --------------------------------------------------------------------------- #
# water_stage_for_severity (policy defaults, V3 §3.2)
# --------------------------------------------------------------------------- #

def test_water_stage_policy_defaults():
    """Policy default water stage heights match V3 §3.2."""
    assert DEFAULT_WATER_STAGE_M["watch"] == 0.5
    assert DEFAULT_WATER_STAGE_M["elevated"] == 2.0
    assert DEFAULT_WATER_STAGE_M["critical"] == 5.0


def test_water_stage_for_severity_valid():
    assert water_stage_for_severity("watch") == 0.5
    assert water_stage_for_severity("elevated") == 2.0
    assert water_stage_for_severity("critical") == 5.0
    assert water_stage_for_severity("informational") == 0.5


def test_water_stage_for_severity_invalid():
    with pytest.raises(ValueError, match="unknown severity"):
        water_stage_for_severity("unknown")


# --------------------------------------------------------------------------- #
# is_exposed_by_hand
# --------------------------------------------------------------------------- #

def test_is_exposed_by_hand_below_stage():
    """A cell with HAND below the water stage is exposed."""
    hand = np.array([[1.0, 3.0, 10.0]])
    assert is_exposed_by_hand(hand, 0, 0, 2.0) is True   # 1.0 ≤ 2.0
    assert is_exposed_by_hand(hand, 0, 1, 2.0) is False  # 3.0 > 2.0
    assert is_exposed_by_hand(hand, 0, 2, 2.0) is False  # 10.0 > 2.0


def test_is_exposed_by_hand_equal_stage():
    """A cell with HAND exactly equal to the water stage is exposed."""
    hand = np.array([[2.0]])
    assert is_exposed_by_hand(hand, 0, 0, 2.0) is True


def test_is_exposed_by_hand_out_of_bounds():
    """Out-of-bounds coordinates return False (not exposed)."""
    hand = np.array([[1.0]])
    assert is_exposed_by_hand(hand, -1, 0, 2.0) is False
    assert is_exposed_by_hand(hand, 0, 5, 2.0) is False


# --------------------------------------------------------------------------- #
# hand_exposure_mask
# --------------------------------------------------------------------------- #

def test_hand_exposure_mask_threshold():
    """hand_exposure_mask produces a boolean mask where HAND ≤ stage."""
    hand = np.array([[0.0, 1.0, 2.0, 3.0, 5.0]])
    mask = hand_exposure_mask(hand, 2.0)
    assert mask.tolist() == [[True, True, True, False, False]]


def test_hand_exposure_mask_zero_stage():
    """With h_water_stage=0, only channel cells (HAND=0) are exposed."""
    hand = np.array([[0.0, 0.5, 1.0]])
    mask = hand_exposure_mask(hand, 0.0)
    assert mask.tolist() == [[True, False, False]]


# --------------------------------------------------------------------------- #
# _trace_hand (core logic with synthetic arrays)
# --------------------------------------------------------------------------- #

def test_trace_hand_flat_dem_all_zero():
    """A flat DEM with all-channel produces HAND=0 everywhere."""
    fdir = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32)
    dem = np.full((3, 3), 100.0, dtype=np.float32)
    channel_mask = np.ones((3, 3), dtype=bool)
    hand = _trace_hand(fdir, dem, channel_mask)
    assert np.allclose(hand, 0.0)


def test_trace_hand_hilltop_above_channel():
    """A hilltop cell draining to a channel has HAND = elevation difference."""
    # 3x3 grid: cell (0,0) is a hilltop at 200m, drains east to (0,1) at 150m,
    # which drains to (0,2) at 100m (channel).
    fdir = np.array([
        [1, 1, 0],   # (0,0)->E, (0,1)->E, (0,2)=channel/pit
        [4, 4, 4],   # row 1 flows south
        [4, 4, 4],   # row 2 flows south
    ], dtype=np.int32)
    dem = np.array([
        [200.0, 150.0, 100.0],
        [180.0, 140.0, 90.0],
        [160.0, 130.0, 80.0],
    ], dtype=np.float32)
    channel_mask = np.zeros((3, 3), dtype=bool)
    channel_mask[0, 2] = True  # (0,2) is the channel

    hand = _trace_hand(fdir, dem, channel_mask)
    # (0,0) drains to (0,2): HAND = 200 - 100 = 100
    assert hand[0, 0] == pytest.approx(100.0)
    # (0,1) drains to (0,2): HAND = 150 - 100 = 50
    assert hand[0, 1] == pytest.approx(50.0)
    # (0,2) is channel: HAND = 0
    assert hand[0, 2] == 0.0


def test_trace_hand_channel_cells_zero():
    """All channel cells have HAND = 0."""
    fdir = np.array([[1, 1], [1, 1]], dtype=np.int32)
    dem = np.array([[100.0, 50.0], [80.0, 40.0]], dtype=np.float32)
    channel_mask = np.ones((2, 2), dtype=bool)
    hand = _trace_hand(fdir, dem, channel_mask)
    assert np.allclose(hand, 0.0)


def test_trace_hand_pit_drains_to_zero():
    """A cell draining to a pit (no channel) has HAND = 0."""
    fdir = np.array([[-1, -1], [-1, -1]], dtype=np.int32)  # all pits
    dem = np.array([[200.0, 150.0], [100.0, 50.0]], dtype=np.float32)
    channel_mask = np.zeros((2, 2), dtype=bool)
    hand = _trace_hand(fdir, dem, channel_mask)
    assert np.allclose(hand, 0.0)


def test_trace_hand_boundary_drains_to_zero():
    """A cell draining off the grid boundary has HAND = 0."""
    # (0,0) flows west (off grid), (0,1) flows west to (0,0)
    fdir = np.array([[16, 16]], dtype=np.int32)  # 16 = W
    dem = np.array([[200.0, 150.0]], dtype=np.float32)
    channel_mask = np.zeros((1, 2), dtype=bool)
    hand = _trace_hand(fdir, dem, channel_mask)
    assert np.allclose(hand, 0.0)


# --------------------------------------------------------------------------- #
# compute_hand (full pysheds pipeline with synthetic DEM)
# --------------------------------------------------------------------------- #

@pysheds_required
def test_compute_hand_flat_dem(tmp_path):
    """A flat DEM produces HAND=0 everywhere (all cells are at drainage level)."""
    dem_path = tmp_path / "flat_dem.tif"
    _write_dem(dem_path, np.full((20, 20), 100.0, dtype=np.float32))

    result = compute_hand(dem_path, channel_threshold=10.0)
    assert isinstance(result, HANDResult)
    assert result.hand.shape == (20, 20)
    assert result.hand.dtype == np.float32
    # Flat DEM → all cells drain to pits → HAND = 0
    assert np.allclose(result.hand, 0.0)


@pysheds_required
def test_compute_hand_gradient_dem(tmp_path):
    """A DEM with a gradient produces non-zero HAND on hilltops."""
    # Create a DEM that slopes from NW (high) to SE (low)
    dem_data = np.tile(
        np.linspace(200, 100, 30, dtype=np.float32), (30, 1)
    )
    dem_path = tmp_path / "gradient_dem.tif"
    _write_dem(dem_path, dem_data)

    result = compute_hand(dem_path, channel_threshold=50.0)
    assert result.hand.shape == (30, 30)
    # Some cells should have HAND > 0 (hilltops above channels)
    assert result.hand.max() > 0.0
    # Channel cells have HAND = 0
    assert result.hand[result.channel_mask].max() == 0.0


@pysheds_required
def test_compute_hand_caches_results(tmp_path):
    """compute_hand caches results per DEM path (same as corridor.py)."""
    dem_path = tmp_path / "cache_dem.tif"
    _write_dem(dem_path, np.full((15, 15), 100.0, dtype=np.float32))

    r1 = compute_hand(dem_path, channel_threshold=10.0)
    r2 = compute_hand(dem_path, channel_threshold=10.0)
    assert r1 is r2  # cached — same object


@pysheds_required
def test_compute_hand_different_thresholds_not_cached(tmp_path):
    """Different channel thresholds produce different cached results."""
    dem_path = tmp_path / "threshold_dem.tif"
    _write_dem(dem_path, np.full((15, 15), 100.0, dtype=np.float32))

    r1 = compute_hand(dem_path, channel_threshold=10.0)
    r2 = compute_hand(dem_path, channel_threshold=100.0)
    assert r1 is not r2
    assert r1.channel_threshold == 10.0
    assert r2.channel_threshold == 100.0


@pysheds_required
def test_compute_hand_channel_mask_shape(tmp_path):
    """The channel mask matches the DEM shape."""
    dem_path = tmp_path / "shape_dem.tif"
    _write_dem(dem_path, np.full((25, 30), 100.0, dtype=np.float32))

    result = compute_hand(dem_path, channel_threshold=10.0)
    assert result.channel_mask.shape == (25, 30)
    assert result.channel_mask.dtype == bool


@pysheds_required
def test_compute_hand_hand_nonnegative(tmp_path):
    """HAND values are always non-negative."""
    dem_data = np.random.RandomState(42).uniform(100, 500, (20, 20)).astype(np.float32)
    dem_path = tmp_path / "random_dem.tif"
    _write_dem(dem_path, dem_data)

    result = compute_hand(dem_path, channel_threshold=50.0)
    assert result.hand.min() >= 0.0
