"""Tests for the Level 2 Phase 2 HAND post-filter pipeline.

Tests cover:
  - D8 flow direction computation (pure numpy)
  - Flow accumulation
  - HAND computation from in-memory DEM arrays
  - apply_hand_filter post-segmentation filter
"""

from __future__ import annotations

import numpy as np
import pytest


class TestD8FlowDirection:
    """Tests for the pure-numpy D8 flow direction."""

    def test_simple_slope(self):
        """A DEM sloping south should flow south (direction 4)."""
        from siren.geo.hand import _d8_flow_direction_numpy
        dem = np.array([
            [10, 10, 10],
            [5, 5, 5],
            [0, 0, 0],
        ], dtype=np.float32)
        fdir = _d8_flow_direction_numpy(dem, pixel_size_m=10.0)
        # Center cell (1,1) should flow south (4)
        assert fdir[1, 1] == 4

    def test_flat_area_no_flow(self):
        """Flat areas should have direction 0 (no flow)."""
        from siren.geo.hand import _d8_flow_direction_numpy
        dem = np.ones((5, 5), dtype=np.float32) * 100
        fdir = _d8_flow_direction_numpy(dem, pixel_size_m=10.0)
        assert (fdir == 0).all()

    def test_diagonal_flow(self):
        """A DEM sloping southeast should flow SE (direction 2)."""
        from siren.geo.hand import _d8_flow_direction_numpy
        dem = np.array([
            [10, 8, 6],
            [8, 6, 4],
            [6, 4, 2],
        ], dtype=np.float32)
        fdir = _d8_flow_direction_numpy(dem, pixel_size_m=10.0)
        # Center cell (1,1) should flow SE (2) — steepest descent
        assert fdir[1, 1] == 2

    def test_pit_has_no_flow(self):
        """A local minimum (pit) should have direction 0."""
        from siren.geo.hand import _d8_flow_direction_numpy
        dem = np.array([
            [10, 10, 10],
            [10, 0, 10],
            [10, 10, 10],
        ], dtype=np.float32)
        fdir = _d8_flow_direction_numpy(dem, pixel_size_m=10.0)
        assert fdir[1, 1] == 0  # pit — no downslope neighbor


class TestFlowAccumulation:
    """Tests for flow accumulation."""

    def test_uniform_slope_accumulation(self):
        """On a uniform south slope, accumulation should increase southward."""
        from siren.geo.hand import _d8_flow_direction_numpy, _flow_accumulation_numpy
        dem = np.array([
            [4, 4, 4],
            [3, 3, 3],
            [2, 2, 2],
            [1, 1, 1],
        ], dtype=np.float32)
        fdir = _d8_flow_direction_numpy(dem, pixel_size_m=10.0)
        acc = _flow_accumulation_numpy(fdir, dem)
        # Bottom row should have the highest accumulation
        assert acc[3, 1] > acc[0, 1]
        # Each cell in the bottom row receives from 4 cells above
        assert acc[3, 1] == 4.0

    def test_flat_area_accumulation(self):
        """Flat areas should have accumulation = 1 (only self)."""
        from siren.geo.hand import _d8_flow_direction_numpy, _flow_accumulation_numpy
        dem = np.ones((5, 5), dtype=np.float32) * 100
        fdir = _d8_flow_direction_numpy(dem, pixel_size_m=10.0)
        acc = _flow_accumulation_numpy(fdir, dem)
        assert (acc == 1.0).all()


class TestComputeHandFromArray:
    """Tests for HAND computation from in-memory DEM arrays."""

    def test_flat_dem_hand_zero(self):
        """A flat DEM should produce HAND = 0 everywhere."""
        from siren.geo.hand import compute_hand_from_array
        dem = np.ones((32, 32), dtype=np.float32) * 100
        hand = compute_hand_from_array(dem, pixel_size_m=10.0, channel_threshold=10.0)
        assert hand.shape == (32, 32)
        assert (hand == 0).all()

    def test_hand_nonnegative(self):
        """HAND values should always be non-negative."""
        from siren.geo.hand import compute_hand_from_array
        # Create a synthetic terrain with a valley
        y, x = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        dem = 100 + 50 * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / 200)
        dem = dem.astype(np.float32)
        hand = compute_hand_from_array(dem, pixel_size_m=10.0, channel_threshold=10.0)
        assert (hand >= 0).all()

    def test_hand_shape_preserved(self):
        """HAND should have the same shape as the input DEM."""
        from siren.geo.hand import compute_hand_from_array
        dem = np.random.rand(64, 64).astype(np.float32) * 100
        hand = compute_hand_from_array(dem, pixel_size_m=10.0, channel_threshold=10.0)
        assert hand.shape == (64, 64)

    def test_hand_real_chip(self):
        """HAND computation should work on a real Sen1Floods11 chip DEM."""
        pytest.importorskip("rasterio")
        from siren.ml.dem_fetch import (
            fetch_dem_for_bounds, coregister_dem_to_grid, pixel_size_m_from_transform,
            DEM_CACHE_DIR,
        )
        from siren.geo.hand import compute_hand_from_array
        import rasterio
        from siren.ml.dataset import SEN1FLOODS11_ROOT

        # Check if DEM tiles are cached
        if not list(DEM_CACHE_DIR.glob("*.tif")):
            pytest.skip("No Copernicus DEM tiles cached")

        # Find a test chip
        chip_path = SEN1FLOODS11_ROOT / "train" / "S1" / "Pakistan_909806_S1Hand.tif"
        if not chip_path.exists():
            pytest.skip("Pakistan chip not found")

        with rasterio.open(str(chip_path)) as src:
            bounds = src.bounds
            chip_crs = str(src.crs)
            chip_transform = src.transform
            chip_shape = (src.height, src.width)
            center_lat = (bounds.top + bounds.bottom) / 2

        dem_paths = fetch_dem_for_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top,
        )
        if not dem_paths:
            pytest.skip("No DEM tiles for this chip")

        dem = coregister_dem_to_grid(dem_paths, chip_crs, chip_transform, chip_shape)
        dx_m, dy_m = pixel_size_m_from_transform(chip_transform, chip_crs, center_lat)
        px = (dx_m + dy_m) / 2

        hand = compute_hand_from_array(dem, pixel_size_m=px, channel_threshold=100.0)
        assert hand.shape == dem.shape
        assert (hand >= 0).all()
        # Pakistan Indus plain is flat — HAND should be low
        assert hand.max() < 50.0


class TestApplyHandFilter:
    """Tests for the apply_hand_filter post-segmentation filter."""

    def test_filter_zeros_impossible_elevations(self):
        """Predictions at high HAND should be zeroed out."""
        from siren.geo.hand import apply_hand_filter
        prob = np.ones((10, 10), dtype=np.float32) * 0.9
        hand = np.zeros((10, 10), dtype=np.float32)
        hand[5:, :] = 30.0  # High HAND in bottom half
        filtered = apply_hand_filter(prob, hand, stage_threshold_m=15.0)
        # Top half should be unchanged
        assert (filtered[:5, :] == 0.9).all()
        # Bottom half should be zeroed
        assert (filtered[5:, :] == 0.0).all()

    def test_filter_preserves_low_hand(self):
        """Predictions at low HAND should be preserved."""
        from siren.geo.hand import apply_hand_filter
        prob = np.ones((10, 10), dtype=np.float32) * 0.8
        hand = np.full((10, 10), 5.0, dtype=np.float32)  # All low HAND
        filtered = apply_hand_filter(prob, hand, stage_threshold_m=15.0)
        assert (filtered == 0.8).all()

    def test_filter_shape_mismatch_raises(self):
        """Shape mismatch between prob_mask and hand_grid should raise."""
        from siren.geo.hand import apply_hand_filter
        prob = np.ones((10, 10), dtype=np.float32)
        hand = np.zeros((20, 20), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            apply_hand_filter(prob, hand, stage_threshold_m=15.0)

    def test_filter_threshold_boundary(self):
        """HAND exactly at threshold should be preserved (<=)."""
        from siren.geo.hand import apply_hand_filter
        prob = np.ones((5, 5), dtype=np.float32) * 0.5
        hand = np.full((5, 5), 15.0, dtype=np.float32)  # Exactly at threshold
        filtered = apply_hand_filter(prob, hand, stage_threshold_m=15.0)
        # HAND == 15 should be preserved (filter is > threshold, not >=)
        assert (filtered == 0.5).all()

    def test_filter_does_not_add_predictions(self):
        """The filter should never increase predictions."""
        from siren.geo.hand import apply_hand_filter
        prob = np.random.rand(20, 20).astype(np.float32) * 0.3
        hand = np.random.rand(20, 20).astype(np.float32) * 50
        filtered = apply_hand_filter(prob, hand, stage_threshold_m=10.0)
        assert (filtered <= prob).all()
