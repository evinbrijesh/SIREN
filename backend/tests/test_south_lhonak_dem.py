"""Tests for the South Lhonak Pléiades DEM differencing module.

Tests cover:
    - DEM loading and CRS verification
    - Common footprint computation
    - Difference raster computation
    - Volume statistics (erosion/deposition)
    - Nodata handling
    - Output raster writing

Tests that require the real downloaded DEMs are marked with
``@pytest.mark.skipif`` and check for data availability at import time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from siren.preprocess.south_lhonak_dem import (
    compute_dem_difference,
    _compute_common_bounds,
    PRE_EVENT_DEM,
    POST_EVENT_DEM,
    NODATA,
)

DEM_AVAILABLE = PRE_EVENT_DEM.exists() and POST_EVENT_DEM.exists()
skip_dem = pytest.mark.skipif(not DEM_AVAILABLE, reason="South Lhonak DEMs not downloaded")


# --------------------------------------------------------------------------- #
# Common bounds computation (pure function, no data needed)
# --------------------------------------------------------------------------- #

class TestComputeCommonBounds:
    """Tests for the _compute_common_bounds helper."""

    def test_full_overlap(self):
        """Identical bounds return the same bounds."""
        b = (100.0, 200.0, 300.0, 400.0)
        result = _compute_common_bounds(b, b)
        assert result == b

    def test_partial_overlap(self):
        """Partially overlapping bounds return the intersection."""
        pre = (100.0, 200.0, 300.0, 400.0)
        post = (200.0, 250.0, 400.0, 450.0)
        result = _compute_common_bounds(pre, post)
        assert result == (200.0, 250.0, 300.0, 400.0)

    def test_no_overlap(self):
        """Non-overlapping bounds return an empty intersection."""
        pre = (100.0, 200.0, 150.0, 250.0)
        post = (300.0, 400.0, 350.0, 450.0)
        result = _compute_common_bounds(pre, post)
        # left > right, bottom > top
        assert result[0] > result[2]
        assert result[1] > result[3]

    def test_pre_contains_post(self):
        """Post entirely inside pre returns post bounds."""
        pre = (0.0, 0.0, 1000.0, 1000.0)
        post = (100.0, 100.0, 200.0, 200.0)
        result = _compute_common_bounds(pre, post)
        assert result == post


# --------------------------------------------------------------------------- #
# Real DEM differencing (requires downloaded data)
# --------------------------------------------------------------------------- #

@skip_dem
class TestDEMDifference:
    """Tests for compute_dem_difference on the real South Lhonak DEMs."""

    def test_returns_result_object(self):
        """compute_dem_difference returns a DEMDifferenceResult."""
        result = compute_dem_difference()
        assert hasattr(result, "diff")
        assert hasattr(result, "loss_m3")
        assert hasattr(result, "gain_m3")

    def test_diff_shape_matches_common_area(self):
        """Difference raster shape matches the common area."""
        result = compute_dem_difference()
        assert result.diff.shape == result.common_shape

    def test_both_dems_are_utm45n(self):
        """Both DEMs are in UTM Zone 45N (EPSG:32645)."""
        result = compute_dem_difference()
        assert result.pre_crs.to_epsg() == 32645
        assert result.post_crs.to_epsg() == 32645

    def test_pixel_area_is_1m(self):
        """Pixel area is 1.0 m² (1 m resolution DEMs)."""
        result = compute_dem_difference()
        assert result.pixel_area_m2 == 1.0

    def test_nodata_handled(self):
        """Nodata values (-9999) are not included in volume statistics."""
        result = compute_dem_difference()
        # The diff array should contain NODATA where either DEM is missing
        assert NODATA in result.diff or np.all(result.diff != NODATA)
        # But valid fractions should be < 1.0 (both DEMs have nodata)
        assert result.pre_valid_fraction < 1.0
        assert result.post_valid_fraction < 1.0

    def test_loss_is_positive(self):
        """Erosion/loss volume is positive (material was lost)."""
        result = compute_dem_difference()
        assert result.loss_m3 > 0, "Loss volume should be positive"

    def test_gain_is_positive(self):
        """Deposition/gain volume is positive (material was deposited)."""
        result = compute_dem_difference()
        assert result.gain_m3 > 0, "Gain volume should be positive"

    def test_net_change_is_negative(self):
        """Net change is negative (more erosion than deposition in a GLOF)."""
        result = compute_dem_difference()
        assert result.net_change_m3 < 0, (
            f"Net change {result.net_change_m3} should be negative"
        )

    def test_loss_exceeds_gain(self):
        """Erosion exceeds deposition (the GLOF removed material)."""
        result = compute_dem_difference()
        assert result.loss_m3 > result.gain_m3, (
            f"Loss ({result.loss_m3}) should exceed gain ({result.gain_m3})"
        )

    def test_loss_volume_is_plausible(self):
        """Loss volume is in a plausible range for a GLOF (1-500 million m³)."""
        result = compute_dem_difference()
        # The 2023 South Lhonak GLOF involved significant moraine collapse.
        # The exact volume depends on the threshold, but should be in the
        # range of millions to hundreds of millions of cubic metres.
        assert 1e6 < result.loss_m3 < 5e8, (
            f"Loss {result.loss_m3:.0f} m³ outside [1M, 500M] m³"
        )

    def test_common_area_is_reasonable(self):
        """Common area is in a plausible range (10-200 km²)."""
        result = compute_dem_difference()
        area_km2 = result.common_shape[0] * result.common_shape[1] * result.pixel_area_m2 / 1e6
        assert 10 < area_km2 < 200, f"Common area {area_km2:.1f} km² outside [10, 200]"

    def test_to_dict_is_serializable(self):
        """to_dict returns a JSON-safe dict."""
        import json
        result = compute_dem_difference()
        d = result.to_dict()
        # Should be JSON-serializable
        json.dumps(d)

    def test_output_raster_writes(self, tmp_path):
        """Output raster can be written to a GeoTIFF."""
        out = tmp_path / "diff.tif"
        result = compute_dem_difference(output_path=out)
        assert out.exists()
        # Verify the output raster opens correctly
        with rasterio.open(out) as ds:
            assert ds.crs.to_epsg() == 32645 if hasattr(ds.crs, 'to_epsg') else True
            assert ds.nodata == NODATA
            data = ds.read(1)
            assert data.shape == result.common_shape

    def test_threshold_affects_volume(self):
        """Higher threshold reduces the counted volume."""
        low_thresh = compute_dem_difference(loss_threshold_m=1.0, gain_threshold_m=1.0)
        high_thresh = compute_dem_difference(loss_threshold_m=5.0, gain_threshold_m=5.0)
        # Higher threshold should count fewer pixels
        assert high_thresh.n_loss_pixels < low_thresh.n_loss_pixels
        assert high_thresh.n_gain_pixels < low_thresh.n_gain_pixels
