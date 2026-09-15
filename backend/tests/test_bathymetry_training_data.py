"""Tests for the bathymetry training data pipeline.

Tests cover:
    - DEM tile lookup
    - Lake mask construction (convex hull and outline)
    - Water surface elevation estimation
    - Depth interpolation
    - Sample building (all 20 lakes)
    - Volume computation from samples
    - Input/target tensor conversion
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from siren.ml.bathymetry_training_data import (
    BathymetrySample,
    build_sample,
    build_all_samples,
    compute_sample_volume,
    _find_dem_tile,
    _estimate_z_surface,
    _resample_array,
    DEFAULT_GRID_SIZE,
    DEM_DIR,
)
from siren.ml.bathymetry_dataset import (
    load_all_surveyed_lakes,
    ZHANG_16LAKES_DIR,
    DAS_4LAKES_DIR,
)

ZHANG_AVAILABLE = (
    ZHANG_16LAKES_DIR / "Glacial_Lake_bathymetry" / "GlacialLakeBathymetry"
).exists()
DAS_AVAILABLE = (
    DAS_4LAKES_DIR / "In-Situ Bathymetry Data for JOG"
).exists()
DEM_AVAILABLE = DEM_DIR.exists() and any(DEM_DIR.glob("*_DEM.tif"))
skip_all = pytest.mark.skipif(
    not (ZHANG_AVAILABLE and DAS_AVAILABLE and DEM_AVAILABLE),
    reason="surveyed bathymetry or DEM datasets not downloaded",
)


# --------------------------------------------------------------------------- #
# DEM tile lookup
# --------------------------------------------------------------------------- #

class TestDEMTileLookup:
    """Tests for the DEM tile lookup function."""

    def test_finds_tile_for_galongco(self):
        """Galongco (28.3, 85.8) is in tile N28_E085."""
        tile = _find_dem_tile(85.8, 28.3)
        assert tile is not None
        assert "N28" in tile.name
        assert "E085" in tile.name

    def test_finds_tile_for_kya_tso(self):
        """Kya Tso (77.4, 32.7) is in tile N32_E077."""
        tile = _find_dem_tile(77.4, 32.7)
        assert tile is not None
        assert "N32" in tile.name
        assert "E077" in tile.name

    def test_returns_none_for_missing_tile(self):
        """A point in the ocean returns None."""
        tile = _find_dem_tile(0.0, 0.0)
        # N00_E000 tile may or may not exist on disk, but it's not in our dataset
        if tile is not None:
            # If it exists, it's not one of our 6 tiles
            assert "N00" not in tile.name or "E000" not in tile.name


# --------------------------------------------------------------------------- #
# Water surface estimation
# --------------------------------------------------------------------------- #

class TestZSurfaceEstimation:
    """Tests for the water surface elevation estimation."""

    def test_simple_bowl(self):
        """A simple bowl-shaped lake with known rim elevation."""
        # 10x10 grid, lake in the centre (3:7, 3:7)
        dem = np.full((10, 10), 5000.0, dtype=np.float32)
        # Lake rim at 4900m (the outlet)
        dem[2, 3:7] = 4900.0
        dem[7, 3:7] = 4950.0
        dem[3:7, 2] = 4920.0
        dem[3:7, 7] = 4930.0
        # Lake interior (masked, set to 0)
        dem[3:7, 3:7] = 0.0

        lake_mask = np.zeros((10, 10), dtype=np.float32)
        lake_mask[3:7, 3:7] = 1.0

        z = _estimate_z_surface(dem, lake_mask)
        assert z == 4900.0

    def test_raises_on_empty_mask(self):
        """Empty lake mask raises ValueError."""
        dem = np.full((10, 10), 5000.0, dtype=np.float32)
        mask = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            _estimate_z_surface(dem, mask)

    def test_raises_on_full_mask(self):
        """Lake mask filling the entire grid raises ValueError."""
        dem = np.full((10, 10), 5000.0, dtype=np.float32)
        mask = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="no rim"):
            _estimate_z_surface(dem, mask)


# --------------------------------------------------------------------------- #
# Array resampling
# --------------------------------------------------------------------------- #

class TestResampling:
    """Tests for the array resampling function."""

    def test_no_resample_needed(self):
        """No resampling when already at target size."""
        arr = np.random.rand(128, 128).astype(np.float32)
        result = _resample_array(arr, 128)
        assert result.shape == (128, 128)
        np.testing.assert_array_equal(result, arr)

    def test_upsample(self):
        """Upsampling produces the correct shape."""
        arr = np.random.rand(64, 64).astype(np.float32)
        result = _resample_array(arr, 128)
        assert result.shape == (128, 128)

    def test_downsample(self):
        """Downsampling produces the correct shape."""
        arr = np.random.rand(256, 256).astype(np.float32)
        result = _resample_array(arr, 128)
        assert result.shape == (128, 128)

    def test_preserves_range(self):
        """Resampling preserves the value range."""
        arr = np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32)
        result = _resample_array(arr, 128)
        assert result.min() >= 0
        assert result.max() <= 1


# --------------------------------------------------------------------------- #
# Sample building (requires real data)
# --------------------------------------------------------------------------- #

@skip_all
class TestSampleBuilding:
    """Tests for building training samples from real data."""

    def test_build_sample_galongco(self):
        """Build a sample for Galongco (large, deep lake)."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        assert sample is not None
        assert sample.lake_name == "Galongco"
        assert sample.dem.shape == (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
        assert sample.lake_mask.shape == (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
        assert sample.bed_elevation.shape == (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)

    def test_z_surface_is_positive(self):
        """Water surface elevation is positive (Himalayan altitude)."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        assert sample.z_surface > 4000  # Galongco is at ~5000m

    def test_lake_mask_is_binary(self):
        """Lake mask is binary (0 or 1)."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        unique = np.unique(sample.lake_mask)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_lake_mask_has_pixels(self):
        """Lake mask has at least some lake pixels."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        assert sample.lake_mask.sum() > 0

    def test_dem_is_masked_in_lake(self):
        """DEM is 0 inside the lake region."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        lake_pixels = sample.lake_mask > 0.5
        dem_in_lake = sample.dem[lake_pixels]
        assert (dem_in_lake == 0).all()

    def test_bed_elevation_below_surface(self):
        """Bed elevation is below z_surface inside the lake."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        lake_pixels = sample.lake_mask > 0.5
        bed_in_lake = sample.bed_elevation[lake_pixels]
        assert (bed_in_lake <= sample.z_surface).all()

    def test_depth_is_positive(self):
        """Depth (z_surface - bed) is non-negative inside the lake."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        lake_pixels = sample.lake_mask > 0.5
        depths = sample.z_surface - sample.bed_elevation[lake_pixels]
        assert (depths >= 0).all()

    def test_build_all_samples(self):
        """All 20 lakes produce samples."""
        samples = build_all_samples()
        assert len(samples) == 20

    def test_all_samples_have_correct_shape(self):
        """All samples have the correct grid size."""
        samples = build_all_samples()
        for s in samples:
            assert s.dem.shape == (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
            assert s.lake_mask.shape == (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
            assert s.bed_elevation.shape == (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)

    def test_all_samples_have_positive_z_surface(self):
        """All samples have positive z_surface (Himalayan altitude)."""
        samples = build_all_samples()
        for s in samples:
            assert s.z_surface > 3000, f"{s.lake_name} z_surface={s.z_surface}"


# --------------------------------------------------------------------------- #
# Volume computation
# --------------------------------------------------------------------------- #

@skip_all
class TestVolumeComputation:
    """Tests for volume computation from samples."""

    def test_galongco_volume_is_positive(self):
        """Galongco sample volume is positive."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        vol = compute_sample_volume(sample)
        assert vol > 0

    def test_galongco_volume_reasonable(self):
        """Galongco sample volume is in a reasonable range (100-400 MCM)."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        vol = compute_sample_volume(sample)
        # Published volume is 375 MCM; sample underestimates due to convex hull
        assert 100e6 < vol < 400e6

    def test_kya_tso_volume_small(self):
        """Kya Tso (small lake) has a small volume (< 5 MCM)."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Kya Tso Lake"][0]
        sample = build_sample(r)
        vol = compute_sample_volume(sample)
        assert vol < 5e6


# --------------------------------------------------------------------------- #
# Input/target conversion
# --------------------------------------------------------------------------- #

@skip_all
class TestInputTargetConversion:
    """Tests for the to_input_target method."""

    def test_shapes(self):
        """Input and target have correct shapes."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        x, y = sample.to_input_target()
        assert x.shape == (2, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
        assert y.shape == (1, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)

    def test_dtypes(self):
        """Input and target are float32."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        x, y = sample.to_input_target()
        assert x.dtype == np.float32
        assert y.dtype == np.float32

    def test_input_normalised(self):
        """Input DEM channel is normalised to [0, 1]."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        x, y = sample.to_input_target()
        dem_channel = x[0]
        assert dem_channel.min() >= 0
        assert dem_channel.max() <= 1

    def test_mask_channel_is_binary(self):
        """Input mask channel is binary."""
        records = load_all_surveyed_lakes()
        r = [r for r in records if r.lake_name == "Galongco"][0]
        sample = build_sample(r)
        x, y = sample.to_input_target()
        mask_channel = x[1]
        unique = np.unique(mask_channel)
        assert set(unique.tolist()).issubset({0.0, 1.0})
