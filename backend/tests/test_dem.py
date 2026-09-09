"""Tests for preprocess/dem.py — DEM utilities for the 4-channel tensor (V3 §2.2).

The core slope_degrees / surface_normal primitives are tested in test_rtc.py;
here we test the dem.py module's re-exports and the new convenience functions
(load_dem, compute_slope_channel, stack_terrain_channels) that the ML dataset
pipeline uses to build channels 2-3 of the 4-channel tensor.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_origin


def test_dem_module_reexports_slope_degrees():
    """dem.py re-exports slope_degrees from rtc.py (canonical implementation)."""
    from siren.preprocess.dem import slope_degrees
    from siren.preprocess.rtc import slope_degrees as rtc_slope_degrees

    assert slope_degrees is rtc_slope_degrees


def test_dem_module_reexports_surface_normal():
    """dem.py re-exports surface_normal from rtc.py."""
    from siren.preprocess.dem import surface_normal
    from siren.preprocess.rtc import surface_normal as rtc_surface_normal

    assert surface_normal is rtc_surface_normal


def test_stack_terrain_channels_shape():
    """stack_terrain_channels produces (2, H, W) from two 2D arrays."""
    from siren.preprocess.dem import stack_terrain_channels

    dem = np.zeros((4, 4), dtype=np.float32)
    slope = np.zeros((4, 4), dtype=np.float32)
    stacked = stack_terrain_channels(dem, slope)
    assert stacked.shape == (2, 4, 4)
    assert stacked.dtype == np.float32


def test_stack_terrain_channels_preserves_values():
    """stack_terrain_channels preserves the input values (unnormalized)."""
    from siren.preprocess.dem import stack_terrain_channels

    dem = np.full((3, 3), 5000.0, dtype=np.float32)
    slope = np.full((3, 3), 30.0, dtype=np.float32)
    stacked = stack_terrain_channels(dem, slope)
    assert np.allclose(stacked[0], 5000.0)  # DEM in metres
    assert np.allclose(stacked[1], 30.0)    # Slope in degrees


def test_stack_terrain_channels_rejects_mismatched_shapes():
    """stack_terrain_channels raises on shape mismatch."""
    from siren.preprocess.dem import stack_terrain_channels

    dem = np.zeros((4, 4), dtype=np.float32)
    slope = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        stack_terrain_channels(dem, slope)


def test_load_dem_reads_raster_and_pixel_size(tmp_path):
    """load_dem returns elevation array + pixel size from a GeoTIFF."""
    from siren.preprocess.dem import load_dem, DEFAULT_PIXEL_SIZE_M

    dem_path = tmp_path / "test_dem.tif"
    dem = np.arange(25, dtype=np.float32).reshape(5, 5) * 100.0  # 0..2400 m
    transform = from_origin(86.0, 28.0, 30.0, 30.0)  # 30 m pixels
    with rasterio.open(
        str(dem_path), "w",
        driver="GTiff", height=5, width=5, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(dem, 1)

    elev, px = load_dem(dem_path)
    assert elev.shape == (5, 5)
    assert elev.dtype == np.float32
    assert np.allclose(elev, dem)
    assert px == pytest.approx(30.0, abs=0.01)


def test_compute_slope_channel_native_resolution(tmp_path):
    """compute_slope_channel returns dem + slope at native resolution."""
    from siren.preprocess.dem import compute_slope_channel

    dem_path = tmp_path / "test_dem.tif"
    # Flat DEM → slope = 0 everywhere
    dem = np.zeros((8, 8), dtype=np.float32)
    transform = from_origin(86.0, 28.0, 30.0, 30.0)
    with rasterio.open(
        str(dem_path), "w",
        driver="GTiff", height=8, width=8, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(dem, 1)

    dem_out, slope_out = compute_slope_channel(dem_path)
    assert dem_out.shape == (8, 8)
    assert slope_out.shape == (8, 8)
    assert np.allclose(slope_out, 0.0)  # flat → 0°


def test_compute_slope_channel_resamples_to_target(tmp_path):
    """compute_slope_channel resamples DEM to a target chip grid."""
    from siren.preprocess.dem import compute_slope_channel

    dem_path = tmp_path / "test_dem.tif"
    # 100x100 DEM with a gradient
    dem = np.tile(np.arange(100, dtype=np.float32) * 30.0, (100, 1))
    transform = from_origin(86.0, 28.0, 30.0, 30.0)
    with rasterio.open(
        str(dem_path), "w",
        driver="GTiff", height=100, width=100, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(dem, 1)

    # Resample to 64x64 (Sen1Floods11 chips are 512x512, use small for speed)
    dem_out, slope_out = compute_slope_channel(dem_path, target_shape=(64, 64))
    assert dem_out.shape == (64, 64)
    assert slope_out.shape == (64, 64)
    # Gradient preserved → interior slope > 0
    assert slope_out[32, 32] > 0.0


def test_compute_slope_channel_known_gradient(tmp_path):
    """A 45° gradient (rise == run) produces ~45° slope."""
    from siren.preprocess.dem import compute_slope_channel

    dem_path = tmp_path / "test_dem.tif"
    # 30 m rise per 30 m pixel → 45° slope
    dem = np.tile(np.arange(16, dtype=np.float32) * 30.0, (16, 1))
    transform = from_origin(86.0, 28.0, 30.0, 30.0)
    with rasterio.open(
        str(dem_path), "w",
        driver="GTiff", height=16, width=16, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(dem, 1)

    _, slope_out = compute_slope_channel(dem_path)
    # Interior pixels should be ~45° (edges have boundary effects)
    assert abs(slope_out[8, 8] - 45.0) < 2.0
