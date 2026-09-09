"""Tests for ml/dataset.py 4-channel stacking (ADR-011 / V3 §2.1).

The full WaterSegmentationDataset requires Sen1Floods11 data on disk, which
is not available in the test environment. These tests exercise the new
``coregister_dem_to_chip`` function and the 4-channel stacking logic against
synthetic rasters, verifying the DEM co-registration + slope derivation that
feeds channels 2-3 of the 4-channel tensor.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def _write_raster(path, data, crs="EPSG:4326", transform=None):
    """Helper: write a 2D array as a single-band GeoTIFF."""
    h, w = data.shape
    if transform is None:
        transform = from_origin(86.0, 28.0, 0.01, 0.01)
    with rasterio.open(
        str(path), "w",
        driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs=crs, transform=transform,
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def test_coregister_dem_to_chip_same_grid(tmp_path):
    """coregister_dem_to_chip returns the DEM when grids already match."""
    from siren.ml.dataset import coregister_dem_to_chip

    dem_data = np.arange(25, dtype=np.float32).reshape(5, 5) * 100.0
    dem_path = tmp_path / "dem.tif"
    transform = from_origin(86.0, 28.0, 0.01, 0.01)
    _write_raster(dem_path, dem_data, transform=transform)

    result = coregister_dem_to_chip(
        dem_path, "EPSG:4326", transform, (5, 5)
    )
    assert result.shape == (5, 5)
    assert result.dtype == np.float32
    assert np.allclose(result, dem_data)


def test_coregister_dem_to_chip_resamples_to_smaller_grid(tmp_path):
    """coregister_dem_to_chip resamples a 100x100 DEM to a 50x50 chip grid."""
    from siren.ml.dataset import coregister_dem_to_chip

    dem_data = np.tile(np.arange(100, dtype=np.float32) * 10.0, (100, 1))
    dem_path = tmp_path / "dem.tif"
    dem_transform = from_origin(86.0, 28.0, 0.001, 0.001)
    _write_raster(dem_path, dem_data, transform=dem_transform)

    # Chip grid: 50x50 at the same extent but coarser pixels
    chip_transform = from_origin(86.0, 28.0, 0.002, 0.002)
    result = coregister_dem_to_chip(
        dem_path, "EPSG:4326", chip_transform, (50, 50)
    )
    assert result.shape == (50, 50)
    assert result.dtype == np.float32
    # The gradient should be preserved (values increase eastward)
    assert result[25, 40] > result[25, 10]


def test_coregister_dem_to_chip_different_crs(tmp_path):
    """coregister_dem_to_chip reprojects across CRS (EPSG:4326 -> EPSG:3857)."""
    from siren.ml.dataset import coregister_dem_to_chip
    from rasterio.warp import transform_bounds

    dem_data = np.full((20, 20), 5000.0, dtype=np.float32)
    dem_path = tmp_path / "dem.tif"
    dem_transform = from_origin(86.0, 28.0, 0.01, 0.01)
    _write_raster(dem_path, dem_data, crs="EPSG:4326", transform=dem_transform)

    # Compute the DEM bounds in Web Mercator for an overlapping chip grid
    dem_bounds = (86.0, 27.8, 86.2, 28.0)  # west, south, east, north
    x_min, y_min, x_max, y_max = transform_bounds(
        "EPSG:4326", "EPSG:3857", *dem_bounds
    )
    pixel_size = (x_max - x_min) / 10
    chip_transform = from_origin(x_min, y_max, pixel_size, pixel_size)
    result = coregister_dem_to_chip(
        dem_path, "EPSG:3857", chip_transform, (10, 10)
    )
    assert result.shape == (10, 10)
    # Flat DEM → all overlapping values should be ~5000
    assert np.allclose(result, 5000.0, atol=100.0)


def test_dataset_accepts_dem_path_parameter():
    """WaterSegmentationDataset.__init__ accepts dem_path without error.

    We can't instantiate the full dataset (needs Sen1Floods11 on disk),
    but we verify the constructor signature accepts the parameter.
    """
    import inspect
    from siren.ml.dataset import WaterSegmentationDataset

    sig = inspect.signature(WaterSegmentationDataset.__init__)
    assert "dem_path" in sig.parameters
    assert sig.parameters["dem_path"].default is None


def test_dataset_4channel_stacking_logic(tmp_path):
    """Verify the 4-channel stacking logic produces the correct tensor shape.

    This tests the coregistration + slope + normalize_tensor pipeline that
    the dataset's __getitem__ uses, without needing the full Sen1Floods11
    directory structure.
    """
    from siren.ml.dataset import coregister_dem_to_chip
    from siren.preprocess.dem import slope_degrees, DEFAULT_PIXEL_SIZE_M
    from siren.ml.contract import normalize_tensor, SAR_CHANNELS

    # Synthetic DEM with a gradient
    dem_data = np.tile(np.arange(32, dtype=np.float32) * 30.0, (32, 1))
    dem_path = tmp_path / "dem.tif"
    transform = from_origin(86.0, 28.0, 30.0, 30.0)
    _write_raster(dem_path, dem_data, transform=transform)

    # Simulate a chip at the same grid
    chip_shape = (32, 32)
    dem = coregister_dem_to_chip(dem_path, "EPSG:4326", transform, chip_shape)
    slope = slope_degrees(dem, 30.0)

    # Synthetic SAR (2, H, W) in dB
    sar = np.random.RandomState(42).uniform(-30, 0, (2, 32, 32)).astype(np.float32)

    # Stack into 4-channel tensor (the dataset's logic)
    tensor = np.stack([sar[0], sar[1], dem, slope], axis=0).astype(np.float32)
    assert tensor.shape == (4, 32, 32)

    # Normalise via the 4-channel contract
    norm = normalize_tensor(tensor)
    assert norm.shape == (SAR_CHANNELS, 32, 32)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
