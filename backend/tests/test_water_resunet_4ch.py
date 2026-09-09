"""Tests for the Level 2 4-channel WaterResUNet training pipeline.

Tests cover:
  - DEM tile naming and bounds computation (dem_fetch)
  - Pixel size conversion (geographic → metres)
  - Dataset 4-channel mode (channel count, shape, normalization)
  - Checkpoint loading and metadata
  - Evaluation metrics (IoU, precision, recall, F1)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"


# ---------------------------------------------------------------------------
# DEM fetcher tests
# ---------------------------------------------------------------------------

class TestDemFetch:
    """Tests for the Copernicus DEM tile fetcher."""

    def test_tile_name_north_east(self):
        from siren.ml.dem_fetch import _tile_name
        assert _tile_name(27, 86) == "Copernicus_DSM_COG_10_N27_00_E086_00_DEM"

    def test_tile_name_south_west(self):
        from siren.ml.dem_fetch import _tile_name
        assert _tile_name(-1, -2) == "Copernicus_DSM_COG_10_S01_00_W002_00_DEM"

    def test_tile_name_zero(self):
        from siren.ml.dem_fetch import _tile_name
        assert _tile_name(0, 0) == "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"

    def test_tiles_for_bounds_single_tile(self):
        from siren.ml.dem_fetch import tiles_for_bounds
        # Small bbox within a single 1°×1° tile
        tiles = tiles_for_bounds(86.5, 27.5, 86.8, 27.8)
        assert tiles == [(27, 86)]

    def test_tiles_for_bounds_multi_tile(self):
        from siren.ml.dem_fetch import tiles_for_bounds
        # Bbox spanning 2°×2° → 3×3 = 9 tiles
        tiles = tiles_for_bounds(86.0, 27.0, 88.0, 29.0)
        assert (27, 86) in tiles
        assert (28, 87) in tiles
        assert (29, 88) in tiles
        assert len(tiles) == 9

    def test_pixel_size_geographic(self):
        """Pixel size in metres for EPSG:4326 should be ~10m near equator."""
        from siren.ml.dem_fetch import pixel_size_m_from_transform
        from rasterio.transform import from_bounds

        # 512×512 chip at ~10m resolution near equator
        transform = from_bounds(-1.15, 9.43, -1.10, 9.47, 512, 512)
        dx_m, dy_m = pixel_size_m_from_transform(transform, "EPSG:4326", 9.45)
        # ~10m per pixel (latitude slightly less due to bounds)
        assert 8.0 < dx_m < 12.0
        assert 8.0 < dy_m < 12.0

    def test_pixel_size_geographic_high_latitude(self):
        """Pixel size should shrink in longitude at high latitudes."""
        from siren.ml.dem_fetch import pixel_size_m_from_transform
        from rasterio.transform import from_bounds

        # Same degree resolution at 60°N — longitude pixel should be ~half
        transform = from_bounds(86.0, 60.0, 86.05, 60.05, 512, 512)
        dx_m, dy_m = pixel_size_m_from_transform(transform, "EPSG:4326", 60.0)
        # At 60°N, cos(60°) = 0.5, so dx should be ~half of dy
        assert dx_m < dy_m * 0.6  # longitude pixel much smaller


# ---------------------------------------------------------------------------
# Dataset 4-channel tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sen1floods11_available():
    """Check if Sen1Floods11 data is available on disk."""
    from siren.ml.dataset import SEN1FLOODS11_ROOT
    if not SEN1FLOODS11_ROOT.exists():
        pytest.skip("Sen1Floods11 data not available")
    return SEN1FLOODS11_ROOT


@pytest.fixture
def dem_tiles_cached():
    """Check if Copernicus DEM tiles are cached."""
    from siren.ml.dem_fetch import DEM_CACHE_DIR
    tiles = list(DEM_CACHE_DIR.glob("*.tif"))
    if len(tiles) == 0:
        pytest.skip("No Copernicus DEM tiles cached")
    return DEM_CACHE_DIR


class TestDataset4Channel:
    """Tests for the 4-channel WaterSegmentationDataset."""

    def test_2channel_legacy(self, sen1floods11_available):
        """Legacy 2-channel mode returns (2, H, W) tensors."""
        from siren.ml.dataset import WaterSegmentationDataset
        ds = WaterSegmentationDataset(split="train", strategy="event_holdout")
        assert len(ds) > 0
        item = ds[0]
        assert item["sar"].shape[0] == 2  # 2 channels
        assert item["sar"].shape[1] == 512
        assert item["sar"].shape[2] == 512
        assert "dem" not in item
        assert "slope" not in item

    def test_4channel_copernicus(self, sen1floods11_available, dem_tiles_cached):
        """4-channel mode with Copernicus DEM returns (4, H, W) tensors."""
        from siren.ml.dataset import WaterSegmentationDataset
        ds = WaterSegmentationDataset(
            split="train", strategy="event_holdout", use_copernicus_dem=True,
        )
        assert len(ds) > 0
        item = ds[0]
        assert item["sar"].shape[0] == 4  # 4 channels
        assert item["sar"].shape[1] == 512
        assert item["sar"].shape[2] == 512
        assert "dem" in item
        assert "slope" in item
        assert item["dem"].shape == (512, 512)
        assert item["slope"].shape == (512, 512)

    def test_4channel_normalization(self, sen1floods11_available, dem_tiles_cached):
        """4-channel tensors should be normalized to [0, 1]."""
        from siren.ml.dataset import WaterSegmentationDataset
        ds = WaterSegmentationDataset(
            split="train", strategy="event_holdout", use_copernicus_dem=True,
        )
        item = ds[0]
        sar = item["sar"]
        assert sar.min() >= 0.0
        assert sar.max() <= 1.0

    def test_4channel_dem_unnormalized(self, sen1floods11_available, dem_tiles_cached):
        """DEM channel in the output dict should be in metres (unnormalized)."""
        from siren.ml.dataset import WaterSegmentationDataset
        ds = WaterSegmentationDataset(
            split="train", strategy="event_holdout", use_copernicus_dem=True,
        )
        item = ds[0]
        dem = item["dem"]
        # DEM should be in metres — not [0, 1]
        assert dem.min() >= 0.0
        # Most DEMs are < 2000m for Sen1Floods11 chips (Ghana, Nigeria, etc.)
        assert dem.max() < 9000.0

    def test_4channel_slope_range(self, sen1floods11_available, dem_tiles_cached):
        """Slope should be in degrees [0, 90]."""
        from siren.ml.dataset import WaterSegmentationDataset
        ds = WaterSegmentationDataset(
            split="train", strategy="event_holdout", use_copernicus_dem=True,
        )
        item = ds[0]
        slope = item["slope"]
        assert slope.min() >= 0.0
        assert slope.max() <= 90.0

    def test_dem_cache_hit(self, sen1floods11_available, dem_tiles_cached):
        """Second access to the same chip should use the in-memory cache."""
        from siren.ml.dataset import WaterSegmentationDataset
        ds = WaterSegmentationDataset(
            split="train", strategy="event_holdout", use_copernicus_dem=True,
        )
        _ = ds[0]  # First access — populates cache
        assert len(ds._dem_cache) > 0
        item = ds[0]  # Second access — should use cache
        assert item["sar"].shape[0] == 4

    def test_event_holdout_split_disjoint(self, sen1floods11_available):
        """Event-holdout splits should have no event overlap."""
        from siren.ml.dataset import WaterSegmentationDataset
        train_ds = WaterSegmentationDataset(split="train", strategy="event_holdout")
        val_ds = WaterSegmentationDataset(split="val", strategy="event_holdout")
        test_ds = WaterSegmentationDataset(split="test", strategy="event_holdout")

        train_events = train_ds.events()
        val_events = val_ds.events()
        test_events = test_ds.events()

        assert len(train_events & val_events) == 0, "train/val event overlap"
        assert len(train_events & test_events) == 0, "train/test event overlap"
        assert len(val_events & test_events) == 0, "val/test event overlap"


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

class TestCheckpoint:
    """Tests for the trained checkpoint and metadata."""

    def test_checkpoint_exists(self):
        """The 4ch checkpoint should exist after training."""
        ckpt = CHECKPOINT_DIR / "water_resunet_4ch_v1.pt"
        if not ckpt.exists():
            pytest.skip("4ch checkpoint not trained yet")
        assert ckpt.stat().st_size > 0

    def test_metadata_exists(self):
        """The metadata sidecar should exist and be valid JSON."""
        meta = CHECKPOINT_DIR / "water_resunet_4ch_v1.meta.json"
        if not meta.exists():
            pytest.skip("4ch metadata not found")
        with open(meta) as f:
            data = json.load(f)
        assert "model" in data
        assert data["model"] == "WaterResUNet"
        assert data["in_channels"] == 4
        assert "test_iou_4ch" in data
        assert "test_iou_2ch" in data
        assert "gate_passed" in data
        assert "shadow_only" in data

    def test_checkpoint_loads(self):
        """The checkpoint should load into a WaterResUNet with correct channels."""
        pytest.importorskip("torch")
        from siren.ml.model import WaterResUNet

        ckpt = CHECKPOINT_DIR / "water_resunet_4ch_v1.pt"
        if not ckpt.exists():
            pytest.skip("4ch checkpoint not trained yet")

        import torch
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        assert state["in_channels"] == 4

        model = WaterResUNet(in_channels=4, base_channels=state.get("base_channels", 32))
        model.load_state_dict(state["model_state"])
        assert model.in_channels == 4


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for the evaluation metrics."""

    def test_iou_perfect(self):
        from siren.ml.train_water_resunet import iou_score
        pred = np.ones((64, 64))
        target = np.ones((64, 64))
        assert iou_score(pred, target) == 1.0

    def test_iou_no_overlap(self):
        from siren.ml.train_water_resunet import iou_score
        pred = np.ones((64, 64))
        target = np.zeros((64, 64))
        assert iou_score(pred, target) == 0.0

    def test_iou_partial(self):
        from siren.ml.train_water_resunet import iou_score
        pred = np.zeros((64, 64))
        pred[:32] = 1
        target = np.zeros((64, 64))
        target[32:] = 1
        # Half overlap: intersection=0, union=64*64
        assert iou_score(pred, target) == 0.0

    def test_iou_half_overlap(self):
        from siren.ml.train_water_resunet import iou_score
        pred = np.zeros((64, 64))
        pred[:32] = 1
        target = np.zeros((64, 64))
        target[:48] = 1
        # intersection = 32*64, union = 48*64
        expected = (32 * 64) / (48 * 64)
        assert abs(iou_score(pred, target) - expected) < 1e-6

    def test_iou_with_valid_mask(self):
        from siren.ml.train_water_resunet import iou_score
        pred = np.ones((64, 64))
        target = np.ones((64, 64))
        valid = np.zeros((64, 64))
        valid[:32] = 1  # Only top half is valid
        # All valid pixels are predicted correctly
        assert iou_score(pred, target, valid) == 1.0

    def test_precision_recall_f1(self):
        from siren.ml.train_water_resunet import precision_recall_f1
        pred = np.array([[1, 1, 0], [0, 0, 0]])
        target = np.array([[1, 0, 0], [0, 0, 0]])
        metrics = precision_recall_f1(pred, target)
        # TP=1, FP=1, FN=0
        assert abs(metrics["precision"] - 0.5) < 1e-6
        assert abs(metrics["recall"] - 1.0) < 1e-6
        assert abs(metrics["f1"] - 2/3) < 1e-6
