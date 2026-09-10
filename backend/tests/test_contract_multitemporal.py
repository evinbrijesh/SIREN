"""Tests for the 6-channel multi-temporal contract and dataset (V3 §2.8).

Tests the normalization functions, synthetic Δσ⁰ construction, HAND
normalization, round-trip properties, and the MultiTemporalWaterDataset
class structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from siren.ml.contract import (
    SAR_CHANNELS_MULTITEMPORAL,
    CHANNEL_NAMES_MULTITEMPORAL,
    DELTA_SAR_DB_MIN,
    DELTA_SAR_DB_MAX,
    HAND_MAX_M,
    normalize_delta_sar,
    denormalize_delta_sar,
    normalize_hand,
    denormalize_hand,
    normalize_tensor_multitemporal,
    denormalize_tensor_multitemporal,
    normalize_sar,
    normalize_slope,
)


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

class TestContractConstants:
    """Verify the 6-channel contract constants are correct."""

    def test_channel_count(self):
        assert SAR_CHANNELS_MULTITEMPORAL == 6

    def test_channel_names(self):
        assert CHANNEL_NAMES_MULTITEMPORAL == (
            "VV_post", "VH_post", "dVV", "dVH", "HAND", "Slope",
        )

    def test_delta_sar_range(self):
        """Δσ⁰ range covers flood drops (-10 dB) to construction (+5 dB)."""
        assert DELTA_SAR_DB_MIN == -15.0
        assert DELTA_SAR_DB_MAX == 5.0

    def test_hand_max(self):
        """HAND max preserves resolution in the critical 0-10 m range."""
        assert HAND_MAX_M == 200.0


# ---------------------------------------------------------------------------
# Δσ⁰ normalization
# ---------------------------------------------------------------------------

class TestDeltaSarNormalization:
    """Test Δσ⁰ (temporal difference) normalization."""

    def test_flood_drop_maps_low(self):
        """Flood water Δσ⁰ = -8 dB maps toward 0.0 (strong change signal)."""
        arr = np.array([[-8.0]], dtype=np.float32)
        norm = normalize_delta_sar(arr)
        # -8 dB in [-15, 5] → (-8 - (-15)) / 20 = 7/20 = 0.35
        assert 0.3 < norm[0, 0] < 0.4

    def test_no_change_maps_high(self):
        """No change Δσ⁰ = 0 dB maps toward 0.75."""
        arr = np.array([[0.0]], dtype=np.float32)
        norm = normalize_delta_sar(arr)
        # 0 in [-15, 5] → 15/20 = 0.75
        assert abs(norm[0, 0] - 0.75) < 0.01

    def test_clamping(self):
        """Values outside the range are clamped."""
        arr = np.array([[-20.0, 10.0]], dtype=np.float32)
        norm = normalize_delta_sar(arr)
        assert norm[0, 0] == 0.0  # clamped to min
        assert norm[0, 1] == 1.0  # clamped to max

    def test_nan_to_zero_change(self):
        """NaN values are treated as no change (0 dB)."""
        arr = np.array([[np.nan]], dtype=np.float32)
        norm = normalize_delta_sar(arr)
        assert abs(norm[0, 0] - 0.75) < 0.01

    def test_round_trip(self):
        """normalize → denormalize round-trips within the clamp range."""
        arr = np.array([[-10.0, -5.0, 0.0, 3.0]], dtype=np.float32)
        norm = normalize_delta_sar(arr)
        recovered = denormalize_delta_sar(norm)
        np.testing.assert_allclose(recovered, arr, atol=1e-5)

    def test_3d_shape(self):
        """3D array (C, H, W) normalizes correctly."""
        arr = np.random.uniform(-10, 3, size=(2, 4, 4)).astype(np.float32)
        norm = normalize_delta_sar(arr)
        assert norm.shape == (2, 4, 4)
        assert norm.dtype == np.float32
        assert norm.min() >= 0.0 and norm.max() <= 1.0


# ---------------------------------------------------------------------------
# HAND normalization
# ---------------------------------------------------------------------------

class TestHandNormalization:
    """Test HAND (Height Above Nearest Drainage) normalization."""

    def test_in_channel_maps_zero(self):
        """HAND = 0 m (in drainage channel) maps to 0.0."""
        arr = np.array([[0.0]], dtype=np.float32)
        norm = normalize_hand(arr)
        assert norm[0, 0] == 0.0

    def test_floodplain_low(self):
        """HAND = 3 m (floodplain) maps to a low value (0.015)."""
        arr = np.array([[3.0]], dtype=np.float32)
        norm = normalize_hand(arr)
        assert abs(norm[0, 0] - 3.0 / 200.0) < 1e-5

    def test_hilltop_high(self):
        """HAND = 200 m (hilltop) maps to 1.0."""
        arr = np.array([[200.0]], dtype=np.float32)
        norm = normalize_hand(arr)
        assert norm[0, 0] == 1.0

    def test_clamping(self):
        """Values above 200 m are clamped to 1.0."""
        arr = np.array([[500.0]], dtype=np.float32)
        norm = normalize_hand(arr)
        assert norm[0, 0] == 1.0

    def test_nan_to_zero(self):
        """NaN values map to 0 (in-channel)."""
        arr = np.array([[np.nan]], dtype=np.float32)
        norm = normalize_hand(arr)
        assert norm[0, 0] == 0.0

    def test_round_trip(self):
        """normalize → denormalize round-trips within the clamp range."""
        arr = np.array([[0.0, 10.0, 50.0, 200.0]], dtype=np.float32)
        norm = normalize_hand(arr)
        recovered = denormalize_hand(norm)
        np.testing.assert_allclose(recovered, arr, atol=1e-3)

    def test_scale_invariance(self):
        """HAND is scale-invariant: 3 m at sea level = 3 m at 4500 m."""
        hand_sea_level = np.array([[3.0]], dtype=np.float32)
        hand_himalaya = np.array([[3.0]], dtype=np.float32)
        assert normalize_hand(hand_sea_level)[0, 0] == normalize_hand(hand_himalaya)[0, 0]


# ---------------------------------------------------------------------------
# 6-channel tensor normalization
# ---------------------------------------------------------------------------

class TestTensorMultitemporalNormalization:
    """Test the full 6-channel tensor normalization."""

    def test_3d_shape(self):
        """3D array (6, H, W) normalizes correctly."""
        arr = np.random.uniform(-25, 0, size=(6, 4, 4)).astype(np.float32)
        # Fix channels to their expected ranges
        arr[0:2] = np.random.uniform(-25, 0, size=(2, 4, 4))  # VV, VH post
        arr[2:4] = np.random.uniform(-10, 2, size=(2, 4, 4))  # ΔVV, ΔVH
        arr[4] = np.random.uniform(0, 100, size=(4, 4))       # HAND
        arr[5] = np.random.uniform(0, 45, size=(4, 4))        # Slope
        norm = normalize_tensor_multitemporal(arr)
        assert norm.shape == (6, 4, 4)
        assert norm.dtype == np.float32
        assert norm.min() >= 0.0 and norm.max() <= 1.0

    def test_4d_shape(self):
        """4D array (B, 6, H, W) normalizes correctly."""
        arr = np.random.uniform(-20, 0, size=(2, 6, 4, 4)).astype(np.float32)
        arr[:, 2:4] = np.random.uniform(-10, 2, size=(2, 2, 4, 4))
        arr[:, 4] = np.random.uniform(0, 100, size=(2, 4, 4))
        arr[:, 5] = np.random.uniform(0, 45, size=(2, 4, 4))
        norm = normalize_tensor_multitemporal(arr)
        assert norm.shape == (2, 6, 4, 4)
        assert norm.min() >= 0.0 and norm.max() <= 1.0

    def test_wrong_channels_raises(self):
        """Wrong number of channels raises ValueError."""
        arr = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="6 channels"):
            normalize_tensor_multitemporal(arr)

    def test_round_trip(self):
        """normalize → denormalize round-trips for all 6 channels."""
        arr = np.zeros((6, 4, 4), dtype=np.float32)
        arr[0:2] = np.random.uniform(-25, 0, size=(2, 4, 4))
        arr[2:4] = np.random.uniform(-10, 2, size=(2, 4, 4))
        arr[4] = np.random.uniform(0, 100, size=(4, 4))
        arr[5] = np.random.uniform(0, 45, size=(4, 4))
        norm = normalize_tensor_multitemporal(arr)
        recovered = denormalize_tensor_multitemporal(norm)
        np.testing.assert_allclose(recovered, arr, atol=1e-3)

    def test_channel_order_preserved(self):
        """Channel order is (VV_post, VH_post, dVV, dVH, HAND, Slope)."""
        arr = np.zeros((6, 1, 1), dtype=np.float32)
        arr[0] = -20.0  # VV_post
        arr[1] = -25.0  # VH_post
        arr[2] = -8.0   # dVV (flood drop)
        arr[3] = -6.0   # dVH
        arr[4] = 5.0    # HAND
        arr[5] = 30.0   # Slope
        norm = normalize_tensor_multitemporal(arr)
        # Verify each channel uses the correct normalization
        np.testing.assert_allclose(norm[0], normalize_sar(arr[0:1])[0], atol=1e-5)
        np.testing.assert_allclose(norm[2], normalize_delta_sar(arr[2:3])[0], atol=1e-5)
        np.testing.assert_allclose(norm[4], normalize_hand(arr[4:5])[0], atol=1e-5)
        np.testing.assert_allclose(norm[5], normalize_slope(arr[5:6])[0], atol=1e-5)


# ---------------------------------------------------------------------------
# Synthetic Δσ⁰ construction
# ---------------------------------------------------------------------------

class TestSyntheticDeltaSar:
    """Test the synthetic Δσ⁰ construction for training augmentation."""

    def test_water_pixels_have_negative_delta(self):
        """Water pixels get a large negative Δσ⁰ (flood drop signal)."""
        from siren.ml.dataset import _synthetic_delta_sar, _SYNTHETIC_FLOOD_DROP_DB
        rng = np.random.default_rng(42)
        sar_post = np.zeros((2, 4, 4), dtype=np.float32)
        water = np.ones((4, 4), dtype=np.float32)  # all water
        delta = _synthetic_delta_sar(sar_post, water, rng)
        # All pixels should be approximately -8 dB (±noise)
        assert delta.mean() < -5.0  # well below 0
        assert delta.std() < 3.0   # noise is small

    def test_land_pixels_have_near_zero_delta(self):
        """Land pixels get Δσ⁰ ≈ 0 (no change)."""
        from siren.ml.dataset import _synthetic_delta_sar
        rng = np.random.default_rng(42)
        sar_post = np.zeros((2, 4, 4), dtype=np.float32)
        water = np.zeros((4, 4), dtype=np.float32)  # all land
        delta = _synthetic_delta_sar(sar_post, water, rng)
        # All pixels should be approximately 0 (±noise)
        assert abs(delta.mean()) < 1.0
        assert delta.std() < 3.0

    def test_reproducible_with_seed(self):
        """Same seed produces same Δσ⁰ (Hard Rule 6)."""
        from siren.ml.dataset import _synthetic_delta_sar
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        sar = np.random.uniform(-25, 0, (2, 4, 4)).astype(np.float32)
        water = np.random.randint(0, 2, (4, 4)).astype(np.float32)
        d1 = _synthetic_delta_sar(sar, water, rng1)
        d2 = _synthetic_delta_sar(sar, water, rng2)
        np.testing.assert_array_equal(d1, d2)

    def test_shape_preserved(self):
        """Output shape matches input SAR shape."""
        from siren.ml.dataset import _synthetic_delta_sar
        rng = np.random.default_rng(42)
        sar = np.zeros((2, 8, 8), dtype=np.float32)
        water = np.zeros((8, 8), dtype=np.float32)
        delta = _synthetic_delta_sar(sar, water, rng)
        assert delta.shape == (2, 8, 8)
        assert delta.dtype == np.float32


# ---------------------------------------------------------------------------
# MultiTemporalWaterDataset structure
# ---------------------------------------------------------------------------

class TestMultiTemporalWaterDataset:
    """Test the MultiTemporalWaterDataset class structure."""

    def test_class_exists(self):
        """The MultiTemporalWaterDataset class is importable."""
        from siren.ml.dataset import MultiTemporalWaterDataset
        assert MultiTemporalWaterDataset is not None

    def test_class_docstring(self):
        """The class documents the 6-channel tensor format."""
        from siren.ml.dataset import MultiTemporalWaterDataset
        doc = MultiTemporalWaterDataset.__doc__
        assert "6-channel" in doc
        assert "VV_post" in doc
        assert "HAND" in doc
        assert "Δσ⁰" in doc or "delta" in doc.lower()

    def test_init_accepts_pre_sar_dir(self):
        """The constructor accepts a pre_sar_dir for real paired SAR."""
        import inspect
        from siren.ml.dataset import MultiTemporalWaterDataset
        sig = inspect.signature(MultiTemporalWaterDataset.__init__)
        assert "pre_sar_dir" in sig.parameters
        assert "seed" in sig.parameters

    def test_init_accepts_dem_path(self):
        """The constructor accepts dem_path for DEM co-registration."""
        import inspect
        from siren.ml.dataset import MultiTemporalWaterDataset
        sig = inspect.signature(MultiTemporalWaterDataset.__init__)
        assert "dem_path" in sig.parameters
        assert "use_copernicus_dem" in sig.parameters


# ---------------------------------------------------------------------------
# Model compatibility
# ---------------------------------------------------------------------------

class TestModelCompatibility:
    """Test that WaterResUNet accepts 6 input channels."""

    def test_model_accepts_6_channels(self):
        """WaterResUNet(in_channels=6) creates successfully."""
        import torch
        from siren.ml.model import WaterResUNet
        model = WaterResUNet(in_channels=6, base_channels=8)
        assert model.in_channels == 6

    def test_model_forward_6_channels(self):
        """WaterResUNet(in_channels=6) forward pass produces correct output."""
        import torch
        from siren.ml.model import WaterResUNet
        model = WaterResUNet(in_channels=6, base_channels=8)
        x = torch.randn(2, 6, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_model_forward_with_features_6_channels(self):
        """forward_with_features works with 6 channels (for DANN)."""
        import torch
        from siren.ml.model import WaterResUNet
        model = WaterResUNet(in_channels=6, base_channels=8)
        x = torch.randn(1, 6, 64, 64)
        logits, features = model.forward_with_features(x)
        assert logits.shape == (1, 1, 64, 64)
        assert features.ndim == 4  # bottleneck features


# ---------------------------------------------------------------------------
# Training script structure
# ---------------------------------------------------------------------------

class TestTrainingScript:
    """Test the 6-channel training script structure."""

    def test_module_exists(self):
        """The training script module exists and imports."""
        from siren.ml import train_water_resunet_6ch
        assert hasattr(train_water_resunet_6ch, "main")
        assert hasattr(train_water_resunet_6ch, "train_model")
        assert hasattr(train_water_resunet_6ch, "evaluate_model")

    def test_main_has_argparse(self):
        """The main function accepts CLI arguments."""
        import inspect
        from siren.ml.train_water_resunet_6ch import main
        # main should accept argv list
        sig = inspect.signature(main)
        assert "argv" in sig.parameters or len(sig.parameters) == 0

    def test_iou_score_function(self):
        """The IoU score function works correctly."""
        from siren.ml.train_water_resunet_6ch import iou_score
        pred = np.array([[1, 1, 0], [0, 1, 1]])
        target = np.array([[1, 0, 0], [0, 1, 1]])
        iou = iou_score(pred, target)
        # intersection = 3, union = 4 → 0.75
        assert abs(iou - 0.75) < 1e-5
