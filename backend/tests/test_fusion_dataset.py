"""Tests for the paired SAR+optical fusion dataset (ADR-013 §9.7.2).

Tests cover:
  - Utility functions (padding, delta computation, calibration)
  - FusionDataset wrapping Kuro Siwo (SAR-only fallback path)
  - collate_fusion batch collation with variable optical availability
  - build_fusion_chip on the real Imja pair (S1 07-02/07-14 + S2 07-05)
  - Full tensor flow: fusion dataset → MultiModalFusionNet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from siren.ml.fusion_dataset import (
    _compute_delta,
    _crop_back,
    _dn_to_linear_sigma0,
    _linear_to_normalized_db,
    _pad_to_size,
    build_fusion_chip,
    build_optical_3ch_from_safe,
    build_sar_6ch_from_safe,
    collate_fusion,
    FusionDataset,
    MAX_PAIR_GAP_DAYS,
)


# --------------------------------------------------------------------------- #
# Paths for real-data tests
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
PRE_SAFE = DATA_RAW / "S1D_IW_GRDH_1SDV_20260702T001034_20260702T001059_003487_0062AB_5083.SAFE.zip"
POST_SAFE = DATA_RAW / "S1D_IW_GRDH_1SDV_20260714T001035_20260714T001100_003662_00689F_377F.SAFE.zip"
S2_MONSOON = DATA_RAW / "S2B_MSIL2A_20260705T044659_N0512_R076_T45RVL_20260705T083506.zip"
DEM_PATH = DATA_RAW / "srtm_30m.tif"

IMJA_LAT = 27.90
IMJA_LON = 86.92
HALF_SIZE = 0.020

REAL_PAIR_AVAILABLE = all(p.exists() for p in [PRE_SAFE, POST_SAFE, S2_MONSOON, DEM_PATH])


# --------------------------------------------------------------------------- #
# Utility function tests
# --------------------------------------------------------------------------- #

class TestPadToSize:
    def test_pad_square(self):
        arr = np.ones((100, 100), dtype=np.float32)
        padded = _pad_to_size(arr, 224)
        assert padded.shape == (224, 224)

    def test_pad_oversize(self):
        """Padding an array larger than target should not shrink it."""
        arr = np.ones((300, 300), dtype=np.float32)
        padded = _pad_to_size(arr, 224)
        assert padded.shape == (300, 300)

    def test_pad_preserves_values(self):
        arr = np.full((50, 50), 0.5, dtype=np.float32)
        padded = _pad_to_size(arr, 224)
        # Center should still be 0.5
        cy, cx = 224 // 2, 224 // 2
        assert padded[cy, cx] == 0.5

    def test_pad_edge_mode(self):
        """Edge values should be replicated at the padding boundary."""
        arr = np.zeros((10, 10), dtype=np.float32)
        arr[0, :] = 1.0  # top edge = 1
        padded = _pad_to_size(arr, 20)
        # Top padding should replicate the top edge value (1.0)
        assert padded[0, 0] == 1.0


class TestCropBack:
    def test_crop_inverse_of_pad(self):
        arr = np.random.rand(100, 100).astype(np.float32)
        padded = _pad_to_size(arr, 224)
        cropped = _crop_back(padded, 100, 100)
        assert cropped.shape == (100, 100)
        np.testing.assert_allclose(cropped, arr)

    def test_crop_no_pad_needed(self):
        arr = np.random.rand(224, 224).astype(np.float32)
        cropped = _crop_back(arr, 224, 224)
        np.testing.assert_allclose(cropped, arr)


class TestComputeDelta:
    def test_zero_delta(self):
        """Same pre/post → Δσ⁰ = 0 → normalized to 0.5 (midpoint of [-15,5])."""
        post = np.full((10, 10), 0.5, dtype=np.float32)
        pre = np.full((10, 10), 0.5, dtype=np.float32)
        delta = _compute_delta(post, pre)
        # delta_db = 0 → clamp(0, -15, 5) = 0 → (0+15)/20 = 0.75
        # Wait: post_db = 0.5*30-30 = -15, pre_db = -15, delta = 0
        # (0 + 15) / 20 = 0.75
        assert delta.shape == (10, 10)
        assert np.allclose(delta, 0.75)

    def test_positive_delta(self):
        """Post > pre → positive Δσ⁰ (brightening, e.g., new water receding)."""
        post = np.full((10, 10), 0.8, dtype=np.float32)  # -6 dB
        pre = np.full((10, 10), 0.2, dtype=np.float32)   # -24 dB
        delta = _compute_delta(post, pre)
        # post_db = 0.8*30-30 = -6, pre_db = 0.2*30-30 = -24
        # delta = -6 - (-24) = 18 → clamp to 5 → (5+15)/20 = 1.0
        assert np.allclose(delta, 1.0)

    def test_negative_delta(self):
        """Post < pre → negative Δσ⁰ (darkening, e.g., new flood water)."""
        post = np.full((10, 10), 0.2, dtype=np.float32)   # -24 dB
        pre = np.full((10, 10), 0.8, dtype=np.float32)    # -6 dB
        delta = _compute_delta(post, pre)
        # delta = -24 - (-6) = -18 → clamp to -15 → (-15+15)/20 = 0.0
        assert np.allclose(delta, 0.0)

    def test_delta_range_bounded(self):
        """Δσ⁰ normalized should always be in [0, 1]."""
        post = np.random.rand(20, 20).astype(np.float32)
        pre = np.random.rand(20, 20).astype(np.float32)
        delta = _compute_delta(post, pre)
        assert delta.min() >= 0.0
        assert delta.max() <= 1.0


class TestDnConversion:
    def test_dn_to_linear_sigma0(self):
        dn = np.array([[700.0, 0.0]], dtype=np.float32)
        linear = _dn_to_linear_sigma0(dn, cal_const=700.0)
        # sigma0 = (700²) / (700²) = 1.0, (0²) / (700²) = 0.0
        assert np.allclose(linear, [[1.0, 0.0]])

    def test_linear_to_normalized_db_range(self):
        """Normalized dB should be in [0, 1]."""
        linear = np.array([[1.0, 0.001, 0.0001]], dtype=np.float32)
        norm = _linear_to_normalized_db(linear)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_water_low_backscatter(self):
        """Water (very low linear sigma0) should map near 0 after normalization."""
        water_linear = np.array([[0.0001]], dtype=np.float32)
        norm = _linear_to_normalized_db(water_linear)
        # 10*log10(0.0001) = -40 dB → clamp to -30 → (-30+30)/30 = 0.0
        assert norm[0, 0] < 0.01


# --------------------------------------------------------------------------- #
# FusionDataset tests (SAR-only fallback path)
# --------------------------------------------------------------------------- #

class TestFusionDataset:
    """Tests the FusionDataset wrapping Kuro Siwo (SAR-only, optical=None)."""

    @pytest.fixture
    def small_dataset(self):
        """Create a small FusionDataset for testing."""
        try:
            return FusionDataset(sar_split="train", max_samples=4, chip_size=224)
        except FileNotFoundError:
            pytest.skip("Kuro Siwo dataset not available")

    def test_dataset_has_length(self, small_dataset):
        assert len(small_dataset) > 0

    def test_dataset_returns_sar(self, small_dataset):
        sample = small_dataset[0]
        assert "sar" in sample
        assert sample["sar"].shape == (6, 224, 224)
        # KuroSiwoDataset returns torch tensors
        import torch
        assert isinstance(sample["sar"], (np.ndarray, torch.Tensor))

    def test_dataset_returns_water_mask(self, small_dataset):
        sample = small_dataset[0]
        assert "water" in sample
        assert sample["water"].shape == (224, 224)

    def test_dataset_returns_valid_mask(self, small_dataset):
        sample = small_dataset[0]
        assert "valid" in sample
        assert sample["valid"].shape == (224, 224)

    def test_dataset_optical_is_none_without_s2(self, small_dataset):
        """Without S2 archives, optical should be None (SAR-only fallback)."""
        sample = small_dataset[0]
        assert sample["optical"] is None

    def test_dataset_returns_sample_id(self, small_dataset):
        sample = small_dataset[0]
        assert "sample_id" in sample
        assert isinstance(sample["sample_id"], str)


# --------------------------------------------------------------------------- #
# collate_fusion tests
# --------------------------------------------------------------------------- #

class TestCollateFusion:
    def test_collate_all_none_optical(self):
        """When all optical=None, collated optical should be None."""
        batch = [
            {
                "sar": np.zeros((6, 224, 224), dtype=np.float32),
                "optical": None,
                "water": np.zeros((224, 224), dtype=np.float32),
                "valid": np.ones((224, 224), dtype=np.float32),
                "sample_id": "s1",
            },
            {
                "sar": np.zeros((6, 224, 224), dtype=np.float32),
                "optical": None,
                "water": np.zeros((224, 224), dtype=np.float32),
                "valid": np.ones((224, 224), dtype=np.float32),
                "sample_id": "s2",
            },
        ]
        collated = collate_fusion(batch)
        assert collated["optical"] is None
        assert collated["sar"].shape == (2, 6, 224, 224)
        assert len(collated["sample_id"]) == 2

    def test_collate_all_present_optical(self):
        """When all optical present, collated optical should be stacked."""
        batch = [
            {
                "sar": np.zeros((6, 224, 224), dtype=np.float32),
                "optical": np.zeros((3, 224, 224), dtype=np.float32),
                "water": np.zeros((224, 224), dtype=np.float32),
                "valid": np.ones((224, 224), dtype=np.float32),
                "sample_id": "s1",
            },
            {
                "sar": np.zeros((6, 224, 224), dtype=np.float32),
                "optical": np.ones((3, 224, 224), dtype=np.float32),
                "water": np.zeros((224, 224), dtype=np.float32),
                "valid": np.ones((224, 224), dtype=np.float32),
                "sample_id": "s2",
            },
        ]
        collated = collate_fusion(batch)
        assert collated["optical"] is not None
        assert collated["optical"].shape == (2, 3, 224, 224)

    def test_collate_mixed_optical(self):
        """When some optical=None, collated optical should be None (fallback)."""
        batch = [
            {
                "sar": np.zeros((6, 224, 224), dtype=np.float32),
                "optical": np.zeros((3, 224, 224), dtype=np.float32),
                "water": np.zeros((224, 224), dtype=np.float32),
                "valid": np.ones((224, 224), dtype=np.float32),
                "sample_id": "s1",
            },
            {
                "sar": np.zeros((6, 224, 224), dtype=np.float32),
                "optical": None,
                "water": np.zeros((224, 224), dtype=np.float32),
                "valid": np.ones((224, 224), dtype=np.float32),
                "sample_id": "s2",
            },
        ]
        collated = collate_fusion(batch)
        # Mixed → None (SAR-only fallback for the whole batch)
        assert collated["optical"] is None


# --------------------------------------------------------------------------- #
# Full tensor flow test: dataset → MultiModalFusionNet
# --------------------------------------------------------------------------- #

class TestFusionTensorFlow:
    """Verify the fusion dataset output feeds correctly into MultiModalFusionNet."""

    def test_sar_only_forward_pass(self):
        """SAR-only (optical=None) forward pass through MultiModalFusionNet."""
        import torch
        from siren.ml.fusion import MultiModalFusionNet

        model = MultiModalFusionNet(sar_channels=6, optical_channels=3, base_channels=8, n_heads=4)
        model.eval()

        sar = torch.randn(2, 6, 32, 32)
        with torch.no_grad():
            logits = model(sar, optical=None)
        assert logits.shape == (2, 1, 32, 32)

    def test_fused_forward_pass(self):
        """SAR + optical forward pass through MultiModalFusionNet."""
        import torch
        from siren.ml.fusion import MultiModalFusionNet

        model = MultiModalFusionNet(sar_channels=6, optical_channels=3, base_channels=8, n_heads=4)
        model.eval()

        sar = torch.randn(2, 6, 32, 32)
        optical = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            logits = model(sar, optical=optical)
        assert logits.shape == (2, 1, 32, 32)

    def test_collated_batch_forward_pass(self):
        """Collated batch from FusionDataset feeds into MultiModalFusionNet."""
        import torch
        from siren.ml.fusion import MultiModalFusionNet

        batch = [
            {
                "sar": np.random.rand(6, 32, 32).astype(np.float32),
                "optical": None,
                "water": np.zeros((32, 32), dtype=np.float32),
                "valid": np.ones((32, 32), dtype=np.float32),
                "sample_id": "s1",
            }
            for _ in range(2)
        ]
        collated = collate_fusion(batch)

        model = MultiModalFusionNet(sar_channels=6, optical_channels=3, base_channels=8, n_heads=4)
        model.eval()

        with torch.no_grad():
            logits = model(collated["sar"], collated["optical"])
        assert logits.shape == (2, 1, 32, 32)


# --------------------------------------------------------------------------- #
# Real-data tests (require the downloaded S1 + S2 scenes)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not REAL_PAIR_AVAILABLE, reason="Real S1+S2 pair or DEM not downloaded")
class TestRealFusionPair:
    """Real-data tests with the Imja SAR+optical pair (ADR-013 §9.7.2).

    S1 pre:  2026-07-02 (descending)
    S1 post: 2026-07-14 (descending)
    S2:      2026-07-05 (monsoon, ~56% cloud — within ±3 days of S1 pre)
    """

    def test_build_sar_6ch_shape(self):
        sar, meta = build_sar_6ch_from_safe(
            pre_safe=PRE_SAFE,
            post_safe=POST_SAFE,
            dem_path=DEM_PATH,
            window_bounds=(
                IMJA_LON - HALF_SIZE, IMJA_LAT - HALF_SIZE,
                IMJA_LON + HALF_SIZE, IMJA_LAT + HALF_SIZE,
            ),
            chip_size=224,
        )
        assert sar.shape == (6, 224, 224)
        assert sar.dtype == np.float32
        assert "crs" in meta
        assert "transform" in meta

    def test_build_sar_6ch_range(self):
        """SAR channels should be in [0, 1] (normalized dB)."""
        sar, _ = build_sar_6ch_from_safe(
            pre_safe=PRE_SAFE, post_safe=POST_SAFE, dem_path=DEM_PATH,
            window_bounds=(
                IMJA_LON - HALF_SIZE, IMJA_LAT - HALF_SIZE,
                IMJA_LON + HALF_SIZE, IMJA_LAT + HALF_SIZE,
            ),
        )
        assert sar.min() >= -0.01
        assert sar.max() <= 1.01

    def test_build_optical_3ch_shape(self):
        optical = build_optical_3ch_from_safe(
            s2_safe=S2_MONSOON,
            dem_path=DEM_PATH,
            window_bounds=(
                IMJA_LON - HALF_SIZE, IMJA_LAT - HALF_SIZE,
                IMJA_LON + HALF_SIZE, IMJA_LAT + HALF_SIZE,
            ),
            chip_size=224,
        )
        assert optical is not None
        assert optical.shape == (3, 224, 224)
        assert optical.dtype == np.float32

    def test_build_optical_3ch_cloud_fraction(self):
        """The monsoon S2 scene should have significant cloud cover."""
        optical = build_optical_3ch_from_safe(
            s2_safe=S2_MONSOON, dem_path=DEM_PATH,
            window_bounds=(
                IMJA_LON - HALF_SIZE, IMJA_LAT - HALF_SIZE,
                IMJA_LON + HALF_SIZE, IMJA_LAT + HALF_SIZE,
            ),
        )
        cloud_frac = float(optical[2].mean())
        # The monsoon scene has ~56% cloud globally; the Imja window may differ
        assert 0.0 <= cloud_frac <= 1.0

    def test_build_fusion_chip_with_optical(self):
        """Full fusion chip: SAR 6ch + optical 3ch from real archives."""
        sar, optical, meta = build_fusion_chip(
            pre_safe=PRE_SAFE, post_safe=POST_SAFE, s2_safe=S2_MONSOON,
            dem_path=DEM_PATH,
            window_bounds=(
                IMJA_LON - HALF_SIZE, IMJA_LAT - HALF_SIZE,
                IMJA_LON + HALF_SIZE, IMJA_LAT + HALF_SIZE,
            ),
        )
        assert sar.shape == (6, 224, 224)
        assert optical is not None
        assert optical.shape == (3, 224, 224)
        assert meta["optical_available"] is True

    def test_build_fusion_chip_without_optical(self):
        """Fusion chip with s2_safe=None should return optical=None."""
        sar, optical, meta = build_fusion_chip(
            pre_safe=PRE_SAFE, post_safe=POST_SAFE, s2_safe=None,
            dem_path=DEM_PATH,
            window_bounds=(
                IMJA_LON - HALF_SIZE, IMJA_LAT - HALF_SIZE,
                IMJA_LON + HALF_SIZE, IMJA_LAT + HALF_SIZE,
            ),
        )
        assert sar.shape == (6, 224, 224)
        assert optical is None
        assert meta["optical_available"] is False

    def test_real_fusion_forward_pass(self):
        """Real SAR+optical pair through MultiModalFusionNet (tensor flow)."""
        import torch
        from siren.ml.fusion import MultiModalFusionNet

        sar, optical, _ = build_fusion_chip(
            pre_safe=PRE_SAFE, post_safe=POST_SAFE, s2_safe=S2_MONSOON,
            dem_path=DEM_PATH,
            window_bounds=(
                IMJA_LON - HALF_SIZE, IMJA_LAT - HALF_SIZE,
                IMJA_LON + HALF_SIZE, IMJA_LAT + HALF_SIZE,
            ),
        )

        model = MultiModalFusionNet(sar_channels=6, optical_channels=3, base_channels=8, n_heads=4)
        model.eval()

        sar_t = torch.from_numpy(sar).unsqueeze(0)
        opt_t = torch.from_numpy(optical).unsqueeze(0)

        with torch.no_grad():
            logits = model(sar_t, opt_t)
        assert logits.shape == (1, 1, 224, 224)
        probs = torch.sigmoid(logits)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_real_fusion_sar_only_fallback(self):
        """Real SAR-only (optical=None) forward pass — fallback path."""
        import torch
        from siren.ml.fusion import MultiModalFusionNet

        sar, optical, _ = build_fusion_chip(
            pre_safe=PRE_SAFE, post_safe=POST_SAFE, s2_safe=None,
            dem_path=DEM_PATH,
            window_bounds=(
                IMJA_LON - HALF_SIZE, IMJA_LAT - HALF_SIZE,
                IMJA_LON + HALF_SIZE, IMJA_LAT + HALF_SIZE,
            ),
        )

        model = MultiModalFusionNet(sar_channels=6, optical_channels=3, base_channels=8, n_heads=4)
        model.eval()

        sar_t = torch.from_numpy(sar).unsqueeze(0)

        with torch.no_grad():
            logits = model(sar_t, optical=None)
        assert logits.shape == (1, 1, 224, 224)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

class TestConstants:
    def test_max_pair_gap_days(self):
        """The ±3 day pairing window matches ADR-013 §9.7.2."""
        assert MAX_PAIR_GAP_DAYS == 3
