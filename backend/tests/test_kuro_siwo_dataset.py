"""Tests for the Kuro Siwo GRD dataset adapter (PRD v4.7 §17.3).

Tests the local .tar shard reader, 6-channel tensor construction,
label conversion, and truncated-tar handling against synthetic fixtures.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from siren.ml.kuro_siwo_dataset import (
    KuroSiwoDataset,
    _build_tensor,
    _index_shard,
    _linear_to_db,
    KURO_SIWO_ROOT,
)


# --------------------------------------------------------------------------- #
# Helpers: build a tiny synthetic .tar shard
# --------------------------------------------------------------------------- #

def _make_sample_arrays(
    sample_id: str,
    chip_size: int = 16,
    flood_fraction: float = 0.1,
    has_permanent_water: bool = True,
) -> dict[str, bytes]:
    """Create synthetic Kuro Siwo sample arrays as in-memory .npy bytes."""
    rng = np.random.RandomState(hash(sample_id) & 0xFFFF)
    h, w = chip_size, chip_size

    # SAR in linear scale (0 to ~1)
    flood_vv = rng.uniform(0.01, 0.5, size=(1, h, w)).astype(np.float32)
    flood_vh = rng.uniform(0.005, 0.2, size=(1, h, w)).astype(np.float32)
    sec1_vv = rng.uniform(0.01, 0.5, size=(1, h, w)).astype(np.float32)
    sec1_vh = rng.uniform(0.005, 0.2, size=(1, h, w)).astype(np.float32)
    sec2_vv = rng.uniform(0.01, 0.5, size=(1, h, w)).astype(np.float32)
    sec2_vh = rng.uniform(0.005, 0.2, size=(1, h, w)).astype(np.float32)

    # DEM
    dem = rng.uniform(100, 500, size=(1, h, w)).astype(np.float32)

    # Mask: 0=no water, 1=permanent, 2=flood
    mask = np.zeros((1, h, w), dtype=np.float32)
    n_flood = int(h * w * flood_fraction)
    flood_idx = rng.choice(h * w, n_flood, replace=False)
    mask.flat[flood_idx] = 2
    if has_permanent_water:
        n_perm = int(h * w * 0.05)
        perm_idx = rng.choice(
            [i for i in range(h * w) if i not in flood_idx],
            n_perm, replace=False,
        )
        mask.flat[perm_idx] = 1

    # Valid mask (all valid)
    valid_mask = np.ones((1, h, w), dtype=np.float32)

    # Info JSON
    info = {
        "flood_date": "2020-10-05 18:00:00",
        "pflood": flood_fraction * 100,
        "pwater": 5.0 if has_permanent_water else 0.0,
        "grid_id": f"test-{sample_id}",
    }

    arrays = {
        "flood_vv": flood_vv, "flood_vh": flood_vh,
        "sec1_vv": sec1_vv, "sec1_vh": sec1_vh,
        "sec2_vv": sec2_vv, "sec2_vh": sec2_vh,
        "dem": dem, "mask": mask, "valid_mask": valid_mask,
    }

    result = {}
    for name, arr in arrays.items():
        buf = io.BytesIO()
        np.save(buf, arr)
        result[f"{sample_id}.{name}.npy"] = buf.getvalue()
    result[f"{sample_id}.info.json"] = json.dumps(info).encode()

    return result


def _write_shard(path: Path, samples: list[dict[str, bytes]]) -> None:
    """Write a list of sample file dicts into a .tar shard."""
    with tarfile.open(str(path), "w") as tar:
        for sample_files in samples:
            for name, data in sample_files.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def _write_truncated_shard(path: Path, samples: list[dict[str, bytes]]) -> None:
    """Write a .tar shard and truncate it after the first sample."""
    full_path = path.parent / (path.name + ".full")
    _write_shard(full_path, samples)
    # Truncate: keep only the first sample's data (~first half)
    with open(full_path, "rb") as f:
        data = f.read()
    # Find a truncation point after the first sample
    # Each file has a 512-byte header + data padded to 512 bytes
    # First sample has 10 files, so ~10 * (512 + ceil(size/512)*512)
    # Just truncate to 60% of the file
    truncate_at = int(len(data) * 0.6)
    with open(path, "wb") as f:
        f.write(data[:truncate_at])
    full_path.unlink()


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_kuro_siwo_root(tmp_path):
    """Create a minimal Kuro Siwo directory structure with 2 train + 1 test shard."""
    root = tmp_path / "Kuro Siwo"
    train_dir = root / "train_GRD"
    test_dir = root / "test_GRD"
    train_dir.mkdir(parents=True)
    test_dir.mkdir()

    # Train: 2 shards, 3 samples each
    for shard_idx in range(2):
        samples = [_make_sample_arrays(f"{i:06d}") for i in range(3)]
        _write_shard(train_dir / f"shard-{shard_idx:05d}.tar", samples)

    # Test: 1 shard, 2 samples
    samples = [_make_sample_arrays(f"{i:06d}", flood_fraction=0.2) for i in range(2)]
    _write_shard(test_dir / "shard-00004.tar", samples)

    return root


@pytest.fixture
def truncated_root(tmp_path):
    """Create a Kuro Siwo directory with one truncated shard."""
    root = tmp_path / "Kuro Siwo"
    train_dir = root / "train_GRD"
    train_dir.mkdir(parents=True)

    samples = [_make_sample_arrays(f"{i:06d}") for i in range(5)]
    _write_truncated_shard(train_dir / "shard-00000.tar", samples)

    return root


class TestLinearToDb:
    def test_basic_conversion(self):
        """10 * log10(x) conversion."""
        assert _linear_to_db(np.array([1.0])) == pytest.approx(0.0, abs=0.1)
        assert _linear_to_db(np.array([0.1])) == pytest.approx(-10.0, abs=0.1)
        assert _linear_to_db(np.array([0.01])) == pytest.approx(-20.0, abs=0.1)

    def test_zero_handling(self):
        """Zero values don't cause -inf."""
        result = _linear_to_db(np.array([0.0]))
        assert np.isfinite(result)
        assert result < -30


class TestBuildTensor:
    def test_output_shape_and_dtype(self):
        """6-channel tensor with correct shape and dtype."""
        h, w = 16, 16
        flood_vv = np.random.rand(1, h, w).astype(np.float32) * 0.5
        flood_vh = np.random.rand(1, h, w).astype(np.float32) * 0.2
        pre_vv = np.random.rand(1, h, w).astype(np.float32) * 0.5
        pre_vh = np.random.rand(1, h, w).astype(np.float32) * 0.2

        tensor = _build_tensor(flood_vv, flood_vh, pre_vv, pre_vh)
        assert tensor.shape == (6, h, w)
        assert tensor.dtype == np.float32

    def test_all_channels_in_unit_range(self):
        """All 6 channels are in [0, 1]."""
        h, w = 16, 16
        flood_vv = np.random.rand(1, h, w).astype(np.float32) * 0.5
        flood_vh = np.random.rand(1, h, w).astype(np.float32) * 0.2
        pre_vv = np.random.rand(1, h, w).astype(np.float32) * 0.5
        pre_vh = np.random.rand(1, h, w).astype(np.float32) * 0.2

        tensor = _build_tensor(flood_vv, flood_vh, pre_vv, pre_vh)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_delta_sigma_computed_from_db(self):
        """Δσ⁰ channels are post_dB - pre_dB, normalized."""
        h, w = 4, 4
        # Post = 0.1 linear, pre = 0.2 linear → ΔVV should be negative (dB)
        flood_vv = np.full((1, h, w), 0.1, dtype=np.float32)
        pre_vv = np.full((1, h, w), 0.2, dtype=np.float32)
        flood_vh = np.full((1, h, w), 0.05, dtype=np.float32)
        pre_vh = np.full((1, h, w), 0.1, dtype=np.float32)

        tensor = _build_tensor(flood_vv, flood_vh, pre_vv, pre_vh)

        # ΔVV in dB: 10*log10(0.1) - 10*log10(0.2) = -10 - (-7) = -3 dB
        # Normalized: (-3 + 15) / 20 = 0.6
        expected_dvv = (np.clip(-3.0, -15, 5) + 15) / 20
        assert tensor[4].mean() == pytest.approx(expected_dvv, abs=0.01)

    def test_identical_pre_post_gives_zero_delta(self):
        """When pre == post, Δσ⁰ = 0 dB → normalized to 0.75."""
        h, w = 4, 4
        vv = np.full((1, h, w), 0.1, dtype=np.float32)
        vh = np.full((1, h, w), 0.05, dtype=np.float32)

        tensor = _build_tensor(vv, vh, vv, vh)
        # 0 dB → (0 + 15) / 20 = 0.75
        assert tensor[4].mean() == pytest.approx(0.75, abs=0.01)
        assert tensor[5].mean() == pytest.approx(0.75, abs=0.01)


class TestIndexShard:
    def test_indexes_all_samples(self, fake_kuro_siwo_root):
        """Index contains all samples from a complete shard."""
        shard = fake_kuro_siwo_root / "train_GRD" / "shard-00000.tar"
        index = _index_shard(shard)
        assert len(index) == 3
        for entry in index:
            assert "sample_id" in entry
            assert "flood_vv" in entry
            assert "mask" in entry

    def test_truncated_shard_returns_partial_index(self, truncated_root):
        """Truncated shards return whatever was indexed before truncation."""
        shard = truncated_root / "train_GRD" / "shard-00000.tar"
        index = _index_shard(shard)
        # Should have at least 1 sample (truncation is at 60%)
        assert len(index) >= 1


class TestKuroSiwoDataset:
    def test_loads_train_split(self, fake_kuro_siwo_root):
        """Train split loads from train_GRD/."""
        ds = KuroSiwoDataset("train", root=fake_kuro_siwo_root)
        assert len(ds) == 6  # 2 shards × 3 samples

    def test_loads_test_split(self, fake_kuro_siwo_root):
        """Test split loads from test_GRD/."""
        ds = KuroSiwoDataset("test", root=fake_kuro_siwo_root)
        assert len(ds) == 2

    def test_missing_directory_raises(self, tmp_path):
        """Missing shard directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            KuroSiwoDataset("train", root=tmp_path / "nonexistent")

    def test_empty_directory_raises(self, tmp_path):
        """Directory with no .tar shards raises FileNotFoundError."""
        (tmp_path / "train_GRD").mkdir()
        with pytest.raises(FileNotFoundError, match="No .tar shards"):
            KuroSiwoDataset("train", root=tmp_path)

    def test_getitem_returns_correct_shapes(self, fake_kuro_siwo_root):
        """__getitem__ returns 6-channel sar, binary water, valid mask."""
        ds = KuroSiwoDataset("train", root=fake_kuro_siwo_root)
        sample = ds[0]

        assert sample["sar"].shape == (6, 16, 16)
        assert sample["water"].shape == (16, 16)
        assert sample["valid"].shape == (16, 16)

    def test_sar_in_unit_range(self, fake_kuro_siwo_root):
        """SAR tensor is in [0, 1]."""
        ds = KuroSiwoDataset("train", root=fake_kuro_siwo_root)
        sample = ds[0]
        assert sample["sar"].min() >= 0.0
        assert sample["sar"].max() <= 1.0

    def test_water_is_binary(self, fake_kuro_siwo_root):
        """Water mask is binary {0, 1}."""
        ds = KuroSiwoDataset("train", root=fake_kuro_siwo_root)
        sample = ds[0]
        unique = set(sample["water"].unique().tolist())
        assert unique.issubset({0.0, 1.0})

    def test_valid_is_binary(self, fake_kuro_siwo_root):
        """Valid mask is binary {0, 1}."""
        ds = KuroSiwoDataset("train", root=fake_kuro_siwo_root)
        sample = ds[0]
        unique = set(sample["valid"].unique().tolist())
        assert unique.issubset({0.0, 1.0})

    def test_water_includes_permanent_and_flood(self, fake_kuro_siwo_root):
        """Water mask includes both permanent water (1) and flood (2)."""
        ds = KuroSiwoDataset("train", root=fake_kuro_siwo_root)
        sample = ds[0]
        # The fixture has both permanent water and flood pixels
        assert sample["water"].sum() > 0

    def test_dem_returned(self, fake_kuro_siwo_root):
        """DEM array is returned (unnormalized, for viz)."""
        ds = KuroSiwoDataset("train", root=fake_kuro_siwo_root)
        sample = ds[0]
        assert "dem" in sample
        assert sample["dem"].shape == (16, 16)

    def test_sample_id_and_flood_date(self, fake_kuro_siwo_root):
        """Sample ID and flood date are returned from metadata."""
        ds = KuroSiwoDataset("train", root=fake_kuro_siwo_root)
        sample = ds[0]
        assert sample["sample_id"] == "000000"
        assert "2020" in sample["flood_date"]

    def test_sec2_pre_event(self, fake_kuro_siwo_root):
        """Using sec2 as pre-event reads the second pre-flood SAR."""
        ds_sec1 = KuroSiwoDataset("train", root=fake_kuro_siwo_root, pre_event="sec1")
        ds_sec2 = KuroSiwoDataset("train", root=fake_kuro_siwo_root, pre_event="sec2")
        s1 = ds_sec1[0]
        s2 = ds_sec2[0]
        # VV_post and VH_post should be the same
        assert torch_allclose(s1["sar"][0], s2["sar"][0])
        # VV_pre and VH_pre should differ (sec1 vs sec2)
        assert not torch_allclose(s1["sar"][2], s2["sar"][2])

    def test_max_samples_cap(self, fake_kuro_siwo_root):
        """max_samples caps the dataset size."""
        ds = KuroSiwoDataset("train", root=fake_kuro_siwo_root, max_samples=3)
        assert len(ds) == 3

    def test_truncated_shards_handled(self, truncated_root):
        """Truncated shards don't crash the dataset."""
        ds = KuroSiwoDataset("train", root=truncated_root)
        assert len(ds) >= 1
        sample = ds[0]
        assert sample["sar"].shape == (6, 16, 16)

    def test_events(self, fake_kuro_siwo_root):
        """events() returns sample ID prefixes."""
        ds = KuroSiwoDataset("train", root=fake_kuro_siwo_root)
        events = ds.events()
        assert len(events) > 0


def torch_allclose(a, b, atol=1e-6):
    """Helper for comparing torch tensors."""
    import torch
    return torch.allclose(a, b, atol=atol)
