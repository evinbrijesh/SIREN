"""Tests for Sentinel-2 optical preprocessing (ADR-013 §9.7.2).

Tests cover:
  - Band extraction from SAFE zip
  - NDWI / MNDWI computation
  - SCL cloud mask parsing (classes 3, 8, 9, 10)
  - Co-registration to a target grid
  - build_optical_input with a reference raster
  - Real-data test with the downloaded S2 L2A scenes
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from siren.preprocess.s2_optical import (
    CLOUD_SCL_CLASSES,
    build_optical_input,
    extract_optical_features,
    _find_band_path,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

DATA_RAW = Path(__file__).parent.parent.parent / "data" / "raw"
S2_MONSOON = DATA_RAW / "S2B_MSIL2A_20260705T044659_N0512_R076_T45RVL_20260705T083506.zip"
S2_PREMONSOON = DATA_RAW / "S2B_MSIL2A_20260526T044659_N0512_R076_T45RVL_20260526T083736.zip"


def _make_synthetic_safe(tmp_path: Path, scl_values: np.ndarray | None = None) -> Path:
    """Create a minimal synthetic S2 L2A SAFE zip for testing.

    Creates B03, B08 (10m) and B11, SCL (20m) as small GeoTIFFs inside a zip.
    """
    if scl_values is None:
        scl_values = np.array([[4, 6], [8, 10]], dtype=np.uint8)  # veg, water, cloud, cirrus

    safe_path = tmp_path / "S2_TEST_SAFE.zip"
    # 10m grid: 4x4 pixels, 10m resolution
    transform_10m = from_bounds(86.0, 27.0, 86.04, 27.04, width=4, height=4)
    # 20m grid: 2x2 pixels, 20m resolution
    transform_20m = from_bounds(86.0, 27.0, 86.04, 27.04, width=2, height=2)

    crs = "EPSG:32645"

    with zipfile.ZipFile(str(safe_path), "w") as zf:
        # B03 (Green) - 10m
        b03 = np.full((4, 4), 1000, dtype=np.uint16)
        b03_buf = io.BytesIO()
        with rasterio.open(
            b03_buf, "w", driver="GTiff", width=4, height=4, count=1,
            dtype="uint16", crs=crs, transform=transform_10m,
        ) as dst:
            dst.write(b03, 1)
        zf.writestr("GRANULE/L2A_T45RVL/IMG_DATA/R10m/T45RVL_B03_10m.jp2", b03_buf.getvalue())

        # B08 (NIR) - 10m
        b08 = np.full((4, 4), 500, dtype=np.uint16)  # lower NIR → positive NDWI (water)
        b08_buf = io.BytesIO()
        with rasterio.open(
            b08_buf, "w", driver="GTiff", width=4, height=4, count=1,
            dtype="uint16", crs=crs, transform=transform_10m,
        ) as dst:
            dst.write(b08, 1)
        zf.writestr("GRANULE/L2A_T45RVL/IMG_DATA/R10m/T45RVL_B08_10m.jp2", b08_buf.getvalue())

        # B11 (SWIR) - 20m
        b11 = np.full((2, 2), 300, dtype=np.uint16)
        b11_buf = io.BytesIO()
        with rasterio.open(
            b11_buf, "w", driver="GTiff", width=2, height=2, count=1,
            dtype="uint16", crs=crs, transform=transform_20m,
        ) as dst:
            dst.write(b11, 1)
        zf.writestr("GRANULE/L2A_T45RVL/IMG_DATA/R20m/T45RVL_B11_20m.jp2", b11_buf.getvalue())

        # SCL - 20m
        scl_buf = io.BytesIO()
        with rasterio.open(
            scl_buf, "w", driver="GTiff", width=2, height=2, count=1,
            dtype="uint8", crs=crs, transform=transform_20m,
        ) as dst:
            dst.write(scl_values.astype(np.uint8), 1)
        zf.writestr("GRANULE/L2A_T45RVL/IMG_DATA/R20m/T45RVL_SCL_20m.jp2", scl_buf.getvalue())

    return safe_path


# --------------------------------------------------------------------------- #
# Unit tests (synthetic data)
# --------------------------------------------------------------------------- #

class TestFindBandPath:
    def test_finds_b03(self, tmp_path):
        safe = _make_synthetic_safe(tmp_path)
        with zipfile.ZipFile(str(safe)) as zf:
            path = _find_band_path(zf, "B03", "10m")
            assert path is not None
            assert "B03" in path
            assert "10m" in path

    def test_finds_scl(self, tmp_path):
        safe = _make_synthetic_safe(tmp_path)
        with zipfile.ZipFile(str(safe)) as zf:
            path = _find_band_path(zf, "SCL", "20m")
            assert path is not None
            assert "SCL" in path

    def test_returns_none_for_missing(self, tmp_path):
        safe = _make_synthetic_safe(tmp_path)
        with zipfile.ZipFile(str(safe)) as zf:
            path = _find_band_path(zf, "B99", "10m")
            assert path is None


class TestExtractOpticalFeatures:
    def test_ndwi_positive_for_water(self, tmp_path):
        """B03 > B08 → positive NDWI (water has high green, low NIR)."""
        safe = _make_synthetic_safe(tmp_path)
        features = extract_optical_features(safe)
        # B03=1000, B08=500 → NDWI = (1000-500)/(1000+500) = 0.333
        assert features["ndwi"].shape == (4, 4)
        assert np.allclose(features["ndwi"], 0.333, atol=0.01)

    def test_mndwi_positive_for_water(self, tmp_path):
        """B03 > B11 → positive MNDWI (water has high green, low SWIR)."""
        safe = _make_synthetic_safe(tmp_path)
        features = extract_optical_features(safe)
        # B03=1000, B11=300 → MNDWI = (1000-300)/(1000+300) = 0.538
        assert features["mndwi"].shape == (4, 4)
        assert np.allclose(features["mndwi"], 0.538, atol=0.05)

    def test_cloud_mask_from_scl(self, tmp_path):
        """SCL classes 8, 9, 10, 3 → cloud_mask = 1; others → 0."""
        scl = np.array([[4, 8], [10, 6]], dtype=np.uint8)  # veg, cloud, cirrus, water
        safe = _make_synthetic_safe(tmp_path, scl_values=scl)
        features = extract_optical_features(safe)
        # SCL is 2x2, resampled to 4x4 (nearest-neighbor)
        assert features["cloud_mask"].shape == (4, 4)
        # Cloud fraction should be ~50% (2 of 4 SCL pixels are cloud)
        assert 0.3 < features["cloud_fraction"] < 0.7

    def test_cloud_shadow_detected(self, tmp_path):
        """SCL class 3 (cloud shadow) is also flagged as cloud."""
        scl = np.array([[3, 4], [4, 4]], dtype=np.uint8)
        safe = _make_synthetic_safe(tmp_path, scl_values=scl)
        features = extract_optical_features(safe)
        assert features["cloud_fraction"] > 0.2  # at least some shadow pixels

    def test_no_clouds_when_all_clear(self, tmp_path):
        """All clear SCL classes → cloud fraction = 0."""
        scl = np.array([[4, 6], [5, 2]], dtype=np.uint8)  # veg, water, soil, dark
        safe = _make_synthetic_safe(tmp_path, scl_values=scl)
        features = extract_optical_features(safe)
        assert features["cloud_fraction"] < 0.01

    def test_output_keys_present(self, tmp_path):
        safe = _make_synthetic_safe(tmp_path)
        features = extract_optical_features(safe)
        assert "ndwi" in features
        assert "mndwi" in features
        assert "cloud_mask" in features
        assert "cloud_fraction" in features
        assert "meta" in features

    def test_ndwi_range_bounded(self, tmp_path):
        """NDWI should be in [-1, 1]."""
        safe = _make_synthetic_safe(tmp_path)
        features = extract_optical_features(safe)
        assert features["ndwi"].min() >= -1.01
        assert features["ndwi"].max() <= 1.01

    def test_mndwi_range_bounded(self, tmp_path):
        """MNDWI should be in [-1, 1]."""
        safe = _make_synthetic_safe(tmp_path)
        features = extract_optical_features(safe)
        assert features["mndwi"].min() >= -1.01
        assert features["mndwi"].max() <= 1.01

    def test_missing_bands_raises(self, tmp_path):
        """Missing bands should raise ValueError."""
        safe = tmp_path / "empty.zip"
        with zipfile.ZipFile(str(safe), "w") as zf:
            zf.writestr("dummy.txt", "test")
        with pytest.raises(ValueError, match="Missing bands"):
            extract_optical_features(safe)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_optical_features(tmp_path / "nonexistent.zip")

    def test_coregister_to_target_grid(self, tmp_path):
        """Reprojection to a target grid should produce the right shape."""
        safe = _make_synthetic_safe(tmp_path)
        features = extract_optical_features(
            safe,
            target_crs="EPSG:32645",
            target_bounds=(470000, 2990000, 510000, 3030000),
            target_shape=(8, 8),
        )
        assert features["ndwi"].shape == (8, 8)
        assert features["mndwi"].shape == (8, 8)
        assert features["cloud_mask"].shape == (8, 8)


class TestBuildOpticalInput:
    def test_output_shape(self, tmp_path):
        """build_optical_input should return (3, H, W)."""
        safe = _make_synthetic_safe(tmp_path)
        # Create a reference raster
        ref_path = tmp_path / "ref.tif"
        transform = from_bounds(86.0, 27.0, 86.04, 27.04, width=4, height=4)
        with rasterio.open(
            str(ref_path), "w", driver="GTiff", width=4, height=4, count=1,
            dtype="float32", crs="EPSG:32645", transform=transform,
        ) as dst:
            dst.write(np.zeros((4, 4), dtype=np.float32), 1)
        result = build_optical_input(safe, ref_path)
        assert result.shape == (3, 4, 4)
        assert result.dtype == np.float32

    def test_channel_order(self, tmp_path):
        """Channel 0 = NDWI, 1 = MNDWI, 2 = cloud_mask."""
        safe = _make_synthetic_safe(tmp_path)
        ref_path = tmp_path / "ref.tif"
        transform = from_bounds(86.0, 27.0, 86.04, 27.04, width=4, height=4)
        with rasterio.open(
            str(ref_path), "w", driver="GTiff", width=4, height=4, count=1,
            dtype="float32", crs="EPSG:32645", transform=transform,
        ) as dst:
            dst.write(np.zeros((4, 4), dtype=np.float32), 1)
        result = build_optical_input(safe, ref_path)
        # NDWI should be ~0.333, MNDWI should be ~0.538
        assert np.allclose(result[0], 0.333, atol=0.05)
        assert np.allclose(result[1], 0.538, atol=0.1)
        # Cloud mask is binary
        assert set(np.unique(result[2])).issubset({0.0, 1.0})


class TestCloudSCLClasses:
    def test_cloud_classes_set(self):
        assert CLOUD_SCL_CLASSES == {3, 8, 9, 10}
        assert 3 in CLOUD_SCL_CLASSES  # cloud shadow
        assert 8 in CLOUD_SCL_CLASSES  # cloud medium prob
        assert 9 in CLOUD_SCL_CLASSES  # cloud high prob
        assert 10 in CLOUD_SCL_CLASSES  # cirrus
        assert 4 not in CLOUD_SCL_CLASSES  # vegetation
        assert 6 not in CLOUD_SCL_CLASSES  # water


# --------------------------------------------------------------------------- #
# Real-data tests (require downloaded S2 scenes)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not S2_MONSOON.exists(), reason="S2 monsoon scene not downloaded")
class TestRealS2Monsoon:
    """Real-data tests with the 2026-07-05 S2 L2A scene (53.68% cloud)."""

    def test_real_ndwi_shape(self):
        features = extract_optical_features(S2_MONSOON)
        assert features["ndwi"].ndim == 2
        assert features["ndwi"].shape[0] > 1000  # full tile is large

    def test_real_ndwi_range(self):
        features = extract_optical_features(S2_MONSOON)
        assert features["ndwi"].min() >= -1.01
        assert features["ndwi"].max() <= 1.01

    def test_real_mndwi_range(self):
        features = extract_optical_features(S2_MONSOON)
        assert features["mndwi"].min() >= -1.01
        assert features["mndwi"].max() <= 1.01

    def test_real_cloud_fraction(self):
        features = extract_optical_features(S2_MONSOON)
        # The STAC metadata says 53.68% cloud
        assert 0.3 < features["cloud_fraction"] < 0.8

    def test_real_cloud_mask_binary(self):
        features = extract_optical_features(S2_MONSOON)
        assert set(np.unique(features["cloud_mask"])).issubset({0.0, 1.0})


@pytest.mark.skipif(not S2_PREMONSOON.exists(), reason="S2 pre-monsoon scene not downloaded")
class TestRealS2PreMonsoon:
    """Real-data tests with the 2026-05-26 S2 L2A scene (15% cloud)."""

    def test_real_cloud_fraction_low(self):
        features = extract_optical_features(S2_PREMONSOON)
        # The STAC metadata says 15% cloud
        assert features["cloud_fraction"] < 0.3

    def test_real_ndwi_shape(self):
        features = extract_optical_features(S2_PREMONSOON)
        assert features["ndwi"].ndim == 2
        assert features["ndwi"].shape[0] > 1000
