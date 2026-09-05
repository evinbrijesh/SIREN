"""Unit tests for the PRD §9.1 quality gate (siren.preprocess.quality)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from siren.preprocess.quality import assess_quality
from tests.fixtures import fixture_path

PRD_KEYS = {"quality_score", "cloud_fraction", "alignment_ok", "usable", "confidence_adjustment"}


def test_output_has_exact_prd_contract_keys():
    out = assess_quality(cloud_fraction=0.11, alignment_error=0.3, sensor="sentinel-2-l2a")
    assert set(out.keys()) == PRD_KEYS


def test_cloud_fraction_at_or_above_threshold_is_unusable():
    # Boundary: exactly 0.20 -> unusable (>= threshold routes to SAR).
    out = assess_quality(cloud_fraction=0.20, alignment_error=0.3, sensor="sentinel-2-l2a")
    assert out["usable"] is False
    # Well above threshold.
    out = assess_quality(cloud_fraction=0.50, alignment_error=0.3, sensor="sentinel-2-l2a")
    assert out["usable"] is False


def test_cloud_fraction_below_threshold_is_usable():
    out = assess_quality(cloud_fraction=0.11, alignment_error=0.3, sensor="sentinel-2-l2a")
    assert out["usable"] is True


def test_sar_input_has_zero_effective_cloud_fraction():
    # SAR is all-weather: even a cloudy input reports cloud_fraction == 0.0.
    out = assess_quality(cloud_fraction=0.90, alignment_error=0.3, sensor="sentinel-1-grd-nrt")
    assert out["cloud_fraction"] == 0.0
    assert out["usable"] is True


def test_sar_tagged_sensor_also_treated_as_all_weather():
    out = assess_quality(cloud_fraction=0.90, alignment_error=0.3, sensor="some-sar-product")
    assert out["cloud_fraction"] == 0.0


def test_confidence_adjustment_formula_optical_archived():
    # sentinel-2-l2a is archived -> freshness weight 0.9.
    cloud = 0.10
    out = assess_quality(cloud_fraction=cloud, alignment_error=0.3, sensor="sentinel-2-l2a")
    expected = round((1.0 - cloud) * 0.9, 2)
    assert out["confidence_adjustment"] == pytest.approx(expected, abs=1e-9)


def test_confidence_adjustment_formula_sar_nrt():
    # sentinel-1-grd-nrt -> SAR (effective cloud 0.0) + NRT freshness 1.0.
    out = assess_quality(cloud_fraction=0.90, alignment_error=0.3, sensor="sentinel-1-grd-nrt")
    expected = round((1.0 - 0.0) * 1.0, 2)
    assert out["confidence_adjustment"] == pytest.approx(expected, abs=1e-9)


def test_alignment_error_below_threshold_is_ok():
    out = assess_quality(cloud_fraction=0.0, alignment_error=0.99, sensor="sentinel-2-l2a")
    assert out["alignment_ok"] is True


def test_alignment_error_at_or_above_threshold_is_not_ok():
    # Boundary: exactly 1.0 px -> not ok (strictly below threshold required).
    out = assess_quality(cloud_fraction=0.0, alignment_error=1.0, sensor="sentinel-2-l2a")
    assert out["alignment_ok"] is False
    out = assess_quality(cloud_fraction=0.0, alignment_error=1.5, sensor="sentinel-2-l2a")
    assert out["alignment_ok"] is False


def test_quality_score_formula_clamped_and_rounded():
    # Clear + aligned: (1-0)*0.6 + 1.0*0.4 = 1.0
    out = assess_quality(cloud_fraction=0.0, alignment_error=0.3, sensor="sentinel-2-l2a")
    assert out["quality_score"] == 1.0
    # Cloudy + misaligned: (1-0.5)*0.6 + 0.5*0.4 = 0.3 + 0.2 = 0.5
    out = assess_quality(cloud_fraction=0.5, alignment_error=1.5, sensor="sentinel-2-l2a")
    assert out["quality_score"] == 0.5


def test_cloudy_optical_fixture_is_unusable():
    """Compute cloud_fraction from the 2-band fixture scene stats (not a mask)
    and assert the gate routes it to SAR (usable == False)."""
    path = fixture_path("rasters/cloudy_optical.tif")
    with rasterio.open(path) as src:
        green = src.read(1).astype(np.float64)
        nir = src.read(2).astype(np.float64)

    # Clouds are bright in BOTH bands (> 0.5 reflectance).
    bright_in_both = (green > 0.5) & (nir > 0.5)
    cloud_fraction = float(bright_in_both.sum()) / float(bright_in_both.size)

    # Fixture: 50x50 bright region in a 100x100 scene -> 0.25.
    assert cloud_fraction == pytest.approx(0.25, abs=1e-6)

    out = assess_quality(
        cloud_fraction=cloud_fraction, alignment_error=0.3, sensor="sentinel-2-l2a"
    )
    assert out["cloud_fraction"] == pytest.approx(0.25, abs=1e-6)
    assert out["usable"] is False  # 0.25 >= 0.20 -> SAR path
    assert set(out.keys()) == PRD_KEYS
