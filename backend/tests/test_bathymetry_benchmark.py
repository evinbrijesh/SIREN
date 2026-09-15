"""Tests for the bathymetry volume estimation and LOO benchmark.

Tests cover:
    - Huggel volume formula consistency with breach_volume.py
    - Power-law fitting (log-space stability)
    - Volume estimation from surveyed points
    - Lake area computation (published > outline > convex hull)
    - LOO benchmark structure and reproducibility
    - Gate evaluation (both methods currently fail the 15% MAPE gate)
"""

from __future__ import annotations

import numpy as np
import pytest

from siren.ml.bathymetry_benchmark import (
    huggel_volume_m3,
    surveyed_volume_m3,
    fit_power_law,
    predict_power_law_volume_m3,
    compute_lake_area_km2,
    build_volume_estimates,
    run_loo_benchmark,
    DAS_PUBLISHED_VOLUMES_MCM,
)
from siren.ml.bathymetry_dataset import (
    load_all_surveyed_lakes,
    ZHANG_16LAKES_DIR,
    DAS_4LAKES_DIR,
)
from siren.risk.breach_volume import HUGGEL_ALPHA, HUGGEL_GAMMA

ZHANG_AVAILABLE = (
    ZHANG_16LAKES_DIR / "Glacial_Lake_bathymetry" / "GlacialLakeBathymetry"
).exists()
DAS_AVAILABLE = (
    DAS_4LAKES_DIR / "In-Situ Bathymetry Data for JOG"
).exists()
skip_all = pytest.mark.skipif(
    not (ZHANG_AVAILABLE and DAS_AVAILABLE),
    reason="surveyed bathymetry datasets not downloaded",
)


# --------------------------------------------------------------------------- #
# Huggel formula consistency
# --------------------------------------------------------------------------- #

class TestHuggelVolume:
    """Tests for the Huggel volume formula."""

    def test_matches_breach_volume_module(self):
        """huggel_volume_m3 matches the formula in breach_volume.py."""
        area_km2 = 1.0
        area_m2 = area_km2 * 1e6
        expected = HUGGEL_ALPHA * (area_m2 ** HUGGEL_GAMMA)
        assert huggel_volume_m3(area_km2) == pytest.approx(expected)

    def test_zero_area_gives_zero_volume(self):
        """Zero area gives zero volume."""
        assert huggel_volume_m3(0.0) == 0.0

    def test_larger_area_gives_larger_volume(self):
        """Volume is monotonically increasing with area."""
        v1 = huggel_volume_m3(0.1)
        v2 = huggel_volume_m3(1.0)
        v3 = huggel_volume_m3(10.0)
        assert v1 < v2 < v3

    def test_known_value_galongco(self):
        """Galongco (5.46 km²) should give ~390 MCM via Huggel."""
        v = huggel_volume_m3(5.46)
        # Published volume is 375.42 MCM; Huggel gives ~390
        assert 350e6 < v < 450e6


# --------------------------------------------------------------------------- #
# Power-law fitting
# --------------------------------------------------------------------------- #

class TestPowerLawFit:
    """Tests for the power-law fitting function."""

    def test_recovers_known_parameters(self):
        """Log-space fitting recovers known power-law parameters."""
        # Generate synthetic data: V = 0.1 * A^1.4
        np.random.seed(42)
        areas_km2 = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        areas_m2 = areas_km2 * 1e6
        volumes_m3 = 0.1 * areas_m2 ** 1.4
        a, b = fit_power_law(areas_km2, volumes_m3)
        assert abs(a - 0.1) < 0.01
        assert abs(b - 1.4) < 0.01

    def test_handles_wide_range(self):
        """Fitting is stable across a wide range of areas and volumes."""
        np.random.seed(42)
        areas_km2 = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        areas_m2 = areas_km2 * 1e6
        volumes_m3 = 0.15 * areas_m2 ** 1.35
        a, b = fit_power_law(areas_km2, volumes_m3)
        assert abs(a - 0.15) < 0.01
        assert abs(b - 1.35) < 0.01

    def test_fallback_on_insufficient_data(self):
        """Fitting returns the initial guess when there are too few points."""
        areas = np.array([1.0])
        volumes = np.array([1e6])
        a, b = fit_power_law(areas, volumes)
        assert a == HUGGEL_ALPHA
        assert b == HUGGEL_GAMMA

    def test_prediction_uses_fitted_params(self):
        """Prediction uses the fitted parameters correctly."""
        areas_km2 = np.array([0.1, 1.0, 10.0])
        areas_m2 = areas_km2 * 1e6
        volumes_m3 = 0.2 * areas_m2 ** 1.3
        a, b = fit_power_law(areas_km2, volumes_m3)
        predicted = predict_power_law_volume_m3(1.0, (a, b))
        expected = 0.2 * (1e6) ** 1.3
        assert predicted == pytest.approx(expected, rel=0.01)


# --------------------------------------------------------------------------- #
# Volume estimation from surveyed data
# --------------------------------------------------------------------------- #

@skip_all
class TestVolumeEstimation:
    """Tests for volume estimation from surveyed bathymetry."""

    def test_build_volume_estimates_returns_all_lakes(self):
        """All 20 lakes have volume estimates."""
        estimates = build_volume_estimates()
        assert len(estimates) == 20

    def test_huggel_volumes_are_positive(self):
        """All Huggel volume estimates are positive."""
        estimates = build_volume_estimates()
        for e in estimates:
            assert e.huggel_volume_m3 > 0, f"{e.lake_name} Huggel volume is non-positive"

    def test_surveyed_volumes_are_positive(self):
        """All surveyed volume estimates are positive."""
        estimates = build_volume_estimates()
        for e in estimates:
            assert e.surveyed_volume_m3 > 0, f"{e.lake_name} surveyed volume is non-positive"

    def test_published_volumes_match_known_values(self):
        """Published volumes match the known values from the literature."""
        estimates = build_volume_estimates()
        by_name = {e.lake_name: e for e in estimates}

        # Galongco: published 375.42 MCM
        assert by_name["Galongco"].published_volume_m3 is not None
        assert abs(by_name["Galongco"].published_volume_m3 - 375.42e6) < 1e6

        # Gepang Gath: published 24.12 MCM
        assert by_name["Gepang Gath Lake"].published_volume_m3 is not None
        assert abs(by_name["Gepang Gath Lake"].published_volume_m3 - 24.12e6) < 0.5e6

    def test_ground_truth_prefers_published(self):
        """Ground truth uses published volume when available."""
        estimates = build_volume_estimates()
        for e in estimates:
            if e.published_volume_m3 is not None:
                assert e.ground_truth_source == "published"
                assert e.ground_truth_volume_m3 == e.published_volume_m3

    def test_area_uses_published_where_available(self):
        """Lake area uses published values from the global compilation."""
        estimates = build_volume_estimates()
        by_name = {e.lake_name: e for e in estimates}
        # Galongco published area is 5.46 km²
        assert abs(by_name["Galongco"].area_km2 - 5.46) < 0.01

    def test_area_uses_outline_for_4lakes(self):
        """4-lake dataset uses outline polygon area."""
        estimates = build_volume_estimates()
        by_name = {e.lake_name: e for e in estimates}
        # Gepang Gath outline area is ~1.073 km²
        assert 1.0 < by_name["Gepang Gath Lake"].area_km2 < 1.2


# --------------------------------------------------------------------------- #
# LOO benchmark
# --------------------------------------------------------------------------- #

@skip_all
class TestLOOBenchmark:
    """Tests for the leave-one-lake-out benchmark."""

    def test_returns_summary_with_correct_structure(self):
        """Benchmark returns a BenchmarkSummary with correct fields."""
        summary = run_loo_benchmark()
        assert hasattr(summary, "n_lakes")
        assert hasattr(summary, "huggel_mape")
        assert hasattr(summary, "regression_mape")
        assert hasattr(summary, "folds")
        assert summary.n_lakes == 20

    def test_number_of_folds_equals_number_of_lakes(self):
        """One fold per lake."""
        summary = run_loo_benchmark()
        assert len(summary.folds) == 20

    def test_each_lake_held_out_exactly_once(self):
        """Each lake appears as the test lake exactly once."""
        summary = run_loo_benchmark()
        test_ids = [f.test_lake_id for f in summary.folds]
        assert len(test_ids) == len(set(test_ids))

    def test_huggel_mape_is_finite(self):
        """Huggel MAPE is a finite number."""
        summary = run_loo_benchmark()
        assert np.isfinite(summary.huggel_mape)
        assert summary.huggel_mape > 0

    def test_regression_mape_is_finite(self):
        """Regression MAPE is a finite number."""
        summary = run_loo_benchmark()
        assert np.isfinite(summary.regression_mape)
        assert summary.regression_mape > 0

    def test_both_methods_fail_gate(self):
        """Both Huggel and regression fail the 15% MAPE gate.

        This is the expected baseline result — area alone is a poor
        predictor of volume. The neural model must beat this once DEM
        data is acquired.
        """
        summary = run_loo_benchmark()
        assert not summary.huggel_passes_gate, (
            f"Huggel unexpectedly passed the gate (MAPE={summary.huggel_mape*100:.1f}%)"
        )
        assert not summary.regression_passes_gate, (
            f"Regression unexpectedly passed the gate (MAPE={summary.regression_mape*100:.1f}%)"
        )

    def test_huggel_mape_below_100_percent(self):
        """Huggel MAPE is below 100% (not catastrophically wrong)."""
        summary = run_loo_benchmark()
        assert summary.huggel_mape < 1.0, (
            f"Huggel MAPE {summary.huggel_mape*100:.1f}% >= 100%"
        )

    def test_to_dict_is_serializable(self):
        """Summary to_dict is JSON-serializable."""
        import json
        summary = run_loo_benchmark()
        d = summary.to_dict()
        json.dumps(d)

    def test_per_fold_apes_are_finite(self):
        """All per-fold APE values are finite."""
        summary = run_loo_benchmark()
        for f in summary.folds:
            assert np.isfinite(f.huggel_ape)
            assert np.isfinite(f.regression_ape)

    def test_galongco_has_low_huggel_ape(self):
        """Galongco (large, deep) has low Huggel APE (< 20%)."""
        summary = run_loo_benchmark()
        galongco = [f for f in summary.folds if "galongco" in f.test_lake_id][0]
        assert galongco.huggel_ape < 0.20, (
            f"Galongco Huggel APE {galongco.huggel_ape*100:.1f}% >= 20%"
        )
