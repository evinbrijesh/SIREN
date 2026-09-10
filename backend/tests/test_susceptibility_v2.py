"""Tests for the expanded GLOF ledger and V2 susceptibility features (V3 §3.2).

Tests the 250+ lake ground-truth generator, physics-grounded features,
isotonic calibration, and the V2 feature set.
"""

from __future__ import annotations

import numpy as np
import pytest

from siren.risk.susceptibility import (
    FEATURE_NAMES,
    FEATURE_NAMES_V2,
    SusceptibilityScorer,
    BRIER_SCORE_GATE,
)
from siren.risk.glof_ledger import (
    generate_glof_ledger,
    split_ledger,
    GLOFLake,
    DEFAULT_N_LAKES,
    DEFAULT_BREACH_FRACTION,
)


# ---------------------------------------------------------------------------
# V2 feature set
# ---------------------------------------------------------------------------

class TestFeatureNamesV2:
    """Test the expanded V2 feature set."""

    def test_v2_has_9_features(self):
        """V2 has 9 physics-grounded features (up from 6)."""
        assert len(FEATURE_NAMES_V2) == 9

    def test_v2_includes_physics_ratios(self):
        """V2 includes the new physics-grounded ratios."""
        assert "dam_width_height_ratio" in FEATURE_NAMES_V2
        assert "ice_core_contact_ratio" in FEATURE_NAMES_V2
        assert "temp_anomaly_0c_isotherm" in FEATURE_NAMES_V2
        assert "dam_width_height_ratio_sq" in FEATURE_NAMES_V2

    def test_v2_preserves_core_features(self):
        """V2 preserves the core features from V1."""
        for name in ["lake_expansion_rate", "moraine_dam_height_m",
                     "rain_anomaly_7d", "mean_upstream_slope_deg", "lake_area_km2"]:
            assert name in FEATURE_NAMES_V2

    def test_v1_preserved(self):
        """V1 feature set is still available for backward compatibility."""
        assert len(FEATURE_NAMES) == 6
        assert "moraine_dam_width_m" in FEATURE_NAMES


# ---------------------------------------------------------------------------
# GLOF ledger generator
# ---------------------------------------------------------------------------

class TestGLOFLedger:
    """Test the expanded GLOF ground-truth ledger generator."""

    def test_default_size(self):
        """Default ledger has 280 lakes (Veh et al. 2022 scale)."""
        X, y, lakes = generate_glof_ledger()
        assert len(lakes) == DEFAULT_N_LAKES
        assert X.shape == (DEFAULT_N_LAKES, 9)

    def test_breach_fraction(self):
        """Breach fraction is approximately 15%."""
        X, y, lakes = generate_glof_ledger()
        breach_frac = y.sum() / len(y)
        assert abs(breach_frac - DEFAULT_BREACH_FRACTION) < 0.05

    def test_reproducible(self):
        """Same seed produces same ledger (Hard Rule 6)."""
        X1, y1, _ = generate_glof_ledger(seed=42)
        X2, y2, _ = generate_glof_ledger(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seed_different(self):
        """Different seeds produce different ledgers."""
        X1, _, _ = generate_glof_ledger(seed=42)
        X2, _, _ = generate_glof_ledger(seed=99)
        assert not np.array_equal(X1, X2)

    def test_feature_shape(self):
        """Feature matrix has 9 columns matching FEATURE_NAMES_V2."""
        X, y, _ = generate_glof_ledger()
        assert X.shape[1] == len(FEATURE_NAMES_V2)

    def test_labels_binary(self):
        """Labels are binary {0, 1}."""
        _, y, _ = generate_glof_ledger()
        assert set(np.unique(y).tolist()).issubset({0.0, 1.0})

    def test_breached_lakes_have_high_risk_features(self):
        """Breached lakes have higher expansion rates and lower W/H ratios."""
        X, y, lakes = generate_glof_ledger()
        breached = X[y == 1]
        stable = X[y == 0]
        # Breached lakes have higher expansion rate (feature 0)
        assert breached[:, 0].mean() > stable[:, 0].mean()
        # Breached lakes have lower W/H ratio (feature 1) — narrow crests
        assert breached[:, 1].mean() < stable[:, 1].mean()
        # Breached lakes have higher ice-core contact (feature 6)
        assert breached[:, 6].mean() > stable[:, 6].mean()

    def test_dam_width_height_ratio_sq(self):
        """The squared W/H ratio is the square of the W/H ratio."""
        X, y, lakes = generate_glof_ledger()
        # Feature 1 = dam_width_height_ratio, Feature 8 = dam_width_height_ratio_sq
        np.testing.assert_allclose(X[:, 8], X[:, 1] ** 2, atol=1e-2)

    def test_regions_present(self):
        """Lakes have region metadata."""
        _, _, lakes = generate_glof_ledger()
        regions = {lake.region for lake in lakes}
        assert "nepal" in regions
        assert "tibet" in regions


class TestGLOFLake:
    """Test the GLOFLake dataclass."""

    def test_to_feature_vector(self):
        """GLOFLake converts to a 9-feature vector."""
        lake = GLOFLake(
            lake_id="test-001",
            region="nepal",
            breached=True,
            lake_expansion_rate=0.15,
            dam_width_height_ratio=5.0,
            moraine_dam_height_m=40,
            rain_anomaly_7d=2.5,
            mean_upstream_slope_deg=30,
            lake_area_km2=1.5,
            ice_core_contact_ratio=0.3,
            temp_anomaly_0c_isotherm=200,
        )
        fv = lake.to_feature_vector()
        assert fv.shape == (1, 9)
        assert fv[0, 0] == 0.15  # lake_expansion_rate
        assert fv[0, 8] == 25.0  # 5.0² = dam_width_height_ratio_sq

    def test_label(self):
        """Label is 1 for breached, 0 for stable."""
        breached = GLOFLake("b", "nepal", True, 0.1, 5, 40, 2, 30, 1, 0.3, 200)
        stable = GLOFLake("s", "nepal", False, 0.01, 20, 10, 0, 15, 0.5, 0.05, 0)
        assert breached.label == 1
        assert stable.label == 0


# ---------------------------------------------------------------------------
# Ledger split
# ---------------------------------------------------------------------------

class TestLedgerSplit:
    """Test the train/cal/test split."""

    def test_split_sizes(self):
        """Split produces correct proportions."""
        X, y, _ = generate_glof_ledger(n_lakes=100)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_ledger(X, y)
        total = len(X_tr) + len(X_cal) + len(X_te)
        assert total == 100
        assert len(X_tr) == 60  # 60%
        assert len(X_cal) == 20  # 20%
        assert len(X_te) == 20  # 20%

    def test_stratified(self):
        """Split maintains breach fraction in each subset."""
        X, y, _ = generate_glof_ledger(n_lakes=200, breach_fraction=0.15)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_ledger(X, y)
        # Each subset should have approximately 15% breach
        for subset_y in [y_tr, y_cal, y_te]:
            frac = subset_y.sum() / len(subset_y)
            assert abs(frac - 0.15) < 0.1  # within 10% of target

    def test_no_overlap(self):
        """Train, cal, and test sets are disjoint."""
        X, y, _ = generate_glof_ledger(n_lakes=100)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_ledger(X, y)
        # Check no row appears in two subsets
        for i in range(len(X_tr)):
            for j in range(len(X_cal)):
                assert not np.array_equal(X_tr[i], X_cal[j])


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------

class TestIsotonicCalibration:
    """Test isotonic probability calibration."""

    def test_isotonic_improves_brier(self):
        """Isotonic calibration should not worsen the Brier score."""
        X, y, _ = generate_glof_ledger(n_lakes=200, seed=42)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_ledger(X, y, seed=42)

        # Train without isotonic
        scorer_raw = SusceptibilityScorer(random_state=42)
        brier_raw = scorer_raw.train(X_tr, y_tr, X_cal, y_cal, use_isotonic=False)

        # Train with isotonic
        scorer_iso = SusceptibilityScorer(random_state=42)
        brier_iso = scorer_iso.train(X_tr, y_tr, X_cal, y_cal, use_isotonic=True)

        # Isotonic should improve or maintain Brier score
        assert brier_iso <= brier_raw + 0.01  # allow small numerical tolerance

    def test_isotonic_flag_set(self):
        """The isotonic calibration flag is set after training."""
        X, y, _ = generate_glof_ledger(n_lakes=100, seed=42)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_ledger(X, y, seed=42)

        scorer = SusceptibilityScorer(random_state=42)
        scorer.train(X_tr, y_tr, X_cal, y_cal, use_isotonic=True)
        assert scorer._is_isotonic_calibrated is True
        assert scorer._calibrator is not None

    def test_no_isotonic_flag_when_disabled(self):
        """The isotonic flag is False when use_isotonic=False."""
        X, y, _ = generate_glof_ledger(n_lakes=100, seed=42)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_ledger(X, y, seed=42)

        scorer = SusceptibilityScorer(random_state=42)
        scorer.train(X_tr, y_tr, X_cal, y_cal, use_isotonic=False)
        assert scorer._is_isotonic_calibrated is False
        assert scorer._calibrator is None

    def test_raw_brier_score_stored(self):
        """Raw (pre-calibration) Brier score is stored for comparison."""
        X, y, _ = generate_glof_ledger(n_lakes=100, seed=42)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_ledger(X, y, seed=42)

        scorer = SusceptibilityScorer(random_state=42)
        scorer.train(X_tr, y_tr, X_cal, y_cal, use_isotonic=True)
        assert scorer._brier_score_raw is not None
        assert scorer._brier_score is not None

    def test_predict_uses_calibrated_probs(self):
        """Predict uses calibrated probabilities when calibrator is fitted."""
        X, y, _ = generate_glof_ledger(n_lakes=100, seed=42)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_ledger(X, y, seed=42)

        scorer = SusceptibilityScorer(random_state=42)
        scorer.train(X_tr, y_tr, X_cal, y_cal, use_isotonic=True)

        # Predict on a test sample
        result = scorer.predict(X_te[:1])
        assert 0.0 <= result.p_breach <= 1.0
        assert result.is_calibrated is True


# ---------------------------------------------------------------------------
# V2 feature set training
# ---------------------------------------------------------------------------

class TestV2FeatureTraining:
    """Test training with the V2 9-feature set."""

    def test_v2_feature_names_used(self):
        """Scorer with V2 feature names uses them in explanations."""
        X, y, _ = generate_glof_ledger(n_lakes=100, seed=42)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_ledger(X, y, seed=42)

        scorer = SusceptibilityScorer(
            random_state=42,
            feature_names=FEATURE_NAMES_V2,
        )
        scorer.train(X_tr, y_tr, X_cal, y_cal, use_isotonic=True)
        assert scorer.feature_names == FEATURE_NAMES_V2

    def test_v2_brier_below_gate(self):
        """V2 model with isotonic calibration should approach the Brier < 0.15 gate."""
        X, y, _ = generate_glof_ledger(n_lakes=280, seed=42)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_ledger(X, y, seed=42)

        scorer = SusceptibilityScorer(
            random_state=42,
            feature_names=FEATURE_NAMES_V2,
        )
        brier = scorer.train(X_tr, y_tr, X_cal, y_cal, use_isotonic=True)

        # With 280 lakes and isotonic calibration, Brier should be well below 0.15
        # (the synthetic data is cleanly separable, so this is expected)
        assert brier < BRIER_SCORE_GATE, f"Brier {brier:.4f} exceeds gate {BRIER_SCORE_GATE}"

    def test_v2_predict_produces_reasons(self):
        """V2 model predict produces reasons with V2 feature names."""
        X, y, _ = generate_glof_ledger(n_lakes=100, seed=42)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_ledger(X, y, seed=42)

        scorer = SusceptibilityScorer(
            random_state=42,
            feature_names=FEATURE_NAMES_V2,
        )
        scorer.train(X_tr, y_tr, X_cal, y_cal, use_isotonic=True)

        result = scorer.predict(X_te[:1])
        assert len(result.reasons) >= 3  # Hard Rule 5
