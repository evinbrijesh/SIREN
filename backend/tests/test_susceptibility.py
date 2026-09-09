"""Tests for the glacial lake breach susceptibility scorer (V3 §3.2, §3.4).

Tests cover XGBoost training, conformal prediction intervals, the width > 0.35
manual-inspection fallback, Brier score reporting, and the shadow-only
acceptance gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from siren.risk.susceptibility import (
    SusceptibilityScorer,
    SusceptibilityResult,
    FEATURE_NAMES,
    DEFAULT_ALPHA,
    INTERVAL_WIDTH_THRESHOLD,
    BRIER_SCORE_GATE,
)


@pytest.fixture
def synthetic_training_data():
    """Synthetic glacial lake features + breach labels for training."""
    rng = np.random.RandomState(42)
    n = 200
    # Features: expansion rate, dam width, dam height, rain anomaly, slope, area
    X = rng.uniform(
        low=[0.0, 10.0, 5.0, -2.0, 0.0, 0.01],
        high=[0.5, 500.0, 50.0, 5.0, 45.0, 5.0],
        size=(n, len(FEATURE_NAMES)),
    ).astype(np.float32)
    # Breach label: higher expansion rate + lower dam height → more likely breach
    log_odds = (
        3.0 * X[:, 0]      # expansion rate
        - 0.05 * X[:, 2]   # dam height (negative: taller dam = safer)
        + 0.3 * X[:, 3]    # rain anomaly
        + 0.02 * X[:, 4]   # slope
    )
    p = 1.0 / (1.0 + np.exp(-log_odds))
    y = (rng.uniform(size=n) < p).astype(np.float32)
    return X, y


@pytest.fixture
def trained_scorer(synthetic_training_data):
    """A trained + calibrated susceptibility scorer."""
    X, y = synthetic_training_data
    rng = np.random.RandomState(123)
    # Split into train + calibration
    idx = rng.permutation(len(X))
    n_cal = len(X) // 3
    cal_idx, train_idx = idx[:n_cal], idx[n_cal:]
    scorer = SusceptibilityScorer(alpha=0.05, random_state=42)
    scorer.train(X[train_idx], y[train_idx], X[cal_idx], y[cal_idx])
    return scorer


# --------------------------------------------------------------------------- #
# Constants and structure
# --------------------------------------------------------------------------- #

def test_feature_names_canonical_order():
    """FEATURE_NAMES matches the V3 §3.2 canonical feature order."""
    assert FEATURE_NAMES == (
        "lake_expansion_rate",
        "moraine_dam_width_m",
        "moraine_dam_height_m",
        "rain_anomaly_7d",
        "mean_upstream_slope_deg",
        "lake_area_km2",
    )


def test_conformal_constants():
    """Conformal prediction constants match V3 §3.4."""
    assert DEFAULT_ALPHA == 0.05       # 95% coverage
    assert INTERVAL_WIDTH_THRESHOLD == 0.35
    assert BRIER_SCORE_GATE == 0.15


def test_susceptibility_result_dataclass():
    """SusceptibilityResult has all required fields."""
    r = SusceptibilityResult(
        p_breach=0.5,
        interval_low=0.3,
        interval_high=0.7,
        interval_width=0.4,
        requires_manual_inspection=True,
    )
    assert r.p_breach == 0.5
    assert r.requires_manual_inspection is True
    assert r.reasons == []
    assert r.feature_contributions == {}


def test_susceptibility_result_to_dict():
    """to_dict produces a JSON-serializable dict with rounded values."""
    r = SusceptibilityResult(
        p_breach=0.823456,
        interval_low=0.712345,
        interval_high=0.934567,
        interval_width=0.222222,
        requires_manual_inspection=False,
        brier_score=0.123456,
        is_calibrated=True,
    )
    d = r.to_dict()
    assert d["p_breach"] == 0.8235
    assert d["interval_low"] == 0.7123
    assert d["requires_manual_inspection"] is False
    assert d["brier_score"] == 0.1235


# --------------------------------------------------------------------------- #
# Training and prediction
# --------------------------------------------------------------------------- #

def test_scorer_untrained_raises():
    """Predicting before training raises RuntimeError."""
    scorer = SusceptibilityScorer()
    with pytest.raises(RuntimeError, match="not been trained"):
        scorer.predict(np.zeros((1, 6)))


def test_scorer_train_without_calibration(synthetic_training_data):
    """Training without a calibration set produces no conformal interval."""
    X, y = synthetic_training_data
    scorer = SusceptibilityScorer(random_state=42)
    brier = scorer.train(X, y)
    assert scorer.is_trained
    assert not scorer.is_calibrated
    assert brier == 0.0  # no calibration set → no Brier score


def test_scorer_train_with_calibration(synthetic_training_data):
    """Training with a calibration set produces a Brier score + conformal q."""
    X, y = synthetic_training_data
    rng = np.random.RandomState(123)
    idx = rng.permutation(len(X))
    n_cal = len(X) // 3
    scorer = SusceptibilityScorer(random_state=42)
    brier = scorer.train(X[idx[n_cal:]], y[idx[n_cal:]], X[idx[:n_cal]], y[idx[:n_cal]])
    assert scorer.is_trained
    assert scorer.is_calibrated
    assert 0.0 <= brier <= 1.0
    assert scorer.brier_score == brier


def test_scorer_predict_returns_result(trained_scorer):
    """predict returns a SusceptibilityResult with valid fields."""
    X = np.array([[0.3, 100.0, 10.0, 2.0, 20.0, 1.0]], dtype=np.float32)
    result = trained_scorer.predict(X)
    assert isinstance(result, SusceptibilityResult)
    assert 0.0 <= result.p_breach <= 1.0
    assert 0.0 <= result.interval_low <= 1.0
    assert 0.0 <= result.interval_high <= 1.0
    assert result.interval_width >= 0.0
    assert result.is_calibrated


def test_scorer_predict_1d_input(trained_scorer):
    """predict handles 1D input by reshaping to (1, n_features)."""
    X = np.array([0.3, 100.0, 10.0, 2.0, 20.0, 1.0], dtype=np.float32)
    result = trained_scorer.predict(X)
    assert 0.0 <= result.p_breach <= 1.0


def test_scorer_predict_interval_contains_point(trained_scorer):
    """The conformal interval contains the point prediction."""
    X = np.array([[0.2, 200.0, 20.0, 1.0, 15.0, 2.0]], dtype=np.float32)
    result = trained_scorer.predict(X)
    assert result.interval_low <= result.p_breach <= result.interval_high


# --------------------------------------------------------------------------- #
# Conformal interval + manual inspection fallback (V3 §3.4)
# --------------------------------------------------------------------------- #

def test_scorer_untrained_no_calibration_wide_interval(synthetic_training_data):
    """Without calibration, the interval is [0, 1] → manual inspection required."""
    X, y = synthetic_training_data
    scorer = SusceptibilityScorer(random_state=42)
    scorer.train(X, y)  # no calibration set
    result = scorer.predict(X[:1])
    assert result.interval_low == 0.0
    assert result.interval_high == 1.0
    assert result.interval_width == 1.0
    assert result.requires_manual_inspection is True


def test_scorer_manual_inspection_when_width_exceeds_threshold(trained_scorer):
    """requires_manual_inspection is True when interval width > 0.35."""
    # The synthetic data may or may not produce a wide interval; test the
    # threshold logic by checking the flag matches the width comparison
    X = np.array([[0.3, 100.0, 10.0, 2.0, 20.0, 1.0]], dtype=np.float32)
    result = trained_scorer.predict(X)
    expected_manual = result.interval_width > INTERVAL_WIDTH_THRESHOLD
    assert result.requires_manual_inspection == expected_manual


def test_scorer_interval_clamped_to_unit_range(trained_scorer):
    """Conformal interval is clamped to [0, 1]."""
    X = np.array([[0.3, 100.0, 10.0, 2.0, 20.0, 1.0]], dtype=np.float32)
    result = trained_scorer.predict(X)
    assert 0.0 <= result.interval_low
    assert result.interval_high <= 1.0


# --------------------------------------------------------------------------- #
# Reasons (Hard Rule 5: ≥3 on elevated+)
# --------------------------------------------------------------------------- #

def test_scorer_reasons_always_present(trained_scorer):
    """Reasons are always present (at least the feature values + probability)."""
    X = np.array([[0.1, 50.0, 30.0, 0.5, 10.0, 0.5]], dtype=np.float32)
    result = trained_scorer.predict(X)
    assert len(result.reasons) >= 3


def test_scorer_reasons_include_brier_score(trained_scorer):
    """When calibrated, reasons include the Brier score + gate status."""
    X = np.array([[0.1, 50.0, 30.0, 0.5, 10.0, 0.5]], dtype=np.float32)
    result = trained_scorer.predict(X)
    brier_reasons = [r for r in result.reasons if "Brier" in r]
    assert len(brier_reasons) == 1


def test_scorer_reasons_include_interval_statement(trained_scorer):
    """Reasons include a conformal interval statement."""
    X = np.array([[0.1, 50.0, 30.0, 0.5, 10.0, 0.5]], dtype=np.float32)
    result = trained_scorer.predict(X)
    interval_reasons = [r for r in result.reasons if "interval" in r.lower()]
    assert len(interval_reasons) >= 1


# --------------------------------------------------------------------------- #
# Acceptance gate (V3 §3.6)
# --------------------------------------------------------------------------- #

def test_scorer_acceptance_gate_untrained():
    """Untrained scorer does not pass the acceptance gate."""
    scorer = SusceptibilityScorer()
    assert not scorer.passes_acceptance_gate()


def test_scorer_acceptance_gate_uncalibrated(synthetic_training_data):
    """Trained but uncalibrated scorer does not pass the acceptance gate."""
    X, y = synthetic_training_data
    scorer = SusceptibilityScorer(random_state=42)
    scorer.train(X, y)
    assert not scorer.passes_acceptance_gate()


def test_scorer_acceptance_gate_calibrated(trained_scorer):
    """A calibrated scorer may pass the gate if Brier < 0.15."""
    # The synthetic data is designed to be learnable, so Brier should be low
    if trained_scorer.brier_score < BRIER_SCORE_GATE:
        assert trained_scorer.passes_acceptance_gate()
    else:
        assert not trained_scorer.passes_acceptance_gate()


# --------------------------------------------------------------------------- #
# Determinism (Hard Rule 6)
# --------------------------------------------------------------------------- #

def test_scorer_deterministic_predictions(synthetic_training_data):
    """Same training data + same random_state → same predictions."""
    X, y = synthetic_training_data
    rng = np.random.RandomState(123)
    idx = rng.permutation(len(X))
    n_cal = len(X) // 3

    scorer1 = SusceptibilityScorer(random_state=42)
    scorer1.train(X[idx[n_cal:]], y[idx[n_cal:]], X[idx[:n_cal]], y[idx[:n_cal]])

    scorer2 = SusceptibilityScorer(random_state=42)
    scorer2.train(X[idx[n_cal:]], y[idx[n_cal:]], X[idx[:n_cal]], y[idx[:n_cal]])

    X_test = np.array([[0.3, 100.0, 10.0, 2.0, 20.0, 1.0]], dtype=np.float32)
    r1 = scorer1.predict(X_test)
    r2 = scorer2.predict(X_test)
    assert r1.p_breach == pytest.approx(r2.p_breach)
    assert r1.interval_low == pytest.approx(r2.interval_low)
    assert r1.interval_high == pytest.approx(r2.interval_high)


# --------------------------------------------------------------------------- #
# TreeSHAP explanation (V3 §3.3)
# --------------------------------------------------------------------------- #

def test_scorer_explain_returns_feature_contributions(trained_scorer):
    """explain() returns a dict mapping feature names to SHAP values."""
    X = np.array([[0.3, 100.0, 10.0, 2.0, 20.0, 1.0]], dtype=np.float32)
    contributions = trained_scorer.explain(X)
    assert isinstance(contributions, dict)
    assert len(contributions) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in contributions
        assert isinstance(contributions[name], float)


def test_scorer_explain_1d_input(trained_scorer):
    """explain() handles 1D input."""
    X = np.array([0.3, 100.0, 10.0, 2.0, 20.0, 1.0], dtype=np.float32)
    contributions = trained_scorer.explain(X)
    assert len(contributions) == len(FEATURE_NAMES)


def test_scorer_explain_untrained_raises():
    """explain() before training raises RuntimeError."""
    scorer = SusceptibilityScorer()
    with pytest.raises(RuntimeError, match="not been trained"):
        scorer.explain(np.zeros((1, 6)))


def test_scorer_predict_includes_shap_reasons(trained_scorer):
    """predict() reasons include TreeSHAP log-odds contributions."""
    X = np.array([[0.3, 100.0, 10.0, 2.0, 20.0, 1.0]], dtype=np.float32)
    result = trained_scorer.predict(X)
    # At least one reason should mention "log-odds" (SHAP contribution format)
    shap_reasons = [r for r in result.reasons if "log-odds" in r]
    assert len(shap_reasons) >= 1


def test_scorer_predict_top3_shap_contributions(trained_scorer):
    """predict() includes the top-3 SHAP contributions in reasons."""
    X = np.array([[0.3, 100.0, 10.0, 2.0, 20.0, 1.0]], dtype=np.float32)
    result = trained_scorer.predict(X)
    shap_reasons = [r for r in result.reasons if "log-odds" in r]
    # Top 3 by absolute value
    assert len(shap_reasons) <= 3


def test_scorer_feature_contributions_in_result(trained_scorer):
    """predict() populates feature_contributions in the result."""
    X = np.array([[0.3, 100.0, 10.0, 2.0, 20.0, 1.0]], dtype=np.float32)
    result = trained_scorer.predict(X)
    assert len(result.feature_contributions) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in result.feature_contributions


def test_scorer_feature_contributions_to_dict(trained_scorer):
    """feature_contributions are serialized in to_dict()."""
    X = np.array([[0.3, 100.0, 10.0, 2.0, 20.0, 1.0]], dtype=np.float32)
    result = trained_scorer.predict(X)
    d = result.to_dict()
    assert "feature_contributions" in d
    assert len(d["feature_contributions"]) == len(FEATURE_NAMES)


def test_scorer_shap_explanations_deterministic(trained_scorer):
    """Same input → same SHAP contributions (determinism, Hard Rule 6)."""
    X = np.array([[0.3, 100.0, 10.0, 2.0, 20.0, 1.0]], dtype=np.float32)
    c1 = trained_scorer.explain(X)
    c2 = trained_scorer.explain(X)
    for name in FEATURE_NAMES:
        assert c1[name] == pytest.approx(c2[name])
