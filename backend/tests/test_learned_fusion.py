"""Tests for risk/learned_fusion.py — Level-6 fused advisory scorer.

The key invariants: the scorer fails closed (no valid checkpoint → no
fabricated probability), missing features degrade loudly (never silently
zero-filled), the score never modifies the canonical hazard/severity,
and runtime demotion via SIREN_ML_DEMOTE reverses the promotion.
"""

from __future__ import annotations

import numpy as np
import pytest

from siren.risk.learned_fusion import (
    LearnedFusionScorer,
    score_imja_observation,
    ADVISORY_THRESHOLD,
    DEFAULT_MODEL_PATH,
)
from siren.risk.shadow_evidence import attach_shadow_evidence


def _make_change_stats(**overrides):
    defaults = {
        "water_area_km2": 1.28,
        "expansion_pct": 25.0,
        "severity": "elevated",
    }
    defaults.update(overrides)
    return defaults


def _make_obs_config(**overrides):
    defaults = {
        "acquired_at": "2026-08-12T12:00:00Z",
        "expansion_pct": 25.0,
        "mean_slope_degrees": 20.0,
    }
    defaults.update(overrides)
    return defaults


# --------------------------------------------------------------------------- #
# Checkpoint loading — fail-closed
# --------------------------------------------------------------------------- #

def test_scorer_missing_checkpoint_fails_closed(tmp_path):
    """No checkpoint → is_loaded False, never a fabricated probability."""
    scorer = LearnedFusionScorer()
    assert scorer.load_checkpoint(tmp_path / "nonexistent.json") is False
    assert scorer.is_loaded is False
    with pytest.raises(RuntimeError, match="no checkpoint loaded"):
        scorer.predict({})


def test_scorer_rejects_disqualified_meta(tmp_path):
    """A checkpoint flagged inference_allowed=False is rejected."""
    import json

    model_path = tmp_path / "model.json"
    model_path.write_text("{}")
    model_path.with_suffix(".meta.json").write_text(json.dumps({
        "status": "gate_evaluated",
        "evaluation_valid": True,
        "inference_allowed": False,
    }))
    scorer = LearnedFusionScorer()
    assert scorer.load_checkpoint(model_path) is False
    assert scorer.is_loaded is False


# --------------------------------------------------------------------------- #
# Real gate-passed checkpoint
# --------------------------------------------------------------------------- #

def test_scorer_loads_gate_passed_checkpoint():
    """The committed fused checkpoint loads and predicts in [0, 1]."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("risk-fusion checkpoint not trained yet")
    scorer = LearnedFusionScorer()
    assert scorer.load_checkpoint() is True
    result = scorer.predict({f: 1.0 for f in (
        "precip_30d_mm", "precip_7d_mm", "max_daily_precip_mm",
        "heavy_rain_days", "api_30", "rain_anom_30d_mm",
        "mdd_30", "mdd_anom_30", "ft_cycles_14",
        "lake_elev_m", "log_lake_area_km2", "log_dist_glacier_m",
        "log_glacier_area_10km", "p_susceptibility", "in_monsoon_window",
    )})
    assert result["is_available"] is True
    assert 0.0 <= result["p_fused"] <= 1.0
    assert result["method"] == "xgboost_risk_fusion_v1"
    assert isinstance(result["reasons"], list) and result["reasons"]


def test_scorer_marks_missing_features_degraded():
    """Missing features → NaN + degraded flag, never silent zeros."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("risk-fusion checkpoint not trained yet")
    scorer = LearnedFusionScorer()
    scorer.load_checkpoint()
    result = scorer.predict({"precip_30d_mm": 100.0})
    assert result["degraded"] is True
    assert "mdd_30" in result["features_missing"]
    assert "p_susceptibility" in result["features_missing"]


# --------------------------------------------------------------------------- #
# score_imja_observation — runtime entry point
# --------------------------------------------------------------------------- #

def test_score_imja_observation_returns_advisory_result():
    """Runtime scorer produces a calibrated advisory result for Imja."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("risk-fusion checkpoint not trained yet")
    result = score_imja_observation(
        _make_obs_config(), _make_change_stats(), p_static=0.5,
    )
    assert result["is_available"] is True
    assert 0.0 <= result["p_fused"] <= 1.0
    assert result["advisory_threshold"] == ADVISORY_THRESHOLD
    assert isinstance(result["elevated_event_probability"], bool)
    # Deterministic severity is echoed for comparison, never modified.
    assert result["deterministic_severity"] == "elevated"


def test_score_imja_observation_monsoon_flag_set():
    """The Jun–Sep monsoon flag is derived from the observation date."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("risk-fusion checkpoint not trained yet")
    result_aug = score_imja_observation(
        _make_obs_config(acquired_at="2026-08-12T00:00:00Z"),
        _make_change_stats(), p_static=0.5,
    )
    result_jan = score_imja_observation(
        _make_obs_config(acquired_at="2026-01-08T00:00:00Z"),
        _make_change_stats(), p_static=0.5,
    )
    assert result_aug["is_available"] is True
    assert result_jan["is_available"] is True
    # Both are valid scores; the flag is a feature input — surfaced via
    # missing-features bookkeeping (neither should list it missing).
    assert "in_monsoon_window" not in result_aug["features_missing"]
    assert "in_monsoon_window" not in result_jan["features_missing"]


def test_score_imja_observation_unparseable_date_degrades():
    """An unparseable date leaves features missing — flagged, not faked."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("risk-fusion checkpoint not trained yet")
    result = score_imja_observation(
        _make_obs_config(acquired_at="not-a-date"),
        _make_change_stats(), p_static=None,
    )
    assert result["is_available"] is True
    assert result["degraded"] is True
    assert "in_monsoon_window" in result["features_missing"]
    assert "p_susceptibility" in result["features_missing"]


# --------------------------------------------------------------------------- #
# Shadow-evidence wiring + promotion/demotion
# --------------------------------------------------------------------------- #

def test_shadow_evidence_includes_learned_fusion():
    """attach_shadow_evidence attaches the fused advisory component."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("risk-fusion checkpoint not trained yet")
    change_stats = _make_change_stats()
    shadow = attach_shadow_evidence(
        change_stats, _make_obs_config(), 5.0, 30.0,
    )
    assert "learned_risk_fusion" in shadow
    fus = shadow["learned_risk_fusion"]
    assert fus["is_available"] is True
    assert fus["promoted"] is True
    assert fus["is_shadow"] is False
    assert "learned_risk_fusion" in shadow["promoted_components"]
    # The prior is fed from the susceptibility shadow result.
    assert np.isfinite(fus["p_fused"])


def test_learned_fusion_never_modifies_canonical_scores():
    """The fused score must not touch hazard or severity (advisory-only)."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("risk-fusion checkpoint not trained yet")
    change_stats = _make_change_stats()
    change_stats["hazard_score"] = 0.75
    attach_shadow_evidence(
        change_stats, _make_obs_config(), 5.0, 30.0,
    )
    assert change_stats["hazard_score"] == 0.75
    assert change_stats["severity"] == "elevated"


def test_learned_fusion_runtime_demotion(monkeypatch):
    """SIREN_ML_DEMOTE=learned_risk_fusion reverses the promotion."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("risk-fusion checkpoint not trained yet")
    monkeypatch.setenv("SIREN_ML_DEMOTE", "learned_risk_fusion")
    result = score_imja_observation(
        _make_obs_config(), _make_change_stats(), p_static=0.5,
    )
    assert result["promoted"] is False
    assert result["is_shadow"] is True


def test_learned_fusion_in_ml_readiness_report():
    """The readiness dashboard reports the promoted fused component."""
    from siren.ml.promotion import get_ml_readiness_report

    report = get_ml_readiness_report()
    fus = report["learned_risk_fusion"]
    assert fus["status"] == "advisory_primary"
    assert fus["gate_passed"] is True
    assert fus["checkpoint"] == "xgboost_risk_fusion.json"
    assert fus["current_metric"]["mean_brier"] < 0.15
