"""Tests for risk/shadow_eval.py — shadow-mode evaluation harness (V3 §3.6, §6).

Tests cover the evaluation report structure, metric computation, gate
status determination, and the convenience function.
"""

from __future__ import annotations

import pytest

from siren.risk.shadow_eval import (
    ShadowModeEvaluator,
    ShadowEvalReport,
    ShadowMetric,
    run_shadow_evaluation,
    IOU_GATE,
    BRIER_GATE,
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

def test_gate_thresholds():
    """ML evaluation gate thresholds match ADR-011."""
    assert IOU_GATE == 0.65
    assert BRIER_GATE == 0.15


# --------------------------------------------------------------------------- #
# ShadowMetric
# --------------------------------------------------------------------------- #

def test_shadow_metric_to_dict():
    """ShadowMetric.to_dict produces a serializable dict."""
    m = ShadowMetric(
        name="brier_score",
        ml_value=0.12,
        deterministic_value=0.20,
        gate_threshold=0.15,
        passes_gate=True,
        description="Calibration quality",
    )
    d = m.to_dict()
    assert d["name"] == "brier_score"
    assert d["ml_value"] == 0.12
    assert d["deterministic_value"] == 0.20
    assert d["gate_threshold"] == 0.15
    assert d["passes_gate"] is True
    assert d["description"] == "Calibration quality"


def test_shadow_metric_none_values():
    """ShadowMetric handles None values gracefully."""
    m = ShadowMetric(
        name="iou",
        ml_value=None,
        gate_threshold=0.65,
        passes_gate=None,
    )
    d = m.to_dict()
    assert d["ml_value"] is None
    assert d["passes_gate"] is None


# --------------------------------------------------------------------------- #
# ShadowEvalReport
# --------------------------------------------------------------------------- #

def test_report_summary_pass():
    """Report summary shows PASS when all gated metrics pass."""
    report = ShadowEvalReport(
        metrics=[
            ShadowMetric(
                name="brier_score",
                ml_value=0.10,
                gate_threshold=0.15,
                passes_gate=True,
            ),
        ],
        overall_passes=True,
        n_observations=3,
    )
    summary = report.summary()
    assert "PASSED" in summary
    assert "load-bearing" in summary


def test_report_summary_fail():
    """Report summary shows FAIL when any gated metric fails."""
    report = ShadowEvalReport(
        metrics=[
            ShadowMetric(
                name="brier_score",
                ml_value=0.20,
                gate_threshold=0.15,
                passes_gate=False,
            ),
        ],
        overall_passes=False,
        n_observations=3,
    )
    summary = report.summary()
    assert "NOT PASSED" in summary
    assert "shadow-only" in summary


def test_report_to_dict():
    """Report.to_dict produces a serializable dict."""
    report = ShadowEvalReport(
        metrics=[
            ShadowMetric(name="brier_score", ml_value=0.10, passes_gate=True),
        ],
        overall_passes=True,
        n_observations=3,
    )
    d = report.to_dict()
    assert d["overall_passes"] is True
    assert d["n_observations"] == 3
    assert len(d["metrics"]) == 1


# --------------------------------------------------------------------------- #
# ShadowModeEvaluator
# --------------------------------------------------------------------------- #

def test_evaluator_initializes_with_defaults():
    """ShadowModeEvaluator initializes with default repo and observations."""
    evaluator = ShadowModeEvaluator()
    assert evaluator.observation_ids is not None
    assert len(evaluator.observation_ids) > 0


def test_evaluator_accepts_custom_observation_ids():
    """ShadowModeEvaluator accepts custom observation IDs."""
    evaluator = ShadowModeEvaluator(observation_ids=["obs-001"])
    assert evaluator.observation_ids == ["obs-001"]


def test_evaluator_evaluate_returns_report():
    """evaluate() returns a ShadowEvalReport."""
    # Use a fresh in-memory repo for isolation
    from siren.db.repo import get_repository
    repo = get_repository()
    evaluator = ShadowModeEvaluator(repo=repo, observation_ids=["obs-001"])
    report = evaluator.evaluate()
    assert isinstance(report, ShadowEvalReport)
    assert report.n_observations == 1
    assert len(report.metrics) > 0


def test_evaluator_evaluate_includes_brier_score():
    """evaluate() includes the Brier score metric."""
    from siren.db.repo import get_repository
    repo = get_repository()
    evaluator = ShadowModeEvaluator(repo=repo, observation_ids=["obs-001"])
    report = evaluator.evaluate()
    metric_names = [m.name for m in report.metrics]
    assert "brier_score" in metric_names


def test_evaluator_evaluate_includes_iou():
    """evaluate() includes the water detection IoU metric."""
    from siren.db.repo import get_repository
    repo = get_repository()
    evaluator = ShadowModeEvaluator(repo=repo, observation_ids=["obs-001"])
    report = evaluator.evaluate()
    metric_names = [m.name for m in report.metrics]
    assert "water_detection_iou" in metric_names


def test_evaluator_evaluate_includes_fno_trigger_rate():
    """evaluate() includes the FNO trigger rate metric."""
    from siren.db.repo import get_repository
    repo = get_repository()
    evaluator = ShadowModeEvaluator(repo=repo, observation_ids=["obs-001"])
    report = evaluator.evaluate()
    metric_names = [m.name for m in report.metrics]
    assert "fno_trigger_rate" in metric_names


def test_evaluator_evaluate_includes_shadow_coverage():
    """evaluate() includes the shadow coverage metric."""
    from siren.db.repo import get_repository
    repo = get_repository()
    evaluator = ShadowModeEvaluator(repo=repo, observation_ids=["obs-001"])
    report = evaluator.evaluate()
    metric_names = [m.name for m in report.metrics]
    assert "shadow_coverage" in metric_names


def test_evaluator_evaluate_handles_pipeline_errors():
    """evaluate() handles pipeline errors gracefully."""
    from siren.db.repo import get_repository
    repo = get_repository()
    # Use a non-existent observation ID — pipeline should handle it
    evaluator = ShadowModeEvaluator(repo=repo, observation_ids=["nonexistent"])
    report = evaluator.evaluate()
    assert report.n_observations == 1
    # Should still produce metrics (with None values where data is missing)


# --------------------------------------------------------------------------- #
# run_shadow_evaluation convenience function
# --------------------------------------------------------------------------- #

def test_run_shadow_evaluation_returns_report():
    """run_shadow_evaluation returns a ShadowEvalReport."""
    from siren.db.repo import get_repository
    # Reset repo state for isolation
    repo = get_repository()
    report = run_shadow_evaluation(observation_ids=["obs-001"])
    assert isinstance(report, ShadowEvalReport)
    assert report.n_observations >= 1
