"""Unit tests for the E3 spectral separability diagnostic helpers.

These cover the pure statistics in siren.ml.s2_spectral_eval — the
scene-loading path is exercised manually against real S2 archives and
reported in models/checkpoints/s2_spectral_eval_*.json.
"""

import numpy as np

from siren.ml.s2_spectral_eval import _auc, _best_threshold, _stats


def test_stats_reports_shape_and_quantiles():
    s = _stats(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s["n"] == 4
    assert s["mean"] == 2.5
    assert s["p50"] == 2.5
    assert s["p05"] <= s["p50"] <= s["p95"]


def test_auc_perfect_separation():
    pos = np.array([0.8, 0.9, 1.0])
    neg = np.array([0.0, 0.1, 0.2])
    assert _auc(pos, neg) == 1.0
    assert _auc(neg, pos) == 0.0


def test_auc_no_separation():
    pos = np.array([0.5, 0.5, 0.5])
    neg = np.array([0.5, 0.5, 0.5])
    assert _auc(pos, neg) == 0.5


def test_auc_inverted_separation():
    """Glacier reads higher MNDWI than the frozen lake — AUC below 0.5
    must surface the inversion, not hide it."""
    pos = np.array([0.0, 0.1])  # lake (lower)
    neg = np.array([0.5, 0.6, 0.7])  # glacier (higher)
    assert _auc(pos, neg) == 0.0


def test_best_threshold_gt_direction():
    pos = np.linspace(0.5, 1.0, 50)
    neg = np.linspace(0.0, 0.4, 50)
    best = _best_threshold(pos, neg)
    assert best["direction"] == "gt"
    assert best["f1"] > 0.95
    assert best["precision"] > 0.9
    assert best["recall"] > 0.9


def test_best_threshold_lt_direction():
    pos = np.linspace(0.0, 0.2, 50)  # pos sits below neg
    neg = np.linspace(0.5, 1.0, 50)
    best = _best_threshold(pos, neg)
    assert best["direction"] == "lt"
    assert best["f1"] > 0.95


def test_best_threshold_class_imbalance_uses_f1():
    """With a huge negative population, 'always neg' would score ~99%
    accuracy — the scan must still find the F1-optimal point."""
    pos = np.linspace(0.6, 1.0, 10)
    neg = np.linspace(0.0, 0.5, 1000)
    best = _best_threshold(pos, neg)
    assert best["direction"] == "gt"
    assert best["recall"] > 0.9
    assert best["f1"] > 0.5
