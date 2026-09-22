"""Tests for ml/eval_severity_mapping.py — Level-6b severity-mapping eval.

The pure helpers (tier tables, monotonicity, boundary metrics, FAR
thresholds) are tested directly; the fold-honest driver is tested on a
small synthetic frame that exercises the real code path end-to-end.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from siren.ml.eval_severity_mapping import (
    boundary_metrics,
    compute_tier_table,
    evaluate,
    is_monotonic,
    recall_at_threshold,
    threshold_at_far,
    tier_index,
    _severities_from_scores,
)
from siren.risk.fusion import SEVERITY_ORDER


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def test_tier_index_matches_severity_order():
    assert [tier_index(s) for s in SEVERITY_ORDER] == [0, 1, 2, 3]
    assert tier_index("informational") < tier_index("elevated")


def test_compute_tier_table_counts_and_rates():
    sev = np.array(["watch", "watch", "elevated", "informational"])
    y = np.array([1, 0, 1, 0])
    table = compute_tier_table(sev, y)
    assert table["watch"] == {"n": 2, "n_events": 1, "event_rate": 0.5}
    assert table["elevated"]["event_rate"] == 1.0
    assert "critical" not in table  # unpopulated tiers are omitted


def test_is_monotonic():
    mono = {
        "informational": {"event_rate": 0.1},
        "watch": {"event_rate": 0.3},
        "elevated": {"event_rate": 0.6},
    }
    non_mono = {
        "informational": {"event_rate": 0.5},
        "watch": {"event_rate": 0.3},
    }
    assert is_monotonic(mono) is True
    assert is_monotonic(non_mono) is False
    # Sparse populated tiers still checked in order.
    assert is_monotonic({
        "informational": {"event_rate": 0.2},
        "critical": {"event_rate": 0.9},
    }) is True


def test_boundary_metrics_recall_and_far():
    sev = np.array([
        "informational", "watch", "elevated", "elevated", "critical",
    ])
    y = np.array([0, 1, 1, 0, 1])
    m = boundary_metrics(sev, y, "elevated")
    # elevated+ flags rows 2,3,4 -> tp=2, fp=1 over 3 events / 2 non-events
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_alarm_rate"] == pytest.approx(0.5)
    assert m["n_flagged"] == 3
    assert m["missed_events"] == 1


def test_threshold_at_far_picks_quantile():
    rng = np.random.RandomState(0)
    y = np.concatenate([np.zeros(100), np.ones(50)])
    p = np.concatenate([rng.uniform(0, 0.6, 100), rng.uniform(0.3, 1.0, 50)])
    thr = threshold_at_far(p, y, 0.10)
    # Flagged non-events must respect the budget.
    flagged_neg = (p[y == 0] >= thr).mean()
    assert flagged_neg <= 0.10 + 1e-9
    assert np.isnan(threshold_at_far(p[y == 1], np.ones(50), 0.10))


def test_recall_at_threshold_counts():
    p = np.array([0.1, 0.6, 0.9, 0.4])
    y = np.array([0, 1, 1, 0])
    r = recall_at_threshold(p, y, 0.5)
    assert r["tp"] == 2 and r["fp"] == 0
    assert r["recall"] == pytest.approx(1.0)
    assert r["false_alarm_rate"] == pytest.approx(0.0)


def test_severities_from_scores_never_critical():
    """Neutral exposure/expansion → critical is unreachable."""
    sev = _severities_from_scores(np.array([0.0, 0.35, 0.6, 0.95]))
    assert list(sev) == ["informational", "watch", "elevated", "elevated"]
    assert "critical" not in sev


# --------------------------------------------------------------------------- #
# End-to-end driver on a synthetic frame
# --------------------------------------------------------------------------- #

def _synthetic_df(n_per_block=40, n_blocks=4, seed=0):
    """Small frame with all required columns and a learnable signal."""
    rng = np.random.RandomState(seed)
    rows = []
    for blk in range(n_blocks):
        for i in range(n_per_block):
            breached = int(rng.rand() < 0.3)
            rain = rng.uniform(0, 80) + breached * 60
            rows.append({
                "sample_id": f"s_{blk}_{i}",
                "lake_id": blk * 100 + (i % 8),
                "event_date": pd.Timestamp("2016-01-01")
                + timedelta(days=int(rng.randint(0, 3000))),
                "lat": 27.0 + blk,
                "lon": 86.0,
                "lake_elev_m": 4000.0,
                "lake_area_km2": 1.0,
                "breached": breached,
                "neg_kind": "event" if breached else "stable_lake",
                "precip_30d_mm": rain,
                "precip_7d_mm": rain / 4,
                "max_daily_precip_mm": rain / 5,
                "heavy_rain_days": rain / 40,
                "api_30": rain * 0.8,
                "rain_anom_30d_mm": rain - 50,
                "mdd_30": 150.0,
                "mdd_anom_30": 5.0,
                "ft_cycles_14": 0,
                "dist_glacier_m": 500.0,
                "glacier_area_10km_m2": 4e7,
                "log_lake_area_km2": 0.0,
                "log_dist_glacier_m": 6.0,
                "log_glacier_area_10km": 4.0,
                "block": blk,
            })
    return pd.DataFrame(rows)


def test_evaluate_returns_report_structure():
    """The driver produces a complete, honest report on synthetic data."""
    df = _synthetic_df()
    results = evaluate(df, temporal_cutoff=pd.Timestamp("2015-01-01"),
                       seed=1)
    assert results["n_test_windows"] > 0
    for arm in ("deterministic", "learned_calibrated", "learned_raw"):
        assert arm in results["tier_tables"]
    for b in ("watch", "elevated"):
        for arm in ("deterministic", "learned", "learned_raw"):
            m = results["boundary_metrics"][b][arm]
            assert 0.0 <= m["recall"] <= 1.0 or np.isnan(m["recall"])
            assert 0.0 <= m["false_alarm_rate"] <= 1.0
    gate = results["gate"]
    assert gate["gate_passed"] == (
        gate["G1_ordinal_coherence"]
        and gate["G2_elevated_boundary_superiority"]
        and gate["G3_recall_at_10pct_far_ge_0.5"]
    )
    assert "unreachable_in_eval" in results


def test_evaluate_skips_degenerate_folds():
    """Blocks without post-cutoff rows or single-class splits are skipped."""
    df = _synthetic_df(n_per_block=20, n_blocks=3, seed=2)
    # Push every row in block 0 before the cutoff -> it must be skipped.
    df.loc[df["block"] == 0, "event_date"] = pd.Timestamp("2001-06-01")
    results = evaluate(df, temporal_cutoff=pd.Timestamp("2015-01-01"),
                       seed=3)
    held = {m["heldout_block"] for m in results["fold_metrics"]}
    assert 0 not in held


def test_committed_report_gate_verdict():
    """The committed severity-mapping report records the honest FAIL."""
    import json
    from siren.ml.eval_severity_mapping import DEFAULT_REPORT_OUT

    if not DEFAULT_REPORT_OUT.exists():
        pytest.skip("severity-mapping report not generated yet")
    report = json.loads(DEFAULT_REPORT_OUT.read_text())
    gate = report["gate"]
    assert gate["G1_ordinal_coherence"] is True
    assert gate["G2_elevated_boundary_superiority"] is True
    assert gate["G3_recall_at_10pct_far_ge_0.5"] is False
    assert gate["gate_passed"] is False
    # The report must document why critical is unreachable.
    assert "critical" in report["unreachable_in_eval"]
