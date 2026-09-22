"""Level 6b — severity-mapping evaluation (DL_PRIMARY_ROADMAP §9.4).

The Level-6 gate established that the fused scorer beats the
deterministic five-factor formula on event *probability*. The remaining
promotion question is whether the learned score can drive the four-class
severity policy — i.e. whether ``classify_severity`` with ``p_fused``
substituted for ``H`` produces tiers that rank real outcomes better than
the deterministic policy on identical held-out windows.

What this evaluation is honest about:

1. **No severity ground truth exists.** The corpus
   (``dynamic_escalation_train.parquet``) carries binary ``breached``
   labels only — HMAGLOFDB impact fields were never ingested. So this
   cannot be a per-class accuracy benchmark. What it *can* measure is
   decision quality: does tier rank order events correctly
   (ordinal coherence), and does the actionable boundary (elevated+ —
   the tier that opens a human review card, PRD §7.6) separate events
   from non-events better than the deterministic policy.

2. **The critical tier is unreachable in this evaluation.** At runtime
   ``classify_severity`` can return critical via (a) the
   ``expansion_pct >= 40`` override — a SAR measurement that does not
   exist for 1960–2024 historical windows — or (b) ``h >= 0.70`` AND
   real exposure (OSM corridor assets / population, also absent). Both
   arms are therefore evaluated with ``exposed_population=0,
   critical_assets=0, expansion_pct=0``; the effective sub-policy under
   test is informational | watch | elevated. The expansion override and
   the exposure leg stay deterministic under any promotion — they are
   physical/exposure facts, not learnable scores. **A passing gate here
   is necessary but NOT sufficient for severity promotion**: the
   critical tier and exposure interaction can only be validated by a
   runtime shadow-agreement study on real SAR observations.

3. **Calibration is fold-honest.** The deployed scorer emits
   Platt-calibrated ``p_fused`` (OOF-fitted sidecar). Inside each fold
   the calibrator is fitted on inner-block OOF predictions of the
   fold's *training* rows only — the test fold is never touched by
   calibration or threshold selection.

Arms compared on identical spatio-temporal folds
(``train_risk_fusion.spatiotemporal_cv`` protocol):

  * **deterministic** — ``classify_severity(h_baseline, 0, 0, 0)``
    where ``h_baseline`` is the five-factor formula on its
    honestly-measurable input subset (measured rain; neutral
    trend/expansion; fixed slope/drainage proxies — same proxies as
    the Level-6 baseline leg).
  * **learned_dropin** — ``classify_severity(p_fused_cal, 0, 0, 0)``:
    the deployed calibrated score through the *unchanged* policy
    thresholds (0.30 / 0.50 / 0.70). This is the actual promotion
    claim — the learned score as a drop-in replacement for H.

Additionally reported (descriptive, not gated): recall at fixed
false-alarm budgets {1%, 5%, 10%} — thresholds selected on the
calibrated inner-OOF training predictions, applied to test. The
deterministic formula has no tunable operating point, so this arm
characterises the learned mapping alone.

Declared gate (this evaluation defines it — no PRD numeric target
exists for severity mapping):

  * **G1 ordinal coherence:** pooled held-out event rate is
    non-decreasing across the tiers the learned policy populates.
  * **G2 actionable-boundary superiority:** at elevated+, learned
    recall > deterministic recall AND learned false-alarm rate <=
    max(deterministic FAR, 0.10). The absolute 10% cap is used instead
    of strict dominance because the deterministic policy under
    measurable-only inputs produces essentially no elevated+ output —
    strict dominance against a zero-output policy is vacuous.
  * **G3 decision usefulness:** pooled recall at a 10% false-alarm
    budget >= 0.50 — a severity mapping that captures less than half
    of held-out events at the declared alarm tolerance is not
    decision-useful. Pooled = each fold's inner-OOF threshold applied
    to its own test rows, outcomes pooled across folds (fold-honest,
    robust to small per-fold event counts).

Usage:
    python -m siren.ml.eval_severity_mapping --eval
    python -m siren.ml.eval_severity_mapping          # writes report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from siren.ml.dataset_dynamic_escalation import (
    DEFAULT_OUT,
    FEATURE_NAMES,
    STATIC_FEATURES,
)
from siren.ml.train_risk_fusion import (
    TEMPORAL_CUTOFF,
    XGBOOST_PARAMS,
    _baseline_h,
    _fit_prior_model,
    _monsoon_flag,
    _oof_prior,
)
from siren.risk.fusion import SEVERITY_ORDER, classify_severity

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_OUT = (
    REPO_ROOT / "models" / "checkpoints" / "severity_mapping_eval_report.json"
)

#: Declared gate constants (see module docstring).
FAR_CAP = 0.10
MIN_RECALL_AT_FAR = 0.50
FAR_GRID = (0.01, 0.05, 0.10)


# --------------------------------------------------------------------------- #
# Pure metric helpers (unit-testable)
# --------------------------------------------------------------------------- #

def tier_index(severity: str) -> int:
    """Ordinal rank of a severity label (informational=0 … critical=3)."""
    return SEVERITY_ORDER.index(severity)


def compute_tier_table(severities: np.ndarray, y: np.ndarray) -> dict:
    """Per-tier window/event counts and event rate.

    Returns {tier: {"n": int, "n_events": int, "event_rate": float}}
    only for populated tiers.
    """
    table = {}
    for tier in SEVERITY_ORDER:
        mask = severities == tier
        n = int(mask.sum())
        if n == 0:
            continue
        n_ev = int(y[mask].sum())
        table[tier] = {
            "n": n,
            "n_events": n_ev,
            "event_rate": float(n_ev / n),
        }
    return table


def is_monotonic(table: dict) -> bool:
    """True when the event rate is non-decreasing across populated tiers."""
    rates = [
        table[t]["event_rate"] for t in SEVERITY_ORDER if t in table
    ]
    return all(b >= a for a, b in zip(rates, rates[1:]))


def boundary_metrics(
    severities: np.ndarray, y: np.ndarray, boundary: str,
) -> dict:
    """Recall / false-alarm / precision / F1 at a severity boundary.

    "Flagged" means tier >= boundary (e.g. elevated+ opens a review
    card). Recall is the fraction of real events flagged; FAR is the
    fraction of non-events flagged.
    """
    thr = tier_index(boundary)
    flagged = np.array(
        [tier_index(s) >= thr for s in severities], dtype=bool
    )
    events = y == 1
    non_events = ~events
    tp = int((flagged & events).sum())
    fp = int((flagged & non_events).sum())
    fn = int((~flagged & events).sum())
    n_events = int(events.sum())
    n_non = int(non_events.sum())
    recall = tp / n_events if n_events else float("nan")
    far = fp / n_non if n_non else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom if denom else float("nan")
    return {
        "boundary": boundary,
        "n_flagged": int(flagged.sum()),
        "n_events": n_events,
        "n_non_events": n_non,
        "recall": float(recall),
        "false_alarm_rate": float(far),
        "precision": float(precision),
        "f1": float(f1),
        "missed_events": fn,
    }


def threshold_at_far(
    p_oof: np.ndarray, y_oof: np.ndarray, far_target: float,
) -> float:
    """Smallest threshold giving FAR <= far_target on OOF predictions.

    Uses the (1 - far_target) quantile of the non-event OOF scores so
    the resulting threshold never exceeds the declared budget on the
    calibration rows. Returns NaN when no non-events exist.
    """
    neg = p_oof[y_oof == 0]
    if len(neg) == 0:
        return float("nan")
    return float(np.quantile(neg, 1.0 - far_target))


def recall_at_threshold(p: np.ndarray, y: np.ndarray, thr: float) -> dict:
    """Test-fold recall/FAR for a pre-selected threshold."""
    flagged = p >= thr
    events = y == 1
    non_events = ~events
    tp = int((flagged & events).sum())
    fp = int((flagged & non_events).sum())
    recall = float(tp / events.sum()) if events.sum() else float("nan")
    far = float(fp / non_events.sum()) if non_events.sum() else float("nan")
    return {
        "threshold": float(thr),
        "recall": recall,
        "false_alarm_rate": far,
        "tp": tp,
        "fp": fp,
    }


def _severities_from_scores(scores: np.ndarray) -> np.ndarray:
    """Map a score vector through the unchanged classify_severity policy.

    Exposure inputs are neutral (0 / 0) and expansion is 0 — the
    critical tier is unreachable on historical windows (see module
    docstring, point 2).
    """
    return np.array(
        [classify_severity(float(s), 0, 0, 0.0) for s in scores]
    )


# --------------------------------------------------------------------------- #
# Fold-honest calibrated predictions
# --------------------------------------------------------------------------- #

def _inner_oof_calibrated(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    blocks_tr: np.ndarray,
    p_test_raw: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Platt calibration fitted inside the training fold.

    Returns (p_test_calibrated, p_oof_calibrated, y_oof, used_fallback).
    The OOF arrays are the calibrated predictions of held-out inner
    training blocks — used for threshold selection so test rows never
    influence calibration or thresholds. On any degenerate inner split
    the raw test scores are returned with used_fallback=True.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    import xgboost as xgb

    unique_blocks = np.unique(blocks_tr)
    if len(unique_blocks) < 2:
        return p_test_raw, np.array([]), np.array([]), True

    splits = list(
        GroupKFold(n_splits=min(4, len(unique_blocks))).split(
            X_tr, y_tr, blocks_tr
        )
    )
    p_oof = np.full(len(y_tr), np.nan)
    for a_idx, b_idx in splits:
        ya, yb = y_tr[a_idx], y_tr[b_idx]
        if len(np.unique(ya)) < 2:
            continue
        sp = (ya == 0).sum() / max((ya == 1).sum(), 1)
        m = xgb.XGBClassifier(
            **XGBOOST_PARAMS, scale_pos_weight=sp, random_state=seed,
        )
        m.fit(X_tr[a_idx], ya)
        p_oof[b_idx] = m.predict_proba(X_tr[b_idx])[:, 1]

    mask = ~np.isnan(p_oof)
    if mask.sum() < 10 or len(np.unique(y_tr[mask])) < 2:
        return p_test_raw, np.array([]), np.array([]), True

    platt = LogisticRegression()
    platt.fit(p_oof[mask].reshape(-1, 1), y_tr[mask])
    p_test_cal = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]
    p_oof_cal = platt.predict_proba(p_oof[mask].reshape(-1, 1))[:, 1]
    return p_test_cal, p_oof_cal, y_tr[mask], False


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #

def evaluate(
    df: pd.DataFrame,
    temporal_cutoff: pd.Timestamp = TEMPORAL_CUTOFF,
    seed: int = 42,
) -> dict:
    """Spatio-temporal holdout: learned severity mapping vs deterministic.

    Identical fold structure to train_risk_fusion.spatiotemporal_cv —
    per block, train on all other blocks, test on rows at/after the
    temporal cutoff.
    """
    import xgboost as xgb

    X_static = df[STATIC_FEATURES].values.astype(np.float32)
    X_base = df[FEATURE_NAMES].values.astype(np.float32)
    monsoon = _monsoon_flag(pd.to_datetime(df["event_date"]))
    y_all = df["breached"].values.astype(int)
    blocks = df["block"].values
    dates = pd.to_datetime(df["event_date"])
    h_baseline = _baseline_h(df)

    pooled = {
        "y": [], "h_baseline": [], "p_raw": [], "p_cal": [],
        "fold": [],
    }
    fold_metrics = []

    for blk in sorted(df["block"].unique()):
        tr = np.where(blocks != blk)[0]
        te = np.where((blocks == blk) & (dates >= temporal_cutoff))[0]
        if len(te) == 0:
            continue
        if len(np.unique(y_all[tr])) < 2 or len(np.unique(y_all[te])) < 2:
            logger.warning("Block %d: single-class split — skipped", blk)
            continue

        prior_tr = _oof_prior(X_static[tr], y_all[tr], blocks[tr], seed)
        prior_model = _fit_prior_model(X_static[tr], y_all[tr], seed + 1)
        prior_te = prior_model.predict_proba(X_static[te])[:, 1]
        X_tr = np.column_stack([X_base[tr], prior_tr, monsoon[tr]])
        X_te = np.column_stack([X_base[te], prior_te, monsoon[te]])

        sp = (y_all[tr] == 0).sum() / max((y_all[tr] == 1).sum(), 1)
        model = xgb.XGBClassifier(
            **XGBOOST_PARAMS, scale_pos_weight=sp,
            random_state=seed + int(blk),
        )
        model.fit(X_tr, y_all[tr])
        p_raw = model.predict_proba(X_te)[:, 1]

        p_cal, p_oof_cal, y_oof, fallback = _inner_oof_calibrated(
            X_tr, y_all[tr], blocks[tr], p_raw, seed + int(blk),
        )

        pooled["y"].append(y_all[te])
        pooled["h_baseline"].append(h_baseline[te])
        pooled["p_raw"].append(p_raw)
        pooled["p_cal"].append(p_cal)
        pooled["fold"].append(np.full(len(te), blk))

        fm = {
            "heldout_block": int(blk),
            "n_test": len(te),
            "n_breached_test": int(y_all[te].sum()),
            "calibration_fallback": fallback,
        }
        # Recall at fixed FAR budgets — thresholds selected on inner-OOF.
        for far_t in FAR_GRID:
            key = f"recall_at_far_{int(far_t * 100)}"
            if len(y_oof) and not fallback:
                thr = threshold_at_far(p_oof_cal, y_oof, far_t)
                fm[key] = recall_at_threshold(p_cal, y_all[te], thr)
                fm[key]["n_events"] = int(y_all[te].sum())
                fm[key]["n_non_events"] = int((y_all[te] == 0).sum())
            else:
                fm[key] = None
        fold_metrics.append(fm)
        logger.info(
            "Block %d: %d test rows (%d events), calibration %s",
            blk, len(te), int(y_all[te].sum()),
            "fallback-raw" if fallback else "platt-inner-oof",
        )

    y = np.concatenate(pooled["y"])
    h = np.concatenate(pooled["h_baseline"])
    p_cal = np.concatenate(pooled["p_cal"])
    p_raw = np.concatenate(pooled["p_raw"])

    sev_det = _severities_from_scores(h)
    sev_learned = _severities_from_scores(p_cal)
    sev_learned_raw = _severities_from_scores(p_raw)

    table_det = compute_tier_table(sev_det, y)
    table_learned = compute_tier_table(sev_learned, y)
    table_learned_raw = compute_tier_table(sev_learned_raw, y)

    boundaries = {}
    for b in ("watch", "elevated"):
        boundaries[b] = {
            "deterministic": boundary_metrics(sev_det, y, b),
            "learned": boundary_metrics(sev_learned, y, b),
            "learned_raw": boundary_metrics(sev_learned_raw, y, b),
        }

    # Recall-at-FAR: per-fold means AND pooled (each fold's inner-OOF
    # threshold applied to its own test rows, outcomes pooled — still
    # fold-honest, less noisy under small per-fold event counts).
    far_curves = {}
    for far_t in FAR_GRID:
        key = f"recall_at_far_{int(far_t * 100)}"
        vals = [m[key] for m in fold_metrics if m.get(key)]
        tp = sum(v["tp"] for v in vals)
        fp = sum(v["fp"] for v in vals)
        n_ev = sum(v["n_events"] for v in vals)
        n_ne = sum(v["n_non_events"] for v in vals)
        far_curves[key] = {
            "far_target": far_t,
            "mean_test_recall": (
                float(np.mean([v["recall"] for v in vals]))
                if vals else None
            ),
            "mean_test_far": (
                float(np.mean([v["false_alarm_rate"] for v in vals]))
                if vals else None
            ),
            "pooled_test_recall": float(tp / n_ev) if n_ev else None,
            "pooled_test_far": float(fp / n_ne) if n_ne else None,
            "n_folds": len(vals),
        }

    g1 = is_monotonic(table_learned)
    det_el = boundaries["elevated"]["deterministic"]
    lr_el = boundaries["elevated"]["learned"]
    g2 = bool(
        lr_el["recall"] > det_el["recall"]
        and lr_el["false_alarm_rate"] <= max(det_el["false_alarm_rate"], FAR_CAP)
    )
    r10 = far_curves["recall_at_far_10"]["pooled_test_recall"]
    g3 = bool(r10 is not None and r10 >= MIN_RECALL_AT_FAR)

    return {
        "component": "learned_risk_fusion",
        "evaluation": "severity_mapping",
        "protocol": (
            "identical spatio-temporal folds to train_risk_fusion: per "
            "block, train on all other blocks, test on rows >= "
            f"{temporal_cutoff.date()}; p_fused calibrated by Platt on "
            "inner-block OOF predictions of the fold's training rows; "
            "thresholds (fixed-FAR arm) selected on the same OOF rows"
        ),
        "arms": {
            "deterministic": (
                "classify_severity(h_five_factor, pop=0, assets=0, "
                "expansion=0) — honestly-measurable input subset only"
            ),
            "learned_dropin": (
                "classify_severity(p_fused_calibrated, pop=0, assets=0, "
                "expansion=0) — unchanged policy thresholds "
                "0.30/0.50/0.70"
            ),
        },
        "unreachable_in_eval": (
            "critical tier: needs expansion_pct >= 40 (SAR) or "
            "h >= 0.70 AND real exposure (OSM) — neither exists for "
            "historical windows; the exposure leg and expansion "
            "override stay deterministic under any promotion"
        ),
        "n_test_windows": int(len(y)),
        "n_test_events": int(y.sum()),
        "tier_tables": {
            "deterministic": table_det,
            "learned_calibrated": table_learned,
            "learned_raw": table_learned_raw,
        },
        "ordinal_monotonic": {
            "deterministic": is_monotonic(table_det),
            "learned_calibrated": g1,
            "learned_raw": is_monotonic(table_learned_raw),
        },
        "boundary_metrics": boundaries,
        "recall_at_fixed_far": far_curves,
        "fold_metrics": fold_metrics,
        "gate": {
            "G1_ordinal_coherence": g1,
            "G2_elevated_boundary_superiority": g2,
            "G3_recall_at_10pct_far_ge_0.5": g3,
            "gate_passed": g1 and g2 and g3,
            "note": (
                "gate declared by this evaluation (no PRD numeric "
                "target exists for severity mapping). Passing is "
                "necessary but NOT sufficient for severity promotion — "
                "the critical tier and exposure interaction require a "
                "runtime shadow-agreement study on real SAR/OSM inputs."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Level-6b severity-mapping evaluation"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_OUT)
    p.add_argument("--output", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--cutoff", default=str(TEMPORAL_CUTOFF.date()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval", action="store_true",
                   help="print metrics without writing the report")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.input.exists():
        logger.error(
            "Dataset not found: %s — run "
            "python -m siren.ml.dataset_dynamic_escalation first",
            args.input,
        )
        return 1

    df = pd.read_parquet(args.input)
    df["event_date"] = pd.to_datetime(df["event_date"])
    logger.info(
        "Loaded %d windows (%d events / %d non-events)",
        len(df), int(df["breached"].sum()),
        int((df["breached"] == 0).sum()),
    )

    results = evaluate(df, pd.Timestamp(args.cutoff), args.seed)
    report = {
        "status": "severity_mapping_evaluated",
        "evaluation_valid": True,
        "dataset": str(args.input),
        "n_samples": len(df),
        **results,
    }

    print("\n" + "=" * 68)
    print("Severity-mapping evaluation — learned p_fused vs deterministic H")
    print("=" * 68)
    for arm in ("deterministic", "learned_calibrated"):
        tbl = results["tier_tables"][arm]
        rates = " | ".join(
            f"{t}:{v['event_rate']:.2f}(n={v['n']})"
            for t, v in tbl.items()
        )
        print(f"  {arm:20s} tiers → event rate: {rates}")
    for b in ("watch", "elevated"):
        det = results["boundary_metrics"][b]["deterministic"]
        lr = results["boundary_metrics"][b]["learned"]
        print(
            f"  {b}+ boundary — det recall {det['recall']:.3f} / "
            f"FAR {det['false_alarm_rate']:.3f} | learned recall "
            f"{lr['recall']:.3f} / FAR {lr['false_alarm_rate']:.3f}"
        )
    for k, v in results["recall_at_fixed_far"].items():
        if v["pooled_test_recall"] is not None:
            print(
                f"  {k}: pooled recall {v['pooled_test_recall']:.3f} "
                f"@ FAR {v['pooled_test_far']:.3f} "
                f"(mean per-fold {v['mean_test_recall']:.3f}, "
                f"{v['n_folds']} folds)"
            )
    g = results["gate"]
    print(
        f"  G1 ordinal: {'PASS' if g['G1_ordinal_coherence'] else 'FAIL'} | "
        f"G2 elevated+ superiority: "
        f"{'PASS' if g['G2_elevated_boundary_superiority'] else 'FAIL'} | "
        f"G3 recall@10%FAR>=0.5: "
        f"{'PASS' if g['G3_recall_at_10pct_far_ge_0.5'] else 'FAIL'}"
    )
    print(f"  GATE: {'PASS' if g['gate_passed'] else 'FAIL'}")
    print("=" * 68)

    if not args.eval:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        logger.info("Report written to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
