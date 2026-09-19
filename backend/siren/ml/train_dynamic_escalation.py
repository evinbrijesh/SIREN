"""Spatio-temporal evaluation + training for the dynamic escalation model.

Trains P(escalation | static morphometrics, trailing-30d weather) on the
dataset built by ``siren.ml.dataset_dynamic_escalation`` (parquet at
data/datasets/dynamic_escalation_train.parquet).

Evaluation protocol — spatio-temporal holdout:
    For each spatial block b (KMeans blocks assigned at dataset build):
        train = all samples NOT in block b
        test  = samples in block b with event_date >= TEMPORAL_CUTOFF
    Rows in block b before the cutoff are unused — the test fold is
    held out in space (whole block) AND time (recent events only).

    Rationale: random or spatial-only CV leaks either geography (same
    sub-range shares climate/geology) or era (reporting density grows
    over time). This protocol estimates true forward-in-time,
    out-of-region generalisation — the deployment condition.

Calibration:
    Same cross-fitted isotonic diagnostic as train_susceptibility_spatial:
    an internal GroupKFold(2) split of each training fold fits the
    isotonic mapping on held-out-half predictions, frozen, applied to test.
    Gate verdict uses the uncalibrated Brier; the calibrated number is
    reported as evidence for adopting a booster+isotonic model definition.

Usage:
    python -m siren.ml.train_dynamic_escalation --eval
    python -m siren.ml.train_dynamic_escalation --save-model
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
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_OUT = (
    REPO_ROOT / "models" / "checkpoints" / "xgboost_dynamic_escalation.json"
)
DEFAULT_REPORT_OUT = (
    REPO_ROOT / "models" / "checkpoints" / "dynamic_escalation_eval_report.json"
)
TEMPORAL_CUTOFF = pd.Timestamp("2015-01-01")
BRIER_GATE = 0.15

XGBOOST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
}


def _oof_probs(X, y, groups, n_blocks, seed):
    """Out-of-fold raw predictions across spatial blocks."""
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_blocks).split(X, y, groups):
        sp = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.XGBClassifier(
            **XGBOOST_PARAMS, scale_pos_weight=sp, random_state=seed,
        )
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def _isotonic_oof(X, y, groups, n_blocks, seed):
    """Out-of-fold raw predictions across blocks -> isotonic mapping.

    Returns (isotonic_thresholds_x, isotonic_thresholds_y) fitted on OOF
    predictions — honest calibrator inputs for a final all-data model.
    """
    from sklearn.isotonic import IsotonicRegression

    oof = _oof_probs(X, y, groups, n_blocks, seed)
    mask = ~np.isnan(oof)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof[mask], y[mask])
    return iso


def _platt_oof(X, y, groups, n_blocks, seed):
    """Platt (logistic) calibration on OOF predictions — more stable than
    isotonic at small fold sizes. Returns (coef, intercept)."""
    from sklearn.linear_model import LogisticRegression

    oof = _oof_probs(X, y, groups, n_blocks, seed)
    mask = ~np.isnan(oof)
    lr = LogisticRegression()
    lr.fit(oof[mask].reshape(-1, 1), y[mask])
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def spatiotemporal_cv(
    df: pd.DataFrame,
    features: list[str],
    temporal_cutoff: pd.Timestamp = TEMPORAL_CUTOFF,
    seed: int = 42,
) -> dict:
    """Held-out-in-space-and-time evaluation. Returns metrics dict."""
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold

    X = df[features].values.astype(np.float32)
    y = df["breached"].values.astype(int)
    blocks = df["block"].values
    dates = pd.to_datetime(df["event_date"])

    fold_metrics, importances = [], []
    for blk in sorted(df["block"].unique()):
        tr_mask = blocks != blk
        te_mask = (blocks == blk) & (dates >= temporal_cutoff)
        tr, te = np.where(tr_mask)[0], np.where(te_mask)[0]
        if len(te) == 0:
            logger.info("Block %d: no post-%s rows — skipped",
                        blk, temporal_cutoff.date())
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            logger.warning("Block %d: single-class split — skipped", blk)
            continue

        sp = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        model = xgb.XGBClassifier(
            **XGBOOST_PARAMS, scale_pos_weight=sp, random_state=seed + int(blk),
        )
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]

        # Cross-fitted calibration: internal 2-block split of training
        brier_cal = float("nan")
        brier_platt = float("nan")
        inner = list(GroupKFold(n_splits=2).split(X[tr], y[tr], blocks[tr]))
        if inner:
            a_idx, b_idx = inner[0]
            tra, trb = tr[a_idx], tr[b_idx]
            if len(np.unique(y[tra])) == 2 and len(np.unique(y[trb])) == 2:
                sp_a = (y[tra] == 0).sum() / max((y[tra] == 1).sum(), 1)
                mdl_a = xgb.XGBClassifier(
                    **XGBOOST_PARAMS, scale_pos_weight=sp_a,
                    random_state=seed + int(blk),
                )
                mdl_a.fit(X[tra], y[tra])
                p_trb = mdl_a.predict_proba(X[trb])[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_trb, y[trb])
                brier_cal = float(
                    brier_score_loss(y[te], iso.predict(p))
                )
                # Platt scaling — stabler than isotonic on small folds
                platt = LogisticRegression().fit(
                    p_trb.reshape(-1, 1), y[trb])
                brier_platt = float(
                    brier_score_loss(y[te], platt.predict_proba(
                        p.reshape(-1, 1))[:, 1])
                )

        m = {
            "heldout_block": int(blk),
            "n_train": len(tr),
            "n_test": len(te),
            "n_breached_test": int(y[te].sum()),
            "test_date_range": [
                str(dates[te].min().date()), str(dates[te].max().date()),
            ],
            "roc_auc": float(roc_auc_score(y[te], p)),
            "pr_auc": float(average_precision_score(y[te], p)),
            "brier_score": float(brier_score_loss(y[te], p)),
            "brier_score_calibrated": (
                brier_cal if not np.isnan(brier_cal) else None
            ),
            "brier_score_platt": (
                brier_platt if not np.isnan(brier_platt) else None
            ),
        }
        fold_metrics.append(m)
        importances.append(model.feature_importances_)
        logger.info(
            "Block %d: AUC=%.3f Brier=%.4f cal=%.4f (test %d, %d events, %s→%s)",
            blk, m["roc_auc"], m["brier_score"],
            m["brier_score_calibrated"] or -1,
            m["n_test"], m["n_breached_test"],
            m["test_date_range"][0], m["test_date_range"][1],
        )

    aucs = [m["roc_auc"] for m in fold_metrics]
    briers = [m["brier_score"] for m in fold_metrics]
    cals = [m["brier_score_calibrated"] for m in fold_metrics
            if m["brier_score_calibrated"] is not None]
    mean_brier = float(np.mean(briers)) if briers else float("nan")

    return {
        "protocol": (
            "spatio-temporal holdout: per block, train on all other blocks, "
            f"test on rows >= {temporal_cutoff.date()}"
        ),
        "temporal_cutoff": str(temporal_cutoff.date()),
        "mean_roc_auc": float(np.mean(aucs)) if aucs else None,
        "std_roc_auc": float(np.std(aucs)) if aucs else None,
        "mean_brier": mean_brier if not np.isnan(mean_brier) else None,
        "std_brier": float(np.std(briers)) if briers else None,
        "mean_brier_calibrated": float(np.mean(cals)) if cals else None,
        "mean_brier_platt": (
            float(np.mean([m["brier_score_platt"] for m in fold_metrics
                           if m["brier_score_platt"] is not None]))
            if any(m["brier_score_platt"] is not None
                   for m in fold_metrics) else None
        ),
        "mean_pr_auc": float(np.mean([m["pr_auc"] for m in fold_metrics]))
        if fold_metrics else None,
        "fold_metrics": fold_metrics,
        "feature_importances": {
            f: float(v) for f, v in
            zip(features, np.mean(importances, axis=0))
        } if importances else {},
        "passes_brier_gate": bool(
            not np.isnan(mean_brier) and mean_brier < BRIER_GATE
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Spatio-temporal eval for the dynamic escalation model"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_OUT,
                   help="dataset parquet from dataset_dynamic_escalation")
    p.add_argument("--output", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--cutoff", default=str(TEMPORAL_CUTOFF.date()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval", action="store_true",
                   help="print metrics without writing the report")
    p.add_argument("--save-model", action="store_true",
                   help="persist booster + OOF isotonic calibration sidecar")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.input.exists():
        logger.error(
            "Dataset not found: %s — run "
            "python -m siren.ml.dataset_dynamic_escalation first "
            "(requires network for the weather fetch stage)", args.input,
        )
        return 1

    df = pd.read_parquet(args.input)
    df["event_date"] = pd.to_datetime(df["event_date"])
    features = [f for f in FEATURE_NAMES if f in df.columns]
    logger.info("Loaded %d samples (%d events / %d non-events), %d features",
                len(df), int(df["breached"].sum()),
                int((df["breached"] == 0).sum()), len(features))

    cutoff = pd.Timestamp(args.cutoff)
    results = spatiotemporal_cv(df, features, cutoff, args.seed)

    report = {
        "status": "gate_evaluated",
        "evaluation_valid": True,
        "inference_allowed": False,
        "brier_gate": BRIER_GATE,
        "gate_metric": (
            "mean_brier (uncalibrated); mean_brier_calibrated is the "
            "cross-fitted in-fold isotonic diagnostic"
        ),
        "features": features,
        "n_samples": len(df),
        "n_breached": int(df["breached"].sum()),
        "n_stable": int((df["breached"] == 0).sum()),
        "dataset": str(args.input),
        **results,
    }

    print("\n" + "=" * 64)
    print("Dynamic escalation — spatio-temporal holdout")
    print("=" * 64)
    print(f"  Mean ROC-AUC: {results['mean_roc_auc']}")
    print(f"  Mean Brier:   {results['mean_brier']} "
          f"(calibrated: {results['mean_brier_calibrated']})")
    print(f"  Gate (<{BRIER_GATE}): "
          f"{'PASS' if results['passes_brier_gate'] else 'FAIL'}")
    for m in results["fold_metrics"]:
        print(f"    block {m['heldout_block']}: AUC={m['roc_auc']:.3f} "
              f"Brier={m['brier_score']:.4f} ({m['n_breached_test']}/"
              f"{m['n_test']} events {m['test_date_range'][0]}→"
              f"{m['test_date_range'][1]})")
    print("=" * 64)

    if not args.eval:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        logger.info("Report written to %s", args.output)

    if args.save_model:
        X = df[features].values.astype(np.float32)
        y = df["breached"].values.astype(int)
        groups = df["block"].values
        sp = (y == 0).sum() / max((y == 1).sum(), 1)

        import xgboost as xgb

        n_blocks = df["block"].nunique()
        iso = _isotonic_oof(X, y, groups, n_blocks, args.seed)
        platt_coef, platt_int = _platt_oof(
            X, y, groups, n_blocks, args.seed)
        model = xgb.XGBClassifier(**XGBOOST_PARAMS, scale_pos_weight=sp,
                                  random_state=args.seed)
        model.fit(X, y)

        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(args.model_output))
        cal_path = args.model_output.with_suffix(".calibration.json")
        cal_path.write_text(json.dumps({
            "method": "platt_crossfit_oof",
            "prefer": "platt",
            "platt": {
                "coef": platt_coef,
                "intercept": platt_int,
                "apply": (
                    "p_cal = sigmoid(coef * p_raw + intercept)"
                ),
            },
            "isotonic": {
                "x_thresholds": iso.X_thresholds_.tolist(),
                "y_thresholds": iso.y_thresholds_.tolist(),
                "apply": (
                    "p_cal = np.interp(p_raw, x_thresholds, y_thresholds)"
                ),
            },
            "base_rate": float(y.mean()),
        }, indent=2))
        logger.info("Booster + calibrator saved to %s", args.model_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
