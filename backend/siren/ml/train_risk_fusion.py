"""Level 6 — learned risk-fusion evaluation + training (DL_PRIMARY_ROADMAP §9).

Trains P(event | static morphometrics, trailing-30d weather window,
stacked susceptibility prior, monsoon-season flag) on the dataset built
by ``siren.ml.dataset_dynamic_escalation`` — the same 887 honest windows
(230 dated GLOF events + stable-lake and within-lake negatives) used for
the Tier-2 escalation model.

What this adds over Tier-2 (``train_dynamic_escalation``):

1. **Stacked static prior.** ``p_susceptibility`` — the output of a
   susceptibility-style model (4 measured morphometrics -> breached)
   — is appended as a feature. Inside every evaluation fold the prior is
   computed *fold-honestly*: train rows get inner-block OOF predictions,
   test rows get a prior model fit only on the training fold. The
   deployed ``xgboost_susceptibility_spatial`` checkpoint is never
   consulted during evaluation (it trained on all breach lakes —
   feeding its outputs into the test fold would leak labels).

2. **Season flag.** ``in_monsoon_window`` — the Jun–Sep declaration from
   ``siren.scope`` — is a measured, inference-time-available quantity
   (the observation date), not a label proxy.

3. **Head-to-head vs the deterministic baseline.** Every fold also
   scores the test rows with the canonical PRD §9.5 five-factor
   ``hazard_score`` using the honestly-measurable subset of its inputs:

       rainfall_24h_mm  <- max_daily_precip_mm
       rainfall_7d_mm   <- precip_7d_mm
       trend_class      <- "uncertain"   (no SAR history exists)
       expansion_pct    <- 0             (no SAR measurement exists)
       mean_slope_deg   <- 31.0          (pipeline MEAN_SLOPE_DEG proxy)
       change_in_drainage <- True        (all monitored lakes drain)

   This is the honest comparison the Level 6 gate asks for: the learned
   scorer must beat the deployed formula on *its own* inputs. H is a
   policy score, not a calibrated probability — the AUC leg is the
   meaningful comparison; the Brier leg is reported for completeness.

Evaluation protocol — identical to ``train_dynamic_escalation``:
per spatial block, train on all other blocks, test on rows with
event_date >= TEMPORAL_CUTOFF (held out in space AND time).

Gate (DL_PRIMARY_ROADMAP §9.2):
    mean raw Brier < 0.15 AND mean fused ROC-AUC > baseline ROC-AUC
    on the same held-out windows.

Usage:
    python -m siren.ml.train_risk_fusion --eval
    python -m siren.ml.train_risk_fusion --save-model
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

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_OUT = (
    REPO_ROOT / "models" / "checkpoints" / "xgboost_risk_fusion.json"
)
DEFAULT_REPORT_OUT = (
    REPO_ROOT / "models" / "checkpoints" / "risk_fusion_eval_report.json"
)
TEMPORAL_CUTOFF = pd.Timestamp("2015-01-01")
BRIER_GATE = 0.15

#: Extra features appended to the Tier-2 vector. The stacked prior is
#: fold-honest at eval time; at runtime the deployed (calibrated)
#: susceptibility score is used — a documented distribution shift.
FUSED_EXTRA_FEATURES = ["p_susceptibility", "in_monsoon_window"]
FUSED_FEATURE_NAMES = FEATURE_NAMES + FUSED_EXTRA_FEATURES

#: Deterministic baseline proxies (documented above — the subset of the
#: five-factor formula's inputs that exist for historical windows).
BASELINE_TREND_CLASS = "uncertain"
BASELINE_SLOPE_DEG = 31.0          # pipeline.MEAN_SLOPE_DEG
BASELINE_IN_DRAINAGE = True

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


def _monsoon_flag(dates: pd.Series) -> np.ndarray:
    """1.0 when the window end date falls in the Jun–Sep monsoon window."""
    from siren.scope import VULNERABLE_MONTHS

    return dates.dt.month.isin(VULNERABLE_MONTHS).astype(np.float32).values


def _baseline_h(df: pd.DataFrame) -> np.ndarray:
    """Deterministic five-factor hazard score per window (honest proxies).

    NaN precipitation maps to 0.0 — a missing weather measurement is
    treated as no measured rain, never interpolated from the label.
    """
    from siren.risk.fusion import hazard_score

    rain_24h = df["max_daily_precip_mm"].fillna(0.0).values
    rain_7d = df["precip_7d_mm"].fillna(0.0).values
    out = np.empty(len(df), dtype=np.float64)
    for i in range(len(df)):
        h, _ = hazard_score(
            trend_class=BASELINE_TREND_CLASS,
            expansion_pct=0.0,
            rainfall_24h_mm=float(rain_24h[i]),
            rainfall_7d_mm=float(rain_7d[i]),
            mean_slope_deg=BASELINE_SLOPE_DEG,
            change_in_drainage=BASELINE_IN_DRAINAGE,
        )
        out[i] = h
    return out


def _fit_prior_model(X_static: np.ndarray, y: np.ndarray, seed: int):
    """Susceptibility-style prior model: static morphometrics -> breached."""
    import xgboost as xgb

    sp = (y == 0).sum() / max((y == 1).sum(), 1)
    m = xgb.XGBClassifier(
        **XGBOOST_PARAMS, scale_pos_weight=sp, random_state=seed,
    )
    m.fit(X_static, y)
    return m


def _oof_prior(
    X_static: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> np.ndarray:
    """GroupKFold-OOF susceptibility prior over a training fold.

    Each row's prior comes from a model that never saw its block —
    the fused model learns on honest, non-overfit prior values.
    """
    from sklearn.model_selection import GroupKFold

    n_blocks = len(np.unique(groups))
    oof = np.full(len(y), np.nan)
    if n_blocks < 2:
        return oof
    for tr_i, te_i in GroupKFold(n_splits=n_blocks).split(
        X_static, y, groups
    ):
        if len(np.unique(y[tr_i])) < 2:
            continue
        m = _fit_prior_model(X_static[tr_i], y[tr_i], seed)
        oof[te_i] = m.predict_proba(X_static[te_i])[:, 1]
    return oof


def _isotonic_oof(X, y, groups, n_blocks, seed):
    """OOF predictions across blocks -> isotonic mapping (final model)."""
    from sklearn.isotonic import IsotonicRegression

    oof = np.full(len(y), np.nan)
    from sklearn.model_selection import GroupKFold

    for tr, te in GroupKFold(n_splits=n_blocks).split(X, y, groups):
        sp = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        import xgboost as xgb

        m = xgb.XGBClassifier(
            **XGBOOST_PARAMS, scale_pos_weight=sp, random_state=seed,
        )
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof[mask], y[mask])
    return iso


def _platt_oof(X, y, groups, n_blocks, seed):
    """Platt (logistic) calibration on block-OOF predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    import xgboost as xgb

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_blocks).split(X, y, groups):
        sp = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.XGBClassifier(
            **XGBOOST_PARAMS, scale_pos_weight=sp, random_state=seed,
        )
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
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
    """Held-out-in-space-and-time evaluation, fused model vs baseline."""
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold

    X_static = df[STATIC_FEATURES].values.astype(np.float32)
    monsoon = _monsoon_flag(pd.to_datetime(df["event_date"]))
    y = df["breached"].values.astype(int)
    blocks = df["block"].values
    dates = pd.to_datetime(df["event_date"])
    h_baseline = _baseline_h(df)

    fold_metrics, importances = [], []
    for blk in sorted(df["block"].unique()):
        tr = np.where(blocks != blk)[0]
        te = np.where((blocks == blk) & (dates >= temporal_cutoff))[0]
        if len(te) == 0:
            logger.info("Block %d: no post-%s rows — skipped",
                        blk, temporal_cutoff.date())
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            logger.warning("Block %d: single-class split — skipped", blk)
            continue

        # Fold-honest stacked prior: train rows get inner-block OOF
        # predictions; test rows get a prior fit only on this fold's
        # training blocks.
        prior_tr = _oof_prior(X_static[tr], y[tr], blocks[tr], seed)
        prior_model = _fit_prior_model(X_static[tr], y[tr], seed + 1)
        prior_te = prior_model.predict_proba(X_static[te])[:, 1]

        X_base = df[FEATURE_NAMES].values.astype(np.float32)
        X_tr = np.column_stack([X_base[tr], prior_tr, monsoon[tr]])
        X_te = np.column_stack([X_base[te], prior_te, monsoon[te]])

        sp = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        model = xgb.XGBClassifier(
            **XGBOOST_PARAMS, scale_pos_weight=sp,
            random_state=seed + int(blk),
        )
        model.fit(X_tr, y[tr])
        p = model.predict_proba(X_te)[:, 1]

        # Tier-2-only comparator on the same fold — isolates the
        # marginal value of the stacked prior + season flag.
        model_t2 = xgb.XGBClassifier(
            **XGBOOST_PARAMS, scale_pos_weight=sp,
            random_state=seed + int(blk),
        )
        model_t2.fit(X_base[tr], y[tr])
        p_t2 = model_t2.predict_proba(X_base[te])[:, 1]

        # Cross-fitted calibration diagnostics (same construction as
        # train_dynamic_escalation: inner 2-block split of the fold).
        brier_cal = float("nan")
        brier_platt = float("nan")
        inner = list(GroupKFold(n_splits=2).split(
            X_tr, y[tr], blocks[tr]))
        if inner:
            a_idx, b_idx = inner[0]
            tra, trb = tr[a_idx], tr[b_idx]
            if len(np.unique(y[tra])) == 2 and len(np.unique(y[trb])) == 2:
                sp_a = (y[tra] == 0).sum() / max((y[tra] == 1).sum(), 1)
                mdl_a = xgb.XGBClassifier(
                    **XGBOOST_PARAMS, scale_pos_weight=sp_a,
                    random_state=seed + int(blk),
                )
                mdl_a.fit(X_tr[a_idx], y[tra])
                p_trb = mdl_a.predict_proba(X_tr[b_idx])[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_trb, y[trb])
                brier_cal = float(
                    brier_score_loss(y[te], iso.predict(p)))
                platt = LogisticRegression().fit(
                    p_trb.reshape(-1, 1), y[trb])
                brier_platt = float(brier_score_loss(
                    y[te],
                    platt.predict_proba(p.reshape(-1, 1))[:, 1]))

        h_te = h_baseline[te]
        m = {
            "heldout_block": int(blk),
            "n_train": len(tr),
            "n_test": len(te),
            "n_breached_test": int(y[te].sum()),
            "test_date_range": [
                str(dates.iloc[te].min().date()),
                str(dates.iloc[te].max().date()),
            ],
            "roc_auc": float(roc_auc_score(y[te], p)),
            "roc_auc_tier2": float(roc_auc_score(y[te], p_t2)),
            "roc_auc_baseline": float(roc_auc_score(y[te], h_te)),
            "pr_auc": float(average_precision_score(y[te], p)),
            "brier_score": float(brier_score_loss(y[te], p)),
            "brier_score_baseline": float(brier_score_loss(y[te], h_te)),
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
            "Block %d: AUC=%.3f (tier2 %.3f, baseline %.3f) "
            "Brier=%.4f (baseline %.4f) — test %d, %d events",
            blk, m["roc_auc"], m["roc_auc_tier2"],
            m["roc_auc_baseline"], m["brier_score"],
            m["brier_score_baseline"], m["n_test"], m["n_breached_test"],
        )

    aucs = [m["roc_auc"] for m in fold_metrics]
    aucs_t2 = [m["roc_auc_tier2"] for m in fold_metrics]
    aucs_bl = [m["roc_auc_baseline"] for m in fold_metrics]
    briers = [m["brier_score"] for m in fold_metrics]
    briers_bl = [m["brier_score_baseline"] for m in fold_metrics]
    cals = [m["brier_score_calibrated"] for m in fold_metrics
            if m["brier_score_calibrated"] is not None]
    platts = [m["brier_score_platt"] for m in fold_metrics
              if m["brier_score_platt"] is not None]
    mean_brier = float(np.mean(briers)) if briers else float("nan")
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    mean_auc_bl = float(np.mean(aucs_bl)) if aucs_bl else float("nan")

    passes_brier = bool(not np.isnan(mean_brier) and mean_brier < BRIER_GATE)
    beats_baseline = bool(
        not np.isnan(mean_auc) and not np.isnan(mean_auc_bl)
        and mean_auc > mean_auc_bl
    )

    return {
        "protocol": (
            "spatio-temporal holdout: per block, train on all other "
            f"blocks, test on rows >= {temporal_cutoff.date()}; "
            "p_susceptibility is a fold-honest inner-block OOF stack "
            "(train) / fold-trained prior (test)"
        ),
        "temporal_cutoff": str(temporal_cutoff.date()),
        "features": features,
        "baseline": {
            "name": "deterministic five-factor hazard_score (PRD §9.5)",
            "proxies": {
                "trend_class": BASELINE_TREND_CLASS,
                "expansion_pct": 0.0,
                "rainfall_24h_mm": "max_daily_precip_mm",
                "rainfall_7d_mm": "precip_7d_mm",
                "mean_slope_deg": BASELINE_SLOPE_DEG,
                "change_in_drainage": BASELINE_IN_DRAINAGE,
            },
            "mean_roc_auc": mean_auc_bl,
            "mean_brier": float(np.mean(briers_bl)) if briers_bl else None,
            "note": (
                "H is a policy score, not a calibrated probability — "
                "the ROC-AUC leg is the meaningful comparison; the "
                "Brier leg is reported for completeness. trend, "
                "expansion and drainage are unmeasurable for "
                "historical windows and held at neutral constants."
            ),
        },
        "mean_roc_auc": mean_auc,
        "std_roc_auc": float(np.std(aucs)) if aucs else None,
        "mean_roc_auc_tier2": (
            float(np.mean(aucs_t2)) if aucs_t2 else None
        ),
        "mean_roc_auc_baseline": mean_auc_bl,
        "mean_brier": mean_brier if not np.isnan(mean_brier) else None,
        "std_brier": float(np.std(briers)) if briers else None,
        "mean_brier_calibrated": float(np.mean(cals)) if cals else None,
        "mean_brier_platt": float(np.mean(platts)) if platts else None,
        "mean_brier_baseline": (
            float(np.mean(briers_bl)) if briers_bl else None
        ),
        "mean_pr_auc": (
            float(np.mean([m["pr_auc"] for m in fold_metrics]))
            if fold_metrics else None
        ),
        "fold_metrics": fold_metrics,
        "feature_importances": {
            f: float(v) for f, v in
            zip(features, np.mean(importances, axis=0))
        } if importances else {},
        "passes_brier_gate": passes_brier,
        "beats_deterministic_baseline": beats_baseline,
        "gate_passed": passes_brier and beats_baseline,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Level-6 fused risk scorer — spatio-temporal eval"
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
                   help="persist booster + OOF calibration sidecar")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.input.exists():
        logger.error(
            "Dataset not found: %s — run "
            "python -m siren.ml.dataset_dynamic_escalation first", args.input,
        )
        return 1

    df = pd.read_parquet(args.input)
    df["event_date"] = pd.to_datetime(df["event_date"])
    features = list(FUSED_FEATURE_NAMES)
    logger.info(
        "Loaded %d samples (%d events / %d non-events), %d fused features",
        len(df), int(df["breached"].sum()),
        int((df["breached"] == 0).sum()), len(features),
    )

    cutoff = pd.Timestamp(args.cutoff)
    results = spatiotemporal_cv(df, features, cutoff, args.seed)

    report = {
        "status": "gate_evaluated",
        "evaluation_valid": True,
        "inference_allowed": False,
        "component": "learned_risk_fusion",
        "brier_gate": BRIER_GATE,
        "gate_metric": (
            "mean_brier (uncalibrated) < 0.15 AND mean_roc_auc > the "
            "deterministic five-factor baseline on identical folds"
        ),
        "n_samples": len(df),
        "n_breached": int(df["breached"].sum()),
        "n_stable": int((df["breached"] == 0).sum()),
        "dataset": str(args.input),
        **results,
    }

    print("\n" + "=" * 68)
    print("Learned risk fusion — spatio-temporal holdout")
    print("=" * 68)
    print(f"  Mean ROC-AUC: {results['mean_roc_auc']:.4f} "
          f"(tier2 {results['mean_roc_auc_tier2']:.4f}, "
          f"baseline {results['mean_roc_auc_baseline']:.4f})")
    print(f"  Mean Brier:   {results['mean_brier']:.4f} "
          f"(baseline {results['mean_brier_baseline']:.4f}, "
          f"platt {results['mean_brier_platt']})")
    print(f"  Brier gate (<{BRIER_GATE}): "
          f"{'PASS' if results['passes_brier_gate'] else 'FAIL'}")
    print(f"  Beats deterministic baseline: "
          f"{'YES' if results['beats_deterministic_baseline'] else 'NO'}")
    print(f"  GATE: {'PASS' if results['gate_passed'] else 'FAIL'}")
    for m in results["fold_metrics"]:
        print(f"    block {m['heldout_block']}: AUC={m['roc_auc']:.3f} "
              f"(base {m['roc_auc_baseline']:.3f}) "
              f"Brier={m['brier_score']:.4f} ({m['n_breached_test']}/"
              f"{m['n_test']} events {m['test_date_range'][0]}→"
              f"{m['test_date_range'][1]})")
    print("=" * 68)

    if not args.eval:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        logger.info("Report written to %s", args.output)

    if args.save_model:
        # Deployed artifact: prior feature for all-data training is the
        # block-OOF susceptibility stack (honest distribution); at
        # inference the runtime substitutes the deployed, calibrated
        # SusceptibilityScorer score — a documented shift.
        X_static = df[STATIC_FEATURES].values.astype(np.float32)
        prior_all = _oof_prior(
            X_static, df["breached"].values.astype(int),
            df["block"].values, args.seed,
        )
        monsoon = _monsoon_flag(df["event_date"])
        X = np.column_stack([
            df[FEATURE_NAMES].values.astype(np.float32),
            prior_all, monsoon,
        ])
        y = df["breached"].values.astype(int)
        groups = df["block"].values
        sp = (y == 0).sum() / max((y == 1).sum(), 1)

        import xgboost as xgb

        n_blocks = df["block"].nunique()
        iso = _isotonic_oof(X, y, groups, n_blocks, args.seed)
        platt_coef, platt_int = _platt_oof(X, y, groups, n_blocks, args.seed)
        model = xgb.XGBClassifier(**XGBOOST_PARAMS, scale_pos_weight=sp,
                                  random_state=args.seed)
        model.fit(X, y)

        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(args.model_output))
        args.model_output.with_suffix(".calibration.json").write_text(
            json.dumps({
                "method": "platt_crossfit_oof",
                "prefer": "platt",
                "platt": {
                    "coef": platt_coef,
                    "intercept": platt_int,
                    "apply": "p_cal = sigmoid(coef * p_raw + intercept)",
                },
                "isotonic": {
                    "x_thresholds": iso.X_thresholds_.tolist(),
                    "y_thresholds": iso.y_thresholds_.tolist(),
                    "apply": (
                        "p_cal = np.interp(p_raw, x_thresholds, "
                        "y_thresholds)"
                    ),
                },
                "base_rate": float(y.mean()),
            }, indent=2)
        )
        args.model_output.with_suffix(".meta.json").write_text(json.dumps({
            "status": "gate_evaluated",
            "evaluation_valid": True,
            # Advisory inference is allowed only when the gate passed —
            # a failed-gate checkpoint self-disqualifies at load time.
            "inference_allowed": bool(results["gate_passed"]),
            "component": "learned_risk_fusion",
            "features": features,
            "gate_report": str(args.output.name),
            "gate_passed": results["gate_passed"],
            "note": (
                "p_susceptibility feature was trained on block-OOF "
                "priors; runtime substitutes the deployed calibrated "
                "SusceptibilityScorer score (documented shift)."
            ),
        }, indent=2))
        logger.info("Booster + calibrator saved to %s", args.model_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
