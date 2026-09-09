"""Offline training script for the XGBoost breach susceptibility model (Level 1).

Trains the XGBoost classifier on the curated GLOF dataset with spatial
cross-validation, computes the true Brier score and ROC-AUC, and exports
the persistent artifact to models/checkpoints/xgboost_susceptibility_v1.json.

Usage:
    python -m siren.ml.train_susceptibility
    python -m siren.ml.train_susceptibility --output models/checkpoints/xgboost_susceptibility_v1.json
    python -m siren.ml.train_susceptibility --eval  # print evaluation metrics without saving

This script is run OFFLINE — it is NOT called at runtime. The pipeline
loads the saved checkpoint via SusceptibilityScorer.load_checkpoint() on
startup (see susceptibility.py).

Spatial cross-validation (V3 §3.2):
    Training and test sets are split by geographic region to prevent spatial
    leakage — lakes from the same region share similar geology, climate,
    and glacial history, so random k-fold would leak information.

Class imbalance (V3 §3.2):
    GLOF events are rare (~15 breached vs ~35 stable in our dataset).
    XGBoost's scale_pos_weight parameter handles this by weighting positive
    (breach) samples inversely to their frequency.

Evaluation metrics:
    - Brier score: calibration quality (target < 0.15, V3 §3.6)
    - ROC-AUC: discrimination ability
    - Per-fold metrics with spatial group breakdown
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from siren.ml.glof_dataset import (
    load_dataset,
    get_feature_matrix,
    get_labels,
    get_spatial_groups,
    dataset_summary,
    FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

# Default checkpoint path
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "checkpoints" / "xgboost_susceptibility_v1.json"
)

# XGBoost hyperparameters tuned for small tabular datasets with class imbalance
XGBOOST_PARAMS = dict(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
)


def train_susceptibility_model(
    output_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    save: bool = True,
) -> dict:
    """Train the XGBoost susceptibility model on the curated GLOF dataset.

    Args:
        output_path: path to save the XGBoost checkpoint.
        save: if True, save the checkpoint to disk.

    Returns:
        Dict with evaluation metrics:
            - brier_score: mean Brier score across CV folds
            - roc_auc: mean ROC-AUC across CV folds
            - fold_metrics: per-fold metrics
            - n_samples: total dataset size
            - n_breached: number of breached lakes
            - n_stable: number of stable lakes
    """
    import xgboost as xgb

    # Load dataset
    df = load_dataset()
    X = get_feature_matrix(df)
    y = get_labels(df)
    groups = get_spatial_groups(df)

    logger.info(dataset_summary(df))

    n_breached = int(y.sum())
    n_stable = int((y == 0).sum())
    scale_pos_weight = n_stable / max(n_breached, 1)

    logger.info("scale_pos_weight=%.2f (class imbalance handling)", scale_pos_weight)

    # Spatial cross-validation (GroupKFold by region)
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics = []
    all_brier = []
    all_auc = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Check that both classes are present in the training set
        if len(np.unique(y_train)) < 2:
            logger.warning("Fold %d: only one class in training set — skipping", fold_idx)
            continue

        # Train XGBoost with class imbalance handling
        model = xgb.XGBClassifier(
            **XGBOOST_PARAMS,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)

        # Evaluate
        p_test = model.predict_proba(X_test)[:, 1]

        # Brier score (calibration quality)
        if len(np.unique(y_test)) < 2:
            brier = float("nan")
            logger.warning("Fold %d: only one class in test set — Brier undefined", fold_idx)
        else:
            brier = float(brier_score_loss(y_test, p_test))
            all_brier.append(brier)

        # ROC-AUC (discrimination ability)
        if len(np.unique(y_test)) < 2:
            auc = float("nan")
            logger.warning("Fold %d: only one class in test set — AUC undefined", fold_idx)
        else:
            auc = float(roc_auc_score(y_test, p_test))
            all_auc.append(auc)

        test_regions = sorted(set(groups[test_idx]))
        fold_metrics.append({
            "fold": fold_idx,
            "brier_score": round(brier, 4) if not np.isnan(brier) else None,
            "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "test_regions": test_regions,
            "n_breached_test": int(y_test.sum()),
            "n_stable_test": int((y_test == 0).sum()),
        })

        logger.info(
            "Fold %d: Brier=%.4f, AUC=%.4f (test regions: %s)",
            fold_idx,
            brier if not np.isnan(brier) else -1,
            auc if not np.isnan(auc) else -1,
            ", ".join(test_regions),
        )

    # Train final model on ALL data for the persistent checkpoint
    final_model = xgb.XGBClassifier(
        **XGBOOST_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )
    final_model.fit(X, y)

    # Compute conformal calibration on the full dataset (in-sample —
    # the CV Brier score above is the honest estimate)
    from siren.risk.susceptibility import SusceptibilityScorer
    scorer = SusceptibilityScorer(random_state=42)
    scorer._model = final_model
    scorer._is_trained = True

    # Calibrate conformal interval on the full dataset
    p_full = final_model.predict_proba(X)[:, 1]
    scores = np.abs(y - p_full)
    n = len(scores)
    alpha = 0.05
    q_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
    q_idx = max(0, min(q_idx, n - 1))
    calibration_q = float(np.sort(scores)[q_idx])

    # In-sample Brier (optimistic — the CV Brier is the honest estimate)
    in_sample_brier = float(brier_score_loss(y, p_full))

    mean_brier = float(np.mean(all_brier)) if all_brier else float("nan")
    mean_auc = float(np.mean(all_auc)) if all_auc else float("nan")

    results = {
        "brier_score_cv": round(mean_brier, 4) if not np.isnan(mean_brier) else None,
        "brier_score_in_sample": round(in_sample_brier, 4),
        "roc_auc_cv": round(mean_auc, 4) if not np.isnan(mean_auc) else None,
        "fold_metrics": fold_metrics,
        "n_samples": len(df),
        "n_breached": n_breached,
        "n_stable": n_stable,
        "scale_pos_weight": round(scale_pos_weight, 2),
        "feature_names": FEATURE_NAMES,
        "xgboost_params": XGBOOST_PARAMS,
        "calibration_q": round(calibration_q, 4),
        "brier_gate": 0.15,
        "passes_brier_gate": (
            not np.isnan(mean_brier) and mean_brier < 0.15
        ) if not np.isnan(mean_brier) else False,
    }

    # Save checkpoint
    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save_model(str(output_path))

        # Save metadata sidecar
        meta_path = output_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(results, indent=2))

        logger.info("Checkpoint saved to %s", output_path)
        logger.info("Metadata saved to %s", meta_path)

    # Print summary
    print("\n" + "=" * 60)
    print("XGBoost Susceptibility Training Results")
    print("=" * 60)
    print(f"  Dataset: {len(df)} lakes ({n_breached} breached, {n_stable} stable)")
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    print(f"  CV folds: {n_splits} (spatial GroupKFold by region)")
    print()
    print(f"  Brier score (CV):     {mean_brier:.4f}" if not np.isnan(mean_brier) else "  Brier score (CV):     N/A")
    print(f"  Brier score (in-sample): {in_sample_brier:.4f}")
    print(f"  ROC-AUC (CV):        {mean_auc:.4f}" if not np.isnan(mean_auc) else "  ROC-AUC (CV):        N/A")
    print(f"  Brier gate (< 0.15):  {'PASS' if results['passes_brier_gate'] else 'FAIL'}")
    print()
    for fm in fold_metrics:
        brier_str = f"{fm['brier_score']:.4f}" if fm['brier_score'] is not None else "N/A"
        auc_str = f"{fm['roc_auc']:.4f}" if fm['roc_auc'] is not None else "N/A"
        print(f"  Fold {fm['fold']}: Brier={brier_str}, AUC={auc_str}, "
              f"test={fm['test_regions']}")
    print("=" * 60)

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the XGBoost breach susceptibility model (Level 1)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Output checkpoint path (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Print evaluation metrics without saving the checkpoint",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    train_susceptibility_model(
        output_path=args.output,
        save=not args.eval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
