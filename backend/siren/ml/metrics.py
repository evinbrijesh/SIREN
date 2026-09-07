"""Segmentation evaluation metrics for the SAR water segmenter.

Per docs/reference/DL_MODEL_AUDIT.md §4: report water/change IoU,
precision, recall, F1 -- not pixel accuracy, which is dominated by the
land-pixel majority class and can look deceptively high on an
all-land prediction.

All metrics are computed with a validity mask that excludes Sen1Floods11
"no data" pixels (label == -1) from both numerator and denominator.
"""

from __future__ import annotations

import numpy as np


def water_confusion_counts(
    pred: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> tuple[int, int, int, int]:
    """Compute TP/FP/FN/TN for the water class over valid pixels only.

    Args:
        pred: binary predicted water mask (0/1)
        target: binary ground-truth water mask (0/1)
        valid: binary validity mask (1 = evaluate this pixel)

    Returns:
        (tp, fp, fn, tn) as plain ints
    """
    pred = pred.astype(bool) & valid.astype(bool)
    target_valid = target.astype(bool) & valid.astype(bool)
    not_pred = (~pred.astype(bool)) & valid.astype(bool)
    not_target = (~target.astype(bool)) & valid.astype(bool)

    tp = int(np.sum(pred & target_valid))
    fp = int(np.sum(pred & not_target))
    fn = int(np.sum(not_pred & target_valid))
    tn = int(np.sum(not_pred & not_target))
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Derive IoU, precision, recall, F1 from confusion counts.

    Returns 0.0 for any ratio with a zero denominator (documented, not
    silently substituted with a misleading value).
    """
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class RunningConfusion:
    """Accumulates TP/FP/FN/TN across an entire dataset split.

    Segmentation IoU must be computed over the whole split's pixel
    counts, not averaged per-chip -- per-chip averaging over-weights
    small/empty-water chips and misrepresents the dataset-level metric.
    """

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0
        self.n_chips = 0

    def update(self, pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> None:
        tp, fp, fn, tn = water_confusion_counts(pred, target, valid)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.n_chips += 1

    def result(self) -> dict[str, float]:
        out = metrics_from_counts(self.tp, self.fp, self.fn, self.tn)
        out["n_chips"] = self.n_chips
        return out
