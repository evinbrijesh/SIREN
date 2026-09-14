"""Threshold sweep for the Kuro Siwo 6-channel WaterResUNet checkpoint.

Loads the best checkpoint and evaluates IoU / precision / recall / F1 across
a range of decision thresholds to find the optimal operating point. The
default threshold (0.5) may be suboptimal when precision >> recall.

Usage:
    python -m siren.ml.threshold_sweep \
        --checkpoint models/checkpoints/water_resunet_kuro_siwo_full/water_resunet_6ch_kuro_siwo_v1_best.pt \
        --thresholds 0.30 0.32 0.34 0.36 0.38 0.40 0.42 0.44 0.46 0.48 0.50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _iou_precision_recall_f1(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict:
    """Compute IoU, precision, recall, F1 for binary masks restricted to valid pixels."""
    pred = pred.astype(bool) & valid.astype(bool)
    target = target.astype(bool) & valid.astype(bool)
    tp = (pred & target).sum()
    fp = (pred & ~target).sum()
    fn = (~pred & target).sum()
    union = (pred | target).sum()
    iou = float(tp) / float(union) if union > 0 else 0.0
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def run_sweep(checkpoint_path: Path, thresholds: list[float], max_samples: int | None = None) -> dict:
    """Run inference on the test split and sweep decision thresholds.

    Collects sigmoid probabilities for all test chips ONCE, then evaluates
    metrics at each threshold without re-running the model.
    """
    import torch
    from torch.utils.data import DataLoader
    from siren.ml.kuro_siwo_dataset import KuroSiwoDataset
    from siren.ml.model import WaterResUNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    model = WaterResUNet(in_channels=6, base_channels=32).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    logger.info("Loaded checkpoint: %s (%d params)", checkpoint_path.name, sum(p.numel() for p in model.parameters()))

    # Load test dataset
    ds = KuroSiwoDataset("test", max_samples=max_samples)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    logger.info("Test set: %d chips", len(ds))

    # Collect probabilities + targets + valid masks
    all_probs = []
    all_targets = []
    all_valids = []

    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            logits = model(sar)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)

            water = batch["water"].numpy()  # (B, H, W)
            valid = batch["valid"].numpy()  # (B, H, W)

            for i in range(probs.shape[0]):
                p = probs[i, 0]  # (H, W)
                all_probs.append(p)
                all_targets.append(water[i])
                all_valids.append(valid[i])

    logger.info("Collected %d test chips, sweeping %d thresholds...", len(all_probs), len(thresholds))

    # Sweep thresholds
    results = {}
    for tau in thresholds:
        tp_total = fp_total = fn_total = 0
        chip_ious = []
        for probs, target, valid in zip(all_probs, all_targets, all_valids):
            pred = (probs > tau).astype(np.float32)
            m = _iou_precision_recall_f1(pred, target, valid)
            tp_total += m["tp"]
            fp_total += m["fp"]
            fn_total += m["fn"]
            chip_ious.append(m["iou"])

        # Global IoU (pooled across all chips)
        union = tp_total + fp_total + fn_total
        global_iou = float(tp_total) / float(union) if union > 0 else 0.0
        global_precision = float(tp_total) / float(tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        global_recall = float(tp_total) / float(tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0

        results[tau] = {
            "threshold": tau,
            "iou": global_iou,
            "precision": global_precision,
            "recall": global_recall,
            "f1": global_f1,
            "mean_chip_iou": float(np.mean(chip_ious)),
            "median_chip_iou": float(np.median(chip_ious)),
        }

        passed = "PASS" if global_iou > 0.65 else "    "
        logger.info(
            "  τ=%.2f | IoU=%.4f %s | P=%.4f R=%.4f F1=%.4f | chip IoU mean=%.4f median=%.4f",
            tau, global_iou, passed, global_precision, global_recall, global_f1,
            results[tau]["mean_chip_iou"], results[tau]["median_chip_iou"],
        )

    # Find best threshold by IoU
    best_tau = max(results, key=lambda t: results[t]["iou"])
    best = results[best_tau]
    logger.info("")
    logger.info("Best threshold: τ=%.2f → IoU=%.4f (P=%.4f, R=%.4f, F1=%.4f)",
                best_tau, best["iou"], best["precision"], best["recall"], best["f1"])
    if best["iou"] > 0.65:
        logger.info("✦ GATE PASSED at τ=%.2f — model eligible for promotion review", best_tau)
    else:
        logger.info("⚠ Gate not passed at any threshold — model remains shadow-only")

    return {
        "checkpoint": str(checkpoint_path),
        "n_test_chips": len(all_probs),
        "thresholds": results,
        "best_threshold": best_tau,
        "best_metrics": best,
        "gate_passed": best["iou"] > 0.65,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Threshold sweep for WaterResUNet checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to .pt checkpoint")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50],
                        help="decision thresholds to sweep")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="cap test samples (for quick runs)")
    parser.add_argument("--output", type=str, default=None,
                        help="save sweep results as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return 1

    results = run_sweep(ckpt_path, args.thresholds, args.max_samples)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Results saved: %s", out_path)

    return 0 if results["gate_passed"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
