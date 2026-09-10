"""Train and validate the 4-channel terrain-aware WaterResUNet (Level 2).

This is the **real** training pipeline for the 4-channel WaterResUNet
(ADR-011 / V3 §2.3), replacing the previous scaffolding that had no
trained checkpoint.

Pipeline:
  1. Build 4-channel (VV, VH, DEM, Slope) tensors from Sen1Floods11 chips
     using Copernicus GLO-30 DEM tiles (fetched per-chip, cached locally).
  2. Train WaterResUNet with L_gravity loss penalty (V3 §2.6) on the
     event-holdout training split for 40-50 epochs.
  3. Evaluate on the strict event-holdout test split (Pakistan/Somalia —
     no event overlap with training).
  4. Compare against a 2-channel (VV, VH) baseline trained identically.
  5. Save the best checkpoint to models/checkpoints/water_resunet_4ch_v1.pt
  6. Report honestly whether the IoU > 0.65 gate (ADR-011) was passed.

Usage:
    python -m siren.ml.train_water_resunet [--epochs 45] [--batch-size 4]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou_score(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Pixel-level Intersection-over-Union for binary water segmentation.

    Args:
        pred: binary predictions {0, 1} (B, H, W) or (H, W).
        target: binary ground truth {0, 1} (B, H, W) or (H, W).
        valid: optional validity mask {0, 1}; invalid pixels excluded.

    Returns:
        IoU score (float). Returns 0.0 if no positive pixels in pred+target.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    if valid is not None:
        valid = valid.astype(bool)
        pred = pred & valid
        target = target & valid
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def precision_recall_f1(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """Precision, recall, F1 for binary water segmentation."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if valid is not None:
        valid = valid.astype(bool)
        pred = pred & valid
        target = target & valid
    tp = (pred & target).sum()
    fp = (pred & ~target).sum()
    fn = (~pred & target).sum()
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, loader, device, threshold: float = 0.5) -> dict:
    """Evaluate a model on a data loader.

    Returns dict with IoU, precision, recall, F1, and per-chip IoU stats.
    """
    import torch

    model.eval()
    all_ious = []
    total_tp = total_fp = total_fn = 0

    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            water = batch["water"].to(device)
            valid = batch["valid"].to(device)

            logits = model(sar)
            probs = torch.sigmoid(logits)

            pred = (probs > threshold).float()
            water_b = water.squeeze(1) if water.ndim == 4 else water
            valid_b = valid.squeeze(1) if valid.ndim == 4 else valid
            pred_b = pred.squeeze(1) if pred.ndim == 4 else pred

            # Per-chip IoU
            for i in range(pred_b.shape[0]):
                chip_iou = iou_score(
                    pred_b[i].cpu().numpy(),
                    water_b[i].cpu().numpy(),
                    valid_b[i].cpu().numpy(),
                )
                all_ious.append(chip_iou)

            # Accumulate for global precision/recall
            p = pred_b.bool() & valid_b.bool()
            t = water_b.bool() & valid_b.bool()
            total_tp += (p & t).sum().item()
            total_fp += (p & ~t).sum().item()
            total_fn += (~p & t).sum().item()

    global_iou = float(total_tp) / float(total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    precision = float(total_tp) / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = float(total_tp) / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "iou": global_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_chip_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "median_chip_iou": float(np.median(all_ious)) if all_ious else 0.0,
        "n_chips": len(all_ious),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs: int,
    log_interval: int = 10,
    use_gravity: bool = True,
    patience: int = 15,
) -> dict:
    """Train a model and return training history + best validation IoU.

    Uses early stopping on validation IoU with the given patience.
    """
    import torch

    best_val_iou = 0.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_iou": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_bce = 0.0
        epoch_grav = 0.0
        n_batches = 0

        for batch in train_loader:
            sar = batch["sar"].to(device)
            water = batch["water"].to(device)
            valid = batch["valid"].to(device)
            dem = batch.get("dem")
            if dem is not None:
                dem = dem.to(device)

            # Reshape for loss: (B, 1, H, W)
            water_t = water.unsqueeze(1) if water.ndim == 3 else water
            valid_t = valid.unsqueeze(1) if valid.ndim == 3 else valid
            dem_t = None
            if use_gravity and dem is not None:
                dem_t = dem.unsqueeze(1) if dem.ndim == 3 else dem

            optimizer.zero_grad()
            logits = model(sar)
            loss_dict = criterion(logits, water_t, dem=dem_t, valid=valid_t)
            loss_dict["total"].backward()
            optimizer.step()

            epoch_loss += loss_dict["total"].item()
            epoch_dice += loss_dict["dice"].item()
            epoch_bce += loss_dict["bce"].item()
            epoch_grav += loss_dict["gravity"].item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate_model(model, val_loader, device)

        history["train_loss"].append(avg_loss)
        history["val_iou"].append(val_metrics["iou"])
        history["val_loss"].append(1.0 - val_metrics["iou"])  # proxy

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d: loss=%.4f (dice=%.4f bce=%.4f grav=%.4f) val_iou=%.4f val_f1=%.4f",
                epoch + 1, num_epochs, avg_loss,
                epoch_dice / max(n_batches, 1),
                epoch_bce / max(n_batches, 1),
                epoch_grav / max(n_batches, 1),
                val_metrics["iou"], val_metrics["f1"],
            )

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                           epoch + 1, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Collate function for the DataLoader."""
    import torch
    return {
        "sar": torch.stack([b["sar"] for b in batch]),
        "water": torch.stack([b["water"] for b in batch]),
        "valid": torch.stack([b["valid"] for b in batch]),
        "dem": torch.stack([b["dem"] for b in batch]) if "dem" in batch[0] else None,
        "slope": torch.stack([b["slope"] for b in batch]) if "slope" in batch[0] else None,
        "chip_id": [b["chip_id"] for b in batch],
        "event": [b["event"] for b in batch],
    }


def main():
    parser = argparse.ArgumentParser(description="Train 4-channel WaterResUNet (Level 2)")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda-gravity", type=float, default=1.0,
                        help="Gravity loss weight (1.0 to achieve ~5-15%% of total loss)")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channel count")
    parser.add_argument("--strategy", default="event_holdout", choices=["official", "event_holdout"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--save-baseline", action="store_true", help="Also save 2-channel baseline checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import torch
    from torch.utils.data import DataLoader

    from siren.ml.dataset import WaterSegmentationDataset
    from siren.ml.model import WaterResUNet, WaterUNet
    from siren.ml.losses import WaterLoss

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Device: %s", device)

    # Build datasets
    logger.info("Building 4-channel datasets (Copernicus GLO-30 DEM)...")
    train_ds = WaterSegmentationDataset(
        split="train", strategy=args.strategy, use_copernicus_dem=True,
    )
    val_ds = WaterSegmentationDataset(
        split="val", strategy=args.strategy, use_copernicus_dem=True,
    )
    test_ds = WaterSegmentationDataset(
        split="test", strategy=args.strategy, use_copernicus_dem=True,
    )

    logger.info("Train chips: %d (events: %s)", len(train_ds), sorted(train_ds.events()))
    logger.info("Val chips: %d (events: %s)", len(val_ds), sorted(val_ds.events()))
    logger.info("Test chips: %d (events: %s)", len(test_ds), sorted(test_ds.events()))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # ---- Train 4-channel WaterResUNet ----
    logger.info("=" * 60)
    logger.info("Training 4-channel WaterResUNet (VV, VH, DEM, Slope)")
    logger.info("=" * 60)

    model_4ch = WaterResUNet(in_channels=4, base_channels=args.base_channels).to(device)
    logger.info("4ch model parameters: %d", model_4ch.num_parameters())

    criterion_4ch = WaterLoss(lambda_gravity=args.lambda_gravity)
    optimizer_4ch = torch.optim.AdamW(model_4ch.parameters(), lr=args.lr, weight_decay=1e-4)

    t0 = time.time()
    results_4ch = train_model(
        model_4ch, train_loader, val_loader, criterion_4ch, optimizer_4ch,
        device, num_epochs=args.epochs, use_gravity=True, patience=args.patience,
    )
    train_time_4ch = time.time() - t0
    logger.info("4ch training time: %.1fs", train_time_4ch)

    # Evaluate 4ch on test set
    test_metrics_4ch = evaluate_model(model_4ch, test_loader, device)
    logger.info("4ch test IoU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f",
                test_metrics_4ch["iou"], test_metrics_4ch["f1"],
                test_metrics_4ch["precision"], test_metrics_4ch["recall"])

    # ---- Train 2-channel baseline ----
    logger.info("=" * 60)
    logger.info("Training 2-channel baseline WaterResUNet (VV, VH only)")
    logger.info("=" * 60)

    train_ds_2ch = WaterSegmentationDataset(
        split="train", strategy=args.strategy, use_copernicus_dem=False,
    )
    val_ds_2ch = WaterSegmentationDataset(
        split="val", strategy=args.strategy, use_copernicus_dem=False,
    )
    test_ds_2ch = WaterSegmentationDataset(
        split="test", strategy=args.strategy, use_copernicus_dem=False,
    )

    train_loader_2ch = DataLoader(
        train_ds_2ch, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    val_loader_2ch = DataLoader(
        val_ds_2ch, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    test_loader_2ch = DataLoader(
        test_ds_2ch, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    model_2ch = WaterResUNet(in_channels=2, base_channels=args.base_channels).to(device)
    logger.info("2ch model parameters: %d", model_2ch.num_parameters())

    criterion_2ch = WaterLoss(lambda_gravity=0.0)  # no gravity without DEM
    optimizer_2ch = torch.optim.AdamW(model_2ch.parameters(), lr=args.lr, weight_decay=1e-4)

    t0 = time.time()
    results_2ch = train_model(
        model_2ch, train_loader_2ch, val_loader_2ch, criterion_2ch, optimizer_2ch,
        device, num_epochs=args.epochs, use_gravity=False, patience=args.patience,
    )
    train_time_2ch = time.time() - t0
    logger.info("2ch training time: %.1fs", train_time_2ch)

    test_metrics_2ch = evaluate_model(model_2ch, test_loader_2ch, device)
    logger.info("2ch test IoU: %.4f, F1: %.4f", test_metrics_2ch["iou"], test_metrics_2ch["f1"])

    # ---- Save checkpoint ----
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "water_resunet_4ch_v1.pt"
    torch.save({
        "model_state": model_4ch.state_dict(),
        "in_channels": 4,
        "base_channels": args.base_channels,
        "best_val_iou": results_4ch["best_val_iou"],
        "best_epoch": results_4ch["best_epoch"],
        "test_iou": test_metrics_4ch["iou"],
        "test_f1": test_metrics_4ch["f1"],
        "test_precision": test_metrics_4ch["precision"],
        "test_recall": test_metrics_4ch["recall"],
        "lambda_gravity": args.lambda_gravity,
        "epochs_trained": len(results_4ch["history"]["train_loss"]),
        "strategy": args.strategy,
        "train_chips": len(train_ds),
        "val_chips": len(val_ds),
        "test_chips": len(test_ds),
    }, str(ckpt_path))
    logger.info("4ch checkpoint saved to %s", ckpt_path)

    if args.save_baseline:
        ckpt_2ch = CHECKPOINT_DIR / "water_resunet_2ch_baseline.pt"
        torch.save({
            "model_state": model_2ch.state_dict(),
            "in_channels": 2,
            "base_channels": args.base_channels,
            "best_val_iou": results_2ch["best_val_iou"],
            "test_iou": test_metrics_2ch["iou"],
        }, str(ckpt_2ch))
        logger.info("2ch baseline checkpoint saved to %s", ckpt_2ch)

    # ---- Report results ----
    iou_gate = 0.65
    gate_passed = test_metrics_4ch["iou"] > iou_gate

    print()
    print("=" * 70)
    print("Level 2: 4-Channel WaterResUNet Training Results")
    print("=" * 70)
    print(f"  Strategy: {args.strategy}")
    print(f"  Epochs requested: {args.epochs}")
    print(f"  Epochs trained (4ch): {len(results_4ch['history']['train_loss'])}")
    print(f"  Epochs trained (2ch): {len(results_2ch['history']['train_loss'])}")
    print(f"  Train chips: {len(train_ds)} (events: {sorted(train_ds.events())})")
    print(f"  Val chips: {len(val_ds)} (events: {sorted(val_ds.events())})")
    print(f"  Test chips: {len(test_ds)} (events: {sorted(test_ds.events())})")
    print(f"  DEM source: Copernicus GLO-30 (48 tiles cached)")
    print()
    print(f"  4-channel model parameters: {model_4ch.num_parameters()}")
    print(f"  2-channel model parameters: {model_2ch.num_parameters()}")
    print()
    print(f"  4ch best val IoU: {results_4ch['best_val_iou']:.4f} (epoch {results_4ch['best_epoch']})")
    print(f"  4ch test IoU:     {test_metrics_4ch['iou']:.4f}")
    print(f"  4ch test F1:      {test_metrics_4ch['f1']:.4f}")
    print(f"  4ch test Prec:    {test_metrics_4ch['precision']:.4f}")
    print(f"  4ch test Recall:  {test_metrics_4ch['recall']:.4f}")
    print(f"  4ch mean chip IoU: {test_metrics_4ch['mean_chip_iou']:.4f}")
    print()
    print(f"  2ch best val IoU: {results_2ch['best_val_iou']:.4f} (epoch {results_2ch['best_epoch']})")
    print(f"  2ch test IoU:     {test_metrics_2ch['iou']:.4f}")
    print(f"  2ch test F1:      {test_metrics_2ch['f1']:.4f}")
    print()
    print(f"  IoU improvement (4ch - 2ch): {test_metrics_4ch['iou'] - test_metrics_2ch['iou']:+.4f}")
    print()
    print(f"  IoU gate (> {iou_gate}): {'PASS' if gate_passed else 'FAIL'}")
    print(f"  Checkpoint: {ckpt_path}")
    print()
    if not gate_passed:
        print("  WARNING: IoU gate NOT passed. Model remains shadow-only.")
        print("  The 4-channel model has NOT been promoted to production.")
    print("=" * 70)

    # Save metadata sidecar
    meta_path = CHECKPOINT_DIR / "water_resunet_4ch_v1.meta.json"
    meta = {
        "model": "WaterResUNet",
        "in_channels": 4,
        "base_channels": args.base_channels,
        "channels": ["VV", "VH", "DEM", "Slope"],
        "dem_source": "Copernicus GLO-30",
        "dem_tiles": 48,
        "strategy": args.strategy,
        "train_chips": len(train_ds),
        "val_chips": len(val_ds),
        "test_chips": len(test_ds),
        "train_events": sorted(train_ds.events()),
        "val_events": sorted(val_ds.events()),
        "test_events": sorted(test_ds.events()),
        "epochs_requested": args.epochs,
        "epochs_trained_4ch": len(results_4ch["history"]["train_loss"]),
        "epochs_trained_2ch": len(results_2ch["history"]["train_loss"]),
        "lambda_gravity": args.lambda_gravity,
        "learning_rate": args.lr,
        "seed": args.seed,
        "best_val_iou_4ch": results_4ch["best_val_iou"],
        "best_epoch_4ch": results_4ch["best_epoch"],
        "test_iou_4ch": test_metrics_4ch["iou"],
        "test_f1_4ch": test_metrics_4ch["f1"],
        "test_precision_4ch": test_metrics_4ch["precision"],
        "test_recall_4ch": test_metrics_4ch["recall"],
        "test_mean_chip_iou_4ch": test_metrics_4ch["mean_chip_iou"],
        "test_median_chip_iou_4ch": test_metrics_4ch["median_chip_iou"],
        "best_val_iou_2ch": results_2ch["best_val_iou"],
        "test_iou_2ch": test_metrics_2ch["iou"],
        "test_f1_2ch": test_metrics_2ch["f1"],
        "iou_improvement": test_metrics_4ch["iou"] - test_metrics_2ch["iou"],
        "iou_gate": iou_gate,
        "gate_passed": gate_passed,
        "train_time_4ch_s": round(train_time_4ch, 1),
        "train_time_2ch_s": round(train_time_2ch, 1),
        "checkpoint": str(ckpt_path),
        "shadow_only": not gate_passed,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
