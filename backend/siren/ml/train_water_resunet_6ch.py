"""Train and validate the 6-channel multi-temporal WaterResUNet (V3 §2.8).

This is the **target architecture** for passing the IoU > 0.65 production
gate (ADR-011). It replaces the static 4-channel (VV, VH, DEM, Slope)
tensor with a 6-channel multi-temporal tensor:

    (VV_post, VH_post, ΔVV, ΔVH, HAND, Slope)

Key architectural changes from the 4-channel model:
  1. **Temporal differencing (Δσ⁰):** eliminates dry-soil false positives
     that dragged the 4-channel test IoU to 0.28 on Pakistan/Somalia.
     Flood water shows Δσ⁰ ≤ -6 dB; dry soil shows Δσ⁰ ≈ 0 dB.
  2. **HAND replaces raw DEM:** prevents the network from memorising
     absolute elevation (Mekong Delta at 5 m vs Himalaya at 4,500 m).
     HAND is scale-invariant — a floodplain at HAND ∈ [0, 3] m is the
     same whether at sea level or high altitude.
  3. **Pretrained encoder init:** optionally initialise the encoder from
     the 4-channel checkpoint (transfer learning) or from scratch.

Pipeline:
  1. Build 6-channel tensors from Sen1Floods11 chips with synthetic Δσ⁰
     (training augmentation) + Copernicus GLO-30 DEM → HAND.
  2. Train WaterResUNet(in_channels=6) with L_gravity loss on the
     event-holdout training split for 40-50 epochs.
  3. Evaluate on the strict event-holdout test split.
  4. Save the best checkpoint to models/checkpoints/water_resunet_6ch_v1.pt
  5. Report whether the IoU > 0.65 gate was passed.

Usage:
    python -m siren.ml.train_water_resunet_6ch [--epochs 45] [--batch-size 4]
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

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"


# ---------------------------------------------------------------------------
# Metrics (shared with 4-channel trainer)
# ---------------------------------------------------------------------------

def iou_score(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Pixel-level IoU for binary water segmentation."""
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


def evaluate_model(model, loader, device, threshold: float = 0.5) -> dict:
    """Evaluate a model on a data loader. Returns IoU, precision, recall, F1."""
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

            for i in range(pred_b.shape[0]):
                chip_iou = iou_score(
                    pred_b[i].cpu().numpy(),
                    water_b[i].cpu().numpy(),
                    valid_b[i].cpu().numpy(),
                )
                all_ious.append(chip_iou)

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
    patience: int = 15,
) -> dict:
    """Train a model and return training history + best validation IoU."""
    import torch

    best_val_iou = 0.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_iou": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            sar = batch["sar"].to(device)
            water = batch["water"].to(device)
            valid = batch["valid"].to(device)

            optimizer.zero_grad()
            logits = model(sar)

            water_b = water.squeeze(1) if water.ndim == 4 else water
            valid_b = valid.squeeze(1) if valid.ndim == 4 else valid

            loss = criterion(logits.squeeze(1), water_b, valid_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        val_iou = val_metrics["iou"]
        history["val_iou"].append(val_iou)
        history["val_loss"].append(1.0 - val_iou)  # proxy

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            logger.info(
                "Epoch %3d/%d | loss=%.4f | val_iou=%.4f | best=%.4f (ep %d)",
                epoch + 1, num_epochs, avg_loss, val_iou, best_val_iou, best_epoch + 1,
            )

        if epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch + 1, patience)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_val_iou": best_val_iou, "best_epoch": best_epoch}


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train 6-channel multi-temporal WaterResUNet (V3 §2.8)",
    )
    parser.add_argument("--epochs", type=int, default=45, help="max training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--base-channels", type=int, default=32, help="base channel width")
    parser.add_argument("--strategy", choices=["official", "event_holdout"],
                        default="event_holdout", help="split strategy")
    parser.add_argument("--dem-path", type=str, default=None,
                        help="path to a single DEM raster (optional)")
    parser.add_argument("--use-copernicus-dem", action="store_true",
                        help="fetch Copernicus GLO-30 DEM per chip")
    parser.add_argument("--pre-sar-dir", type=str, default=None,
                        help="directory of pre-event SAR scenes (for real Δσ⁰)")
    parser.add_argument("--init-from-4ch", type=str, default=None,
                        help="path to 4-channel checkpoint for transfer learning")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save-name", type=str, default="water_resunet_6ch_v1.pt",
                        help="checkpoint filename")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    import torch
    from torch.utils.data import DataLoader

    from siren.ml.dataset import MultiTemporalWaterDataset
    from siren.ml.model import WaterResUNet
    from siren.ml.losses import GravityLoss

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Datasets ---
    logger.info("Loading 6-channel multi-temporal datasets (strategy=%s)...", args.strategy)
    common_kwargs = {
        "strategy": args.strategy,
        "dem_path": args.dem_path,
        "use_copernicus_dem": args.use_copernicus_dem,
        "pre_sar_dir": args.pre_sar_dir,
        "seed": args.seed,
    }
    train_ds = MultiTemporalWaterDataset("train", **common_kwargs)
    val_ds = MultiTemporalWaterDataset("val", **common_kwargs)
    test_ds = MultiTemporalWaterDataset("test", **common_kwargs)

    logger.info("Train: %d chips (%d events)", len(train_ds), len(train_ds.events()))
    logger.info("Val:   %d chips (%d events)", len(val_ds), len(val_ds.events()))
    logger.info("Test:  %d chips (%d events)", len(test_ds), len(test_ds.events()))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Model ---
    model = WaterResUNet(in_channels=6, base_channels=args.base_channels).to(device)
    logger.info("Model: WaterResUNet(in_channels=6, base_channels=%d)", args.base_channels)
    logger.info("Parameters: %d", model.num_parameters())

    # Optional: transfer learning from 4-channel checkpoint
    if args.init_from_4ch:
        ckpt_path = Path(args.init_from_4ch)
        if ckpt_path.exists():
            logger.info("Loading 4-channel checkpoint for transfer: %s", ckpt_path)
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
            # The first conv layer (enc1) expects 4 channels; we need to adapt to 6.
            # Strategy: copy enc1 weights for channels 0-1 (VV, VH), randomly init
            # channels 2-5 (ΔVV, ΔVH, HAND, Slope).
            if "enc1.conv1.weight" in ckpt:
                old_w = ckpt["enc1.conv1.weight"]  # (c_out, 4, 3, 3)
                new_w = model.enc1.conv1.weight.data  # (c_out, 6, 3, 3)
                new_w[:, :2] = old_w[:, :2]  # copy VV, VH
                # ΔVV, ΔVH, HAND, Slope are randomly initialised
                logger.info("Transferred VV/VH encoder weights; initialised Δσ⁰ + HAND + Slope")
            # Copy all other layers
            for k, v in ckpt.items():
                if k != "enc1.conv1.weight" and k in model.state_dict():
                    model.state_dict()[k].copy_(v)
            logger.info("Transfer learning complete")
        else:
            logger.warning("4-channel checkpoint not found: %s", ckpt_path)

    # --- Loss + Optimizer ---
    criterion = GravityLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # --- Train ---
    logger.info("Starting training (epochs=%d, lr=%.1e, batch_size=%d)...",
                args.epochs, args.lr, args.batch_size)
    t0 = time.time()
    train_result = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=args.epochs, patience=15,
    )
    train_time = time.time() - t0
    logger.info("Training complete in %.1f minutes", train_time / 60)

    # --- Evaluate on test set ---
    logger.info("Evaluating on event-holdout test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    logger.info("Test IoU: %.4f (gate: > 0.65)", test_metrics["iou"])
    logger.info("Test precision: %.4f, recall: %.4f, F1: %.4f",
                test_metrics["precision"], test_metrics["recall"], test_metrics["f1"])
    logger.info("Mean chip IoU: %.4f, median: %.4f",
                test_metrics["mean_chip_iou"], test_metrics["median_chip_iou"])

    # --- Save checkpoint ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / args.save_name
    torch.save(model.state_dict(), str(ckpt_path))
    logger.info("Checkpoint saved: %s", ckpt_path)

    # --- Save metadata ---
    meta = {
        "architecture": "WaterResUNet",
        "variant": "6-channel multi-temporal (V3 §2.8)",
        "in_channels": 6,
        "channel_names": ["VV_post", "VH_post", "dVV", "dVH", "HAND", "Slope"],
        "base_channels": args.base_channels,
        "num_parameters": model.num_parameters(),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "strategy": args.strategy,
            "seed": args.seed,
            "best_val_iou": train_result["best_val_iou"],
            "best_epoch": train_result["best_epoch"],
            "train_time_seconds": train_time,
        },
        "test_metrics": test_metrics,
        "gate": {
            "target_iou": 0.65,
            "passed": test_metrics["iou"] > 0.65,
        },
        "checkpoint_path": str(ckpt_path),
        "checkpoint_size_mb": ckpt_path.stat().st_size / 1e6,
    }
    meta_path = CHECKPOINT_DIR / args.save_name.replace(".pt", ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Metadata saved: %s", meta_path)

    # --- Report ---
    print(f"\n{'=' * 60}")
    print(f"6-Channel Multi-Temporal WaterResUNet — Training Complete")
    print(f"{'=' * 60}")
    print(f"  Test IoU:     {test_metrics['iou']:.4f}  (gate: > 0.65)")
    print(f"  Precision:    {test_metrics['precision']:.4f}")
    print(f"  Recall:       {test_metrics['recall']:.4f}")
    print(f"  F1:           {test_metrics['f1']:.4f}")
    print(f"  Mean chip IoU: {test_metrics['mean_chip_iou']:.4f}")
    print(f"  Best val IoU: {train_result['best_val_iou']:.4f} (epoch {train_result['best_epoch'] + 1})")
    print(f"  Parameters:   {model.num_parameters():,}")
    print(f"  Train time:   {train_time / 60:.1f} min")
    print(f"  Checkpoint:   {ckpt_path} ({meta['checkpoint_size_mb']:.1f} MB)")
    if test_metrics["iou"] > 0.65:
        print(f"\n  ✦ GATE PASSED — IoU > 0.65 — model eligible for load-bearing role")
    else:
        print(f"\n  ⚠ GATE NOT PASSED — IoU ≤ 0.65 — model remains shadow-only")
    print(f"{'=' * 60}")

    return 0 if test_metrics["iou"] > 0.65 else 1


if __name__ == "__main__":
    sys.exit(main())
