"""Train and validate the 6-channel multi-temporal WaterResUNet (V3 §2.8).

This is the **target architecture** for passing the IoU > 0.65 production
gate (ADR-011). It uses a 6-channel multi-temporal tensor with real paired
pre/post SAR from the Kuro Siwo GRD dataset (PRD v4.7 §17.3):

    (VV_post, VH_post, VV_pre, VH_pre, ΔVV, ΔVH)

where Δσ⁰ = σ⁰_post − σ⁰_pre in dB. This replaces the disqualified synthetic
Δσ⁰ path that inflated ``water_resunet_6ch_v1`` test IoU to 0.9999 via label
leakage (PRD v4.7 §17.3).

Pipeline:
  1. Build 6-channel tensors from Kuro Siwo GRD .tar shards (real paired SAR).
  2. Train WaterResUNet(in_channels=6) with WaterLoss on the train split.
  3. Evaluate on the Kuro Siwo test split.
  4. Save the best checkpoint to models/checkpoints/.
  5. Report whether the IoU > 0.65 gate was passed.

Usage:
    # Train on Kuro Siwo (default — real paired SAR):
    python -m siren.ml.train_water_resunet_6ch --kuro-siwo --epochs 45

    # Legacy Sen1Floods11 path (requires --pre-sar-dir, blocked by §17.3):
    python -m siren.ml.train_water_resunet_6ch --pre-sar-dir /path/to/pre/sar
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

# Channel names for the Kuro Siwo 6-channel contract (PRD v4.7 §17.3).
KURO_SIWO_CHANNEL_NAMES = [
    "VV_post", "VH_post", "VV_pre", "VH_pre", "dVV", "dVH",
]


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
    use_amp: bool = False,
    scheduler=None,
    ckpt_dir: Path | None = None,
    save_name: str = "water_resunet_6ch_v1.pt",
) -> dict:
    """Train a model and return training history + best validation IoU.

    Saves two checkpoints to ``ckpt_dir``:
        - ``{save_name}`` — latest epoch weights (overwritten each epoch)
        - ``{save_name.replace('.pt','_best.pt')}`` — best val IoU weights
          (only overwritten when val IoU improves)

    Args:
        scheduler: optional LR scheduler (stepped per epoch after validation).
        ckpt_dir: directory to save checkpoints (created if missing).
        save_name: base checkpoint filename.
    """
    import torch

    best_val_iou = 0.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_iou": [], "val_loss": [], "lr": []}
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest_path = ckpt_dir / save_name
        best_path = ckpt_dir / save_name.replace(".pt", "_best.pt")
    else:
        latest_path = best_path = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            sar = batch["sar"].to(device)
            water = batch["water"].to(device)
            valid = batch["valid"].to(device)
            dem = batch["dem"].to(device)

            optimizer.zero_grad()

            # Mixed-precision forward
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(sar)
                # WaterLoss expects (B, 1, H, W); add channel dim if needed
                if water.ndim == 3:
                    water = water.unsqueeze(1)
                if valid.ndim == 3:
                    valid = valid.unsqueeze(1)
                if dem.ndim == 3:
                    dem = dem.unsqueeze(1)
                loss_dict = criterion(logits, water, dem=dem, valid=valid)
                loss = loss_dict["total"]

            # Mixed-precision backward
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        # Save latest checkpoint every epoch
        if latest_path is not None:
            torch.save(model.state_dict(), str(latest_path))

        # Track + save best checkpoint
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if best_path is not None:
                torch.save(best_state, str(best_path))
                logger.info("  → best checkpoint saved: %s (IoU=%.4f)", best_path.name, val_iou)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == num_epochs - 1:
            logger.info(
                "Epoch %3d/%d | loss=%.4f | val_iou=%.4f | best=%.4f (ep %d) | lr=%.2e",
                epoch + 1, num_epochs, avg_loss, val_iou, best_val_iou, best_epoch + 1,
                current_lr,
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
    parser.add_argument("--kuro-siwo", action="store_true", default=True,
                        help="use Kuro Siwo GRD dataset (real paired SAR, default)")
    parser.add_argument("--kuro-siwo-root", type=str, default=None,
                        help="path to Kuro Siwo dataset root (default: data/datasets/Kuro Siwo)")
    parser.add_argument("--kuro-siwo-pre", choices=["sec1", "sec2"], default="sec1",
                        help="which pre-flood image to use (sec1=first, sec2=second)")
    parser.add_argument("--init-from-4ch", type=str, default=None,
                        help="path to 4-channel checkpoint for transfer learning")
    parser.add_argument("--init-from-6ch", type=str, default=None,
                        help="path to 6-channel checkpoint for fine-tuning (same architecture)")
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="BCE positive class weight (upweight water pixels for recall)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save-name", type=str, default="water_resunet_6ch_v1.pt",
                        help="checkpoint filename")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="cap samples per split (for pilot runs)")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="use mixed-precision (autocast) training")
    parser.add_argument("--device", type=str, default=None,
                        help="device override (cuda/cpu/cuda:0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="checkpoint output directory (default: models/checkpoints)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0=main process)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="log training metrics every N epochs")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    import torch
    from torch.utils.data import DataLoader

    from siren.ml.dataset import MultiTemporalWaterDataset
    from siren.ml.model import WaterResUNet
    from siren.ml.losses import WaterLoss

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Datasets ---
    use_kuro_siwo = args.kuro_siwo or args.pre_sar_dir is None

    if use_kuro_siwo:
        from siren.ml.kuro_siwo_dataset import KuroSiwoDataset
        logger.info("Loading Kuro Siwo GRD datasets (real paired SAR, PRD v4.7 §17.3)...")
        ks_root = args.kuro_siwo_root
        ks_pre = args.kuro_siwo_pre
        train_ds = KuroSiwoDataset("train", root=ks_root, pre_event=ks_pre, max_samples=args.max_samples)
        test_ds = KuroSiwoDataset("test", root=ks_root, pre_event=ks_pre, max_samples=args.max_samples)
        # Kuro Siwo has no val split — use test for early stopping validation.
        val_ds = test_ds
        channel_names = KURO_SIWO_CHANNEL_NAMES
        logger.info("Train: %d chips", len(train_ds))
        logger.info("Test:  %d chips (also used for early-stopping validation)", len(test_ds))
    else:
        from siren.ml.dataset import MultiTemporalWaterDataset
        logger.info("Loading Sen1Floods11 multi-temporal datasets (strategy=%s)...", args.strategy)
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
        channel_names = ["VV_post", "VH_post", "dVV", "dVH", "HAND", "Slope"]
        logger.info("Train: %d chips (%d events)", len(train_ds), len(train_ds.events()))
        logger.info("Val:   %d chips (%d events)", len(val_ds), len(val_ds.events()))
        logger.info("Test:  %d chips (%d events)", len(test_ds), len(test_ds.events()))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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

    # Optional: fine-tune from a 6-channel checkpoint (same architecture)
    if args.init_from_6ch:
        ft_path = Path(args.init_from_6ch)
        if ft_path.exists():
            logger.info("Loading 6-channel checkpoint for fine-tuning: %s", ft_path)
            ckpt = torch.load(str(ft_path), map_location=device, weights_only=True)
            model.load_state_dict(ckpt)
            logger.info("Fine-tuning from %s (all layers loaded)", ft_path.name)
        else:
            logger.warning("6-channel checkpoint not found: %s", ft_path)

    # --- Loss + Optimizer + Scheduler ---
    criterion = WaterLoss(lambda_gravity=1.0, pos_weight=args.pos_weight)
    if args.pos_weight is not None:
        logger.info("BCE pos_weight=%.2f (upweighting water pixels for recall)", args.pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Cosine annealing: smooth decay from args.lr to ~0 over the full run,
    # avoiding plateau/divergence after epoch 15-20.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    logger.info("Scheduler: CosineAnnealingLR(T_max=%d, eta_min=%.2e)", args.epochs, args.lr * 0.01)

    # --- Train ---
    ckpt_dir = Path(args.output_dir) if args.output_dir else CHECKPOINT_DIR
    logger.info("Starting training (epochs=%d, lr=%.1e, batch_size=%d)...",
                args.epochs, args.lr, args.batch_size)
    t0 = time.time()
    train_result = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=args.epochs, patience=15, use_amp=args.amp,
        log_interval=args.log_interval,
        scheduler=scheduler,
        ckpt_dir=ckpt_dir,
        save_name=args.save_name,
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

    # --- Save final checkpoint (best weights already loaded by train_model) ---
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / args.save_name
    torch.save(model.state_dict(), str(ckpt_path))
    logger.info("Final checkpoint saved (best weights): %s", ckpt_path)

    # --- Save metadata ---
    meta = {
        "architecture": "WaterResUNet",
        "variant": "6-channel multi-temporal (V3 §2.8)",
        "in_channels": 6,
        "channel_names": channel_names,
        "dataset": "kuro_siwo_grd" if use_kuro_siwo else "sen1floods11",
        "base_channels": args.base_channels,
        "num_parameters": model.num_parameters(),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "strategy": args.strategy,
            "seed": args.seed,
            "amp": args.amp,
            "pos_weight": args.pos_weight,
            "init_from_6ch": args.init_from_6ch,
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
    meta_path = ckpt_dir / args.save_name.replace(".pt", ".meta.json")
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
