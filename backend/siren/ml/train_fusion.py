"""Train and evaluate the multi-modal SAR+optical fusion model (ADR-013 §9.7.2).

The MultiModalFusionNet fuses 6-channel SAR (all-weather) with 3-channel
optical (NDWI, MNDWI, cloud_mask) via cloud-gated cross-attention. When
optical is unavailable (cloud-blocked or no S2 coverage), the model falls
back to SAR-only.

Gate (ADR-013 §9.7.2): event-held-out IoU > 0.75 AND precision ≥ 0.85 on
real paired SAR+optical data. Until this gate passes, the SAR-only 6-channel
model (ADR-011.1, IoU 0.62) remains the primary segmenter.

Two modes:

1. **Training** (``--train``): trains on Kuro Siwo 6-channel SAR. Optical
   is None for all training chips (no paired S2 for Kuro Siwo chips yet) —
   this exercises the SAR-only fallback path of the fusion model. When
   paired S2 optical chips become available, pass ``--s2-archives`` to
   attach optical features.

2. **Inference** (``--inference``): runs the fusion model on the real Imja
   pair (S1 07-02/07-14 + S2 07-05), producing a water mask with the
   cloud-gated cross-attention fusion. Reports the cloud fraction and
   whether optical was used.

Usage:
    # Train (SAR-only fallback path):
    python -m siren.ml.train_fusion --train --epochs 45

    # Inference on the real Imja pair:
    python -m siren.ml.train_fusion --inference

    # Full pipeline: train then evaluate on real pair:
    python -m siren.ml.train_fusion --train --epochs 45 --inference
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
FUSION_CKPT_NAME = "multimodal_fusion_v1.pt"

# Real Imja pair paths (ADR-013 §9.7.2)
PRE_SAFE = REPO_ROOT / "data/raw/S1D_IW_GRDH_1SDV_20260702T001034_20260702T001059_003487_0062AB_5083.SAFE.zip"
POST_SAFE = REPO_ROOT / "data/raw/S1D_IW_GRDH_1SDV_20260714T001035_20260714T001100_003662_00689F_377F.SAFE.zip"
S2_MONSOON = REPO_ROOT / "data/raw/S2B_MSIL2A_20260705T044659_N0512_R076_T45RVL_20260705T083506.zip"
DEM_PATH = REPO_ROOT / "data/raw/srtm_30m.tif"

# Imja Tsho window (matches imja_integration_test.py)
IMJA_LAT = 27.90
IMJA_LON = 86.92
HALF_SIZE = 0.020  # ~2.2 km half-width

CHIP_SIZE = 224
THRESHOLD = 0.30  # ADR-011.1 calibrated threshold

# Gate criteria (ADR-013 §9.7.2)
GATE_IOU = 0.75
GATE_PRECISION = 0.85


# ---------------------------------------------------------------------------
# Metrics (shared with train_water_resunet_6ch.py)
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
    """Evaluate the fusion model on a data loader."""
    import torch

    model.eval()
    all_ious = []
    total_tp = total_fp = total_fn = 0

    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            water = batch["water"].to(device)
            valid = batch["valid"].to(device)
            optical = batch.get("optical")
            if optical is not None:
                optical = optical.to(device)

            logits = model(sar, optical)
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
    save_name: str = FUSION_CKPT_NAME,
) -> dict:
    """Train the fusion model. Returns training history + best validation IoU."""
    import torch

    best_val_iou = 0.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_iou": [], "val_loss": [], "lr": []}
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            sar = batch["sar"].to(device)
            water = batch["water"].to(device)
            valid = batch["valid"].to(device)
            optical = batch.get("optical")
            if optical is not None:
                optical = optical.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(sar, optical)
                water_b = water.squeeze(1) if water.ndim == 4 else water
                valid_b = valid.squeeze(1) if valid.ndim == 4 else valid
                loss = criterion(logits, water_b.unsqueeze(1), valid_b.unsqueeze(1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    "Epoch %d/%d batch %d — loss %.4f",
                    epoch + 1, num_epochs, batch_idx + 1, loss.item(),
                )

        avg_loss = running_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device, threshold=THRESHOLD)
        val_iou = val_metrics["iou"]
        val_loss = 0.0  # computed in eval if needed
        history["val_iou"].append(val_iou)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_iou=%.4f val_precision=%.4f",
            epoch + 1, num_epochs, avg_loss, val_iou, val_metrics["precision"],
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            if ckpt_dir is not None:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                best_path = ckpt_dir / save_name.replace(".pt", "_best.pt")
                torch.save(best_state, str(best_path))
                logger.info("  → saved best checkpoint to %s", best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch + 1, patience)
                break

        if scheduler is not None:
            scheduler.step()

    # Save final checkpoint
    if ckpt_dir is not None and best_state is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_path = ckpt_dir / save_name
        torch.save(best_state, str(final_path))
        logger.info("Saved final checkpoint to %s", final_path)

    return {
        "history": history,
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
    }


# ---------------------------------------------------------------------------
# Inference on real Imja pair
# ---------------------------------------------------------------------------

def run_inference(
    ckpt_path: Path | None = None,
    device_str: str | None = None,
) -> dict:
    """Run the fusion model on the real Imja SAR+optical pair.

    Builds a (sar_6ch, optical_3ch) pair from:
        - S1 pre:  2026-07-02 (descending)
        - S1 post: 2026-07-14 (descending)
        - S2:      2026-07-05 (monsoon, ~56% cloud)

    The S2 scene is within ±3 days of the pre-event SAR (07-02), satisfying
    the ADR-013 §9.7.2 temporal pairing requirement.

    Args:
        ckpt_path: path to a trained fusion checkpoint, or None to use
            random weights (tensor-flow verification only).
        device_str: "cuda" or "cpu" (auto-detected if None).

    Returns:
        Dict with water_mask, cloud_fraction, optical_used, model_info.
    """
    import torch
    from siren.ml.fusion import MultiModalFusionNet
    from siren.ml.fusion_dataset import build_fusion_chip

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    # --- Load model ---
    model = MultiModalFusionNet(
        sar_channels=6, optical_channels=3, base_channels=32, n_heads=4, dropout=0.0,
    ).to(device)

    if ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        logger.info("Loaded checkpoint: %s (%d params)", ckpt_path, sum(p.numel() for p in model.parameters()))
    else:
        logger.warning("No checkpoint — using random weights (tensor-flow verification only)")

    model.eval()

    # --- Build fusion chip from real SAFE archives ---
    window_bounds = (
        IMJA_LON - HALF_SIZE, IMJA_LAT - HALF_SIZE,
        IMJA_LON + HALF_SIZE, IMJA_LAT + HALF_SIZE,
    )

    logger.info("Building SAR 6-channel from S1 %s + S1 %s...", PRE_SAFE.stem[:20], POST_SAFE.stem[:20])
    s2_path = S2_MONSOON if S2_MONSOON.exists() else None
    if s2_path is not None:
        logger.info("Attaching S2 optical: %s", s2_path.name)
    else:
        logger.warning("No S2 scene available — running SAR-only")

    sar_6ch, optical_3ch, meta = build_fusion_chip(
        pre_safe=PRE_SAFE,
        post_safe=POST_SAFE,
        s2_safe=s2_path,
        dem_path=DEM_PATH,
        window_bounds=window_bounds,
        chip_size=CHIP_SIZE,
    )

    logger.info("SAR chip shape: %s, dtype: %s", sar_6ch.shape, sar_6ch.dtype)
    if optical_3ch is not None:
        cloud_frac = float(optical_3ch[2].mean())
        logger.info("Optical chip shape: %s, cloud fraction: %.1f%%", optical_3ch.shape, cloud_frac * 100)
    else:
        cloud_frac = None
        logger.info("Optical: None (SAR-only fallback)")

    # --- Run inference ---
    sar_tensor = torch.from_numpy(sar_6ch).unsqueeze(0).to(device)
    optical_tensor = None
    if optical_3ch is not None:
        optical_tensor = torch.from_numpy(optical_3ch).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(sar_tensor, optical_tensor)
        probs = torch.sigmoid(logits)

    water_mask = (probs > THRESHOLD).float().squeeze().cpu().numpy()
    water_frac = float(water_mask.mean())

    logger.info("Inference complete — water fraction: %.1f%%", water_frac * 100)
    logger.info("  Prob range: [%.4f, %.4f]", float(probs.min()), float(probs.max()))

    return {
        "water_mask": water_mask,
        "prob_map": probs.squeeze().cpu().numpy(),
        "cloud_fraction": cloud_frac,
        "optical_used": optical_3ch is not None,
        "threshold": THRESHOLD,
        "model_params": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(ckpt_path) if ckpt_path and ckpt_path.exists() else None,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def evaluate_gate(metrics: dict) -> dict:
    """Check whether the fusion model passes the ADR-013 §9.7.2 gate.

    Gate: event-held-out IoU > 0.75 AND precision ≥ 0.85.
    """
    iou = metrics.get("iou", 0.0)
    precision = metrics.get("precision", 0.0)
    passed = iou > GATE_IOU and precision >= GATE_PRECISION
    return {
        "gate_iou_threshold": GATE_IOU,
        "gate_precision_threshold": GATE_PRECISION,
        "iou": iou,
        "precision": precision,
        "gate_passed": passed,
        "note": (
            "Gate requires event-held-out evaluation on real paired SAR+optical. "
            "Current evaluation is on a single real pair (no held-out set)."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the multi-modal SAR+optical fusion model (ADR-013 §9.7.2)",
    )
    parser.add_argument("--train", action="store_true", help="Train the fusion model")
    parser.add_argument("--inference", action="store_true", help="Run inference on the real Imja pair")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap training samples (debugging)")
    parser.add_argument("--ckpt-dir", type=str, default=str(CHECKPOINT_DIR), help="Checkpoint directory")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Specific checkpoint for inference")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.train and not args.inference:
        logger.error("Specify --train and/or --inference")
        return 1

    import torch
    from torch.utils.data import DataLoader

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- Training ---
    if args.train:
        from siren.ml.fusion import MultiModalFusionNet
        from siren.ml.fusion_dataset import FusionDataset, collate_fusion
        from siren.ml.losses import WaterLoss

        logger.info("=== Training MultiModalFusionNet ===")
        logger.info("Device: %s", device)

        model = MultiModalFusionNet(
            sar_channels=6, optical_channels=3, base_channels=32, n_heads=4, dropout=0.0,
        ).to(device)
        logger.info("Model: %d parameters", sum(p.numel() for p in model.parameters()))

        train_ds = FusionDataset(
            sar_split="train", max_samples=args.max_samples, chip_size=CHIP_SIZE,
        )
        val_ds = FusionDataset(
            sar_split="test", max_samples=args.max_samples, chip_size=CHIP_SIZE,
        )
        logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fusion, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fusion, num_workers=0,
        )

        criterion = WaterLoss(lambda_gravity=0.1, dice_weight=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        ckpt_dir = Path(args.ckpt_dir) / "multimodal_fusion"
        result = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=args.epochs, scheduler=scheduler, ckpt_dir=ckpt_dir,
        )
        logger.info(
            "Training complete — best val IoU: %.4f at epoch %d",
            result["best_val_iou"], result["best_epoch"],
        )

    # --- Inference ---
    if args.inference:
        logger.info("=== Inference on real Imja pair ===")
        ckpt_path = None
        if args.ckpt_path:
            ckpt_path = Path(args.ckpt_path)
        elif args.train:
            ckpt_path = Path(args.ckpt_dir) / "multimodal_fusion" / FUSION_CKPT_NAME
        else:
            # Try default checkpoint location
            default_ckpt = Path(args.ckpt_dir) / "multimodal_fusion" / FUSION_CKPT_NAME
            ckpt_path = default_ckpt if default_ckpt.exists() else None

        result = run_inference(ckpt_path=ckpt_path, device_str=args.device)
        logger.info("Inference result:")
        logger.info("  Water fraction: %.1f%%", float(result["water_mask"].mean()) * 100)
        logger.info("  Cloud fraction: %s", f"{result['cloud_fraction']:.1%}" if result["cloud_fraction"] is not None else "N/A")
        logger.info("  Optical used: %s", result["optical_used"])
        logger.info("  Model params: %d", result["model_params"])
        logger.info("  Checkpoint: %s", result["checkpoint"] or "None (random weights)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
