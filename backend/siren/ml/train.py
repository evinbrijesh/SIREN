"""Training script for the single-date SAR water segmentation U-Net
(ADR-010 Stage 1, replacing the disqualified bi-temporal Siamese U-Net).

Trains WaterUNet (siren.ml.model.WaterUNet) on Sen1Floods11 hand-labeled
chips using the frozen input contract (siren.ml.contract.normalize_sar).
Not run at demo time -- produces weights saved to
data/processed/water_unet_weights.pt, loaded by ChangeDetectionEngine.

Two split strategies are supported (siren.ml.dataset.WaterSegmentationDataset):

  "official"      -- Sen1Floods11's own train/valid/test CSVs. Chip-level:
                      every flood event appears in all three splits, so
                      this number is literature-comparable but has known
                      event-geography leakage between train and test
                      (see docs/reference/DL_MODEL_AUDIT.md §4).
  "event_holdout" -- siren.ml.build_event_holdout_split's train/val/test.
                      Entire flood events are held out; no event appears
                      in more than one split. This is the strict
                      generalization number ADR-010 requires before any
                      ML output is displayed as evidence.

Model selection (best checkpoint) is done on water-class IoU on the
validation split -- never on training loss (docs/reference/DL_MODEL_AUDIT.md
finding: "all three training loops ... select by training loss").

Usage:
    # Train + evaluate with the strict event-level holdout (default,
    # recommended before treating any result as a generalization claim):
    python -m siren.ml.train --strategy event_holdout --epochs 40

    # Train + evaluate with the literature-comparable official split:
    python -m siren.ml.train --strategy official --epochs 40

    # Run both strategies back-to-back and write a comparison report:
    python -m siren.ml.train --both --epochs 40

    # Quick smoke test (synthetic data, 2 epochs, no real dataset needed):
    python -m siren.ml.train --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Deterministic training (Hard Rule 6)
SEED = 42

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
DEFAULT_SAVE_PATH = DATA_ROOT / "processed" / "water_unet_weights.pt"
REPORT_PATH = DATA_ROOT / "processed" / "water_unet_eval_report.json"


def set_seed(seed: int = SEED) -> None:
    """Set all RNG seeds for reproducible training."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _dice_bce_loss(logits, target, valid, dice_weight: float = 0.5):
    """Soft Dice + BCE, both computed over valid pixels only."""
    import torch
    import torch.nn.functional as F

    logits = logits.squeeze(1)  # (B, H, W)
    probs = torch.sigmoid(logits)

    # BCE, masked to valid pixels
    bce_per_px = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce_per_px * valid).sum() / valid.sum().clamp_min(1.0)

    # Soft Dice, masked to valid pixels
    probs_v = probs * valid
    target_v = target * valid
    intersection = (probs_v * target_v).sum(dim=(1, 2))
    union = probs_v.sum(dim=(1, 2)) + target_v.sum(dim=(1, 2))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    dice_loss = 1.0 - dice.mean()

    return dice_weight * dice_loss + (1 - dice_weight) * bce


def _evaluate(model, loader, device, threshold: float = 0.5) -> dict[str, Any]:
    """Run the model over a full split and compute dataset-level metrics."""
    import torch
    from siren.ml.metrics import RunningConfusion

    model.eval()
    acc = RunningConfusion()
    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            water = batch["water"].numpy()
            valid = batch["valid"].numpy()

            logits = model(sar)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            pred = (probs >= threshold).astype(np.uint8)

            for i in range(pred.shape[0]):
                acc.update(pred[i], water[i], valid[i])

    return acc.result()


def train_one_strategy(
    strategy: str,
    epochs: int = 40,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda",
    save_path: Path | None = None,
    num_workers: int = 2,
) -> dict[str, Any]:
    """Train WaterUNet under one split strategy and evaluate on its test split.

    Returns a report dict: {strategy, best_val_iou, best_epoch, test_metrics, ...}
    """
    import torch
    from torch.utils.data import DataLoader

    from siren.ml.dataset import WaterSegmentationDataset
    from siren.ml.model import WaterUNet

    set_seed(SEED)
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    if save_path is None:
        save_path = DATA_ROOT / "processed" / f"water_unet_weights_{strategy}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = WaterSegmentationDataset(split="train", strategy=strategy)
    val_ds = WaterSegmentationDataset(split="val", strategy=strategy)
    test_ds = WaterSegmentationDataset(split="test", strategy=strategy)

    logger.info(
        f"[{strategy}] train={len(train_ds)} chips {sorted(train_ds.events())}, "
        f"val={len(val_ds)} chips {sorted(val_ds.events())}, "
        f"test={len(test_ds)} chips {sorted(test_ds.events())}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = WaterUNet(in_channels=2).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_iou = -1.0
    best_epoch = -1
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            sar = batch["sar"].to(device_t)
            water = batch["water"].to(device_t)
            valid = batch["valid"].to(device_t)

            optimizer.zero_grad()
            logits = model(sar)
            loss = _dice_bce_loss(logits, water, valid)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_metrics = _evaluate(model, val_loader, device_t)
        history.append({"epoch": epoch + 1, "train_loss": round(avg_loss, 4), "val_iou": val_metrics["iou"]})
        logger.info(
            f"[{strategy}] epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f} — val IoU: {val_metrics['iou']:.4f}"
        )

        # Model selection by VAL IoU, never by training loss (audit finding).
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "val_iou": best_val_iou,
                    "in_channels": 2,
                    "dataset": "Sen1Floods11-handlabeled",
                    "split_strategy": strategy,
                    "architecture": "WaterUNet",
                    "input_contract": "ml/contract.py:normalize_sar",
                    "model_selection": "val_water_iou",
                },
                str(save_path),
            )

    # Reload the best checkpoint (by val IoU) before final test evaluation.
    checkpoint = torch.load(str(save_path), map_location=device_t, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = _evaluate(model, test_loader, device_t)

    logger.info(
        f"[{strategy}] BEST epoch={best_epoch} val_iou={best_val_iou:.4f} — "
        f"TEST iou={test_metrics['iou']:.4f} precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
    )

    return {
        "strategy": strategy,
        "train_events": sorted(train_ds.events()),
        "val_events": sorted(val_ds.events()),
        "test_events": sorted(test_ds.events()),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "test_metrics": test_metrics,
        "history": history,
        "weights_path": str(save_path),
    }


def smoke_test(epochs: int = 2, device: str = "cpu") -> Path:
    """Quick training smoke test on synthetic data.

    Verifies the architecture compiles, forward/backward passes work, and
    weights can be saved/loaded. Does NOT produce meaningful predictions
    and is NOT evaluated against real accuracy metrics.
    """
    import torch
    from siren.ml.model import WaterUNet

    set_seed(SEED)
    device_t = torch.device(device)

    save_path = DEFAULT_SAVE_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Synthetic single-date data: 8 chips, 2 channels (VV/VH), 64x64
    sar = torch.rand(8, 2, 64, 64)
    water = (torch.rand(8, 64, 64) > 0.7).float()
    valid = torch.ones(8, 64, 64)

    model = WaterUNet(in_channels=2).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    logger.info(f"Smoke test: {epochs} epochs on synthetic data")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(sar.to(device_t))
        loss = _dice_bce_loss(logits, water.to(device_t), valid.to(device_t))
        loss.backward()
        optimizer.step()
        logger.info(f"Smoke epoch {epoch + 1}/{epochs} — loss: {loss.item():.4f}")

    torch.save(
        {"state_dict": model.state_dict(), "epoch": epochs, "smoke_test": True, "in_channels": 2},
        str(save_path),
    )
    logger.info(f"Smoke test weights saved to {save_path}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WaterUNet on Sen1Floods11 (ADR-010 Stage 1)")
    parser.add_argument("--strategy", choices=["official", "event_holdout"], default="event_holdout")
    parser.add_argument("--both", action="store_true", help="Run both strategies and write a comparison report")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.smoke_test:
        smoke_test(epochs=2, device="cpu" if args.device == "cuda" else args.device)
        return

    strategies = ["official", "event_holdout"] if args.both else [args.strategy]
    reports = {}
    for strategy in strategies:
        reports[strategy] = train_one_strategy(
            strategy=strategy,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            num_workers=args.num_workers,
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(reports, f, indent=2)
    logger.info(f"Evaluation report written to {REPORT_PATH}")

    for strategy, report in reports.items():
        tm = report["test_metrics"]
        print(
            f"\n=== {strategy} ===\n"
            f"  train events: {report['train_events']}\n"
            f"  val events:   {report['val_events']}\n"
            f"  test events:  {report['test_events']}\n"
            f"  best epoch:   {report['best_epoch']} (val IoU={report['best_val_iou']:.4f})\n"
            f"  TEST metrics: IoU={tm['iou']:.4f}  Precision={tm['precision']:.4f}  "
            f"Recall={tm['recall']:.4f}  F1={tm['f1']:.4f}  (n={tm['n_chips']} chips)"
        )
    if args.both:
        print(
            "\nNOTE: 'official' is literature-comparable but has event-geography leakage "
            "between train and test (all 10 flood events appear in every official split). "
            "'event_holdout' is the strict generalization number — no event overlaps between "
            "train/val/test — and is the number ADR-010 requires before any ML output is "
            "displayed as evidence."
        )


if __name__ == "__main__":
    main()
