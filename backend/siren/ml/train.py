"""Training script for the Siamese U-Net on the Sen1Floods11 flood benchmark.

Sen1Floods11 is a public dataset of Sentinel-1 SAR and Sentinel-2 optical
chips with hand-labeled flood masks (Bonafilia et al., 2020).
  Repo: https://github.com/cloudtostreet/Sen1Floods11

This script fine-tunes the Siamese U-Net for bi-temporal change detection
on the Sen1Floods11 chips. It is NOT run at demo time — it produces trained
weights that are saved to data/processed/siamese_unet_weights.pt and loaded
by the ChangeDetectionEngine at runtime.

Usage:
    python -m siren.ml.train --sen1floods11-dir /path/to/Sen1Floods11 --epochs 50

    # Quick smoke test (synthetic data, 2 epochs):
    python -m siren.ml.train --smoke-test

Requires: torch, torchvision, rasterio, numpy (already in the dependency
whitelist except torch, which was approved for the ML stretch goal).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Deterministic training (Hard Rule 6)
SEED = 42


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


def train(
    sen1floods11_dir: Path,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    in_channels: int = 2,
    save_path: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Train the Siamese U-Net on Sen1Floods11 SAR water segmentation chips.

    Creates synthetic bi-temporal pairs from single-image S1 chips (see
    Sen1Floods11Dataset docstring for the pairing strategy).

    Args:
        sen1floods11_dir: Root of the Sen1Floods11 dataset
        epochs: Number of training epochs
        batch_size: Mini-batch size (4 for 512x512 on 8GB VRAM)
        lr: Learning rate (AdamW)
        in_channels: Input channels (2 for VV+VH SAR)
        save_path: Where to save the trained weights
        device: "cpu" or "cuda"

    Returns:
        Path to the saved weights file
    """
    import torch
    from torch.utils.data import DataLoader

    from siren.ml.model import SiameseUNet

    set_seed(SEED)
    device_t = torch.device(device)

    if save_path is None:
        save_path = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "processed"
            / "siamese_unet_weights.pt"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Build dataset and dataloader
    dataset = Sen1Floods11Dataset(sen1floods11_dir, split="train", in_channels=in_channels)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    # Build model, optimizer, loss
    model = SiameseUNet(in_channels=in_channels).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    logger.info(f"Training SiameseUNet on {len(dataset)} chips for {epochs} epochs")
    logger.info(f"Device: {device_t}, Batch size: {batch_size}, LR: {lr}, Channels: {in_channels}")

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            t0 = batch["t0"].to(device_t)
            t1 = batch["t1"].to(device_t)
            mask = batch["mask"].to(device_t)

            optimizer.zero_grad()
            logits = model(t0, t1)
            loss = criterion(logits.squeeze(1), mask.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best weights so far
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
                "in_channels": in_channels,
                "dataset": "Sen1Floods11-handlabeled",
            }, str(save_path))

        logger.info(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f} (best: {best_loss:.4f})")

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    logger.info(f"Saved trained weights to {save_path}")
    return save_path


def smoke_test(epochs: int = 2, device: str = "cpu") -> Path:
    """Quick training smoke test on synthetic data.

    Verifies the architecture compiles, forward/backward passes work, and
    weights can be saved/loaded. Does NOT produce meaningful predictions.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from siren.ml.model import SiameseUNet

    set_seed(SEED)
    device_t = torch.device(device)

    save_path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "processed"
        / "siamese_unet_weights.pt"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Synthetic bi-temporal data: 8 chips, 3 channels, 64x64
    t0 = torch.randn(8, 3, 64, 64)
    t1 = torch.randn(8, 3, 64, 64)
    masks = (torch.rand(8, 64, 64) > 0.7).float()

    model = SiameseUNet(in_channels=3).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    logger.info(f"Smoke test: {epochs} epochs on synthetic data")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(t0.to(device_t), t1.to(device_t))
        loss = criterion(logits.squeeze(1), masks.to(device_t))
        loss.backward()
        optimizer.step()
        logger.info(f"Smoke epoch {epoch + 1}/{epochs} — loss: {loss.item():.4f}")

    torch.save({"state_dict": model.state_dict(), "epoch": epochs, "smoke_test": True}, str(save_path))
    logger.info(f"Smoke test weights saved to {save_path}")
    return save_path


class Sen1Floods11Dataset:
    """Dataset loader for Sen1Floods11 SAR water segmentation.

    Sen1Floods11 provides single-image Sentinel-1 chips with binary water
    labels (0=land, 1=water, -1=no data). To train the Siamese U-Net
    (which expects bi-temporal inputs), we create synthetic pre-flood/flood
    pairs:

      t1 = actual S1 chip (flood state)
      t0 = S1 chip with water pixels replaced by median land backscatter
           (simulating the pre-flood dry state)
      mask = (label == 1) — binary water mask

    The model learns to detect |F1 - F0| at water locations, which
    represents the "change" from dry to wet.

    Expected directory structure:
        sen1floods11_dir/
          train/
            S1/         # Sentinel-1 VV+VH chips (*.tif)
            Label/      # Hand-labeled water masks (*.tif)
            pairs.csv   # S1_filename,Label_filename pairs
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        in_channels: int = 2,
        chip_size: int = 512,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.in_channels = in_channels
        self.chip_size = chip_size
        self.pairs = self._load_pairs()

    def _load_pairs(self) -> list[dict]:
        """Load the pairs CSV (S1,Label format for Sen1Floods11 v1.1)."""
        import csv

        csv_path = self.root / self.split / "pairs.csv"
        if not csv_path.exists():
            return []

        pairs = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                s1_file, label_file = row[0], row[1]
                pairs.append({
                    "s1": self.root / self.split / "S1" / s1_file,
                    "label": self.root / self.split / "Label" / label_file,
                })
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        import rasterio
        import torch

        pair = self.pairs[idx]

        with rasterio.open(str(pair["s1"])) as src:
            s1 = src.read()  # (2, 512, 512) — VV + VH in dB
        with rasterio.open(str(pair["label"])) as src:
            label = src.read(1)  # (512, 512) — 0=land, 1=water, -1=nodata

        # Handle NaN: replace with median of valid pixels
        if np.any(np.isnan(s1)):
            valid = s1[~np.isnan(s1)]
            if len(valid) > 0:
                s1 = np.where(np.isnan(s1), np.median(valid), s1)
            else:
                s1 = np.zeros_like(s1, dtype=np.float32)

        # Clip to typical SAR dB range and normalize to [0, 1]
        s1 = np.clip(s1, -50, 20)
        s1 = ((s1 - (-50)) / (20 - (-50))).astype(np.float32)

        # Binary water mask: 1 = water, 0 = land/nodata
        mask = (label == 1).astype(np.float32)

        # Create synthetic pre-flood baseline (t0):
        # Replace water pixels with median land backscatter
        land_pixels = s1[:, mask == 0]
        if land_pixels.size > 0:
            land_median = np.median(land_pixels, axis=1, keepdims=True)  # (C, 1)
        else:
            land_median = s1.mean(axis=(1, 2), keepdims=True)  # fallback

        t0 = s1.copy()
        water_mask_2d = mask.astype(bool)
        if water_mask_2d.any():
            t0[:, water_mask_2d] = land_median

        t1 = s1

        # Adjust channels if needed
        if t0.shape[0] != self.in_channels:
            t0 = self._adjust_channels(t0, self.in_channels)
            t1 = self._adjust_channels(t1, self.in_channels)

        return {
            "t0": torch.from_numpy(t0).float(),
            "t1": torch.from_numpy(t1).float(),
            "mask": torch.from_numpy(mask),
        }

    @staticmethod
    def _adjust_channels(arr: np.ndarray, target: int) -> np.ndarray:
        """Adjust channel count: truncate or replicate to match target."""
        c = arr.shape[0]
        if c == target:
            return arr
        if c > target:
            return arr[:target]
        reps = [target // c] + [1] * (target - (target // c) * c)
        return np.concatenate(
            [np.repeat(arr, reps[0], axis=0)] + [arr[:1]] * reps[1], axis=0
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Siamese U-Net on Sen1Floods11")
    parser.add_argument("--sen1floods11-dir", type=str, help="Path to Sen1Floods11 dataset root")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size (4 for 512x512 on 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--in-channels", type=int, default=2, help="Input channels (2=VV+VH SAR, 3=RGB, 4=RGB+NIR)")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test instead of real training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.smoke_test:
        smoke_test(epochs=2, device=args.device)
    elif args.sen1floods11_dir:
        train(
            sen1floods11_dir=Path(args.sen1floods11_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            in_channels=args.in_channels,
            device=args.device,
        )
    else:
        parser.error("Either --sen1floods11-dir or --smoke-test is required")


if __name__ == "__main__":
    main()
