"""Training script for the SegFormer Stage 2 land-cover classifier.

Classifies changed-pixel crops into 5 functional categories:
  0: Water (floodwater / lake expansion)
  1: Debris flow (mud/rock torrent)
  2: Snowmelt (benign seasonal melt)
  3: Shadow / cloud anomaly (sensor artifact — discard)
  4: Bare rock / landslide

Since Sen1Floods11 only has binary water/land labels, we use a weak-labeling
strategy based on SAR backscatter statistics and spatial context:

  - Water: pixels labeled as water (label==1) — very low backscatter (< -20 dB)
  - Shadow: pixels with very low backscatter but NOT labeled as water, on steep slopes
  - Snowmelt: pixels with moderate backscatter in high-altitude chips (near snowline)
  - Bare rock: pixels with high backscatter (> -10 dB) on steep terrain
  - Debris: pixels with moderate-high backscatter in transition zones (edge of water)

This is a weak supervision approach — the labels are noisy but sufficient for
a demo-level classifier that filters obvious false alarms.

Usage:
    python -m siren.ml.train_segformer --sen1floods11-dir ../data/raw/Sen1Floods11 --epochs 50 --device cuda

    # Quick smoke test:
    python -m siren.ml.train_segformer --smoke-test --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)

SEED = 42
DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "segformer_classifier_weights.pt"
)

# Class names (must match model.py SegFormerHead)
CLASS_NAMES = ["water", "debris", "snowmelt", "shadow", "bare_rock"]
NUM_CLASSES = 5
CROP_SIZE = 64  # Crop size for changed regions


def set_seed(seed: int = SEED) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _assign_weak_labels(
    s1: np.ndarray,
    label: np.ndarray,
) -> np.ndarray:
    """Assign weak pseudo-labels to pixels based on SAR backscatter statistics.

    Args:
        s1: S1 SAR chip (2, H, W) in dB — VV and VH
        label: Binary water mask (H, W) — 1=water, 0=land, -1=nodata

    Returns:
        label_map: (H, W) int8 array with class indices 0-4
    """
    h, w = label.shape
    label_map = np.full((h, w), -1, dtype=np.int8)  # -1 = no label

    vv = s1[0]  # VV polarization
    vh = s1[1]  # VH polarization

    # Handle NaN
    valid = ~np.isnan(vv) & ~np.isnan(vh)
    if not valid.any():
        return label_map

    # Class 0: Water — labeled water or very low backscatter
    water_mask = (label == 1) | ((vv < -22) & valid)
    label_map[water_mask] = 0

    # Compute local statistics for remaining pixels
    vv_median = np.median(vv[valid])
    vh_median = np.median(vh[valid])

    # Class 3: Shadow — very low VV but not water, typically on steep terrain
    # In SAR, shadow appears as very dark regions (no return) adjacent to terrain
    shadow_mask = (vv < -25) & ~water_mask & valid
    label_map[shadow_mask] = 3

    # Class 4: Bare rock — high backscatter (strong return from rough surfaces)
    bare_rock_mask = (vv > -8) & ~water_mask & ~shadow_mask & valid
    label_map[bare_rock_mask] = 4

    # Class 1: Debris — moderate-high backscatter near water edges
    # (transition zones between water and land)
    from scipy import ndimage
    water_edge = ndimage.binary_dilation(water_mask, iterations=2) & ~water_mask
    debris_mask = water_edge & (vv > -15) & (vv < -8) & ~shadow_mask & ~bare_rock_mask & valid
    label_map[debris_mask] = 1

    # Class 2: Snowmelt — moderate backscatter, not near water, moderate VH
    # (wet snow has moderate backscatter, distinguishable from dry rock)
    remaining = (label_map == -1) & valid
    snowmelt_mask = remaining & (vv > -18) & (vv < -10) & (vh > -22) & ~water_edge
    label_map[snowmelt_mask] = 2

    # Assign remaining unlabeled valid pixels to the closest class
    remaining = (label_map == -1) & valid
    if remaining.any():
        # Default to bare rock for high backscatter, snowmelt for moderate
        label_map[remaining & (vv > vv_median)] = 4
        label_map[remaining & (vv <= vv_median)] = 2

    return label_map


def _extract_crops(
    s1: np.ndarray,
    label_map: np.ndarray,
    crop_size: int = CROP_SIZE,
    max_crops_per_class: int = 10,
) -> list[tuple[np.ndarray, int]]:
    """Extract random crops from the chip, labeled by their dominant class.

    For each class present in the label map, extract random crops where
    that class is the majority.
    """
    h, w = label_map.shape
    crops = []

    for cls in range(NUM_CLASSES):
        cls_mask = (label_map == cls)
        if cls_mask.sum() < crop_size * crop_size * 0.1:
            continue  # Skip if too few pixels of this class

        # Find regions where this class is dominant
        cls_density = ndimage_uniform_filter(cls_mask.astype(np.float32), size=crop_size)

        # Sample crops from high-density regions
        high_density = np.argwhere(cls_density > 0.3)
        if len(high_density) == 0:
            continue

        n_crops = min(max_crops_per_class, len(high_density))
        indices = np.random.choice(len(high_density), n_crops, replace=False)

        for idx in indices:
            cy, cx = high_density[idx]
            y0 = max(0, cy - crop_size // 2)
            x0 = max(0, cx - crop_size // 2)
            y1 = min(h, y0 + crop_size)
            x1 = min(w, x0 + crop_size)

            crop = s1[:, y0:y1, x0:x1]
            if crop.shape[1] < crop_size or crop.shape[2] < crop_size:
                # Pad to crop_size
                pad_h = crop_size - crop.shape[1]
                pad_w = crop_size - crop.shape[2]
                crop = np.pad(crop, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

            crops.append((crop, cls))

    return crops


def ndimage_uniform_filter(arr, size):
    """Wrapper for scipy uniform_filter."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(arr, size=size)


class SegFormerDataset:
    """Dataset of weak-labeled crops from Sen1Floods11 for SegFormer training.

    Extracts random crops from S1 chips and assigns pseudo-labels using
    backscatter statistics. Each crop is (2, CROP_SIZE, CROP_SIZE) with
    a class label 0-4.
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        crop_size: int = CROP_SIZE,
        max_crops_per_chip: int = 20,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.crops: list[tuple[np.ndarray, int]] = []
        self._build(max_crops_per_chip)

    def _build(self, max_crops_per_chip: int) -> None:
        import csv
        import rasterio

        csv_path = self.root / self.split / "pairs.csv"
        if not csv_path.exists():
            return

        with open(csv_path) as f:
            reader = csv.reader(f)
            pairs = list(reader)

        for s1_file, label_file in pairs:
            s1_path = self.root / self.split / "S1" / s1_file
            label_path = self.root / self.split / "Label" / label_file
            if not s1_path.exists() or not label_path.exists():
                continue

            with rasterio.open(str(s1_path)) as src:
                s1 = src.read()
            with rasterio.open(str(label_path)) as src:
                label = src.read(1)

            # Handle NaN
            if np.any(np.isnan(s1)):
                valid = s1[~np.isnan(s1)]
                if len(valid) > 0:
                    s1 = np.where(np.isnan(s1), np.median(valid), s1)
                else:
                    continue

            # Normalize to [0, 1]
            s1 = np.clip(s1, -50, 20)
            s1 = ((s1 - (-50)) / (20 - (-50))).astype(np.float32)

            # Assign weak labels
            # Re-denormalize for label assignment (need dB values)
            s1_db = s1 * 70 - 50
            label_map = _assign_weak_labels(s1_db, label)

            # Extract crops
            chip_crops = _extract_crops(
                s1, label_map, crop_size=self.crop_size,
                max_crops_per_class=max_crops_per_chip // NUM_CLASSES,
            )
            self.crops.extend(chip_crops)

        logger.info(f"SegFormer dataset: {len(self.crops)} crops from {len(pairs)} chips")

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> dict:
        import torch

        crop, cls = self.crops[idx]
        return {
            "image": torch.from_numpy(crop).float(),
            "label": torch.tensor(cls, dtype=torch.long),
        }


def train(
    sen1floods11_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    in_channels: int = 2,
    save_path: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Train the SegFormer head on weak-labeled Sen1Floods11 crops.

    Args:
        sen1floods11_dir: Root of the Sen1Floods11 dataset
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate (AdamW)
        in_channels: Input channels (2 for VV+VH SAR)
        save_path: Where to save the trained weights
        device: "cpu" or "cuda"

    Returns:
        Path to the saved weights file
    """
    import torch
    from torch.utils.data import DataLoader

    from siren.ml.model import SegFormerHead

    set_seed(SEED)
    device_t = torch.device(device)

    if save_path is None:
        save_path = DEFAULT_WEIGHTS_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Build dataset
    dataset = SegFormerDataset(sen1floods11_dir, split="train")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    # Build model
    model = SegFormerHead(in_channels=in_channels, num_classes=NUM_CLASSES).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Training SegFormer on {len(dataset)} crops for {epochs} epochs")
    logger.info(f"Device: {device_t}, Batch size: {batch_size}, LR: {lr}")

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0

        for batch in loader:
            images = batch["image"].to(device_t)
            labels = batch["label"].to(device_t)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
                "accuracy": accuracy,
                "in_channels": in_channels,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "dataset": "Sen1Floods11-weak-labeled",
            }, str(save_path))

        logger.info(
            f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f} "
            f"acc: {accuracy:.2%} (best loss: {best_loss:.4f})"
        )

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    logger.info(f"Saved trained weights to {save_path}")
    return save_path


def smoke_test(epochs: int = 2, device: str = "cpu") -> Path:
    """Quick smoke test on synthetic data."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from siren.ml.model import SegFormerHead

    set_seed(SEED)
    device_t = torch.device(device)
    save_path = DEFAULT_WEIGHTS_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Synthetic crops: 32 samples, 2 channels, 64x64
    images = torch.randn(32, 2, CROP_SIZE, CROP_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (32,))

    model = SegFormerHead(in_channels=2, num_classes=NUM_CLASSES).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Smoke test: {epochs} epochs on synthetic data")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(images.to(device_t))
        loss = criterion(logits, labels.to(device_t))
        loss.backward()
        optimizer.step()
        logger.info(f"Smoke epoch {epoch + 1}/{epochs} — loss: {loss.item():.4f}")

    torch.save({"state_dict": model.state_dict(), "epoch": epochs, "smoke_test": True}, str(save_path))
    logger.info(f"Smoke test weights saved to {save_path}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SegFormer land-cover classifier")
    parser.add_argument("--sen1floods11-dir", type=str, help="Path to Sen1Floods11 dataset root")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--in-channels", type=int, default=2, help="Input channels (2=VV+VH)")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
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
