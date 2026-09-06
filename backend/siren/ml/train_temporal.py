"""Training script for the ConvLSTM temporal trend classifier.

Creates synthetic temporal sequences from Sen1Floods11 SAR chips by
progressively expanding water regions over N timesteps. This simulates
different glacial lake expansion rates:

  - stable:     water area stays roughly constant (< 2% change)
  - slowly:     water area increases gradually (2-15% over the sequence)
  - rapidly:    water area increases sharply (15-50% over the sequence)
  - uncertain:  water area fluctuates (increases then decreases)

Each sequence has T=4 timesteps (matching the demo's 4 observations:
baseline + 3 observations).

Usage:
    python -m siren.ml.train_temporal --sen1floods11-dir ../data/raw/Sen1Floods11 --epochs 50 --device cuda

    # Quick smoke test (synthetic data, 2 epochs):
    python -m siren.ml.train_temporal --smoke-test --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)

SEED = 42
DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "convlstm_trend_weights.pt"
)

# Trend class names (must match temporal.py)
TREND_CLASSES = ["stable", "slowly", "rapidly", "uncertain"]


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


def _normalize_s1(s1: np.ndarray) -> np.ndarray:
    """Normalize SAR dB values to [0, 1]."""
    s1 = np.clip(s1, -50, 20)
    return ((s1 - (-50)) / (20 - (-50))).astype(np.float32)


def _create_temporal_sequence(
    s1: np.ndarray,
    label: np.ndarray,
    trend_class: str,
    seq_len: int = 4,
) -> tuple[np.ndarray, int]:
    """Create a synthetic temporal sequence from a single S1 chip.

    Strategy: start with a "baseline" (reduced water), then progressively
    add water pixels to simulate the trend class. Classes are made highly
    distinct to ensure the ConvLSTM can learn clear decision boundaries:

      stable:    0-3% random variation (noise around baseline)
      slowly:    8-25% monotonic growth
      rapidly:   30-70% monotonic growth
      uncertain: grow 15-30% then shrink back below baseline

    Args:
        s1: S1 SAR chip (2, H, W) in dB
        label: Binary water mask (H, W) — 1=water, 0=land
        trend_class: One of "stable", "slowly", "rapidly", "uncertain"
        seq_len: Number of timesteps

    Returns:
        (sequence, class_idx) — sequence is (seq_len, 1, H, W) water masks,
        class_idx is the integer label
    """
    h, w = label.shape
    water_mask = (label == 1).astype(np.float32)
    total_water = int(water_mask.sum())

    if total_water == 0:
        # No water in this chip — use it as a "stable" (all zeros) sequence
        seq = np.zeros((seq_len, 1, h, w), dtype=np.float32)
        return seq, 0  # stable

    # Find land pixels adjacent to water (for realistic expansion)
    from scipy import ndimage
    water_dilated = ndimage.binary_dilation(water_mask, iterations=5)
    expansion_zone = water_dilated & ~water_mask.astype(bool)
    expansion_pixels = np.argwhere(expansion_zone)

    class_idx = TREND_CLASSES.index(trend_class)

    if trend_class == "stable":
        # Water area stays roughly constant — tiny random variation (±3%)
        seq = np.zeros((seq_len, 1, h, w), dtype=np.float32)
        for t in range(seq_len):
            # Just use the base water mask with negligible noise
            seq[t, 0] = water_mask
        return seq, class_idx

    if trend_class == "uncertain":
        # Fluctuating — water increases 20% then decreases below baseline
        seq = np.zeros((seq_len, 1, h, w), dtype=np.float32)
        peak = seq_len // 2
        max_growth = np.random.uniform(0.20, 0.35)
        for t in range(seq_len):
            if t <= peak:
                frac = (t + 1) / (peak + 1) * max_growth
            else:
                # Shrink back — end below the starting point
                frac = max_growth * (1 - (t - peak) / (seq_len - peak)) * 0.5
            n_add = int(total_water * frac)
            mask_t = water_mask.copy()
            if t > peak:
                # Remove some original water pixels to simulate recession
                water_pixels = np.argwhere(water_mask == 1)
                n_remove = int(total_water * frac * 0.3)
                if n_remove > 0 and len(water_pixels) > 0:
                    rm_idx = np.random.choice(len(water_pixels), min(n_remove, len(water_pixels)), replace=False)
                    for i in rm_idx:
                        mask_t[water_pixels[i][0], water_pixels[i][1]] = 0.0
            if n_add > 0 and len(expansion_pixels) > 0:
                idx = np.random.choice(len(expansion_pixels), min(n_add, len(expansion_pixels)), replace=False)
                for i in idx:
                    mask_t[expansion_pixels[i][0], expansion_pixels[i][1]] = 1.0
            seq[t, 0] = mask_t
        return seq, class_idx

    # slowly or rapidly — monotonic progressive expansion
    if trend_class == "slowly":
        max_expansion = np.random.uniform(0.08, 0.25)  # 8-25% growth
    else:  # rapidly
        max_expansion = np.random.uniform(0.30, 0.70)  # 30-70% growth

    seq = np.zeros((seq_len, 1, h, w), dtype=np.float32)
    # Start from a reduced baseline (50% of current water) to show growth
    baseline_frac = np.random.uniform(0.4, 0.6)
    baseline_water = water_mask.copy()
    water_pixels = np.argwhere(water_mask == 1)
    n_keep = int(total_water * baseline_frac)
    keep_idx = np.random.choice(len(water_pixels), n_keep, replace=False)
    baseline_water[:] = 0
    for i in keep_idx:
        baseline_water[water_pixels[i][0], water_pixels[i][1]] = 1.0

    for t in range(seq_len):
        frac = (t + 1) / seq_len * max_expansion
        n_add = int(total_water * frac)
        mask_t = baseline_water.copy()
        if n_add > 0 and len(expansion_pixels) > 0:
            idx = np.random.choice(
                len(expansion_pixels),
                min(n_add, len(expansion_pixels)),
                replace=False,
            )
            for i in idx:
                mask_t[expansion_pixels[i][0], expansion_pixels[i][1]] = 1.0
        seq[t, 0] = mask_t

    return seq, class_idx


class TemporalSequenceDataset:
    """Dataset of synthetic temporal sequences from Sen1Floods11.

    For each S1 chip, creates 4 sequences (one per trend class) by
    synthetically expanding water regions at different rates.
    """

    def __init__(self, root: Path | str, split: str = "train", seq_len: int = 4) -> None:
        self.root = Path(root)
        self.split = split
        self.seq_len = seq_len
        self.sequences: list[tuple[np.ndarray, int]] = []
        self._build()

    def _build(self) -> None:
        import csv
        import rasterio
        from scipy import ndimage

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

            # Skip chips with no water
            if (label == 1).sum() < 50:
                continue

            # Downsample to 128x128 for memory efficiency
            from scipy.ndimage import zoom
            if label.shape[0] != 128:
                zoom_factor = 128 / label.shape[0]
                label = zoom(label, zoom_factor, order=0).astype(np.int16)

            # Create one sequence per trend class
            for trend in TREND_CLASSES:
                seq, cls = _create_temporal_sequence(s1, label, trend, self.seq_len)
                self.sequences.append((seq, cls))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        import torch

        seq, cls = self.sequences[idx]
        return {
            "sequence": torch.from_numpy(seq).float(),
            "label": torch.tensor(cls, dtype=torch.long),
        }


def train(
    sen1floods11_dir: Path,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    seq_len: int = 4,
    save_path: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Train the ConvLSTM temporal trend classifier.

    Args:
        sen1floods11_dir: Root of the Sen1Floods11 dataset
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate (AdamW)
        seq_len: Number of timesteps per sequence
        save_path: Where to save the trained weights
        device: "cpu" or "cuda"

    Returns:
        Path to the saved weights file
    """
    import torch
    from torch.utils.data import DataLoader

    from siren.ml.temporal import ConvLSTMTrendClassifier

    set_seed(SEED)
    device_t = torch.device(device)

    if save_path is None:
        save_path = DEFAULT_WEIGHTS_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Build dataset
    dataset = TemporalSequenceDataset(sen1floods11_dir, split="train", seq_len=seq_len)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    # Build model
    model = ConvLSTMTrendClassifier(
        in_channels=1,
        encoder_dim=32,
        lstm_dim=64,
        num_classes=4,
    ).to(device_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Training ConvLSTM on {len(dataset)} sequences for {epochs} epochs")
    logger.info(f"Device: {device_t}, Batch size: {batch_size}, LR: {lr}, Seq len: {seq_len}")

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0

        for batch in loader:
            seq = batch["sequence"].to(device_t)
            labels = batch["label"].to(device_t)

            optimizer.zero_grad()
            logits = model(seq)
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
                "seq_len": seq_len,
                "num_classes": 4,
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

    from siren.ml.temporal import ConvLSTMTrendClassifier

    set_seed(SEED)
    device_t = torch.device(device)
    save_path = DEFAULT_WEIGHTS_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Synthetic sequences: 16 samples, 4 timesteps, 1 channel, 64x64
    sequences = torch.randn(16, 4, 1, 64, 64)
    labels = torch.randint(0, 4, (16,))

    model = ConvLSTMTrendClassifier(in_channels=1, encoder_dim=16, lstm_dim=32).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Smoke test: {epochs} epochs on synthetic data")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(sequences.to(device_t))
        loss = criterion(logits, labels.to(device_t))
        loss.backward()
        optimizer.step()
        logger.info(f"Smoke epoch {epoch + 1}/{epochs} — loss: {loss.item():.4f}")

    torch.save({"state_dict": model.state_dict(), "epoch": epochs, "smoke_test": True}, str(save_path))
    logger.info(f"Smoke test weights saved to {save_path}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ConvLSTM temporal trend classifier")
    parser.add_argument("--sen1floods11-dir", type=str, help="Path to Sen1Floods11 dataset root")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=4, help="Sequence length (timesteps)")
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
            seq_len=args.seq_len,
            device=args.device,
        )
    else:
        parser.error("Either --sen1floods11-dir or --smoke-test is required")


if __name__ == "__main__":
    main()
