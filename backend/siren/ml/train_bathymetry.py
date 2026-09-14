"""Training script for the neural bathymetry inversion model (ADR-013 §9.7.1).

Trains a BathymetryUNet to predict submerged lake bed elevation from
surrounding moraine DEM contours and lake boundary shape features.

Data sources (in priority order):
    1. Millan et al. (2022) consensus ice-thickness estimates
    2. Farinotti et al. (2019) ITMIX 2 bed-inversion datasets
    3. Surveyed lake bathymetry (if available)
    4. Synthetic parabolic basins (architecture testing only — NOT for
       production training; real training requires real bathymetric data)

Gate: < 15% MAPE on held-out lakes with known bathymetry.

Usage:
    # Architecture test on synthetic data
    python -m siren.ml.train_bathymetry --synthetic --epochs 50

    # Real training (requires Millan/Farinotti data in data/raw/)
    python -m siren.ml.train_bathymetry --data-dir data/raw/millan --epochs 200
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from siren.ml.bathymetry import (
    BathymetryUNet,
    generate_synthetic_bathymetry_data,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"


def train_on_synthetic(
    model: BathymetryUNet,
    n_samples: int = 500,
    grid_size: int = 64,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> dict:
    """Train on synthetic parabolic basin data (architecture testing only).

    This is for verifying the architecture can learn the bed-elevation
    prediction task. Real training requires Millan/Farinotti data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Generating %d synthetic bathymetry samples...", n_samples)
    dems, masks, beds = generate_synthetic_bathymetry_data(
        n_samples=n_samples, grid_size=grid_size, seed=seed,
    )

    # Normalize
    dem_min = dems.min()
    dem_max = dems.max()
    dems_norm = (dems - dem_min) / (dem_max - dem_min + 1e-6)
    beds_norm = (beds - dem_min) / (dem_max - dem_min + 1e-6)

    # Build dataset
    x = np.stack([dems_norm, masks], axis=1).astype(np.float32)  # (N, 2, H, W)
    y = beds_norm[:, np.newaxis].astype(np.float32)  # (N, 1, H, W)

    # Split: 80/20 train/val
    n_train = int(0.8 * n_samples)
    train_ds = TensorDataset(
        torch.from_numpy(x[:n_train]), torch.from_numpy(y[:n_train]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x[n_train:]), torch.from_numpy(y[n_train:]),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += F.mse_loss(pred, yb).item()
        val_loss /= len(val_loader)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d: train_loss=%.6f, val_loss=%.6f, best=%.6f@%d",
                epoch + 1, epochs, train_loss, val_loss, best_val_loss, best_epoch,
            )

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "n_samples": n_samples,
        "data_source": "synthetic_parabolic",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train neural bathymetry model")
    parser.add_argument("--synthetic", action="store_true",
                        help="Train on synthetic data (architecture testing only)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to real bathymetry data (Millan/Farinotti)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--n-down", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of synthetic samples (synthetic mode only)")
    parser.add_argument("--save-name", type=str, default="bathymetry_unet_v1.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = BathymetryUNet(
        in_channels=2,
        base_channels=args.base_channels,
        n_down=args.n_down,
    ).to(device)
    logger.info("Model: %d params", model.num_parameters())

    if args.synthetic:
        result = train_on_synthetic(
            model, n_samples=args.n_samples, grid_size=args.grid_size,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=device, seed=args.seed,
        )
    elif args.data_dir:
        # Real data training — requires Millan/Farinotti loader
        # TODO: implement real data loader when datasets are available
        logger.error("Real data training not yet implemented — requires Millan/Farinotti data loader")
        return 1
    else:
        parser.error("Must specify --synthetic or --data-dir")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINT_DIR / args.save_name
    torch.save(model.state_dict(), str(save_path))
    logger.info("Saved checkpoint: %s", save_path)

    # Save metadata
    meta = {
        "architecture": "BathymetryUNet",
        "in_channels": 2,
        "base_channels": args.base_channels,
        "n_down": args.n_down,
        "num_parameters": model.num_parameters(),
        "training": result,
        "gate": {
            "target_mape": 0.15,
            "passed": False,
            "adr": "ADR-013",
            "note": "Gate evaluation requires real bathymetric data (Millan/Farinotti)",
        },
        "checkpoint_path": str(save_path),
    }
    meta_path = CHECKPOINT_DIR / args.save_name.replace(".pt", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata: %s", meta_path)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
