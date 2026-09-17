"""Training script for the neural bathymetry inversion model (ADR-013 §9.7.1).

Trains a BathymetryUNet to predict submerged lake bed elevation from
surrounding moraine DEM contours and lake boundary shape features.

Data sources (in priority order):
    1. Surveyed lake bathymetry (20 lakes, 117k points — Zhang 2023 + Das 2025)
    2. Millan et al. (2022) consensus ice-thickness estimates (proxy)
    3. Farinotti et al. (2019) ITMIX 2 bed-inversion datasets (proxy)
    4. Synthetic parabolic basins (architecture testing only — NOT for
       production training; real training requires real bathymetric data)

Gate: < 15% MAPE on held-out lakes with known bathymetry.

Usage:
    # Architecture test on synthetic data
    python -m siren.ml.train_bathymetry --synthetic --epochs 50

    # Real-data LOO benchmark (Huggel vs regression on surveyed volumes)
    python -m siren.ml.train_bathymetry --benchmark

    # Real-data LOO training and evaluation with neural model
    python -m siren.ml.train_bathymetry --train-real --epochs 200

    # Transfer learning (E2): pre-train on ~2000 synthetic basins that
    # mimic the real-sample contract, then fine-tune per LOO fold
    python -m siren.ml.train_bathymetry --pretrain-synthetic --epochs 40
    python -m siren.ml.train_bathymetry --train-real --epochs 60 --lr 1e-4 \
        --pretrain-weights models/checkpoints/bathymetry_unet_synth_pretrain.pt

    # Real training with external data (requires DEM data for the surveyed lakes)
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
    generate_transfer_bathymetry_data,
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


def _normalize_sample(dem: np.ndarray, mask: np.ndarray, bed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample normalisation matching BathymetrySample.to_input_target."""
    dem_min = float(dem[dem > 0].min()) if (dem > 0).any() else 0.0
    dem_max = float(dem.max())
    if dem_max > dem_min:
        dem_norm = np.clip((dem - dem_min) / (dem_max - dem_min), 0.0, 1.0)
        bed_norm = np.clip((bed - dem_min) / (dem_max - dem_min), 0.0, 1.0)
    else:
        dem_norm = np.zeros_like(dem)
        bed_norm = np.zeros_like(bed)
    x = np.stack([dem_norm, mask], axis=0).astype(np.float32)
    y = bed_norm[np.newaxis].astype(np.float32)
    return x, y


def train_transfer_pretrain(
    model: BathymetryUNet,
    n_samples: int = 2000,
    grid_size: int = 128,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    seed: int = 123,
) -> dict:
    """Pre-train on synthetic basins matching the real-sample contract (E2).

    Uses ``generate_transfer_bathymetry_data`` — relief terrain, masked
    lake DEM channel, blobby outlines, shoreline-following bowls — with
    the same per-sample normalisation and lake-masked MSE loss the real
    LOO loop uses, so the learned features transfer. The resulting
    checkpoint is a weight initialisation for ``run_loo_neural``
    (--pretrain-weights), not a deployable model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(
        "Generating %d transfer basins (grid=%d, seed=%d)...",
        n_samples, grid_size, seed,
    )
    dems, masks, beds = generate_transfer_bathymetry_data(
        n_samples=n_samples, grid_size=grid_size, seed=seed,
    )

    xs, ys = [], []
    for i in range(n_samples):
        x, y = _normalize_sample(dems[i], masks[i], beds[i])
        xs.append(x)
        ys.append(y)
    x_all = np.stack(xs)
    y_all = np.stack(ys)

    n_train = int(0.9 * n_samples)
    train_ds = TensorDataset(
        torch.from_numpy(x_all[:n_train]), torch.from_numpy(y_all[:n_train]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_all[n_train:]), torch.from_numpy(y_all[n_train:]),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            mask = xb[:, 1:2]
            loss = F.mse_loss(pred * mask, yb * mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                mask = xb[:, 1:2]
                val_loss += F.mse_loss(pred * mask, yb * mask).item()
        val_loss /= len(val_loader)

        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "Pretrain epoch %d/%d: train=%.6f val=%.6f best=%.6f@%d",
                epoch + 1, epochs, train_loss, val_loss, best_val_loss, best_epoch,
            )

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "n_samples": n_samples,
        "data_source": "synthetic_transfer_basins",
    }


def run_loo_benchmark() -> dict:
    """Run the leave-one-lake-out volume estimation benchmark.

    Compares the Huggel et al. (2002) empirical area-volume power law against
    a fitted power-law regression on 20 surveyed Himalayan glacial lakes.

    Returns:
        Benchmark summary as a dict.
    """
    from siren.ml.bathymetry_benchmark import run_loo_benchmark as _run

    summary = _run()
    logger.info(
        "LOO Benchmark: %d lakes (%d with published volumes)",
        summary.n_lakes,
        summary.n_with_published,
    )
    logger.info(
        "  Huggel:     MAPE=%.1f%%  median=%.1f%%  gate=%s",
        summary.huggel_mape * 100,
        summary.huggel_median_ape * 100,
        "PASS" if summary.huggel_passes_gate else "FAIL",
    )
    logger.info(
        "  Regression: MAPE=%.1f%%  median=%.1f%%  gate=%s",
        summary.regression_mape * 100,
        summary.regression_median_ape * 100,
        "PASS" if summary.regression_passes_gate else "FAIL",
    )
    logger.info("  Gate target: %.0f%%", summary.gate_target_mape * 100)

    return summary.to_dict()


def run_loo_neural(
    epochs: int = 200,
    batch_size: int = 4,
    lr: float = 1e-3,
    base_channels: int = 32,
    n_down: int = 3,
    grid_size: int = 128,
    seed: int = 42,
    device: torch.device | None = None,
    pretrained_weights: Path | None = None,
) -> dict:
    """Run leave-one-lake-out training and evaluation with the neural model.

    For each of the 20 surveyed lakes:
        1. Hold out one lake as the test set.
        2. Train the BathymetryUNet on the remaining 19 lakes.
        3. Predict the held-out lake's bed elevation.
        4. Compute the predicted volume from the predicted bed elevation.
        5. Compare against the sample volume (from surveyed depths) and
           the published volume (where available).

    The sample volume uses the same lake mask as the prediction, so the
    comparison is fair (both use the convex hull of surveyed points). The
    published volume comparison is also reported for context.

    When ``pretrained_weights`` is given, each fold fine-tunes from the
    synthetic-basin pretrained checkpoint instead of a random init — the
    E2 transfer-learning path (ADR-013 §9.7.1). Use a lower ``--lr``
    (e.g. 1e-4) for fine-tuning than for pretraining.

    Returns:
        Dict with per-fold results and aggregate MAPE.
    """
    from siren.ml.bathymetry_training_data import build_all_samples, compute_sample_volume
    from siren.ml.bathymetry_benchmark import build_volume_estimates

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Building training samples...")
    samples = build_all_samples(grid_size=grid_size)
    if len(samples) < 2:
        logger.error("Need at least 2 samples for LOO, got %d", len(samples))
        return {"error": "insufficient samples", "n_samples": len(samples)}

    estimates = build_volume_estimates()
    est_by_name = {e.lake_name: e for e in estimates}

    # Sort by lake_id for reproducibility
    samples_sorted = sorted(samples, key=lambda s: s.lake_id)

    folds: list[dict] = []
    for i, test_sample in enumerate(samples_sorted):
        train_samples = [s for j, s in enumerate(samples_sorted) if j != i]

        # Build training tensors
        train_x = np.stack([s.to_input_target()[0] for s in train_samples], axis=0)
        train_y = np.stack([s.to_input_target()[1] for s in train_samples], axis=0)

        train_x_t = torch.from_numpy(train_x).to(device)
        train_y_t = torch.from_numpy(train_y).to(device)

        # Build test tensor
        test_x, test_y = test_sample.to_input_target()
        test_x_t = torch.from_numpy(test_x[np.newaxis]).to(device)

        # Fresh model for each fold (optionally from pretrained weights —
        # the transfer-learning path starts each fold from the synthetic-
        # basin initialisation rather than random weights)
        model = BathymetryUNet(
            in_channels=2,
            base_channels=base_channels,
            n_down=n_down,
        ).to(device)
        if pretrained_weights is not None:
            state = torch.load(
                str(pretrained_weights), map_location=device, weights_only=True
            )
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5,
        )

        # Train
        model.train()
        for epoch in range(epochs):
            # Mini-batch over training lakes
            indices = np.random.permutation(len(train_samples))
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                xb = train_x_t[batch_idx]
                yb = train_y_t[batch_idx]
                pred = model(xb)
                # Only compute loss inside the lake mask
                mask = xb[:, 1:2]  # (B, 1, H, W)
                loss = F.mse_loss(pred * mask, yb * mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Predict
        model.eval()
        with torch.no_grad():
            pred_norm = model(test_x_t).cpu().numpy()[0, 0]  # (H, W)

        # Denormalise
        dem = test_sample.dem.astype(np.float32)
        dem_min = float(dem[dem > 0].min()) if (dem > 0).any() else 0.0
        dem_max = float(dem.max())
        if dem_max > dem_min:
            pred_bed = pred_norm * (dem_max - dem_min) + dem_min
        else:
            pred_bed = np.full_like(pred_norm, test_sample.z_surface)

        # Compute predicted volume
        lake_pixels = test_sample.lake_mask > 0.5
        left, bottom, right, top = test_sample.bounds
        width_m = (right - left) * 111320 * np.cos(np.radians((top + bottom) / 2))
        height_m = (top - bottom) * 111320
        pixel_area = (width_m * height_m) / (grid_size * grid_size)

        pred_depths = np.maximum(0.0, test_sample.z_surface - pred_bed)
        pred_volume = float(pred_depths[lake_pixels].sum() * pixel_area)

        # Ground truth volumes
        sample_volume = compute_sample_volume(test_sample, pixel_area)
        est = est_by_name.get(test_sample.lake_name)
        published_volume = est.published_volume_m3 if est else None
        gt_volume = est.ground_truth_volume_m3 if est else sample_volume

        # APEs
        sample_ape = abs(pred_volume - sample_volume) / sample_volume if sample_volume > 0 else float("inf")
        published_ape = (
            abs(pred_volume - published_volume) / published_volume
            if published_volume and published_volume > 0
            else None
        )
        gt_ape = abs(pred_volume - gt_volume) / gt_volume if gt_volume > 0 else float("inf")

        fold = {
            "test_lake_id": test_sample.lake_id,
            "test_lake_name": test_sample.lake_name,
            "pred_volume_m3": pred_volume,
            "sample_volume_m3": sample_volume,
            "published_volume_m3": published_volume,
            "gt_volume_m3": gt_volume,
            "sample_ape": sample_ape,
            "published_ape": published_ape,
            "gt_ape": gt_ape,
        }
        folds.append(fold)
        logger.info(
            "Fold %d/%d: %s pred=%.2f MCM  sample=%.2f MCM  gt=%.2f MCM  "
            "sample_APE=%.1f%%  gt_APE=%.1f%%",
            i + 1, len(samples_sorted), test_sample.lake_name,
            pred_volume / 1e6, sample_volume / 1e6, gt_volume / 1e6,
            sample_ape * 100, gt_ape * 100,
        )

    # Aggregate
    sample_mapes = [f["sample_ape"] for f in folds if np.isfinite(f["sample_ape"])]
    gt_mapes = [f["gt_ape"] for f in folds if np.isfinite(f["gt_ape"])]
    published_mapes = [f["published_ape"] for f in folds if f["published_ape"] is not None]

    sample_mape = float(np.mean(sample_mapes)) if sample_mapes else float("inf")
    gt_mape = float(np.mean(gt_mapes)) if gt_mapes else float("inf")
    published_mape = float(np.mean(published_mapes)) if published_mapes else None

    result = {
        "n_lakes": len(samples_sorted),
        "n_folds": len(folds),
        "epochs": epochs,
        "grid_size": grid_size,
        "base_channels": base_channels,
        "n_down": n_down,
        "sample_mape": sample_mape,
        "gt_mape": gt_mape,
        "published_mape": published_mape,
        "gate_target_mape": 0.15,
        "sample_gate_passed": sample_mape < 0.15,
        "gt_gate_passed": gt_mape < 0.15,
        "pretrained_weights": str(pretrained_weights) if pretrained_weights else None,
        "folds": folds,
    }

    logger.info(
        "LOO Neural: %d lakes, sample MAPE=%.1f%%, gt MAPE=%.1f%%, "
        "published MAPE=%s, gate=%.0f%%",
        len(samples_sorted),
        sample_mape * 100,
        gt_mape * 100,
        f"{published_mape * 100:.1f}%" if published_mape else "N/A",
        15,
    )

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train neural bathymetry model")
    parser.add_argument("--synthetic", action="store_true",
                        help="Train on synthetic data (architecture testing only)")
    parser.add_argument("--pretrain-synthetic", action="store_true",
                        help="Pre-train on transfer-contract synthetic basins "
                             "(E2 transfer learning — initialisation for LOO fine-tuning)")
    parser.add_argument("--pretrain-weights", type=str, default=None,
                        help="Checkpoint to initialise each LOO fold from "
                             "(use with --train-real; e.g. the "
                             "bathymetry_unet_synth_pretrain.pt output of "
                             "--pretrain-synthetic)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run the LOO volume estimation benchmark (Huggel vs regression)")
    parser.add_argument("--train-real", action="store_true",
                        help="Run LOO training and evaluation with the neural model on real surveyed data")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to real bathymetry data (requires DEM — not yet available)")
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

    if args.benchmark:
        result = run_loo_benchmark()
        # Save benchmark report
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = CHECKPOINT_DIR / "bathymetry_loo_benchmark.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Benchmark report saved: %s", report_path)
        return 0

    if args.train_real:
        pretrained = Path(args.pretrain_weights) if args.pretrain_weights else None
        if pretrained is not None and not pretrained.exists():
            logger.error("Pretrained weights not found: %s", pretrained)
            return 1
        result = run_loo_neural(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            base_channels=args.base_channels,
            n_down=args.n_down,
            grid_size=args.grid_size,
            seed=args.seed,
            pretrained_weights=pretrained,
        )
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        report_name = (
            "bathymetry_loo_neural_transfer.json"
            if pretrained is not None
            else "bathymetry_loo_neural.json"
        )
        report_path = CHECKPOINT_DIR / report_name
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Neural LOO report saved: %s", report_path)
        return 0

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
    elif args.pretrain_synthetic:
        result = train_transfer_pretrain(
            model, n_samples=args.n_samples, grid_size=args.grid_size,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=device, seed=args.seed,
        )
        if args.save_name == "bathymetry_unet_v1.pt":
            args.save_name = "bathymetry_unet_synth_pretrain.pt"
    elif args.data_dir:
        # Real data training requires surrounding DEM terrain for each lake.
        # The surveyed bathymetry points (Zhang 2023 + Das 2025) provide
        # depth ground truth, but the BathymetryUNet input contract is
        # (DEM, lake_mask) → bed_elevation. Without DEMs for the surveyed
        # lakes, we cannot train the neural model on real data yet.
        #
        # The LOO benchmark (--benchmark) establishes the baseline:
        # Huggel MAPE ~76%, regression MAPE ~82%. The neural model must
        # beat this once DEM data is acquired.
        logger.error(
            "Real data training requires surrounding DEM terrain for each "
            "surveyed lake. The current dataset has depth points but no "
            "DEMs. Run --benchmark for the volume estimation baseline, or "
            "acquire DEMs for the 20 surveyed lakes first."
        )
        return 1
    else:
        parser.error(
            "Must specify --synthetic, --pretrain-synthetic, --benchmark, "
            "--train-real, or --data-dir"
        )

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
            "note": (
                "Gate evaluation requires DEM data for the 20 surveyed lakes. "
                "LOO benchmark baseline: Huggel MAPE ~76%, regression MAPE ~82%. "
                "The neural model must beat Huggel once DEMs are acquired."
            ),
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
