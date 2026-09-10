"""Evaluate 2-channel WaterResUNet with and without HAND post-filter (Level 2 Phase 2).

This script loads the trained 2-channel baseline checkpoint, runs inference
on the event-holdout test set (Pakistan/Somalia), and compares:

  1. Raw 2-channel predictions (no post-filter)
  2. 2-channel + deterministic HAND post-filter (stage_threshold_m)

The HAND post-filter zeros out water predictions at locations where the
Height Above Nearest Drainage exceeds the maximum plausible flood surge
stage, eliminating false positives at impossible elevations.

This is the deterministic hydraulic filter approach recommended by the
Level 2 analysis: instead of forcing the neural network to learn fluid
dynamics from 4 input channels, let the vision model handle backscatter
segmentation while hydraulic physics enforces elevation limits.

Usage:
    python -m siren.ml.eval_hand_filter [--stage-threshold 15.0]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"


def main():
    parser = argparse.ArgumentParser(description="Evaluate HAND post-filter on 2ch baseline")
    parser.add_argument("--stage-threshold", type=float, default=15.0,
                        help="HAND stage threshold in metres (default 15)")
    parser.add_argument("--channel-threshold", type=float, default=100.0,
                        help="Flow accumulation threshold for channel identification")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--model", default="2ch", choices=["2ch", "4ch"],
                        help="Which checkpoint to evaluate: 2ch baseline or 4ch terrain-aware")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import torch
    from torch.utils.data import DataLoader

    from siren.ml.dataset import WaterSegmentationDataset
    from siren.ml.model import WaterResUNet
    from siren.ml.train_water_resunet import evaluate_model, iou_score, collate_fn
    from siren.geo.hand import compute_hand_from_array, apply_hand_filter
    from siren.ml.dem_fetch import (
        fetch_dem_for_bounds, coregister_dem_to_grid, pixel_size_m_from_transform,
    )
    import rasterio

    # Device
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"
    logger.info("Device: %s", device)

    # Load checkpoint based on model selection
    if args.model == "4ch":
        ckpt_path = CHECKPOINT_DIR / "water_resunet_4ch_v1.pt"
        in_channels = 4
        use_copernicus_dem = True
        model_label = "4-channel (VV, VH, DEM, Slope)"
    else:
        ckpt_path = CHECKPOINT_DIR / "water_resunet_2ch_baseline.pt"
        in_channels = 2
        use_copernicus_dem = False
        model_label = "2-channel (VV, VH)"

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = WaterResUNet(in_channels=in_channels, base_channels=state.get("base_channels", 32)).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    logger.info("Loaded %s checkpoint (test IoU from training: %.4f)", model_label, state.get("test_iou", 0))

    # Build test dataset
    test_ds = WaterSegmentationDataset(
        split="test", strategy="event_holdout", use_copernicus_dem=use_copernicus_dem,
    )
    logger.info("Test chips: %d (events: %s)", len(test_ds), sorted(test_ds.events()))

    # Evaluate raw predictions (no HAND filter)
    logger.info("Evaluating raw %s predictions (no HAND filter)...", model_label)
    raw_ious = []
    raw_tp = raw_fp = raw_fn = 0
    per_chip_raw = []

    # Evaluate with HAND filter
    logger.info("Evaluating %s + HAND post-filter (threshold=%.1fm)...", model_label, args.stage_threshold)
    filtered_ious = []
    filtered_tp = filtered_fp = filtered_fn = 0
    per_chip_filtered = []
    chips_filtered_count = 0

    t0 = time.time()
    with torch.no_grad():
        for idx in range(len(test_ds)):
            item = test_ds[idx]
            sar = item["sar"].unsqueeze(0).to(device)  # (1, 2, H, W)
            water = item["water"].numpy()  # (H, W)
            valid = item["valid"].numpy()  # (H, W)
            chip_id = item["chip_id"]
            event = item["event"]

            # Model prediction
            logits = model(sar)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)
            pred_raw = (probs > 0.5).astype(np.float32)

            # Compute HAND for this chip
            # Need to re-open the chip to get bounds for DEM fetch
            s1_path = test_ds.rows[idx]
            from siren.ml.dataset import SEN1FLOODS11_ROOT
            s1_full_path = SEN1FLOODS11_ROOT / test_ds.rows[idx][2] / "S1" / test_ds.rows[idx][0]
            with rasterio.open(str(s1_full_path)) as src:
                chip_bounds = src.bounds
                chip_crs = str(src.crs)
                chip_transform = src.transform
                chip_shape = (src.height, src.width)
                center_lat = (chip_bounds.top + chip_bounds.bottom) / 2

            dem_paths = fetch_dem_for_bounds(
                chip_bounds.left, chip_bounds.bottom,
                chip_bounds.right, chip_bounds.top,
            )
            if dem_paths:
                dem = coregister_dem_to_grid(
                    dem_paths, chip_crs, chip_transform, chip_shape,
                )
                dx_m, dy_m = pixel_size_m_from_transform(
                    chip_transform, chip_crs, center_lat,
                )
                px = (dx_m + dy_m) / 2
                hand = compute_hand_from_array(
                    dem, pixel_size_m=px, channel_threshold=args.channel_threshold,
                )
                # Apply HAND filter
                pred_filtered = apply_hand_filter(
                    probs, hand, stage_threshold_m=args.stage_threshold,
                )
                pred_filtered = (pred_filtered > 0.5).astype(np.float32)
                n_filtered = int((pred_raw > 0).sum() - (pred_filtered > 0).sum())
                if n_filtered > 0:
                    chips_filtered_count += 1
            else:
                pred_filtered = pred_raw
                n_filtered = 0

            # Compute IoU for raw and filtered
            valid_bool = valid.astype(bool)
            raw_iou = iou_score(pred_raw, water, valid)
            filt_iou = iou_score(pred_filtered, water, valid)
            raw_ious.append(raw_iou)
            filtered_ious.append(filt_iou)
            per_chip_raw.append((chip_id, event, raw_iou))
            per_chip_filtered.append((chip_id, event, filt_iou, n_filtered))

            # Accumulate global TP/FP/FN
            raw_p = pred_raw.astype(bool) & valid_bool
            raw_t = water.astype(bool) & valid_bool
            raw_tp += (raw_p & raw_t).sum()
            raw_fp += (raw_p & ~raw_t).sum()
            raw_fn += (~raw_p & raw_t).sum()

            filt_p = pred_filtered.astype(bool) & valid_bool
            filt_t = water.astype(bool) & valid_bool
            filtered_tp += (filt_p & filt_t).sum()
            filtered_fp += (filt_p & ~filt_t).sum()
            filtered_fn += (~filt_p & filt_t).sum()

            if (idx + 1) % 10 == 0:
                logger.info("  Processed %d/%d chips...", idx + 1, len(test_ds))

    eval_time = time.time() - t0

    # Compute global IoU
    raw_global_iou = float(raw_tp) / float(raw_tp + raw_fp + raw_fn) if (raw_tp + raw_fp + raw_fn) > 0 else 0
    filt_global_iou = float(filtered_tp) / float(filtered_tp + filtered_fp + filtered_fn) if (filtered_tp + filtered_fp + filtered_fn) > 0 else 0

    # Compute per-event IoU
    events = sorted(set(e for _, e, *_ in per_chip_raw))
    per_event_raw = {}
    per_event_filt = {}
    for event in events:
        raw_vals = [iou for _, e, iou in per_chip_raw if e == event]
        filt_vals = [iou for _, e, iou, _ in per_chip_filtered if e == event]
        per_event_raw[event] = float(np.mean(raw_vals)) if raw_vals else 0
        per_event_filt[event] = float(np.mean(filt_vals)) if filt_vals else 0

    # Report
    iou_delta = filt_global_iou - raw_global_iou
    print()
    print("=" * 70)
    print(f"Level 2 Phase 2: HAND Post-Filter Evaluation ({model_label})")
    print("=" * 70)
    print(f"  Test chips: {len(test_ds)} (events: {sorted(test_ds.events())})")
    print(f"  HAND stage threshold: {args.stage_threshold}m")
    print(f"  Channel threshold: {args.channel_threshold}")
    print(f"  Evaluation time: {eval_time:.1f}s")
    print(f"  Chips with predictions removed: {chips_filtered_count}/{len(test_ds)}")
    print()
    print(f"  Raw {args.model} IoU (global):     {raw_global_iou:.4f}")
    print(f"  {args.model} + HAND IoU (global):   {filt_global_iou:.4f}")
    print(f"  IoU delta:                 {iou_delta:+.4f}")
    print()
    print(f"  Raw mean chip IoU:         {np.mean(raw_ious):.4f}")
    print(f"  Filtered mean chip IoU:    {np.mean(filtered_ious):.4f}")
    print()
    print("  Per-event mean chip IoU:")
    for event in events:
        r = per_event_raw[event]
        f = per_event_filt[event]
        d = f - r
        print(f"    {event:15s}: raw={r:.4f}, filtered={f:.4f}, delta={d:+.4f}")
    print()
    print(f"  Raw TP/FP/FN:     {raw_tp}/{raw_fp}/{raw_fn}")
    print(f"  Filtered TP/FP/FN: {filtered_tp}/{filtered_fp}/{filtered_fn}")
    print(f"  FP removed: {raw_fp - filtered_fp}")
    print(f"  TP removed: {raw_tp - filtered_tp} (should be ~0)")
    print("=" * 70)

    # Save results
    results_path = CHECKPOINT_DIR / f"hand_filter_eval_{args.model}.json"
    results = {
        "model": args.model,
        "stage_threshold_m": args.stage_threshold,
        "channel_threshold": args.channel_threshold,
        "test_chips": len(test_ds),
        "chips_filtered": chips_filtered_count,
        "raw_global_iou": raw_global_iou,
        "filtered_global_iou": filt_global_iou,
        "iou_delta": iou_delta,
        "raw_mean_chip_iou": float(np.mean(raw_ious)),
        "filtered_mean_chip_iou": float(np.mean(filtered_ious)),
        "per_event_raw": per_event_raw,
        "per_event_filtered": per_event_filt,
        "raw_tp": int(raw_tp), "raw_fp": int(raw_fp), "raw_fn": int(raw_fn),
        "filtered_tp": int(filtered_tp), "filtered_fp": int(filtered_fp), "filtered_fn": int(filtered_fn),
        "fp_removed": int(raw_fp - filtered_fp),
        "tp_removed": int(raw_tp - filtered_tp),
        "eval_time_s": round(eval_time, 1),
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
