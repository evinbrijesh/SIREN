"""Multi-geometry + multi-season adapter fine-tune (v2).

Extends the single-pair Himalayan adapter to be pass-geometry and season
invariant. Trains on ALL available Imja SAR pairs with liquid-water
semantics:

  * unfrozen descending pairs (07-02/14, 08-19/09-12) — gold labels
  * unfrozen ascending pair (08-11/09-16) — gold labels
  * frozen descending pairs (11-09/21, 01-08/20) — all-zero labels
    (frozen lake = no liquid water)

The encoder stays frozen (low-level SAR features are transferable);
the bottleneck + decoder + head are fine-tuned to learn pass-invariant,
season-aware water boundaries.

Honesty notes:
  * Frozen pairs contribute NEGATIVE examples only (frozen lake → no
    liquid water). This teaches the model that ice ≠ water.
  * Ascending pair has a different incidence angle; including it teaches
    pass-geometry invariance.
  * Gold labels are AI-assisted preliminary annotations — human QA is
    still required before operational certification.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch

from siren.ml.engine import _detect_architecture
from siren.ml.losses import bce_loss, dice_loss
from siren.ml.model import WaterResUNet

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CKPT_IN = (
    REPO_ROOT
    / "models"
    / "checkpoints"
    / "water_resunet_kuro_siwo_full"
    / "water_resunet_6ch_kuro_siwo_v1.pt"
)
OUT_CKPT = (
    REPO_ROOT
    / "models"
    / "checkpoints"
    / "water_resunet_6ch_himalayan_adapter_multigeometry.pt"
)
REPORT_OUT = (
    REPO_ROOT
    / "models"
    / "checkpoints"
    / "adapter_multigeometry_report.json"
)

# Training pairs: (pair_name, t0_date, t1_date, season, label_source)
#   season: "unfrozen" | "frozen" | "frozen_shoulder"
#   label_source: "auto" | "gold" | "zero" | "inventory"
# Frozen pairs get all-zero labels (no liquid water). Eval pairs are held out.
# NOTE: eval pairs are monsoon_2025_desc (08-29→09-10), monsoon_2025_desc2
# (08-29→09-22), unfrozen_desc (08-19→09-12), monsoon_asc (08-11→09-16).
# No training pair may share a date with an eval pair.
TRAIN_PAIRS = [
    ("early_desc", "20260702", "20260714", "unfrozen", "inventory"),
    ("unfrozen_desc2_a", "20260702", "20260726", "unfrozen", "auto"),
    ("unfrozen_desc2_b", "20260714", "20260807", "unfrozen", "auto"),
    ("shoulder", "20251109", "20251121", "frozen", "zero"),
    ("winter", "20260108", "20260120", "frozen", "zero"),
    # 2025 ascending training pairs — teach pass-geometry invariance
    ("asc_2025_a", "20250902", "20250914", "unfrozen", "inventory"),
    ("asc_2025_b", "20250914", "20250926", "unfrozen", "inventory"),
]

# Auto labels for non-eval dates (SAR date -> GeoTIFF)
AUTO_LABELS = {
    "20260726": PROCESSED_DIR / "imja_autolabel_20260725.tif",
    "20260807": PROCESSED_DIR / "imja_autolabel_20260811.tif",
}

CHIP = 96
STRIDE = 48  # overlapping chips for more training data


def _sar_cache(date_str: str, tag: str) -> Path:
    return PROCESSED_DIR / f"imja_{tag}_{date_str}_sar_vv_vh_db.tif"


def _labels_on_sar_grid(label_path: Path, cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Sample a label GeoTIFF onto the SAR grid. Returns (water, valid)."""
    from siren.detect.sar import sar_grid_lonlat, sar_grid_sample

    ll = sar_grid_lonlat(str(cache_path))
    if ll is None:
        raise RuntimeError(f"no GCP geolocation in {cache_path}")
    lon, lat = ll
    lbl = sar_grid_sample(str(label_path), lon, lat, fill=255.0)
    return lbl == 1, lbl < 255


def _zero_labels(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """All-zero labels (frozen lake = no liquid water)."""
    with rasterio.open(cache_path) as d:
        h, w = d.height, d.width
    return np.zeros((h, w), dtype=bool), np.ones((h, w), dtype=bool)


def _extract_chips(
    tensor6: np.ndarray,
    water: np.ndarray,
    valid: np.ndarray,
    chip: int = CHIP,
    stride: int = STRIDE,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Extract overlapping chips centred on the lake + background negatives."""
    h, w = water.shape
    half = chip // 2
    xs, ys, vs = [], [], []

    # Lake-centred chips: find pixels where water > 0 or near the ROI centre
    # (Imja lake is at ~centre of the 250x250 ROI on the SAR grid)
    lake_rows, lake_cols = np.where(water)
    if len(lake_rows) > 0:
        # Use lake centroid + jittered positions
        cr, cc = int(lake_rows.mean()), int(lake_cols.mean())
        offsets = [(-half, -half), (0, -half), (half, -half),
                   (-half, 0), (0, 0), (half, 0),
                   (-half, half), (0, half), (half, half)]
        for dr, dc in offsets:
            r0, c0 = cr - half + dr, cc - half + dc
            if 0 <= r0 and r0 + chip <= h and 0 <= c0 and c0 + chip <= w:
                xs.append(tensor6[:, r0:r0 + chip, c0:c0 + chip])
                ys.append(water[r0:r0 + chip, c0:c0 + chip].astype(np.uint8))
                vs.append(valid[r0:r0 + chip, c0:c0 + chip].astype(np.float32))
    else:
        # No water (frozen) — still extract chips at the lake centre for negatives
        # Imja lake is at approximately row ~600, col ~1000 on the desc grid
        # For asc grid it's different — find via the label's valid region centroid
        vrows, vcols = np.where(valid)
        if len(vrows) > 0:
            cr, cc = int(vrows.mean()), int(vcols.mean())
            for dr in range(-2 * half, 3 * half, half):
                for dc in range(-2 * half, 3 * half, half):
                    r0, c0 = cr - half + dr, cc - half + dc
                    if 0 <= r0 and r0 + chip <= h and 0 <= c0 and c0 + chip <= w:
                        xs.append(tensor6[:, r0:r0 + chip, c0:c0 + chip])
                        ys.append(water[r0:r0 + chip, c0:c0 + chip].astype(np.uint8))
                        vs.append(valid[r0:r0 + chip, c0:c0 + chip].astype(np.float32))

    # Background negatives: strided chips with no water
    rng = np.random.RandomState(42)
    bg_count = 0
    for r0 in range(0, h - chip + 1, stride * 4):
        for c0 in range(0, w - chip + 1, stride * 4):
            if water[r0:r0 + chip, c0:c0 + chip].sum() == 0:
                if bg_count < 5:
                    xs.append(tensor6[:, r0:r0 + chip, c0:c0 + chip])
                    ys.append(water[r0:r0 + chip, c0:c0 + chip].astype(np.uint8))
                    vs.append(valid[r0:r0 + chip, c0:c0 + chip].astype(np.float32))
                    bg_count += 1

    return xs, ys, vs


def _inventory_labels(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Inventory-polygon labels on the SAR grid (weak positives)."""
    from scipy.ndimage import binary_dilation
    from siren.ml.himalayan_lake_dataset import (
        lake_grid_positions, load_lake_inventory, rasterize_lake_labels,
    )
    lakes = load_lake_inventory(min_elev_m=4000, min_area_km2=0.02)
    positions, _, _ = lake_grid_positions(lakes, str(cache_path))
    label = rasterize_lake_labels(lakes, positions, str(cache_path)) > 0
    # Dilate slightly so shoreline water pixels are included
    water = binary_dilation(label, iterations=2)
    valid = np.ones_like(water, dtype=bool)
    return water, valid


def build_training_tensors() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Build (x, y, v) training tensors from all SAR pairs."""
    from siren.ml.contract import build_kuro_siwo_tensor

    xs_all, ys_all, vs_all = [], [], []
    manifest = []

    for pair_name, t0, t1, season, label_src in TRAIN_PAIRS:
        tag = "asc" if pair_name.endswith("_asc") else "desc"
        cache_t0 = _sar_cache(t0, tag)
        cache_t1 = _sar_cache(t1, tag)
        if not cache_t0.exists() or not cache_t1.exists():
            logger.warning("missing SAR cache for %s (%s/%s)", pair_name, t0, t1)
            continue

        with rasterio.open(cache_t0) as d:
            pre_db = d.read().astype(np.float32)
        with rasterio.open(cache_t1) as d:
            post_db = d.read().astype(np.float32)
        tensor6 = build_kuro_siwo_tensor(pre_db[:2], post_db[:2])

        if label_src == "inventory":
            water, valid = _inventory_labels(cache_t1)
            logger.info("%s t1=%s: inventory label water=%d px", pair_name, t1, water.sum())
        elif label_src == "auto" and t1 in AUTO_LABELS:
            water, valid = _labels_on_sar_grid(AUTO_LABELS[t1], cache_t1)
            logger.info("%s t1=%s: auto label water=%d px", pair_name, t1, water.sum())
        else:
            water, valid = _zero_labels(cache_t1)
            logger.info("%s t1=%s: frozen — zero labels", pair_name, t1)

        xs, ys, vs = _extract_chips(tensor6, water, valid)
        for i, (x, y, v) in enumerate(zip(xs, ys, vs)):
            xs_all.append(x)
            ys_all.append(y)
            vs_all.append(v)
            manifest.append({
                "pair": pair_name, "t0": t0, "t1": t1, "season": season,
                "chip_id": f"{pair_name}_{i:03d}",
                "pos_px": int(y.sum()),
                "kind": "lake" if y.sum() > 0 else "background",
            })

    if not xs_all:
        raise RuntimeError("no training chips extracted")

    x = np.stack(xs_all).astype(np.float32)
    y = np.stack(ys_all).astype(np.uint8)
    v = np.stack(vs_all).astype(np.float32)
    logger.info("total chips=%d  pos_frac=%.4f", len(x), float(y.sum() / y.size))
    return x, y, v, manifest


def run_finetune(
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 8,
    threshold: float = 0.30,
    pos_weight_cap: float = 15.0,
    seed: int = 42,
    out_ckpt: Path | str = OUT_CKPT,
    report_out: Path | str = REPORT_OUT,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x, y, v, manifest = build_training_tensors()

    state = torch.load(str(CKPT_IN), map_location="cpu", weights_only=True)
    arch, in_ch, base = _detect_architecture(state)
    model = WaterResUNet(in_channels=in_ch, base_channels=base)
    model.load_state_dict(state)

    # Freeze encoder, fine-tune bottleneck + decoder + head
    frozen, trainable = [], []
    for name, p in model.named_parameters():
        if name.startswith("enc"):
            p.requires_grad = False
            frozen.append(name)
        else:
            trainable.append(name)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("frozen=%d tensors  trainable=%.1fM/%.1fM params", len(frozen), n_train / 1e6, n_total / 1e6)

    pos = float((y * v).sum())
    neg = float(((1 - y) * v).sum())
    pos_weight = torch.tensor([min(neg / max(pos, 1.0), pos_weight_cap)])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.float32))[:, None]
    vt = torch.from_numpy(v.astype(np.float32))[:, None]

    model.train()
    history = []
    for ep in range(epochs):
        perm = np.random.permutation(len(x))
        ep_loss = 0.0
        for i in range(0, len(perm), batch_size):
            j = perm[i:i + batch_size]
            logits = model(xt[j])
            loss = bce_loss(logits, yt[j], vt[j], pos_weight) + dice_loss(logits, yt[j], vt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss)
        history.append(round(ep_loss / max(len(perm) // batch_size, 1), 4))
        logger.info("epoch %d  loss=%.4f", ep, history[-1])

    out_ckpt = Path(out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)

    report = {
        "experiment": "multi-geometry + multi-season adapter fine-tune",
        "labels": {
            "positives": "gold liquid-water labels (unfrozen) + all-zero (frozen)",
            "negatives": "all non-water pixels incl. glacier/moraine/slope + frozen lake",
            "chips": len(x),
            "pos_pixel_frac": round(float(y.sum() / y.size), 5),
        },
        "training": {
            "epochs": epochs, "lr": lr,
            "frozen_tensors": len(frozen),
            "trainable_params": n_train,
            "total_params": n_total,
            "pos_weight": float(pos_weight),
            "loss_history": history,
        },
        "pairs": manifest,
        "checkpoint_out": str(out_ckpt),
        "limitations": [
            "Gold labels are AI-assisted preliminary — human QA required.",
            "Frozen pairs contribute negatives only (ice ≠ water).",
            "Ascending pair is a single geometry — pass invariance is partial.",
            "Shadow-only — not wired into runtime pending gate on held-out data.",
        ],
    }
    Path(report_out).write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-ckpt", default=str(OUT_CKPT))
    ap.add_argument("--report-out", default=str(REPORT_OUT))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = run_finetune(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        out_ckpt=args.out_ckpt, report_out=args.report_out,
    )
    print(json.dumps(report["training"], indent=1))


if __name__ == "__main__":
    main()
