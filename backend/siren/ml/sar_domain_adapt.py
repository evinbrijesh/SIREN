"""Weak-label domain adaptation of the SAR water model on the Imja scene.

The gate-passed Kuro Siwo checkpoint is out-of-distribution on the high-
Himalaya scene (61k raw detections vs ~2.5k rule px; ~6% geographic
overlap). No labelled high-altitude SAR dataset exists on disk (audit:
docs/DATA_LICENSES.md), so this experiment fine-tunes the checkpoint on
*weak labels* assembled for the descending Imja pair:

    label = 1  verified lake footprints (union of obs-*_expansion_mask.tif
               + baseline_water_mask.tif) ∪ SCL class 6 (water)
    label = 0  SCL clear-land classes {4,5,7} and snow/ice {11} outside
               lake vicinity — the hard-negative population the model
               currently confuses for water
    ignore     outside AOI, SCL cloud/shadow/unclassified, no SCL coverage

SCL labels come from the *optical* sensor — an independent source from
the deterministic SAR rule mask used for evaluation. Residual
circularity: the lake positives derive from the same scenario masks the
rule path uses, so post-finetune overlap gains over the lake itself are
expected by construction. The informative metrics are the *spread*
metrics — raw positive count and detections on glacier/steep terrain —
which are driven by the independent SCL negatives.

Protocol:
    1. Build labels on the GCP-geolocated SAR grid.
    2. Extract 128x128 chips over the AOI, block-split train/val.
    3. Fine-tune WaterResUNet-6ch from the gate-passed checkpoint
       (low LR, few epochs, masked BCE + Dice).
    4. Evaluate full-scene before/after: raw px, terrain-gated px,
       geographic overlap vs the rule mask.

Usage:
    python -m siren.ml.sar_domain_adapt [--epochs 12] [--lr 1e-4]
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"

SAR_T0 = PROCESSED_DIR / "imja_desc_20260702_sar_vv_vh_db.tif"
SAR_T1 = PROCESSED_DIR / "imja_desc_20260714_sar_vv_vh_db.tif"
AOI_GEOJSON = DATA_DIR / "assets" / "dudh_koshi_aoi.geojson"
RGI_SHP = (
    "/vsizip/"
    + str(
        DATA_DIR
        / "datasets"
        / "RGI2000-v7.0-G-15_south_asia_east.zip"
        / "RGI2000-v7.0-G-15_south_asia_east.shp"
    )
)
# Clear 2025-11-22 scene — 84% clear over the AOI, best SCL quality.
S2_NOV = DATA_DIR / "raw" / (
    "S2C_MSIL2A_20251122T045131_N0511_R076_T45RVL_20251122T083010.SAFE.zip"
)
RULE_MASK = PROCESSED_DIR / "obs-003_expansion_mask.tif"

CKPT = (
    CHECKPOINT_DIR
    / "water_resunet_kuro_siwo_full"
    / "water_resunet_6ch_kuro_siwo_v1_best.pt"
)
OUT_CKPT = CHECKPOINT_DIR / "water_resunet_6ch_imja_weaklabel.pt"

SCL_CLEAR_LAND = {4, 5, 7}   # vegetation, bare soil, unclassified-land
SCL_WATER = 6
SCL_SNOW_ICE = 11
CHIP = 96
STRIDE = 48


# --------------------------------------------------------------------------- #
# Label assembly on the SAR grid
# --------------------------------------------------------------------------- #

def _scl_to_tmp_tif(s2_zip: Path, lon: np.ndarray, lat: np.ndarray) -> Path:
    """Extract the 20 m SCL band onto an EPSG:4326 grid covering the SAR
    pixel geolocations, written to a temp GeoTIFF for sar_grid_sample."""
    from siren.preprocess.s2_optical import _find_band_path

    west, east = float(np.nanmin(lon)), float(np.nanmax(lon))
    south, north = float(np.nanmin(lat)), float(np.nanmax(lat))
    # ~90 m SAR pitch -> 0.001 deg cells, generous padding
    w = int((east - west) / 0.001) + 2
    h = int((north - south) / 0.001) + 2
    dst_transform = from_bounds(west, south, east, north, w, h)
    out = np.zeros((h, w), dtype=np.uint8)

    with zipfile.ZipFile(str(s2_zip)) as zf:
        scl_path = _find_band_path(zf, "SCL", "20m")
        with zf.open(scl_path) as f:
            with rasterio.open(f) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=out,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest,
                )

    tmp = Path(tempfile.mkstemp(suffix="_scl.tif")[1])
    with rasterio.open(
        tmp, "w", driver="GTiff", width=w, height=h, count=1,
        dtype="uint8", crs="EPSG:4326", transform=dst_transform,
    ) as dst:
        dst.write(out, 1)
    return tmp


def build_weak_labels(
    sar_path: Path = SAR_T1,
    s2_zip: Path = S2_NOV,
) -> dict[str, np.ndarray]:
    """Assemble weak labels on the SAR grid.

    Returns dict with:
      label  int8 (H, W): 1 water, 0 non-water, -1 ignore
      lon/lat (H, W) geolocation grids (for later eval reuse)
      aoi, lake, glac masks for diagnostics
    """
    from scipy.ndimage import binary_dilation
    from siren.detect.sar import (
        sar_grid_lonlat,
        sar_grid_polygon_mask,
        sar_grid_sample,
    )

    ll = sar_grid_lonlat(str(sar_path))
    if ll is None:
        raise RuntimeError(f"no GCP geolocation in {sar_path}")
    lon, lat = ll

    aoi = sar_grid_polygon_mask(str(AOI_GEOJSON), lon, lat)
    glac = sar_grid_polygon_mask(RGI_SHP, lon, lat)

    lake = np.zeros(aoi.shape, dtype=bool)
    lake |= sar_grid_sample(
        str(PROCESSED_DIR / "baseline_water_mask.tif"), lon, lat
    ) > 0
    for p in sorted(PROCESSED_DIR.glob("obs-*_expansion_mask.tif")):
        lake |= sar_grid_sample(str(p), lon, lat) > 0
    lake_vic = binary_dilation(lake, iterations=3) if lake.any() else lake

    scl_tif = _scl_to_tmp_tif(s2_zip, lon, lat)
    try:
        scl = sar_grid_sample(str(scl_tif), lon, lat, fill=0).astype(np.uint8)
    finally:
        scl_tif.unlink(missing_ok=True)

    label = np.full(aoi.shape, -1, dtype=np.int8)
    inside = aoi & (scl > 0)
    label[inside & np.isin(scl, list(SCL_CLEAR_LAND) + [SCL_SNOW_ICE])] = 0
    label[inside & (scl == SCL_WATER)] = 1
    label[lake] = 1                       # verified footprints win
    label[lake_vic & (scl == SCL_SNOW_ICE)] = 1  # frozen lake surface
    label[glac & ~lake_vic & (label == 1)] = 0   # glacier ≠ water off-lake
    label[~aoi] = -1

    return {"label": label, "lon": lon, "lat": lat,
            "aoi": aoi, "lake": lake, "glac": glac, "scl": scl}


# --------------------------------------------------------------------------- #
# Chip extraction
# --------------------------------------------------------------------------- #

def extract_chips(
    tensor6: np.ndarray,
    label: np.ndarray,
    aoi: np.ndarray,
    chip: int = CHIP,
    stride: int = STRIDE,
    min_labelled: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile the AOI bounding box into (6, C, C) chips + label/valid maps.

    Returns (x, y, v): float32 chips, int8 labels, float32 valid masks.
    """
    ys, xs = np.where(aoi)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    x_l, y_l, v_l = [], [], []
    for r in range(r0, max(r1 - chip + 1, r0 + 1), stride):
        for c in range(c0, max(c1 - chip + 1, c0 + 1), stride):
            y = label[r:r + chip, c:c + chip]
            if y.shape != (chip, chip):
                continue
            v = (y >= 0).astype(np.float32)
            if v.mean() < min_labelled:
                continue
            x_l.append(tensor6[:, r:r + chip, c:c + chip])
            y_l.append(np.clip(y, 0, 1).astype(np.float32))
            v_l.append(v)
    return (
        np.stack(x_l) if x_l else np.empty((0, 6, chip, chip), np.float32),
        np.stack(y_l) if y_l else np.empty((0, chip, chip), np.float32),
        np.stack(v_l) if v_l else np.empty((0, chip, chip), np.float32),
    )


# --------------------------------------------------------------------------- #
# Fine-tune + eval
# --------------------------------------------------------------------------- #

def evaluate_scene(model, pre_db, post_db, threshold, lon, lat, aoi, glac, lake_vic):
    """Full-scene probability map + the same metrics the pipeline reports."""
    import torch
    from siren.ml.contract import build_kuro_siwo_tensor
    from siren.detect.sar import sar_grid_sample

    def prob_of(pre, post):
        t6 = build_kuro_siwo_tensor(pre, post)
        # Pad to a multiple of 16 for the UNet skip connections (same as
        # the engine's _pad_to_multiple), crop back after inference.
        h, w = t6.shape[1:]
        padded = np.pad(t6, ((0, 0), (0, (16 - h % 16) % 16), (0, (16 - w % 16) % 16)))
        with torch.no_grad():
            logits = model(torch.from_numpy(padded)[None]).cpu().numpy()[0, 0]
        logits = logits[:h, :w]
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))

    # Match the runtime shadow-mask path exactly: predict_change_mask is
    # water_t1 & ~water_t0 (expansion only), where t0 is evaluated with
    # itself (Δσ⁰ = 0) under the multi-temporal contract.
    p_t1 = prob_of(pre_db, post_db)
    p_t0 = prob_of(pre_db, pre_db)
    ml = (p_t1 >= threshold) & (p_t0 < threshold)

    gated = ml & aoi & ~(glac & ~lake_vic)
    rule = sar_grid_sample(str(RULE_MASK), lon, lat) > 0
    overlap = int((gated & rule).sum())
    rule_px = int(rule.sum())
    return {
        "raw_px": int(ml.sum()),
        "water_extent_px": int((p_t1 >= threshold).sum()),
        "water_extent_on_glacier_px": int(
            ((p_t1 >= threshold) & glac & ~lake_vic).sum()
        ),
        "gated_px": int(gated.sum()),
        "on_glacier_px": int((ml & glac & ~lake_vic).sum()),
        "rule_overlap_px": overlap,
        "rule_overlap_pct": round(overlap / rule_px * 100, 1) if rule_px else 0.0,
    }


def run_experiment(
    epochs: int = 12,
    lr: float = 1e-4,
    batch_size: int = 8,
    threshold: float = 0.30,
    pos_weight_cap: float = 20.0,
    seed: int = 42,
) -> dict:
    import torch
    from scipy.ndimage import binary_dilation

    from siren.ml.contract import build_kuro_siwo_tensor
    from siren.ml.engine import _detect_architecture
    from siren.ml.losses import bce_loss, dice_loss
    from siren.ml.metrics import metrics_from_counts, water_confusion_counts
    from siren.ml.model import WaterResUNet

    torch.manual_seed(seed)
    np.random.seed(seed)

    with rasterio.open(SAR_T0) as d:
        pre_db = d.read().astype(np.float32)
    with rasterio.open(SAR_T1) as d:
        post_db = d.read().astype(np.float32)

    labels = build_weak_labels(SAR_T1)
    label, lon, lat, aoi = labels["label"], labels["lon"], labels["lat"], labels["aoi"]
    glac, lake = labels["glac"], labels["lake"]
    lake_vic = binary_dilation(lake, iterations=3)

    t6 = build_kuro_siwo_tensor(pre_db, post_db)
    x, y, v = extract_chips(t6, label, aoi)
    if len(x) == 0:
        raise RuntimeError("no labelled chips extracted")
    logger.info(
        "chips=%d  labelled_frac=%.2f  pos_frac=%.4f",
        len(x), float((v > 0).mean()), float((y * v).sum() / v.sum()),
    )

    # Block split: every 5th row-block of chips -> val (coarse spatial sep)
    n_rows = int(np.sqrt(len(x)))
    idx = np.arange(len(x))
    val_mask = (idx // max(n_rows, 1)) % 5 == 4
    tr, va = ~val_mask, val_mask
    if va.sum() == 0:
        va = idx % 5 == 0
        tr = ~va

    state = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    arch, in_ch, base = _detect_architecture(state)
    model = WaterResUNet(in_channels=in_ch, base_channels=base)
    model.load_state_dict(state)

    # ----- before -----
    model.eval()
    before = evaluate_scene(model, pre_db, post_db, threshold, lon, lat, aoi, glac, lake_vic)

    # ----- fine-tune -----
    pos = float((y[tr] * v[tr]).sum())
    neg = float(((1 - y[tr]) * v[tr]).sum())
    pos_weight = torch.tensor([min(neg / max(pos, 1.0), pos_weight_cap)])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(x[tr]); yt = torch.from_numpy(y[tr])[:, None]
    vt = torch.from_numpy(v[tr])[:, None]

    model.train()
    history = []
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        ep_loss = 0.0
        for i in range(0, len(xt), batch_size):
            j = perm[i:i + batch_size]
            logits = model(xt[j])
            loss = bce_loss(logits, yt[j], vt[j], pos_weight) + dice_loss(
                logits, yt[j], vt[j]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss)
        history.append(round(ep_loss / max(len(xt) // batch_size, 1), 4))
        logger.info("epoch %d  loss=%.4f", ep, history[-1])

    # ----- val IoU (weak-label agreement, sanity only) -----
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for i in range(0, int(va.sum()), batch_size):
            xb = torch.from_numpy(x[va][i:i + batch_size])
            yb = y[va][i:i + batch_size]
            vb = v[va][i:i + batch_size]
            p = (torch.sigmoid(model(xb))[:, 0].numpy() >= threshold)
            c = water_confusion_counts(
                p.astype(np.uint8), yb.astype(np.uint8), vb.astype(np.uint8)
            )
            tp += c[0]; fp += c[1]; fn += c[2]; tn += c[3]
    val_metrics = metrics_from_counts(tp, fp, fn, tn)

    # ----- after -----
    after = evaluate_scene(model, pre_db, post_db, threshold, lon, lat, aoi, glac, lake_vic)

    torch.save(model.state_dict(), OUT_CKPT)

    report = {
        "experiment": "weak-label domain adaptation on Imja descending pair",
        "labels": {
            "positives": "verified lake masks + SCL water (independent optical)",
            "negatives": "SCL clear-land + snow/ice off-lake (independent optical)",
            "ignore": "outside AOI, SCL cloud/shadow, glacier is negative off-lake",
            "chips": int(len(x)),
            "train_chips": int(tr.sum()),
            "val_chips": int(va.sum()),
            "pos_pixel_frac": round(float((y * v).sum() / v.sum()), 5),
        },
        "training": {"epochs": epochs, "lr": lr, "pos_weight": float(pos_weight),
                     "loss_history": history},
        "val_weaklabel_metrics": val_metrics,
        "full_scene_before": before,
        "full_scene_after": after,
        "checkpoint_out": str(OUT_CKPT),
        "limitations": [
            "Single scene pair — risk of memorising the label layout rather "
            "than learning a transferable Himalayan water signature.",
            "Lake positives derive from the same scenario masks the rule "
            "path uses — overlap gains over the lake are circular by "
            "construction; the honest metrics are raw_px / on_glacier_px "
            "spread, driven by independent SCL negatives.",
            "SCL at 20 m resampled to ~90 m SAR pitch is noisy at water "
            "edges; the Nov scene's lake is frozen (SCL snow/ice), so the "
            "in-lake positives lean on the verified footprints.",
        ],
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--pos-weight-cap", type=float, default=20.0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = run_experiment(
        args.epochs, args.lr, args.batch_size, args.threshold,
        args.pos_weight_cap,
    )
    out = args.out or CHECKPOINT_DIR / "sar_domain_adapt_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("val_weaklabel_metrics", "full_scene_before",
                       "full_scene_after")}, indent=2))
    logger.info("report: %s", out)


if __name__ == "__main__":
    main()
