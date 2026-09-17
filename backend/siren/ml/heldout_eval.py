"""Held-out evaluation of the high-altitude lake adapter on winter S1 pairs.

ADR-013 honesty note: every prior adapter metric was *in-scene* — the
model fine-tuned on lake chips from the 2026-07-02/07-14 descending pair
and was evaluated on that same pair. The spatial north–south chip split
prevented neighbour leakage but not seasonal/acquisition leakage: one
season (monsoon), one satellite (S1D), one pass geometry.

This script evaluates the base Kuro Siwo checkpoint and the Himalayan
adapter on INDEPENDENT descending pairs:

  pair          dates                 conditions
  ----          -----                 ----------
  shoulder      2025-11-09/2025-11-21 cold season, lakes still largely
                                      open water — recall stays valid
  winter        2026-01-08/2026-01-20 deep winter, Imja frozen + max
                                      snow — stress test for false
                                      positives (recall expected to
                                      drop; a frozen lake is NOT water)

Both pairs are Sentinel-1A (the July pair was S1D), so this also tests
cross-sensor generalisation.

Metrics per pair per checkpoint (same machinery as
``sar_domain_adapt.evaluate_scene``, extended with inventory recall):
  - expansion/water-extent pixel counts, terrain-gated counts
  - water/extent on glacier (the domain-shift signature)
  - recall over verified inventory lake polygons at t1 and t0
    (weak positives — polygons are 2022–2024 median outlines)
  - overlap vs the deterministic scenario masks (caveated: scenario
    masks encode a simulated July event, not winter truth)

Usage:
    python -m siren.ml.heldout_eval [--pair shoulder|winter|both]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"

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

PAIRS = {
    "shoulder": ("20251109", "20251121"),
    "winter": ("20260108", "20260120"),
}

def _checkpoints() -> dict[str, Path]:
    """Base model + every Himalayan adapter variant on disk — re-running
    the eval after a new fine-tune compares all of them automatically."""
    ckpts = {
        "kuro_siwo_base": CHECKPOINT_DIR
        / "water_resunet_kuro_siwo_full"
        / "water_resunet_6ch_kuro_siwo_v1.pt",
    }
    for p in sorted(
        CHECKPOINT_DIR.glob("water_resunet_6ch_himalayan_adapter*.pt")
    ):
        name = p.stem.replace("water_resunet_6ch_", "")
        ckpts[name] = p
    return ckpts

REPORT_OUT = CHECKPOINT_DIR / "heldout_eval_report.json"

# Imja lake — inventory centroid ~ (86.9282°E, 27.8983°N); matched by
# nearest centroid so the report can isolate it from the other ~1,400
# in-swath lakes.
IMJA_LON, IMJA_LAT = 86.9282, 27.8983


def _find_safe(date_str: str) -> Path:
    hits = sorted(RAW_DIR.glob(f"S1*_IW_GRDH_*_{date_str}T00*.SAFE.zip"))
    if not hits:
        raise FileNotFoundError(
            f"no descending S1 SAFE archive for {date_str} in {RAW_DIR}"
        )
    return hits[0]


def calibrated_cache(date_str: str) -> Path:
    """Calibrate a SAFE archive to the (2, H, W) sigma0-dB cache used by
    the July pair — same function, same GCP persistence."""
    from siren.preprocess.sar_calibrate import extract_and_cache_vv_vh_db

    cache = PROCESSED_DIR / f"imja_desc_{date_str}_sar_vv_vh_db.tif"
    extract_and_cache_vv_vh_db(str(_find_safe(date_str)), cache)
    return cache


def _scene_masks(sar_path: Path) -> dict:
    """AOI / glacier / lake / inventory-polygon masks on the SAR grid."""
    from scipy.ndimage import binary_dilation

    from siren.detect.sar import (
        sar_grid_lonlat,
        sar_grid_polygon_mask,
        sar_grid_sample,
    )
    from siren.ml.himalayan_lake_dataset import (
        lake_grid_positions,
        load_lake_inventory,
        rasterize_lake_labels,
    )

    ll = sar_grid_lonlat(str(sar_path))
    if ll is None:
        raise RuntimeError(f"no GCP geolocation in {sar_path}")
    lon, lat = ll

    aoi = sar_grid_polygon_mask(str(AOI_GEOJSON), lon, lat)
    glac = sar_grid_polygon_mask(RGI_SHP, lon, lat)

    lake = np.zeros(aoi.shape, dtype=bool)
    bm = PROCESSED_DIR / "baseline_water_mask.tif"
    if bm.exists():
        lake |= sar_grid_sample(str(bm), lon, lat) > 0
    for p in sorted(PROCESSED_DIR.glob("obs-*_expansion_mask.tif")):
        lake |= sar_grid_sample(str(p), lon, lat) > 0
    lake_vic = binary_dilation(lake, iterations=3) if lake.any() else lake

    lakes = load_lake_inventory(min_elev_m=4000, min_area_km2=0.02)
    positions, _, _ = lake_grid_positions(lakes, str(sar_path))
    inventory = rasterize_lake_labels(lakes, positions, str(sar_path)) > 0

    # Per-area-bin inventory rasters — the tarn-vs-large-lake recall
    # split the stratified-retraining experiment is designed to move.
    from siren.ml.lake_adapter_finetune import AREA_BINS_KM2, AREA_BIN_NAMES
    pos_idx = np.array([p[0] for p in positions], dtype=np.int64)
    areas = lakes["area_km2"].astype(float).to_numpy()[pos_idx]
    inventory_bins = {}
    for b, name in enumerate(AREA_BIN_NAMES):
        lo = 0.0 if b == 0 else AREA_BINS_KM2[b - 1]
        hi = AREA_BINS_KM2[b] if b < len(AREA_BINS_KM2) else np.inf
        sel = [p for p, a in zip(positions, areas) if lo <= a < hi]
        inventory_bins[name] = (
            rasterize_lake_labels(lakes, sel, str(sar_path)) > 0
            if sel else np.zeros(aoi.shape, dtype=bool)
        )

    # Imja-specific raster: nearest centroid to the published position.
    d2 = (lakes["Longitude"].astype(float) - IMJA_LON) ** 2 + (
        lakes["Latitude"].astype(float) - IMJA_LAT
    ) ** 2
    imja_idx = int(np.argmin(d2.to_numpy()))
    imja_pos = [p for p in positions if p[0] == imja_idx]
    imja = (
        rasterize_lake_labels(lakes, imja_pos, str(sar_path)) > 0
        if imja_pos
        else np.zeros(aoi.shape, dtype=bool)
    )

    return {
        "lon": lon, "lat": lat, "aoi": aoi, "glac": glac,
        "lake_vic": lake_vic, "inventory": inventory, "imja": imja,
        "inventory_bins": inventory_bins,
        "n_inventory_lakes": len(positions),
    }


def _prob_map(model, pre_db: np.ndarray, post_db: np.ndarray) -> np.ndarray:
    import torch

    from siren.ml.contract import build_kuro_siwo_tensor

    t6 = build_kuro_siwo_tensor(pre_db, post_db)
    h, w = t6.shape[1:]
    padded = np.pad(
        t6, ((0, 0), (0, (16 - h % 16) % 16), (0, (16 - w % 16) % 16))
    )
    with torch.no_grad():
        logits = model(torch.from_numpy(padded)[None]).cpu().numpy()[0, 0]
    logits = logits[:h, :w]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def _recall(prob: np.ndarray, mask: np.ndarray, threshold: float) -> float | None:
    n = int(mask.sum())
    if n == 0:
        return None
    return round(float(((prob >= threshold) & mask).sum()) / n, 4)


def evaluate_checkpoint(
    ckpt_path: Path,
    pre_db: np.ndarray,
    post_db: np.ndarray,
    masks: dict,
    threshold: float = 0.30,
) -> dict:
    import torch

    from siren.detect.sar import sar_grid_sample
    from siren.ml.engine import _detect_architecture
    from siren.ml.model import WaterResUNet

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    arch, in_ch, base = _detect_architecture(state)
    model = WaterResUNet(in_channels=in_ch, base_channels=base)
    model.load_state_dict(state)
    model.eval()

    p_t1 = _prob_map(model, pre_db, post_db)
    p_t0 = _prob_map(model, pre_db, pre_db)
    water_t1 = p_t1 >= threshold
    expansion = water_t1 & (p_t0 < threshold)

    aoi, glac, lake_vic = masks["aoi"], masks["glac"], masks["lake_vic"]
    gated = expansion & aoi & ~(glac & ~lake_vic)

    rule_path = PROCESSED_DIR / "obs-003_expansion_mask.tif"
    rule = (
        sar_grid_sample(str(rule_path), masks["lon"], masks["lat"]) > 0
        if rule_path.exists()
        else np.zeros_like(aoi)
    )
    rule_px = int(rule.sum())
    overlap = int((gated & rule).sum())

    return {
        "expansion_px": int(expansion.sum()),
        "water_extent_px": int(water_t1.sum()),
        "water_extent_on_glacier_px": int(
            (water_t1 & glac & ~lake_vic).sum()
        ),
        "expansion_on_glacier_px": int((expansion & glac & ~lake_vic).sum()),
        "gated_expansion_px": int(gated.sum()),
        "rule_overlap_px": overlap,
        "rule_overlap_pct": (
            round(overlap / rule_px * 100, 1) if rule_px else None
        ),
        "inventory_lakes_in_swath": masks["n_inventory_lakes"],
        "inventory_recall_t1": _recall(
            p_t1, masks["inventory"], threshold
        ),
        "inventory_recall_t0": _recall(
            p_t0, masks["inventory"], threshold
        ),
        "inventory_recall_t1_by_bin": {
            name: _recall(p_t1, m, threshold)
            for name, m in masks["inventory_bins"].items()
        },
        "imja_recall_t1": _recall(p_t1, masks["imja"], threshold),
        "imja_recall_t0": _recall(p_t0, masks["imja"], threshold),
    }


def evaluate_pair(pair_name: str, threshold: float = 0.30) -> dict:
    t0_str, t1_str = PAIRS[pair_name]
    cache_t0 = calibrated_cache(t0_str)
    cache_t1 = calibrated_cache(t1_str)

    with rasterio.open(cache_t0) as d:
        pre_db = d.read().astype(np.float32)
    with rasterio.open(cache_t1) as d:
        post_db = d.read().astype(np.float32)
    if pre_db.shape != post_db.shape:
        raise RuntimeError(
            f"grid mismatch {pre_db.shape} vs {post_db.shape} — "
            "pair must share the same pass geometry"
        )

    masks = _scene_masks(cache_t1)
    logger.info(
        "%s pair: grid=%s  inventory_lakes=%d  imja_px=%d",
        pair_name, pre_db.shape, masks["n_inventory_lakes"],
        int(masks["imja"].sum()),
    )

    results = {}
    for name, ckpt in _checkpoints().items():
        logger.info("  evaluating %s ...", name)
        results[name] = evaluate_checkpoint(
            ckpt, pre_db, post_db, masks, threshold
        )
    return {
        "pair": pair_name,
        "t0": t0_str,
        "t1": t1_str,
        "grid_shape": list(pre_db.shape),
        "checkpoints": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pair", choices=[*PAIRS, "both"], default="both"
    )
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--out", type=Path, default=REPORT_OUT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    names = list(PAIRS) if args.pair == "both" else [args.pair]
    report = {
        "experiment": "held-out adapter evaluation on independent S1A pairs",
        "independence": (
            "Different season, different satellite (S1A vs S1D), same "
            "descending track. Neither pair contributed training chips."
        ),
        "threshold": args.threshold,
        "pairs": [evaluate_pair(n, args.threshold) for n in names],
        "interpretation": {
            "shoulder": (
                "Lakes largely open water — inventory/imja recall is "
                "meaningful; glacier FPs measure domain-shift suppression."
            ),
            "winter": (
                "Imja frozen — recall expected to collapse (frozen lake "
                "is not liquid water); honest metric is whether glacier/"
                "snow FPs stay suppressed vs the base model."
            ),
        },
        "limitations": [
            "Inventory polygons are 2022–2024 median outlines — recall "
            "against them is weak-positive agreement, not truth.",
            "Scenario rule masks encode a simulated July event; "
            "rule_overlap on winter pairs is for continuity, not truth.",
            "Frozen-lake recall collapse is expected physics (C-band "
            "sees ice surface, not water), not necessarily model failure.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    logger.info("report: %s", args.out)


if __name__ == "__main__":
    main()
