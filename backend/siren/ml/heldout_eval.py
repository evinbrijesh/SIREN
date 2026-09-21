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

A third pair — ``monsoon_asc`` (2026-08-11/2026-09-16, S1D relative
orbit 12, ascending) — adds the missing domain-coverage axis: different
pass geometry *and* unfrozen-season liquid water. All prior eval pairs
share the orbit-121 descending footprint; the orbit-85 ascending scenes
were verified to miss Imja entirely (``ingest/swath_coverage.py``).

Metrics per pair per checkpoint (same machinery as
``sar_domain_adapt.evaluate_scene``, extended with inventory recall):
  - expansion/water-extent pixel counts, terrain-gated counts
  - water/extent on glacier (the domain-shift signature)
  - recall over verified inventory lake polygons at t1 and t0
    (weak positives — polygons are 2022–2024 median outlines)
  - overlap vs the deterministic scenario masks (caveated: scenario
    masks encode a simulated July event, not winter truth)

Usage:
    python -m siren.ml.heldout_eval [--pair shoulder|winter|monsoon_asc|both]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
    # Orbit-12 ascending pair (different pass geometry AND unfrozen
    # season — the strongest domain-coverage eval on disk). Verified
    # to cover Imja by ingest/swath_coverage.py; the orbit-85 ascending
    # scenes miss the lake entirely.
    "monsoon_asc": ("20260811", "20260916"),
    # Unfrozen-season pair on the ro-121 descending deployment track —
    # the clean liquid-water recall number the §9.8 gate needs. Both
    # dates post-date the adapter's July training pair (held-out).
    "unfrozen_desc": ("20260819", "20260912"),
    # Second unfrozen ro-121 pair — the gate wants ≥2 independent
    # Imja pairs. Immediately post-training dates, still monsoon.
    "unfrozen_desc2": ("20260726", "20260807"),
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

# Paired S2 scenes per SAR acquisition date — independent optical labels
# for the §9.8 IoU gate. Each date maps to a LIST of scenes (nearest
# acquisition per tile covering the AOI); the label raster merges them
# per-pixel so eastern tiles fill coverage the Imja tile's cloud masks.
# Dates without a pair report inventory IoU only. SCL class 6 = water;
# SCL_CLEAR classes are valid non-water; everything else (cloud/shadow/
# nodata) is masked out of the IoU.
S2_LABEL_SCENES = {
    # shoulder t0: nearest clear scene (AOI clear frac ~0.99)
    "20251109": [RAW_DIR
    / "S2C_MSIL2A_20251112T045051_N0511_R076_T45RVL_20251112T083114.zip"],
    # shoulder t1: same-day+1 clear scene (AOI clear frac ~0.84)
    "20251121": [RAW_DIR
    / "S2C_MSIL2A_20251122T045131_N0511_R076_T45RVL_20251122T083010.SAFE.zip"],
    # winter t0: nearest clear scene (AOI clear frac ~0.98)
    "20260108": [RAW_DIR
    / "S2C_MSIL2A_20260101T045221_N0511_R076_T45RVL_20260101T082811.zip"],
    # winter t1: nearest clear scene (AOI clear frac ~0.99)
    "20260120": [RAW_DIR
    / "S2C_MSIL2A_20260121T045121_N0511_R076_T45RVL_20260121T083909.zip"],
    # monsoon_asc t1: 2-day-offset scene (t1=09-16, S2=09-18, 34.8% cloud)
    "20260916": [RAW_DIR
    / "S2C_MSIL2A_20260918T044701_N0512_R076_T45RVL_20260918T074909.zip"],
    # unfrozen_desc2 t0: 1-day-offset scenes (t0=07-26, S2=07-25)
    "20260726": [RAW_DIR
    / "S2B_MSIL2A_20260725T044659_N0512_R076_T45RVL_20260725T083358.zip",
                 RAW_DIR
    / "S2B_MSIL2A_20260725T044659_N0512_R076_T45RWL_20260725T083358.zip"],
    # unfrozen_desc2 t1: 4-day-offset scenes (t1=08-07, S2=08-11,
    # RVL tile cloud 34.7%, RWL 32.6%) — closest clear acquisitions
    "20260807": [RAW_DIR
    / "S2A_MSIL2A_20260811T045241_N0512_R076_T45RVL_20260811T100012.SAFE.zip",
                 RAW_DIR
    / "S2A_MSIL2A_20260811T045241_N0512_R076_T45RWL_20260811T100012.zip"],
    # monsoon_asc t0: same-day scenes (RVL 34.7%, RWL 32.6%)
    "20260811": [RAW_DIR
    / "S2A_MSIL2A_20260811T045241_N0512_R076_T45RVL_20260811T100012.SAFE.zip",
                 RAW_DIR
    / "S2A_MSIL2A_20260811T045241_N0512_R076_T45RWL_20260811T100012.zip"],
    # unfrozen_desc t0: 5-day-offset scenes (t0=08-19, S2=08-24,
    # RVL 71%, RWL 62.2%) — clearest acquisitions in the window
    "20260819": [RAW_DIR
    / "S2B_MSIL2A_20260824T044659_N0512_R076_T45RVL_20260824T083447.SAFE.zip",
                 RAW_DIR
    / "S2B_MSIL2A_20260824T044659_N0512_R076_T45RWL_20260824T083447.zip"],
    # unfrozen_desc t1: 4-day-offset scenes (t1=09-12, S2=09-08,
    # RVL 65.7%, RWL 56.6%)
    "20260912": [RAW_DIR
    / "S2C_MSIL2A_20260908T044701_N0512_R076_T45RVL_20260908T094920.SAFE.zip",
                 RAW_DIR
    / "S2C_MSIL2A_20260908T044701_N0512_R076_T45RWL_20260908T094920.zip"],
}
SCL_WATER = 6
SCL_CLEAR = {4, 5, 6, 7, 11}
LABEL_NODATA = 255


def _find_safe(date_str: str) -> Path:
    # Any acquisition time — descending scenes are ~T00:10 UTC, the
    # orbit-12 ascending scenes ~T12:13 UTC. Matches both .SAFE.zip
    # and _COG.zip archives.
    hits = sorted(
        RAW_DIR.glob(f"S1*_IW_GRDH_*_{date_str}T*.zip")
    )
    if not hits:
        raise FileNotFoundError(
            f"no S1 SAFE archive for {date_str} in {RAW_DIR}"
        )
    return hits[0]


def calibrated_cache(date_str: str, tag: str = "desc") -> Path:
    """Calibrate a SAFE archive to the (2, H, W) sigma0-dB cache used by
    the July pair — same function, same GCP persistence."""
    from siren.preprocess.sar_calibrate import extract_and_cache_vv_vh_db

    cache = PROCESSED_DIR / f"imja_{tag}_{date_str}_sar_vv_vh_db.tif"
    extract_and_cache_vv_vh_db(str(_find_safe(date_str)), cache)
    return cache


def s2_label_raster(s2_paths: list[Path], sar_date: str) -> Path:
    """Build (cached) merged SCL water-label GeoTIFF over the AOI at
    ~20 m from every scene in ``s2_paths`` — per pixel the first scene
    with usable SCL wins, so clearer tiles fill another tile's cloud.

    Band values: 1 = water (SCL 6), 0 = valid non-water (SCL_CLEAR),
    255 = unlabelled (cloud/shadow/nodata — masked out of the IoU).
    """
    from rasterio.transform import from_bounds

    from siren.ml.s2_spectral_eval import _aoi_bounds, _read_scl_on_grid

    cache = PROCESSED_DIR / f"s2_water_label_for_{sar_date}.tif"
    if cache.exists():
        return cache

    west, south, east, north = _aoi_bounds()
    cell = 0.0002  # ~22 m in degrees — near SCL's native 20 m
    w = int(np.ceil((east - west) / cell))
    h = int(np.ceil((north - south) / cell))
    label = np.full((h, w), LABEL_NODATA, dtype=np.uint8)
    for s2_path in sorted(s2_paths):
        if not s2_path.exists():
            continue
        scl = _read_scl_on_grid(s2_path, (h, w), (west, south, east, north))
        todo = label == LABEL_NODATA
        label[todo & np.isin(scl, list(SCL_CLEAR))] = 0
        label[todo & (scl == SCL_WATER)] = 1
    with rasterio.open(
        cache, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="uint8", crs="EPSG:4326",
        transform=from_bounds(west, south, east, north, w, h),
        nodata=LABEL_NODATA,
    ) as dst:
        dst.write(label, 1)
    logger.info("built S2 label raster %s from %d scene(s) (water px=%d)",
                cache, len(s2_paths), int((label == 1).sum()))
    return cache


def _optical_labels(date_str: str, masks: dict):
    """Merged SCL water labels sampled onto the SAR grid for one SAR
    date. Returns (water, valid, coverage) or None when no paired S2
    exists."""
    s2_list = [p for p in S2_LABEL_SCENES.get(date_str, []) if p.exists()]
    if not s2_list:
        return None
    from siren.detect.sar import sar_grid_sample

    lbl = sar_grid_sample(
        str(s2_label_raster(s2_list, date_str)), masks["lon"], masks["lat"],
        fill=float(LABEL_NODATA),
    )
    valid = lbl < LABEL_NODATA
    coverage = float(valid[masks["aoi"]].mean()) if masks["aoi"].any() else 0.0
    return lbl == 1, valid, round(coverage, 4)


def _iou(pred: np.ndarray, label: np.ndarray, region: np.ndarray) -> float | None:
    """IoU of pred vs label restricted to region pixels."""
    reg = region.astype(bool)
    union = int(((pred | label) & reg).sum())
    if union == 0:
        return None
    return round(int((pred & label & reg).sum()) / union, 4)


def _precision(
    pred: np.ndarray, label: np.ndarray, region: np.ndarray
) -> float | None:
    """Precision of pred vs label restricted to region pixels — the
    §17.2 P ≥ 0.84 gate metric. None when pred is empty in-region."""
    reg = region.astype(bool)
    n_pred = int((pred & reg).sum())
    if n_pred == 0:
        return None
    return round(int((pred & label & reg).sum()) / n_pred, 4)


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
    labels: dict | None = None,
) -> dict:
    import torch
    from scipy.ndimage import binary_dilation

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
    water_t0 = p_t0 >= threshold
    expansion = water_t1 & ~water_t0

    aoi, glac, lake_vic = masks["aoi"], masks["glac"], masks["lake_vic"]
    gated = expansion & aoi & ~(glac & ~lake_vic)

    # Imja-scoped IoU: evaluated inside a ~10-px (~900 m) ROI around the
    # inventory polygon so scene-wide FPs don't swamp the lake measure.
    imja = masks["imja"]
    imja_roi = binary_dilation(imja, iterations=10) if imja.any() else imja

    rule_path = PROCESSED_DIR / "obs-003_expansion_mask.tif"
    rule = (
        sar_grid_sample(str(rule_path), masks["lon"], masks["lat"]) > 0
        if rule_path.exists()
        else np.zeros_like(aoi)
    )
    rule_px = int(rule.sum())
    overlap = int((gated & rule).sum())

    out = {
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
        # §9.8 IoU gate metrics — vs independent labels
        "inventory_iou_t1": _iou(water_t1, masks["inventory"], aoi),
        "inventory_iou_t0": _iou(water_t0, masks["inventory"], aoi),
        "imja_iou_t1": _iou(water_t1, imja, imja_roi),
        "imja_iou_t0": _iou(water_t0, imja, imja_roi),
        # §17.2 P ≥ 0.84 gate metrics — precision vs the same labels
        "inventory_precision_t1": _precision(
            water_t1, masks["inventory"], aoi
        ),
        "inventory_precision_t0": _precision(
            water_t0, masks["inventory"], aoi
        ),
        "imja_precision_t1": _precision(water_t1, imja, imja_roi),
        "imja_precision_t0": _precision(water_t0, imja, imja_roi),
    }
    for key, pred in (("t1", water_t1), ("t0", water_t0)):
        lab = (labels or {}).get(key)
        if lab is None:
            out[f"optical_iou_{key}"] = None
            out[f"optical_precision_{key}"] = None
            out[f"optical_label_coverage_{key}"] = None
            out[f"imja_optical_iou_{key}"] = None
            out[f"imja_optical_precision_{key}"] = None
            out[f"imja_optical_label_coverage_{key}"] = None
        else:
            water_l, valid_l, cov = lab
            out[f"optical_iou_{key}"] = _iou(pred, water_l, aoi & valid_l)
            out[f"optical_precision_{key}"] = _precision(
                pred, water_l, aoi & valid_l
            )
            # Imja-scoped optical metrics — the gate's fair comparison
            # scope (matches imja_iou); AOI-wide numbers include
            # terrain-gate-able scene FPs.
            imja_scope = imja_roi & valid_l
            out[f"imja_optical_iou_{key}"] = _iou(
                pred, water_l, imja_scope
            )
            out[f"imja_optical_precision_{key}"] = _precision(
                pred, water_l, imja_scope
            )
            out[f"imja_optical_label_coverage_{key}"] = (
                round(float(valid_l[imja_roi].mean()), 4)
                if imja_roi.any()
                else None
            )
            out[f"optical_label_coverage_{key}"] = cov

    # AOI-wide change-product metrics — the operational contract
    # (expansion = water_t1 & ~water_t0). The per-date boundary offset
    # cancels in the difference; measured against the merged optical
    # label change on the both-valid region.
    lab0 = labels.get("t0") if labels else None
    lab1 = labels.get("t1") if labels else None
    for scope_name, extra in (("aoi", None), ("lake", "vic")):
        if lab0 is None or lab1 is None:
            for k in ("exp_px", "tp_px", "fp_px", "label_px",
                      "precision", "recall"):
                out[f"change_{scope_name}_{k}"] = None
            continue
        wl0, vl0, _ = lab0
        wl1, vl1, _ = lab1
        cscope = aoi & vl0 & vl1
        if extra == "vic":
            cscope &= lake_vic
        lab_change = (wl1 & ~wl0) & cscope
        exp_s = expansion & cscope
        tp = int((exp_s & lab_change).sum())
        fp = int((exp_s & ~lab_change).sum())
        n_exp, n_lc = int(exp_s.sum()), int(lab_change.sum())
        out[f"change_{scope_name}_exp_px"] = n_exp
        out[f"change_{scope_name}_tp_px"] = tp
        out[f"change_{scope_name}_fp_px"] = fp
        out[f"change_{scope_name}_label_px"] = n_lc
        out[f"change_{scope_name}_precision"] = (
            round(tp / n_exp, 4) if n_exp else None
        )
        out[f"change_{scope_name}_recall"] = (
            round(tp / n_lc, 4) if n_lc else None
        )
    return out


def evaluate_pair(pair_name: str, threshold: float = 0.30) -> dict:
    t0_str, t1_str = PAIRS[pair_name]
    tag = "asc" if pair_name.endswith("_asc") else "desc"
    cache_t0 = calibrated_cache(t0_str, tag)
    cache_t1 = calibrated_cache(t1_str, tag)

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

    labels = {
        "t0": _optical_labels(t0_str, masks),
        "t1": _optical_labels(t1_str, masks),
    }

    results = {}
    for name, ckpt in _checkpoints().items():
        logger.info("  evaluating %s ...", name)
        results[name] = evaluate_checkpoint(
            ckpt, pre_db, post_db, masks, threshold, labels
        )
    return {
        "pair": pair_name,
        "t0": t0_str,
        "t1": t1_str,
        "grid_shape": list(pre_db.shape),
        "s2_label_scenes": {
            d: [p.name for p in S2_LABEL_SCENES[d] if p.exists()]
            for d in (t0_str, t1_str)
            if d in S2_LABEL_SCENES
            and any(p.exists() for p in S2_LABEL_SCENES[d])
        },
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
        "experiment": "held-out adapter evaluation on independent S1 pairs",
        "independence": (
            "shoulder/winter: S1A ro-121 descending (cross-sensor vs "
            "S1D training pair). monsoon_asc: S1D ro-12 ascending "
            "(different pass geometry — out of the ro-121 deployment "
            "domain per ADR-014). unfrozen_desc: S1D ro-121 descending, "
            "both dates after the adapter's July training pair. No pair "
            "contributed training chips."
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
            "monsoon_asc": (
                "Unfrozen-season ascending pair on a different orbit "
                "(ro=12 vs ro=121 descending) — clean liquid-water "
                "recall plus the strongest geometry-shift test. Ascending "
                "look direction flips layover/shadow vs all prior evals."
            ),
            "unfrozen_desc": (
                "Unfrozen-season pair on the ro-121 deployment track — "
                "the gate's clean liquid-water recall + IoU number. "
                "Both dates post-date the adapter training pair."
            ),
            "unfrozen_desc2": (
                "Second independent unfrozen pair on ro-121 — the "
                "gate's ≥2-pair requirement. Immediately post-training "
                "dates (12–24 days after the training pair)."
            ),
        },
        "limitations": [
            "Inventory polygons are 2022–2024 median outlines — recall "
            "and IoU against them are weak-positive agreement, not "
            "truth; seasonal outlines can differ from same-day extent.",
            "Optical labels are SCL class 6 — ESA's own classifier, not "
            "truth; frozen or debris-covered water can misclassify, and "
            "cloud masks leave unlabelled gaps (masked out of the IoU).",
            "Monsoon optical labels carry a 4–5 day temporal offset from "
            "the SAR date (nearest clear T45RVL acquisition) — lake extent "
            "can drift over the gap, so offset-label IoU is weaker "
            "evidence than same-day labels.",
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
