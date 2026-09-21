"""South Lhonak 2023 GLOF — real-event SAR validation for the segmentation model.

The only on-disk SAR pair that brackets a real GLOF event: the 2023-10-03
South Lhonak moraine failure. Pre-event 2023-09-28, post-event 2023-10-10
(7 days after the burst, which released ~40-50 MCM).

This is an out-of-distribution probe for the Imja-trained adapter: a
different lake, ~130 km east of Imja, in the same SAR frame. The point is
not to inflate metrics but to find where the certified scope ends.

Labels are S2 optical water masks (NDWI > 0.15 inside the inventory
polygon buffered 200 m), from the T45RXL scenes:
    - pre:  2023-09-26 (41% tile cloud, 7% invalid over the ROI)
    - post: 2023-10-09 (35% tile cloud, 5% invalid over the ROI)

Ground truth (Sattar et al. 2025, Science; ICIMOD/DOe reports):
    - Event: moraine failure + lake outburst, 2023-10-03 ~22:30 IST
    - Released volume: ~40-50 MCM; drawdown 10-20 m
    - The lake partially drained; the post-event surface carried massive
      icebergs and debris (documented in the literature) — a hard case
      for SAR water detection because the surface is rough, not specular.

Documented caveats:
    - S2/SAR date offsets: pre 2023-09-26 vs 2023-09-28; post 2023-10-09
      vs 2023-10-10 (≤2 days — acceptable).
    - SAR pixels are ~110 m; the S2 labels are 10 m. Shoreline mixing
      brightens label pixels near the edge.
    - The post-event lake surface is partly covered by icebergs/debris,
      so the S2 NDWI label has holes where the surface was bright.
    - The SRTM DEM tile (data/raw/srtm_30m.tif) covers only the Dudh
      Koshi AOI (86.65-87.00 E) — the runtime slope gate is inoperative
      at South Lhonak (88.19 E). Only the RGI glacier gate applies.

Usage:
    python -m siren.ml.south_lhonak_sar_eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = REPO_ROOT / "data" / "processed"
CANDIDATES = PROCESSED / "gold_label_candidates"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"

# The 2023 South Lhonak SAR pair (descending, orbit 121) — caches are
# named imja_desc_* but the frame spans 87.2-90.1 E.
SAR_T0 = "20230928"
SAR_T1 = "20231010"

# S2-derived water labels (NDWI, inventory-buffered)
LABEL_PRE = CANDIDATES / "south_lhonak_20230926_candidate.tif"
LABEL_POST = CANDIDATES / "south_lhonak_20231009_candidate.tif"

GROUND_TRUTH = {
    "event": "South Lhonak GLOF, 2023-10-03",
    "release_volume_mcm": (40.0, 50.0),
    "drawdown_m_published": (10.0, 20.0),
    "lake_coords": (88.194, 27.9123),
}


def _sample_labels(cache_t1: Path):
    from siren.detect.sar import sar_grid_lonlat, sar_grid_sample

    ll = sar_grid_lonlat(str(cache_t1))
    if ll is None:
        raise RuntimeError(f"no GCPs in {cache_t1}")
    lon, lat = ll
    pre = sar_grid_sample(str(LABEL_PRE), lon, lat, fill=255.0) == 1
    post = sar_grid_sample(str(LABEL_POST), lon, lat, fill=255.0) == 1
    return lon, lat, pre, post


def _terrain_gate(cache_t1: Path, lon, lat, lake_vic):
    """Runtime terrain gates: slope >15 deg + RGI glacier minus lake vicinity.

    The slope gate is inoperative where the SRTM tile does not cover
    (South Lhonak) — reported explicitly in the result.
    """
    from siren.detect.sar import sar_grid_polygon_mask, sar_grid_dem_slope

    dem = REPO_ROOT / "data" / "raw" / "srtm_30m.tif"
    slope = sar_grid_dem_slope(str(cache_t1), str(dem))
    slope_covered = bool((~np.isnan(slope)).any())
    slope_excl = (~np.isnan(slope)) & (slope > 15.0)

    rgi = (
        "/vsizip/"
        + str(
            REPO_ROOT
            / "data/datasets/RGI2000-v7.0-G-15_south_asia_east.zip"
            / "RGI2000-v7.0-G-15_south_asia_east.shp"
        )
    )
    glacier = sar_grid_polygon_mask(rgi, lon, lat)
    glacier_excl = glacier & ~lake_vic
    return slope_excl | glacier_excl, slope_covered, int(slope_excl.sum()), int(glacier_excl.sum())


def evaluate(thresholds: tuple[float, ...] = (0.3, 0.4, 0.5)) -> dict:
    from siren.ml.engine import ChangeDetectionEngine
    from siren.ml.heldout_eval import _iou, _precision, _prob_map
    from siren.ml.promotion import promotion_record

    cache_t0 = PROCESSED / f"imja_desc_{SAR_T0}_sar_vv_vh_db.tif"
    cache_t1 = PROCESSED / f"imja_desc_{SAR_T1}_sar_vv_vh_db.tif"
    for p in (cache_t0, cache_t1, LABEL_PRE, LABEL_POST):
        if not p.exists():
            raise FileNotFoundError(p)

    rec = promotion_record("sar_segmentation_expansion")
    ckpt = CHECKPOINT_DIR / rec["checkpoint"]
    engine = ChangeDetectionEngine(weights_path=ckpt)

    pre_db = rasterio.open(cache_t0).read().astype(np.float32)
    post_db = rasterio.open(cache_t1).read().astype(np.float32)
    prob = np.asarray(_prob_map(engine.model, pre_db, post_db))

    lon, lat, pre_lbl, post_lbl = _sample_labels(cache_t1)
    lake_vic = binary_dilation(pre_lbl | post_lbl, iterations=10)
    excl, slope_covered, n_slope, n_glac = _terrain_gate(cache_t1, lon, lat, lake_vic)
    valid = lake_vic

    results: dict = {
        "event": GROUND_TRUTH["event"],
        "sar_pair": [SAR_T0, SAR_T1],
        "checkpoint": rec["checkpoint"],
        "labels": {
            "pre": LABEL_PRE.name,
            "post": LABEL_POST.name,
            "pre_px_on_sar": int(pre_lbl.sum()),
            "post_px_on_sar": int(post_lbl.sum()),
        },
        "terrain_gate": {
            "slope_gate_active": slope_covered,
            "slope_excluded_px": n_slope,
            "glacier_excluded_px": n_glac,
            "note": (
                "SRTM tile covers only the Dudh Koshi AOI — the slope gate "
                "is inoperative at South Lhonak; only the RGI glacier gate "
                "applies."
                if not slope_covered
                else "both gates active"
            ),
        },
        "thresholds": {},
    }

    dvv = post_db[0] - pre_db[0]
    r, c = np.where(pre_lbl)
    results["sar_signature"] = {
        "vv_pre_db": round(float(pre_db[0][r, c].mean()), 2),
        "vv_post_db": round(float(post_db[0][r, c].mean()), 2),
        "dvv_db": round(float(dvv[r, c].mean()), 2),
    }

    for tau in thresholds:
        pred = (prob >= tau) & ~excl
        tp = int((pred & post_lbl & valid).sum())
        n_lbl = int((post_lbl & valid).sum())
        results["thresholds"][str(tau)] = {
            "post": {
                "iou": _iou(pred, post_lbl, valid),
                "precision": _precision(pred, post_lbl, valid),
                "recall": round(tp / max(n_lbl, 1), 4),
                "pred_px": int((pred & valid).sum()),
                "label_px": n_lbl,
            },
            "pre": {
                "iou": _iou(pred, pre_lbl, valid),
                "recall": round(
                    int((pred & pre_lbl & valid).sum()) / max(int((pre_lbl & valid).sum()), 1), 4
                ),
            },
            "scene_false_positives": int((pred & ~pre_lbl & ~post_lbl).sum()),
            "scene_total_pred": int(pred.sum()),
        }

    out = CHECKPOINT_DIR / "south_lhonak_sar_eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("South Lhonak SAR eval written to %s", out)

    print()
    print("=" * 72)
    print("South Lhonak 2023 GLOF — SAR real-event validation (OOD probe)")
    print("=" * 72)
    print(
        f"  Labels: pre {results['labels']['pre_px_on_sar']} px, "
        f"post {results['labels']['post_px_on_sar']} px on SAR grid"
    )
    sig = results["sar_signature"]
    print(f"  SAR signature at lake: VV pre={sig['vv_pre_db']} post={sig['vv_post_db']} dVV={sig['dvv_db']} dB")
    print(f"  Terrain gate: slope active={slope_covered} (SRTM coverage), glacier excl {n_glac} px")
    for tau, m in results["thresholds"].items():
        p = m["post"]
        print(
            f"  tau={tau}: post IoU={p['iou']} recall={p['recall']} "
            f"({p['pred_px']} pred in ROI / {p['label_px']} label) — "
            f"scene FPs {m['scene_false_positives']} / {m['scene_total_pred']} pred"
        )
    print("=" * 72)
    return results


def main():
    parser = argparse.ArgumentParser(description="South Lhonak 2023 GLOF SAR validation")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate()


if __name__ == "__main__":
    main()
