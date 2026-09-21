"""Operational-primary gate evaluation for SAR segmentation.

Evaluates the promoted segmentation checkpoint against the label registry's
gold-complete held-out pairs. Computes:

  * water-extent IoU / Precision / Recall on the Imja ROI
  * change-product metrics (expansion = water_t1 & ~water_t0, Δp expansion)
  * glacier false-positive rate outside the mapped lake vicinity
  * pass/fail verdict for the operational-primary gate

The evaluation is honest by design:
  * pairs used for adapter training are excluded from the gate
  * only gold-tier labels count for the gate verdict
  * auto/SCL/inventory labels may be reported for diagnostics but never
    satisfy the gate

Output: models/checkpoints/imja_operational_gate_eval.json

Usage:
    python -m siren.ml.operational_gate_eval [--thresholds 0.30,0.40,0.50,0.55]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch

from siren.ml.engine import ChangeDetectionEngine
from siren.ml.label_registry import (
    GOLD_LABELS,
    eval_eligible_pairs,
    operational_gate_status,
)
from siren.ml.promotion import promotion_record

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_OUT = (
    REPO_ROOT / "models" / "checkpoints" / "imja_operational_gate_eval.json"
)

# Operational gate criteria (PRD §17.2 / DL_PRIMARY_ROADMAP.md Level 1)
GATE_IOU = 0.60
GATE_PRECISION = 0.84
GATE_GLACIER_FP_FRAC = 0.05
# Frozen-season pairs are evaluated for glacier-FP control only — the model
# is a liquid-water detector and is not expected to detect ice-covered lake.
GATE_FROZEN_MAX_ROI_WATER_FRAC = 0.10  # predicted liquid water in ROI ≤10%


def _promoted_checkpoint_path() -> Path | None:
    """Resolve the promoted sar_segmentation_expansion checkpoint, if any."""
    rec = promotion_record("sar_segmentation_expansion")
    if rec is None:
        return None
    path = REPO_ROOT / "models" / "checkpoints" / rec["checkpoint"]
    return path if path.exists() else None


def _pair_in_scope(entry: dict[str, Any]) -> tuple[bool, str]:
    """Whether an eval pair sits inside the declared operational scope.

    In scope = monsoon window (Jun-Sep, ``siren/scope.py``) + descending
    orbit + unfrozen surface. Out-of-scope pairs remain documented probes
    — they never gate the promotion, and their results are still reported.
    """
    from siren.scope import in_operational_window

    reasons: list[str] = []
    if not in_operational_window(entry["t1"]):
        reasons.append("outside monsoon window (Jun-Sep)")
    if entry["pair"].endswith("_asc"):
        reasons.append("ascending orbit (component scope: descending)")
    season = entry.get("season", "unfrozen")
    if season != "unfrozen":
        reasons.append(f"season={season}")
    return (not reasons), ("; ".join(reasons) if reasons else "in scope")


def _evaluate_pair(
    pair_name: str,
    t0_date: str,
    t1_date: str,
    engine: ChangeDetectionEngine,
    thresholds: tuple[float, ...],
    season: str = "unfrozen",
) -> dict[str, Any]:
    """Evaluate one held-out pair with gold labels.

    Returns extent + change metrics plus the glacier-FP rate. For frozen
    season pairs the extent metrics are diagnostic only — the gate is the
    glacier-FP budget plus a low-predicted-liquid-water criterion.
    """
    from scipy.ndimage import binary_dilation

    from siren.detect.sar import sar_grid_lonlat, sar_grid_polygon_mask
    from siren.ml.himalayan_lake_dataset import (
        lake_grid_positions,
        load_lake_inventory,
        rasterize_lake_labels,
    )
    from siren.ml.heldout_eval import (
        calibrated_cache,
        _iou,
        _precision,
        _recall,
    )
    from siren.ml.imja_gold_eval import (
        GOLD_LABELS as _GOLD_LABELS,
        _labels_on_grid,
        DP_HI,
        DP_DELTA,
        LABEL_NODATA,
    )

    tag = "asc" if pair_name.endswith("_asc") else "desc"
    cache_t0 = calibrated_cache(t0_date, tag)
    cache_t1 = calibrated_cache(t1_date, tag)

    with rasterio.open(cache_t0) as d:
        pre_db = d.read().astype(np.float32)
    with rasterio.open(cache_t1) as d:
        post_db = d.read().astype(np.float32)

    # Build masks on the SAR grid
    ll = sar_grid_lonlat(str(cache_t1))
    if ll is None:
        raise RuntimeError(f"no GCP geolocation in {cache_t1}")
    lon, lat = ll

    aoi_json = str(REPO_ROOT / "data" / "assets" / "dudh_koshi_aoi.geojson")
    rgi_shp = (
        "/vsizip/"
        + str(
            REPO_ROOT
            / "data"
            / "datasets"
            / "RGI2000-v7.0-G-15_south_asia_east.zip"
            / "RGI2000-v7.0-G-15_south_asia_east.shp"
        )
    )
    aoi = sar_grid_polygon_mask(aoi_json, lon, lat)
    glacier = sar_grid_polygon_mask(rgi_shp, lon, lat)

    # Imja ROI: 10-px dilation of the Imja lake inventory polygon only.
    # Filter to the lake nearest the known Imja coordinates (86.928, 27.898)
    # — the inventory contains ~70 lakes in the AOI and we only want Imja.
    lakes = load_lake_inventory(min_elev_m=4000, min_area_km2=0.02)
    imja_lakes = lakes[
        (lakes["Longitude"] > 86.92) & (lakes["Longitude"] < 86.94)
        & (lakes["Latitude"] > 27.89) & (lakes["Latitude"] < 27.91)
    ]
    positions, _, _ = lake_grid_positions(imja_lakes, str(cache_t1))
    inventory = rasterize_lake_labels(imja_lakes, positions, str(cache_t1)) > 0
    imja_roi = binary_dilation(inventory, iterations=10)

    # Gold labels on the SAR grid
    t0_label = _labels_on_grid(GOLD_LABELS.get(t0_date), {"lon": lon, "lat": lat})
    t1_label = _labels_on_grid(GOLD_LABELS.get(t1_date), {"lon": lon, "lat": lat})
    if t0_label is None or t1_label is None:
        raise RuntimeError(f"missing gold label for {pair_name}")

    t0_water, t0_valid = t0_label
    t1_water, t1_valid = t1_label
    valid_region = imja_roi & t1_valid

    # Run the promoted model. The 6-channel multi-temporal model takes the
    # real pre/post pair; per-date probabilities are obtained by using the
    # same pre-event for both arguments (Δσ⁰ = 0) or the real pair for t1.
    from siren.ml.heldout_eval import _prob_map

    p_t1 = _prob_map(engine.model, pre_db, post_db)
    p_t0 = _prob_map(engine.model, pre_db, pre_db)

    # Terrain gate — same as the runtime shadow-mask gating (ADR-010):
    # AOI polygon + DEM slope >15° + RGI glacier minus mapped-lake vicinity.
    # The gate eval measures the operational output (post-gate), not raw
    # model logits — the terrain gate is part of the deployed system.
    exclusion = np.zeros(p_t1.shape, dtype=bool)
    exclusion |= ~aoi
    dem_path = REPO_ROOT / "data" / "raw" / "srtm_30m.tif"
    if dem_path.exists():
        from siren.detect.sar import sar_grid_dem_slope
        slope = sar_grid_dem_slope(str(cache_t1), str(dem_path))
        steep = (~np.isnan(slope)) & (slope > 15.0)
        exclusion |= steep
    lake_vic = binary_dilation(inventory, iterations=10)
    glac_gate = glacier & ~lake_vic
    exclusion |= glac_gate

    per_tau: dict[str, Any] = {}
    for tau in thresholds:
        pred_t1 = (p_t1 >= tau) & ~exclusion
        pred_t0 = (p_t0 >= tau) & ~exclusion

        # Extent metrics
        extent = {
            "iou": _iou(pred_t1, t1_water, valid_region),
            "precision": _precision(pred_t1, t1_water, valid_region),
            "recall": _recall(p_t1, t1_water & valid_region, tau),
            "pred_px_in_roi": int((pred_t1 & valid_region).sum()),
            "label_px_in_roi": int((t1_water & valid_region).sum()),
        }

        # Change metrics (operational output)
        change_scope = imja_roi & t0_valid & t1_valid
        lab_change = t1_water & ~t0_water
        expansion = pred_t1 & ~pred_t0 & change_scope
        dp = p_t1 - p_t0
        expansion_dp = (
            (p_t1 >= DP_HI) & (dp >= DP_DELTA) & change_scope
        )
        change = {
            "expansion_px": int(expansion.sum()),
            "tp_vs_label_change": int((expansion & lab_change).sum()),
            "fp_vs_label_t1": int((expansion & ~t1_water).sum()),
            "expansion_dp_px": int(expansion_dp.sum()),
            "tp_dp_vs_label_change": int((expansion_dp & lab_change).sum()),
            "fp_dp_vs_label_t1": int((expansion_dp & ~t1_water).sum()),
            "label_change_px": int((lab_change & change_scope).sum()),
        }

        # Glacier false-positive budget: predicted water on RGI glacier outside
        # the mapped lake vicinity (same dilation as the terrain gate).
        outside_lake_vic = ~lake_vic & aoi
        glac_pred = pred_t1 & glacier & outside_lake_vic
        total_pred = pred_t1 & aoi
        roi_pred_frac = (
            float((pred_t1 & valid_region).sum()) / float(valid_region.sum())
            if valid_region.sum() > 0 else None
        )
        glacier_fp = {
            "glacier_pred_px": int(glac_pred.sum()),
            "total_pred_px": int(total_pred.sum()),
            "glacier_fp_frac": (
                round(float(glac_pred.sum()) / float(total_pred.sum()), 4)
                if total_pred.sum() > 0 else None
            ),
            "glacier_fp_frac_of_roi_pred": (
                round(float(glac_pred.sum()) / float((pred_t1 & valid_region).sum()), 4)
                if (pred_t1 & valid_region).sum() > 0 else None
            ),
            "roi_pred_water_frac": round(roi_pred_frac, 4) if roi_pred_frac is not None else None,
        }

        if season == "unfrozen":
            gate_passed = (
                (extent["iou"] is not None and extent["iou"] >= GATE_IOU)
                and (extent["precision"] is not None and extent["precision"] >= GATE_PRECISION)
                and (glacier_fp["glacier_fp_frac"] is not None and glacier_fp["glacier_fp_frac"] < GATE_GLACIER_FP_FRAC)
            )
        else:
            # Frozen/shoulder: liquid water should be near-absent. Gate is
            # glacier-FP control plus a cap on predicted liquid water in the
            # ROI (the lake is ice-covered; true liquid water extent is small).
            gate_passed = (
                (glacier_fp["glacier_fp_frac"] is not None and glacier_fp["glacier_fp_frac"] < GATE_GLACIER_FP_FRAC)
                and (roi_pred_frac is not None and roi_pred_frac <= GATE_FROZEN_MAX_ROI_WATER_FRAC)
            )

        per_tau[str(tau)] = {
            "extent": extent,
            "change": change,
            "glacier_fp": glacier_fp,
            "gate_passed": gate_passed,
        }

    return {
        "pair": pair_name,
        "t0": t0_date,
        "t1": t1_date,
        "season": season,
        "thresholds": per_tau,
        "label_provenance": {
            "t0": str(GOLD_LABELS[t0_date]),
            "t1": str(GOLD_LABELS[t1_date]),
        },
    }


def evaluate_operational_gate(
    thresholds: tuple[float, ...] = (0.30, 0.40, 0.50, 0.55),
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run the operational-primary gate evaluation on all gold-complete pairs.

    Returns a dict with per-pair metrics and an overall gate verdict. If no
    promoted checkpoint exists or fewer than 2 gold-complete pairs exist, the
    report explains why the gate cannot be attempted.
    """
    ckpt = _promoted_checkpoint_path()
    if ckpt is None:
        return {
            "status": "cannot_evaluate",
            "reason": "no promoted sar_segmentation_expansion checkpoint on disk",
            "gate": {
                "iou": GATE_IOU,
                "precision": GATE_PRECISION,
                "glacier_fp_frac": GATE_GLACIER_FP_FRAC,
            },
            "pairs": [],
            "verdict": "not_attempted",
        }

    engine = ChangeDetectionEngine(weights_path=ckpt)
    if not engine.is_ready:
        return {
            "status": "cannot_evaluate",
            "reason": f"promoted checkpoint {ckpt} failed to load",
            "gate": {
                "iou": GATE_IOU,
                "precision": GATE_PRECISION,
                "glacier_fp_frac": GATE_GLACIER_FP_FRAC,
            },
            "pairs": [],
            "verdict": "not_attempted",
        }

    op_status = operational_gate_status()
    gold_complete = op_status.get("gold_complete_pairs", [])
    if len(gold_complete) < 2:
        return {
            "status": "insufficient_labels",
            "reason": op_status.get("next_action", "acquire more gold labels"),
            "gate": {
                "iou": GATE_IOU,
                "precision": GATE_PRECISION,
                "glacier_fp_frac": GATE_GLACIER_FP_FRAC,
            },
            "gold_complete_pairs": gold_complete,
            "pairs": [],
            "verdict": "not_attempted",
        }

    eligible = eval_eligible_pairs("eval")
    results: list[dict[str, Any]] = []
    for entry in eligible:
        if not entry["gold_complete"]:
            continue
        try:
            result = _evaluate_pair(
                entry["pair"],
                entry["t0"],
                entry["t1"],
                engine,
                thresholds,
                season=entry.get("season", "unfrozen"),
            )
            in_scope, scope_note = _pair_in_scope(entry)
            result["in_operational_scope"] = in_scope
            result["scope_note"] = scope_note
            results.append(result)
        except Exception as exc:
            logger.exception("evaluation failed for %s", entry["pair"])
            results.append({
                "pair": entry["pair"],
                "season": entry.get("season"),
                "error": str(exc),
            })

    # Verdict: the gate counts only pairs inside the declared operational
    # scope — monsoon window (Jun-Sep, siren/scope.py) + descending orbit +
    # unfrozen surface. Out-of-scope pairs (frozen/shoulder, ascending,
    # other regions) are documented probes: informative, never gating.
    # The gate passes when ≥2 in-scope pairs pass the extent criteria at
    # the same threshold.
    tau_key = str(thresholds[0])
    in_scope_results = [r for r in results if r.get("in_operational_scope")]
    out_of_scope_results = [
        r for r in results if not r.get("in_operational_scope")
    ]
    verdict = {
        "threshold": tau_key,
        "pairs_evaluated": len(results),
        "in_scope_pairs_evaluated": len(in_scope_results),
        "in_scope_pairs_passed": sum(
            1 for r in in_scope_results
            if r.get("thresholds", {}).get(tau_key, {}).get("gate_passed")
        ),
        "out_of_scope_pairs": [
            {
                "pair": r.get("pair"),
                "note": r.get("scope_note"),
                "gate_passed": r.get("thresholds", {})
                .get(tau_key, {}).get("gate_passed"),
            }
            for r in out_of_scope_results
        ],
    }
    verdict["operational_primary_eligible"] = (
        verdict["in_scope_pairs_passed"] >= 2
    )

    from siren.scope import scope_summary

    report = {
        "status": "evaluated",
        "checkpoint": str(ckpt),
        "checkpoint_architecture": engine.architecture,
        "checkpoint_in_channels": engine.in_channels,
        "gate": {
            "iou": GATE_IOU,
            "precision": GATE_PRECISION,
            "glacier_fp_frac": GATE_GLACIER_FP_FRAC,
        },
        "operational_scope": scope_summary(),
        "gold_complete_pairs": gold_complete,
        "pairs": results,
        "verdict": verdict,
    }

    out = Path(out_path) if out_path else REPORT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    logger.info("operational gate report written to %s", out)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--thresholds",
        default="0.30,0.40,0.50,0.55",
        help="comma-separated decision thresholds",
    )
    ap.add_argument("--out", type=Path, default=REPORT_OUT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    taus = tuple(float(t) for t in args.thresholds.split(","))
    report = evaluate_operational_gate(taus, args.out)
    print(json.dumps(report["verdict"], indent=2))


if __name__ == "__main__":
    main()
