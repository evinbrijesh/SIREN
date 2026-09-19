"""Score adapter checkpoints against hand-verified Imja gold labels.

The §17.2 precision leg is contested by weak label sources (inventory
median polygons flatter the model; SCL under-labels turbid water). The
gold labels — NDWI > 0.15 inside a hand-drawn lake region, boundary
verified against RGB, cloud masked — are the eval's truth tier above
both (ADR-014 gold adjudication, 2026-09-18).

Truth tiers:
  * ``gold`` — S2 2026-08-24 serves ``unfrozen_desc`` t0 (2026-08-19,
    5-day offset); S2 2026-09-08 serves t1 (2026-09-12, 4-day offset).
  * ``auto`` — unverified candidates (NDWI>0.15 inside the inventory
    polygon +200 m buffer) for dates with no gold label: S2 2026-07-25
    serves ``unfrozen_desc2`` t0 (1-day offset) and S2 2026-08-11 serves
    t1 (4-day offset). Tier-2 evidence only — never confused with gold.

Metrics per tier are Imja-scoped (10-px ROI dilation, same as
``imja_iou_*``) over the label-valid region: strict IoU/P/R, tolerant
precision (pred px within 1/2 px of label water — separates a
sub-footprint shoreline offset from scattered FPs), and the
change-product block (expansion = water_t1 & ~water_t0 vs the label
inter-date change — the runtime's actual operational output). A
threshold sweep reports all of it — the ADR-014 adjudication reported
ranges over τ 0.30–0.55.

Caveats carried from the adjudication: centre-sampling the 10 m label
onto the ~90 m SAR grid is stricter than the polygon burn used for the
inventory mask (~1 px bound), and labels carry 1–5 day offsets.

Usage:
    python -m siren.ml.imja_gold_eval [--pair unfrozen_desc] [--threshold 0.30]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import rasterio

from siren.ml.heldout_eval import (
    PAIRS,
    PROCESSED_DIR,
    RAW_DIR,
    _checkpoints,
    _iou,
    _precision,
    _prob_map,
    _scene_masks,
    calibrated_cache,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_OUT = (
    REPO_ROOT / "models" / "checkpoints" / "imja_gold_eval_report.json"
)

# SAR acquisition date -> hand-verified gold label raster (sidecar
# records the offset). Tier 1 truth.
GOLD_LABELS = {
    "20260819": PROCESSED_DIR / "imja_gold_label_20260824.tif",
    "20260912": PROCESSED_DIR / "imja_gold_label_20260908.tif",
}
# SAR date -> paired S2 scene for AUTO-candidate labels (tier 2:
# NDWI>0.15 inside inventory+200m, NOT hand-verified). Only dates
# without a hand-verified gold label are mapped.
AUTO_LABEL_SCENES = {
    "20260726": RAW_DIR
    / "S2B_MSIL2A_20260725T044659_N0512_R076_T45RVL_20260725T083358.zip",
    "20260807": RAW_DIR
    / "S2A_MSIL2A_20260811T045241_N0512_R076_T45RVL_20260811T100012.SAFE.zip",
}
LABEL_NODATA = 255

# Δp expansion decision rule: a pixel is expansion evidence when the
# post-date probability is confident (>= DP_HI) AND it rose by >= DP_DELTA
# vs the pre-date. Catches sub-pixel footprint growth the binary
# extent difference structurally misses (ADR-014-am1 L2 evidence).
DP_HI = 0.5
DP_DELTA = 0.2


def _auto_label_tif(sar_date: str) -> Path | None:
    """Build (cached) the auto-candidate label tif for a SAR date."""
    s2 = AUTO_LABEL_SCENES.get(sar_date)
    if s2 is None or not s2.exists():
        return None
    m = re.search(r"_(\d{8})T", s2.name)
    out = PROCESSED_DIR / f"imja_autolabel_{m.group(1)}.tif"
    if out.exists():
        return out
    from siren.ml.imja_label_roi import (
        auto_candidate_label,
        extract_roi,
        write_label,
    )

    roi = extract_roi(s2)
    write_label(auto_candidate_label(roi), roi, out, {
        "scene": s2.name,
        "sar_date_served": sar_date,
        "method": (
            "auto: NDWI>0.15 inside inventory polygon +200m buffer, "
            "SCL-clear only — UNVERIFIED, tier below hand-verified gold"
        ),
        "tier": "auto_candidate_unverified",
    })
    logger.info("built auto-candidate label %s", out)
    return out


def _labels_on_grid(path: Path | None, masks: dict):
    """Water/valid masks for a label raster sampled onto the SAR grid."""
    if path is None or not Path(path).exists():
        return None
    from siren.detect.sar import sar_grid_sample

    lbl = sar_grid_sample(
        str(path), masks["lon"], masks["lat"], fill=float(LABEL_NODATA)
    )
    return lbl == 1, lbl < LABEL_NODATA


def _gold_labels(date_str: str, masks: dict):
    """Gold water/valid masks sampled onto the SAR grid, or None."""
    return _labels_on_grid(GOLD_LABELS.get(date_str), masks)


def _auto_labels(date_str: str, masks: dict):
    """Auto-candidate water/valid masks on the SAR grid, or None."""
    return _labels_on_grid(_auto_label_tif(date_str), masks)


def evaluate_pair_gold(
    pair_name: str, thresholds: tuple[float, ...] = (0.30,)
) -> dict:
    from scipy.ndimage import binary_dilation

    t0_str, t1_str = PAIRS[pair_name]
    tag = "asc" if pair_name.endswith("_asc") else "desc"
    cache_t0 = calibrated_cache(t0_str, tag)
    cache_t1 = calibrated_cache(t1_str, tag)
    with rasterio.open(cache_t0) as d:
        pre_db = d.read().astype(np.float32)
    with rasterio.open(cache_t1) as d:
        post_db = d.read().astype(np.float32)

    masks = _scene_masks(cache_t1)
    imja = masks["imja"]
    imja_roi = binary_dilation(imja, iterations=10) if imja.any() else imja

    tiers = {
        "gold": {
            "t0": _gold_labels(t0_str, masks),
            "t1": _gold_labels(t1_str, masks),
        },
        "auto": {
            "t0": _auto_labels(t0_str, masks),
            "t1": _auto_labels(t1_str, masks),
        },
    }

    results = {}
    for name, ckpt in _checkpoints().items():
        import torch

        from siren.ml.engine import _detect_architecture
        from siren.ml.model import WaterResUNet

        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        _, in_ch, base = _detect_architecture(state)
        model = WaterResUNet(in_channels=in_ch, base_channels=base)
        model.load_state_dict(state)
        model.eval()

        p_t1 = _prob_map(model, pre_db, post_db)
        p_t0 = _prob_map(model, pre_db, pre_db)

        per_tau = {}
        for tau in thresholds:
            entry = {}
            for tier, labs in tiers.items():
                for key, prob in (("t1", p_t1), ("t0", p_t0)):
                    entry[f"{tier}_{key}"] = _score_tier(
                        prob, tau, labs[key], imja_roi, tier
                    )
                entry[f"{tier}_change"] = _score_change(
                    p_t0, p_t1, tau, labs, imja_roi, tier
                )
            per_tau[str(tau)] = entry
        results[name] = per_tau
        logger.info("%s %s done", pair_name, name)

    return {
        "pair": pair_name,
        "t0": t0_str,
        "t1": t1_str,
        "gold_labels": {
            d: GOLD_LABELS[d].name for d in (t0_str, t1_str) if d in GOLD_LABELS
        },
        "auto_labels": {
            d: AUTO_LABEL_SCENES[d].name
            for d in (t0_str, t1_str) if d in AUTO_LABEL_SCENES
        },
        "checkpoints": results,
    }


def _score_tier(prob, tau, lab, imja_roi, tier):
    """Extent metrics for one tier (gold / auto) on one date, or None."""
    from scipy.ndimage import binary_dilation

    if lab is None:
        return None
    water_l, valid_l = lab
    scope = imja_roi & valid_l
    pred = prob >= tau
    pred_s = pred & scope
    n_pred = int(pred_s.sum())
    # Tolerant precision: a predicted px counts as correct if ANY
    # label-water px lies within `tol` px — separates a sub-footprint
    # shoreline offset from scattered FPs.
    tol = {}
    for t in (1, 2):
        wl_d = binary_dilation(water_l & scope, iterations=t)
        tol[f"{tier}_precision_tol{t}px"] = (
            round(int((pred_s & wl_d).sum()) / n_pred, 4) if n_pred else None
        )
    return {
        f"{tier}_iou": _iou(pred, water_l, scope),
        f"{tier}_precision": _precision(pred, water_l, scope),
        **tol,
        f"{tier}_recall": _recall(prob, water_l & scope, tau),
        f"{tier}_valid_coverage": (
            round(float(valid_l[imja_roi].mean()), 4) if imja_roi.any() else None
        ),
        f"{tier}_px": int((water_l & scope).sum()),
    }


def _score_change(p_t0, p_t1, tau, labs, imja_roi, tier):
    """Change-product metrics — the runtime's actual output
    (water_t1 & ~water_t0): per-date boundary offsets cancel in the
    difference, so this measures the operational contract."""
    if labs["t0"] is None or labs["t1"] is None:
        return None
    g0w, g0v = labs["t0"]
    g1w, g1v = labs["t1"]
    both_v = g0v & g1v
    cscope = imja_roi & both_v
    lab_change = g1w & ~g0w
    exp = ((p_t1 >= tau) & ~(p_t0 >= tau)) & cscope
    # Δp expansion variant — a pixel counts as expansion when the
    # post-date probability is confident AND it rose meaningfully:
    # catches sub-pixel growth (e.g. 30%→80% water within one ~90 m
    # footprint) that the binary extent difference structurally misses.
    dp = p_t1 - p_t0
    exp_dp = ((p_t1 >= DP_HI) & (dp >= DP_DELTA)) & cscope
    return {
        "expansion_px_in_scope": int(exp.sum()),
        f"fp_vs_{tier}_t1": int((exp & ~g1w).sum()),
        f"tp_vs_{tier}_change": int((exp & lab_change).sum()),
        "expansion_dp_px_in_scope": int(exp_dp.sum()),
        f"fp_dp_vs_{tier}_t1": int((exp_dp & ~g1w).sum()),
        f"tp_dp_vs_{tier}_change": int((exp_dp & lab_change).sum()),
        "dp_mass_px_equiv": round(float(np.maximum(0, dp)[cscope].sum()), 1),
        f"{tier}_change_px": int((lab_change & cscope).sum()),
        "scope_valid_frac": (
            round(float(both_v[imja_roi].mean()), 4) if imja_roi.any() else None
        ),
    }


def _recall(prob: np.ndarray, mask: np.ndarray, threshold: float):
    n = int(mask.sum())
    if n == 0:
        return None
    return round(float(((prob >= threshold) & mask).sum()) / n, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", choices=list(PAIRS), default="unfrozen_desc")
    ap.add_argument(
        "--thresholds", default="0.30,0.40,0.50,0.55",
        help="comma-separated decision thresholds",
    )
    ap.add_argument("--out", type=Path, default=REPORT_OUT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    taus = tuple(float(t) for t in args.thresholds.split(","))
    report = {
        "experiment": "adapter scoring vs hand-verified + auto Imja labels",
        "truth_tiers": {
            "gold": (
                "NDWI>0.15 hand-verified labels — above SCL and "
                "inventory median outlines (ADR-014)"
            ),
            "auto": (
                "NDWI>0.15 inside inventory+200m, UNVERIFIED — tier 2, "
                "covers dates with no gold label"
            ),
        },
        "pairs": [evaluate_pair_gold(args.pair, taus)],
        "limitations": [
            "1–5 day label offsets (monsoon shoreline drift).",
            "Centre-sampling 10 m labels onto ~90 m SAR grid is stricter "
            "than the polygon burn (~1 px bound).",
            "Hand-verified gold labels exist only for the unfrozen_desc "
            "dates; unfrozen_desc2 uses the unverified auto tier.",
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["pairs"][0]["checkpoints"], indent=1))


if __name__ == "__main__":
    main()
