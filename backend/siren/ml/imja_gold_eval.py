"""Score adapter checkpoints against hand-verified Imja gold labels.

The §17.2 precision leg is contested by weak label sources (inventory
median polygons flatter the model; SCL under-labels turbid water). The
gold labels — NDWI > 0.15 inside a hand-drawn lake region, boundary
verified against RGB, cloud masked — are the eval's truth tier above
both (ADR-014 gold adjudication, 2026-09-18).

Gold coverage: S2 2026-08-24 serves ``unfrozen_desc`` t0 (2026-08-19,
5-day offset); S2 2026-09-08 serves t1 (2026-09-12, 4-day offset). No
gold labels exist for other pair dates — those report ``null``.

Metrics are Imja-scoped (10-px ROI dilation, same as ``imja_iou_*``)
over the gold-valid region, plus a threshold sweep — the ADR-014
adjudication reported ranges over τ 0.30–0.55.

Caveat carried from the adjudication: centre-sampling the 10 m label
onto the ~90 m SAR grid is stricter than the polygon burn used for the
inventory mask (~1 px bound), and labels carry 4–5 day offsets.

Usage:
    python -m siren.ml.imja_gold_eval [--pair unfrozen_desc] [--threshold 0.30]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio

from siren.ml.heldout_eval import (
    PAIRS,
    PROCESSED_DIR,
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

# SAR acquisition date -> gold label raster (sidecar records the offset).
GOLD_LABELS = {
    "20260819": PROCESSED_DIR / "imja_gold_label_20260824.tif",
    "20260912": PROCESSED_DIR / "imja_gold_label_20260908.tif",
}
LABEL_NODATA = 255


def _gold_labels(date_str: str, masks: dict):
    """Gold water/valid masks sampled onto the SAR grid, or None."""
    from siren.detect.sar import sar_grid_sample

    path = GOLD_LABELS.get(date_str)
    if path is None or not path.exists():
        return None
    lbl = sar_grid_sample(
        str(path), masks["lon"], masks["lat"], fill=float(LABEL_NODATA)
    )
    return lbl == 1, lbl < LABEL_NODATA


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

    gold = {
        "t0": _gold_labels(t0_str, masks),
        "t1": _gold_labels(t1_str, masks),
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
            for key, prob in (("t1", p_t1), ("t0", p_t0)):
                lab = gold[key]
                if lab is None:
                    entry[key] = None
                    continue
                water_l, valid_l = lab
                scope = imja_roi & valid_l
                entry[key] = {
                    "gold_iou": _iou(prob >= tau, water_l, scope),
                    "gold_precision": _precision(prob >= tau, water_l, scope),
                    "gold_recall": _recall(prob, water_l & scope, tau),
                    "gold_valid_coverage": (
                        round(float(valid_l[imja_roi].mean()), 4)
                        if imja_roi.any() else None
                    ),
                    "gold_px": int((water_l & scope).sum()),
                }
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
        "checkpoints": results,
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
        "experiment": "adapter scoring vs hand-verified Imja gold labels",
        "truth_tier": (
            "gold NDWI>0.15 hand-verified labels — above SCL and "
            "inventory median outlines (ADR-014)"
        ),
        "pairs": [evaluate_pair_gold(args.pair, taus)],
        "limitations": [
            "4–5 day label offsets (monsoon shoreline drift).",
            "Centre-sampling 10 m labels onto ~90 m SAR grid is stricter "
            "than the polygon burn (~1 px bound).",
            "Gold labels exist only for the unfrozen_desc dates.",
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["pairs"][0]["checkpoints"], indent=1))


if __name__ == "__main__":
    main()
