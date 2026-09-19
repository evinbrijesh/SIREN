"""Residual-segmentation chip dataset — SAR tensor + deterministic rule mask.

Extends the label-refined Himalayan lake chips with a 7th input channel:
the deterministic rule water mask (``baseline_water_mask.tif``, NDWI on
Sentinel-2) sampled onto each chip's SAR-grid window via GCP geolocation —
the identical mapping the runtime uses to project ``rule_on_sar``.

The corrector learns to *fix* the rule mask rather than replace it:
input = (6ch SAR, rule_mask) → target = label-refined water mask. Held-out
evaluation groups chips by lake_id — a lake and all its chips are either
train or test, never both.

Usage:
    python -m siren.ml.residual_seg_dataset
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "datasets" / "himalayan_chips_residual"
RULE_MASK = REPO_ROOT / "data" / "processed" / "baseline_water_mask.tif"
CHIP = 96

# (refined-chips dir, SAR t1 scene for geolocation, pair tag)
# Gold-eval pair dates (08-19/09-12) are deliberately excluded — they
# are the temporal holdout.
SOURCES = [
    ("himalayan_lake_chips_refined",
     "imja_desc_20260714_sar_vv_vh_db.tif", "p0714"),
    ("himalayan_chips_refined_p0726",
     "imja_desc_20260726_sar_vv_vh_db.tif", "p0726"),
    ("himalayan_chips_refined_p0807",
     "imja_desc_20260807_sar_vv_vh_db.tif", "p0807"),
]


def _rule_on_grid(sar_scene: Path) -> np.ndarray:
    from siren.detect.sar import sar_grid_lonlat, sar_grid_sample

    lon, lat = sar_grid_lonlat(str(sar_scene))
    if lon is None:
        raise RuntimeError(f"SAR cache carries no GCPs: {sar_scene.name}")
    return sar_grid_sample(str(RULE_MASK), lon, lat) > 0


def build() -> dict:
    xs, ys, srcs, manifest_all = [], [], [], []
    n_rule_pos = 0
    for chips_dir, sar_name, pair in SOURCES:
        d = REPO_ROOT / "data" / "datasets" / chips_dir
        chips = np.load(d / "chips.npz")
        manifest = json.loads((d / "manifest.json").read_text())
        rule_on_sar = _rule_on_grid(REPO_ROOT / "data" / "processed" / sar_name)
        h, w = rule_on_sar.shape

        rule_chips = np.zeros((len(manifest), CHIP, CHIP), dtype=np.float32)
        for i, m in enumerate(manifest):
            r0, c0 = int(m["row"]), int(m["col"])
            if r0 + CHIP <= h and c0 + CHIP <= w:
                rule_chips[i] = rule_on_sar[r0:r0 + CHIP, c0:c0 + CHIP]
                n_rule_pos += int(rule_chips[i].sum())
            else:
                logger.warning("%s chip %s out of bounds", pair, m["chip_id"])
            m["pair"] = pair
        xs.append(np.concatenate(
            [chips["x"], rule_chips[:, None]], axis=1))
        ys.append(chips["y"])
        srcs.append(chips["src"])
        manifest_all.extend(manifest)

    x7 = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "chips.npz",
        x=x7, y=y.astype(np.uint8), src=np.concatenate(srcs),
    )
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest_all, indent=2))

    # Rule-vs-label overlap on chips — the number the corrector must beat
    rule_all = x7[:, 6] > 0.5
    inter = float((rule_all & (y > 0)).sum())
    union = float((rule_all | (y > 0)).sum())
    report = {
        "experiment": "residual-seg chip build (multi-date SAR + rule channel)",
        "chips_total": len(x7),
        "pairs": [s[2] for s in SOURCES],
        "rule_pos_px_total": n_rule_pos,
        "label_pos_px_total": int((y > 0).sum()),
        "rule_label_iou": round(inter / union, 4) if union else None,
        "rule_mask": RULE_MASK.name,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("Report: %s", report)
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
