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
CHIPS_IN = REPO_ROOT / "data" / "datasets" / "himalayan_lake_chips_refined"
OUT_DIR = REPO_ROOT / "data" / "datasets" / "himalayan_chips_residual"
SAR_SCENE = (
    REPO_ROOT / "data" / "processed" / "imja_desc_20260714_sar_vv_vh_db.tif"
)
RULE_MASK = REPO_ROOT / "data" / "processed" / "baseline_water_mask.tif"
CHIP = 96


def build() -> dict:
    from siren.detect.sar import sar_grid_lonlat, sar_grid_sample

    chips = np.load(CHIPS_IN / "chips.npz")
    x, y = chips["x"], chips["y"]
    manifest = json.loads((CHIPS_IN / "manifest.json").read_text())

    lon, lat = sar_grid_lonlat(str(SAR_SCENE))
    if lon is None:
        raise RuntimeError("SAR cache carries no GCPs — cannot geolocate")
    rule_on_sar = sar_grid_sample(str(RULE_MASK), lon, lat) > 0

    h, w = lon.shape
    rule_chips = np.zeros((len(manifest), CHIP, CHIP), dtype=np.float32)
    n_rule_pos = 0
    for i, m in enumerate(manifest):
        r0, c0 = int(m["row"]), int(m["col"])
        if r0 + CHIP <= h and c0 + CHIP <= w:
            rule_chips[i] = rule_on_sar[r0:r0 + CHIP, c0:c0 + CHIP]
            n_rule_pos += int(rule_chips[i].sum())
        else:
            logger.warning("chip %s window out of bounds", m["chip_id"])

    x7 = np.concatenate([x, rule_chips[:, None]], axis=1).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "chips.npz",
        x=x7, y=y.astype(np.uint8), src=chips["src"],
    )
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Rule-vs-label overlap on chips — the number the corrector must beat
    inter = float(((rule_chips > 0) & (y > 0)).sum())
    union = float(((rule_chips > 0) | (y > 0)).sum())
    report = {
        "experiment": "residual-seg chip build (SAR + rule-mask channel)",
        "chips_total": len(x),
        "rule_pos_px_total": int(n_rule_pos),
        "label_pos_px_total": int((y > 0).sum()),
        "rule_label_iou": round(inter / union, 4) if union else None,
        "sar_scene": SAR_SCENE.name,
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
