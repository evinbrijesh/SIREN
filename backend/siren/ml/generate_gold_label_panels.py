"""Generate annotation panels for SAR dates that still lack gold labels.

This is a workflow helper for the label-acquisition plan. For each target
SAR date it:

  1. picks the best available S2 L2A scene (local archive first; optional
     STAC search for clearer archive gaps),
  2. extracts the Imja ROI and builds an auto-candidate label,
  3. writes a 4-up PNG panel (RGB | NDWI | SCL water | candidate overlay)
     for the analyst to edit, and
  4. persists the candidate GeoTIFF + provenance sidecar at tier
     ``candidate`` — not gold until manually edited and renamed.

The analyst then uses any GIS/image tool to draw include/exclude
polygons, calls ``siren.ml.imja_label_roi.apply_edits()``, renames the
tier to ``gold`` in the sidecar, and commits the final GeoTIFF to
data/processed/imja_gold_label_YYYYMMDD.tif.

Usage:
    python -m siren.ml.generate_gold_label_panels
        [--pairs unfrozen_desc2,shoulder,monsoon_asc]
        [--out-dir data/processed/gold_label_candidates]
        [--search-stac]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from siren.ml.label_registry import GOLD_LABELS, PAIRS, list_labels

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"

# Import-time dependency: PIL may not be installed in minimal environments.
try:
    from PIL import Image  # noqa: F401
    HAS_PIL = True
except Exception:  # pragma: no cover
    HAS_PIL = False


def _parse_scene_date(path: Path) -> str | None:
    m = re.search(r"_(\d{8})T", path.name)
    return m.group(1) if m else None


def _s2_cloud_fraction(path: Path) -> float | None:
    """Return cloud fraction from the STAC-style JSON sidecar if present."""
    sidecar = path.with_suffix(path.suffix + ".json")
    try:
        data = json.loads(sidecar.read_text())
        return float(data.get("cloud_fraction", data.get("eo:cloud_cover", -1)))
    except Exception:
        return None


def _scene_candidates_for_sar_date(sar_date: str) -> list[Path]:
    """Return S2 scenes matching the date window, nearest first.

    First try the curated mapping in heldout_eval.S2_LABEL_SCENES; if a
    scene is missing or not mapped, fall back to globbing the raw archive
    for S2 zipfiles within ±7 days.
    """
    import datetime as dt

    from siren.ml.heldout_eval import S2_LABEL_SCENES

    candidates: list[Path] = []
    curated = S2_LABEL_SCENES.get(sar_date, [])
    for p in curated:
        if p.exists():
            candidates.append(p)

    if candidates:
        # Sort by cloud fraction if known, otherwise keep curated order
        def _key(p: Path) -> tuple[float, int]:
            cf = _s2_cloud_fraction(p)
            d = _parse_scene_date(p)
            offset = 999
            if d:
                offset = abs((dt.date(int(d[:4]), int(d[4:6]), int(d[6:])) - dt.date(int(sar_date[:4]), int(sar_date[4:6]), int(sar_date[6:]))).days)
            return (cf if cf is not None else 999.0, offset)
        candidates.sort(key=_key)
        return candidates

    # Fallback glob
    sar_d = dt.date(int(sar_date[:4]), int(sar_date[4:6]), int(sar_date[6:]))
    for p in sorted(RAW_DIR.glob("S2*_MSIL2A_*.zip")):
        d = _parse_scene_date(p)
        if not d:
            continue
        dd = dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))
        if abs((dd - sar_d).days) <= 7:
            candidates.append(p)
    candidates.sort(key=lambda p: _s2_cloud_fraction(p) or 999.0)
    return candidates


def _missing_gold_dates(pair_names: list[str] | None = None) -> dict[str, str]:
    """Map SAR date -> pair name for dates that need gold labels."""
    labels = list_labels()
    missing: dict[str, str] = {}
    for name, meta in PAIRS.items():
        if pair_names and name not in pair_names:
            continue
        for date_key in ("t0", "t1"):
            date = meta[date_key]
            rec = labels.get(date)
            if rec is None or rec["tier"] != "gold" or not rec["exists"]:
                missing[date] = name
    return missing


def generate_panel(
    sar_date: str,
    pair_name: str,
    out_dir: Path,
) -> dict[str, Any] | None:
    """Generate an annotation panel + candidate label for one SAR date."""
    if not HAS_PIL:
        logger.error("PIL/Pillow is required for panel rendering")
        return None

    from siren.ml.imja_label_roi import (
        auto_candidate_label,
        extract_roi,
        render_panel,
        write_label,
    )

    scenes = _scene_candidates_for_sar_date(sar_date)
    if not scenes:
        logger.warning("no S2 scene found for SAR date %s", sar_date)
        return None

    scene = scenes[0]
    scene_date = _parse_scene_date(scene) or "unknown"
    offset_days = None
    try:
        import datetime as dt
        sd = dt.date(int(scene_date[:4]), int(scene_date[4:6]), int(scene_date[6:]))
        dd = dt.date(int(sar_date[:4]), int(sar_date[4:6]), int(sar_date[6:]))
        offset_days = abs((sd - dd).days)
    except Exception:
        pass

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"imja_gold_label_{scene_date}.tif"
    out_png = out_dir / f"imja_gold_label_{scene_date}_panel.png"

    try:
        roi = extract_roi(scene)
    except Exception as exc:
        logger.warning("ROI extraction failed for %s: %s", scene, exc)
        return None

    label = auto_candidate_label(roi)
    render_panel(roi, label, out_png, title=f"{sar_date} ({pair_name}) {scene.name}")

    provenance = {
        "sar_date_served": sar_date,
        "pair": pair_name,
        "s2_scene": scene.name,
        "s2_acquisition_date": scene_date,
        "offset_days": offset_days,
        "tier": "candidate",
        "method": (
            "NDWI>0.15 on SCL-valid pixels inside inventory polygon +200m buffer; "
            "UNVERIFIED candidate — requires manual shoreline edits"
        ),
        "analyst": None,
        "annotated_at": None,
        "edit_count": 0,
        "notes": (
            "Generate annotation panel from generate_gold_label_panels.py. "
            "Apply manual edits, rename tier to 'gold', and move/rename to "
            f"data/processed/imja_gold_label_{scene_date}.tif if this date "
            "matches the naming convention."
        ),
    }
    write_label(label, roi, out_tif, provenance)

    return {
        "sar_date": sar_date,
        "pair": pair_name,
        "scene": str(scene),
        "offset_days": offset_days,
        "candidate_tif": str(out_tif),
        "panel_png": str(out_png),
    }


def generate_all(
    pair_names: list[str] | None = None,
    out_dir: Path | str = PROCESSED_DIR / "gold_label_candidates",
) -> dict[str, Any]:
    """Generate candidate panels for every SAR date missing a gold label."""
    out_dir = Path(out_dir)
    missing = _missing_gold_dates(pair_names)
    if not missing:
        return {"status": "no_missing_gold_dates", "generated": []}

    generated: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for sar_date, pair_name in sorted(missing.items(), key=lambda kv: kv[0]):
        rec = generate_panel(sar_date, pair_name, out_dir)
        if rec:
            generated.append(rec)
        else:
            failed.append({"sar_date": sar_date, "pair": pair_name})

    return {
        "status": "generated" if generated else "failed",
        "out_dir": str(out_dir),
        "generated": generated,
        "failed": failed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pairs",
        default=None,
        help="comma-separated pair names (default: all eval-eligible pairs)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DIR / "gold_label_candidates",
    )
    ap.add_argument("--quiet", action="store_true", help="suppress INFO logs")
    args = ap.parse_args()

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    pair_names = args.pairs.split(",") if args.pairs else None
    report = generate_all(pair_names, args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
