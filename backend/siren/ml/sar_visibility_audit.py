"""SAR-visibility audit — which inventory lakes can C-band actually see?

Motivation (ADR-014-am1 probe finding): South Lhonak — the deadliest
recent GLOF site — is SAR-invisible at C-band in a descending geometry:
its surface reads VV ~-7 dB (layover/rough-surface dominated), and its
2023 drain produced ~0 dB of backscatter change. No SAR water detector
(model or deterministic) can monitor such lakes.

This audit scores every in-swath inventory lake for water visibility on
a calibrated SAR cache: fraction of lake-polygon pixels with VV below
the water-dark threshold (grounded on Imja's measured signature:
water VV med -20.1 dB vs non-water -11.6 dB vs South Lhonak -6.9 dB).

Classes:
  monitorable — >50% of lake px water-dark (VV < -15 dB)
  marginal    — 20-50% dark, or median VV in [-15, -12] dB
  invisible   — <20% dark AND median VV > -12 dB (South Lhonak class)

Run:
    python -m siren.ml.sar_visibility_audit [sar_date ...]
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import GCPTransformer
from shapely.geometry import MultiPolygon, Polygon

from siren.ml.heldout_eval import calibrated_cache
from siren.ml.himalayan_lake_dataset import (
    lake_grid_positions,
    load_lake_inventory,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_OUT = (
    REPO_ROOT / "models" / "checkpoints" / "sar_visibility_audit.json"
)

# Water-dark threshold grounded on Imja's measured gold-water signature
# (VV med -20.1, p75 -17.4) vs non-water (-11.6) vs South Lhonak (-6.9).
VV_WATER_DB = -15.0
MIN_LAKE_PX = 3  # ~0.025 km2 at ~90 m pitch — smaller lakes are
#                 # sub-footprint anyway


def _lake_index_raster(lakes, positions, cache_path: str) -> np.ndarray:
    """Rasterize inventory polygons with lake index+1 as burn value."""
    with rasterio.open(cache_path) as src:
        gcps, _ = src.gcps
        h, w = src.height, src.width
    transformer = GCPTransformer(gcps)
    shapes = []
    for burn, (idx, _r, _c) in enumerate(positions, start=1):
        geom = lakes.geometry.iloc[idx]
        if geom is None or geom.is_empty:
            continue
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            verts = list(poly.exterior.coords)
            rows, cols = transformer.rowcol(
                [v[0] for v in verts], [v[1] for v in verts]
            )
            shapes.append(
                (Polygon(zip(cols, rows)), burn)
            )
    if not shapes:
        return np.zeros((h, w), dtype=np.int32)
    return rasterize(shapes, out_shape=(h, w), dtype="int32", fill=0)


def _classify(dark_frac: float, med_vv: float) -> str:
    if dark_frac > 0.5:
        return "monitorable"
    if dark_frac < 0.2 and med_vv > -12.0:
        return "invisible"
    return "marginal"


def audit(sar_date: str, tag: str = "desc") -> dict:
    cache = calibrated_cache(sar_date, tag)
    arr = rasterio.open(cache).read().astype(np.float32)
    vv, vh = arr[0], arr[1]

    lakes = load_lake_inventory(min_elev_m=4000, min_area_km2=0.02)
    positions, _, _ = lake_grid_positions(lakes, str(cache))
    idx_raster = _lake_index_raster(lakes, positions, str(cache))

    rows = []
    counts = {"monitorable": 0, "marginal": 0, "invisible": 0}
    for burn, (idx, _r, _c) in enumerate(positions, start=1):
        m = idx_raster == burn
        n = int(m.sum())
        if n < MIN_LAKE_PX:
            continue
        vv_l, vh_l = vv[m], vh[m]
        dark_frac = float((vv_l < VV_WATER_DB).mean())
        med_vv = float(np.median(vv_l))
        cls = _classify(dark_frac, med_vv)
        counts[cls] += 1
        lake = lakes.iloc[idx]
        rows.append({
            "lake_id": str(lake.get("glac_id", idx)),
            "name": str(
                lake.get("glac_name", lake.get("GL_Name", ""))
            ),
            "lon": round(float(lake["Longitude"]), 5),
            "lat": round(float(lake["Latitude"]), 5),
            "area_km2": round(float(lake["area_km2"]), 3),
            "grid_px": n,
            "vv_median_db": round(med_vv, 2),
            "vh_median_db": round(float(np.median(vh_l)), 2),
            "water_dark_frac": round(dark_frac, 3),
            "class": cls,
        })

    rows.sort(key=lambda r: -r["area_km2"])
    return {
        "sar_date": sar_date,
        "lakes_scored": len(rows),
        "counts": counts,
        "thresholds": {"vv_water_db": VV_WATER_DB, "min_lake_px": MIN_LAKE_PX},
        "lakes": rows,
    }


def main() -> None:
    dates = sys.argv[1:] or ["20260912", "20260807", "20260714"]
    out = {
        "experiment": (
            "SAR-visibility audit — fraction of each inventory lake's "
            "pixels that are water-dark (VV < -15dB) on the calibrated "
            "descending cache; classifies monitorable/marginal/invisible"
        ),
        "audits": [audit(d) for d in dates],
    }
    REPORT_OUT.write_text(json.dumps(out, indent=1))
    for a in out["audits"]:
        print(
            a["sar_date"], a["lakes_scored"], "lakes:",
            a["counts"],
        )
    print("wrote", REPORT_OUT)


if __name__ == "__main__":
    main()
