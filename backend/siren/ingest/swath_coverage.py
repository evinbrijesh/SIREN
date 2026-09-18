"""Verify which on-disk S1 SAFE scenes actually image Imja Tsho.

Live Phase 2 task: "Validate SAR footprint covers Imja Lake — actual pixel
coverage at 86.925°E verified for each track". A scene's filename says
nothing about swath placement; the honest check is the annotation GCP grid
(the measurement TIFFs carry no geotransform). For each SAFE archive this:

  1. reads the VV measurement TIFF GCPs + dimensions via /vsizip/,
  2. parses pass direction + relative orbit from the annotation XML,
  3. builds the footprint polygon by GCP-transforming the image edge
     midpoints + corners (GRD geolocation is non-affine over terrain, so
     the transformed boundary is more honest than four corners alone),
  4. point-tests Imja Tsho and the AOI polygon against that footprint,
  5. reports the swath longitude range at Imja's latitude directly from
     the GCP grid (the eastern-edge measure that decides coverage here).

Report convention matches the other eval scripts: JSON to
``models/checkpoints/``.

Usage:
    python -m siren.ingest.swath_coverage
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import GCPTransformer
from shapely.geometry import Point, Polygon, shape

from siren.preprocess.sar_calibrate import (
    _find_annotation_xml,
    _find_measurement_tiff,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
AOI_GEOJSON = DATA_DIR / "assets" / "dudh_koshi_aoi.geojson"
REPORT_OUT = CHECKPOINT_DIR / "swath_coverage.json"

# Imja Tsho inventory centroid — same reference as heldout_eval.py.
IMJA_LON, IMJA_LAT = 86.9282, 27.8983
# Half-width (deg) of the latitude band used for the swath-edge measure.
LAT_BAND = 0.05


def _parse_orbit_meta(safe_zip: Path) -> dict:
    """Pass direction from the annotation XML; relative orbit from manifest."""
    ann = _find_annotation_xml(str(safe_zip), "vv")
    with zipfile.ZipFile(safe_zip) as z:
        data = z.read(ann).decode("utf-8", errors="replace")
        manifest = z.read(
            [n for n in z.namelist() if n.endswith("manifest.safe")][0]
        ).decode("utf-8", errors="replace")
    pass_m = re.search(r"<pass>(Ascending|Descending)</pass>", data)
    orbit_m = re.search(
        r'relativeOrbitNumber type="start">(\d+)<', manifest
    )
    return {
        "pass": pass_m.group(1) if pass_m else "Unknown",
        "relative_orbit": int(orbit_m.group(1)) if orbit_m else None,
    }


def _footprint_polygon(transformer: GCPTransformer, h: int, w: int):
    """Swath footprint in EPSG:4326 from transformed boundary points."""
    # Corners + edge midpoints — the swath boundary is gently curved in
    # ground space, so 8 points bound it tightly without dense sampling.
    edges = [
        (0.0, 0.0), (0.0, w / 2), (0.0, float(w - 1)),
        (h / 2, float(w - 1)), (float(h - 1), float(w - 1)),
        (float(h - 1), w / 2), (float(h - 1), 0.0), (h / 2, 0.0),
    ]
    xs, ys = transformer.xy(
        [r for r, _ in edges], [c for _, c in edges]
    )
    return Polygon(zip(xs, ys))


def scene_coverage(safe_zip: Path, aoi_geom=None) -> dict:
    """Coverage report for one SAFE archive."""
    inner = _find_measurement_tiff(str(safe_zip), "vv")
    with rasterio.open(f"/vsizip/{safe_zip}/{inner}") as src:
        gcps, _ = src.gcps
        h, w = src.height, src.width

    transformer = GCPTransformer(gcps)
    footprint = _footprint_polygon(transformer, h, w)

    # Imja pixel via the GCP inverse — sanity cross-check on the
    # polygon test (inverse polynomial can extrapolate off-swath, so
    # the polygon test is authoritative).
    rows, cols = transformer.rowcol([IMJA_LON], [IMJA_LAT])
    imja_rc = (float(rows[0]), float(cols[0]))
    in_bounds = (0 <= imja_rc[0] < h) and (0 <= imja_rc[1] < w)

    covers_imja = bool(footprint.contains(Point(IMJA_LON, IMJA_LAT)))

    # Swath longitude range in Imja's latitude band, straight from the
    # GCP grid — the eastern edge is what decides Imja coverage on the
    # western-swath ascending tracks.
    glon = np.array([g.x for g in gcps])
    glat = np.array([g.y for g in gcps])
    band = np.abs(glat - IMJA_LAT) <= LAT_BAND
    lon_at_imja_lat = (
        [round(float(glon[band].min()), 4), round(float(glon[band].max()), 4)]
        if band.any()
        else None
    )

    meta = _parse_orbit_meta(safe_zip)
    date_m = re.search(r"_(\d{8})T\d{6}", safe_zip.name)

    out = {
        "scene": safe_zip.name,
        "date": date_m.group(1) if date_m else None,
        "satellite": safe_zip.name[:3],
        **meta,
        "grid_hw": [h, w],
        "n_gcps": len(gcps),
        "covers_imja": covers_imja and in_bounds,
        "imja_pixel_rc": [round(imja_rc[0], 1), round(imja_rc[1], 1)],
        "swath_lon_at_imja_lat": lon_at_imja_lat,
    }
    if aoi_geom is not None:
        out["aoi_overlap_frac"] = round(
            float(aoi_geom.intersection(footprint).area / aoi_geom.area), 4
        )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    aoi_geom = None
    if AOI_GEOJSON.exists():
        fc = json.loads(AOI_GEOJSON.read_text())
        feat = fc["features"][0] if fc.get("features") else fc
        aoi_geom = shape(feat["geometry"])

    scenes = sorted(RAW_DIR.glob("S1*_IW_GRDH_*.SAFE.zip"))
    if not scenes:
        raise FileNotFoundError(f"no S1 SAFE archives in {RAW_DIR}")

    results = []
    for z in scenes:
        logger.info("checking %s", z.name)
        results.append(scene_coverage(z, aoi_geom))

    covering = [r["date"] for r in results if r["covers_imja"]]
    report = {
        "experiment": "S1 swath footprint verification at Imja Tsho",
        "imja_lonlat": [IMJA_LON, IMJA_LAT],
        "method": (
            "GCP-grid geolocation (annotation), boundary transformed "
            "through GCPTransformer; Imja point-tested against the swath "
            "polygon. swath_lon_at_imja_lat is the min/max GCP longitude "
            f"within ±{LAT_BAND}° of Imja latitude."
        ),
        "scenes": results,
        "imja_covering_dates": covering,
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    logger.info("report: %s", REPORT_OUT)


if __name__ == "__main__":
    main()
