"""OSM Overpass extract CLI.

Usage:
    python -m siren.ingest.overpass --bbox 86.65,27.65,87.00,27.98 --out data/assets/osm_infrastructure.geojson

Queries the Overpass API for critical facilities inside the bbox and writes a
GeoJSON FeatureCollection to --out:
  - settlements (place=village/hamlet) + buildings
  - roads (highway)
  - bridges (highway=bridge or bridge=yes)
  - wells / drinking-water points (amenity=drinking_water, man_made=water_well)
  - clinics / hospitals (amenity=hospital/clinic)
  - schools (amenity=school)
  - shelters (amenity=shelter)

A provenance sidecar `<out>.json` is written beside the output:
    {source, bbox, acquired_at, retries, scene_id, download_url}

Retry logic with exponential backoff on transient failures (max 3 retries).

Offline-safe (ADR-004): if the network is unavailable, prints a clear message
and exits with code 0 — never crashes. Prep-time acquisition script only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MAX_RETRIES = 3
BASE_DELAY = 1.0


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--bbox must be 'lon_min,lat_min,lon_max,lat_max', got {s!r}"
        )
    try:
        lon_min, lat_min, lon_max, lat_max = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--bbox values must be numeric: {s!r}") from exc
    if not (-180.0 <= lon_min < lon_max <= 180.0):
        raise argparse.ArgumentTypeError("--bbox lon range invalid (lon_min < lon_max)")
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        raise argparse.ArgumentTypeError("--bbox lat range invalid (lat_min < lat_max)")
    return lon_min, lat_min, lon_max, lat_max


def provenance_path_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def write_provenance(
    sidecar: Path,
    *,
    source: str,
    bbox: tuple[float, float, float, float],
    acquired_at: str,
    retries: int,
    scene_id: str,
    download_url: str,
) -> None:
    sidecar.write_text(
        json.dumps(
            {
                "source": source,
                "bbox": list(bbox),
                "acquired_at": acquired_at,
                "retries": retries,
                "scene_id": scene_id,
                "download_url": download_url,
            },
            indent=2,
        )
    )


def retry(fn, *args, max_retries: int = MAX_RETRIES, base_delay: float = BASE_DELAY, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs), attempt
        except urllib.error.HTTPError as exc:
            last_exc = exc
            transient = exc.code == 429 or 500 <= exc.code < 600
            if transient and attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise
    raise last_exc  # pragma: no cover


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Overpass operations
# --------------------------------------------------------------------------- #
def build_query(bbox: tuple[float, float, float, float]) -> str:
    """Build an Overpass QL query for critical facilities (out:json, out geom)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    # Overpass bbox order: south, west, north, east
    b = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    return f"""[out:json][timeout:60];
(
  node["place"~"village|hamlet"]({b});
  way["place"~"village|hamlet"]({b});
  way["building"]({b});
  node["highway"]({b});
  way["highway"]({b});
  way["bridge"]({b});
  way["highway"="bridge"]({b});
  node["bridge"="yes"]({b});
  node["amenity"="drinking_water"]({b});
  node["man_made"="water_well"]({b});
  node["amenity"~"hospital|clinic"]({b});
  way["amenity"~"hospital|clinic"]({b});
  node["amenity"="school"]({b});
  way["amenity"="school"]({b});
  node["amenity"="shelter"]({b});
  way["amenity"="shelter"]({b});
);
out geom;"""


def query_overpass(bbox: tuple[float, float, float, float]) -> dict:
    """POST the Overpass query and return the parsed JSON response."""
    data = urllib.parse.urlencode({"data": build_query(bbox)}).encode()
    req = urllib.request.Request(OVERPASS_URL, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def overpass_to_geojson(elements: list[dict]) -> dict:
    """Convert Overpass JSON elements to a GeoJSON FeatureCollection."""
    features = []
    for el in elements:
        etype = el.get("type")
        geom = None
        if etype == "node":
            if "lon" in el and "lat" in el:
                geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif etype == "way":
            coords = [[g["lon"], g["lat"]] for g in el.get("geometry", [])]
            if len(coords) < 2:
                continue
            if coords[0] == coords[-1] and len(coords) >= 4:
                geom = {"type": "Polygon", "coordinates": [coords]}
            else:
                geom = {"type": "LineString", "coordinates": coords}
        else:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "id": el.get("id"),
                    "osm_type": etype,
                    "tags": el.get("tags", {}),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m siren.ingest.overpass",
        description="Extract critical facilities from OpenStreetMap via the Overpass API.",
    )
    p.add_argument("--bbox", required=True, type=parse_bbox,
                   help="bbox 'lon_min,lat_min,lon_max,lat_max'")
    p.add_argument("--out", default="data/assets/osm_infrastructure.geojson",
                   help="output GeoJSON path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data, retries = retry(query_overpass, args.bbox)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"✗ Network unavailable or Overpass error: {exc}", file=sys.stderr)
        print("  Offline-safe exit (no file written).", file=sys.stderr)
        return 0

    fc = overpass_to_geojson(data.get("elements", []))
    out_path.write_text(json.dumps(fc, indent=2))
    write_provenance(
        provenance_path_for(out_path),
        source="osm-overpass",
        bbox=args.bbox,
        acquired_at=_now_iso(),
        retries=retries,
        scene_id="overpass-extract",
        download_url=OVERPASS_URL,
    )
    print(f"✓ Wrote {out_path} ({len(fc['features'])} features, retries={retries})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
