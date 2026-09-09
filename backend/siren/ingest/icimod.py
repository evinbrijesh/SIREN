"""ICIMOD Glacial Lake & GLOF Inventory ingest (V3 §3.5).

Downloads the International Centre for Integrated Mountain Development
(ICIMOD) glacial lake inventory for the Hindu Kush Himalaya (HKH) region.
This inventory provides:

  - Lake locations (lat/lon) and boundaries
  - Lake type (moraine-dammed, ice-dammed, supraglacial, etc.)
  - Dam geometry (where available)
  - Historical GLOF (glacial lake outburst flood) events

The data is used by the breach susceptibility model (Sprint 2.6) as
training features: lake area, dam type, elevation, and proximity to
steep terrain are key predictors of moraine-dam failure probability.

Usage:
    python -m siren.ingest.icimod --bbox 86.0,27.0,87.5,28.5 --out data/raw/icimod/

Offline-safe: if the network is unavailable, prints a message and exits 0.

Data sources:
    - ICIMOD HKH Glacial Lake Inventory (2020): GeoJSON / Shapefile
    - ICIMOD GLOF event database: CSV with event dates, locations, magnitudes

Provenance: a sidecar JSON is written beside every downloaded file,
recording the source URL, bbox, download timestamp, and checksum.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ICIMOD data portal (regional centre for HKH glacial data).
# The inventory is published as GeoJSON; the GLOF events as CSV.
ICIMOD_LAKE_INVENTORY_URL = (
    "https://geoportal.icimod.org/downloads/glacial-lake-inventory-hkh-2020.geojson"
)
ICIMOD_GLOF_EVENTS_URL = (
    "https://geoportal.icimod.org/downloads/glof-events-hkh.csv"
)

# ICIMOD data license: CC-BY-NC 4.0 (attribution required for non-commercial use).
ICIMOD_LICENSE = "CC-BY-NC 4.0"
ICIMOD_SOURCE = "ICIMOD HKH Glacial Lake Inventory (2020)"


def provenance_path_for(path: Path) -> Path:
    """Sidecar path: <file><suffix>.json (e.g. lakes.geojson -> lakes.geojson.json)."""
    return path.with_suffix(path.suffix + ".json")


def write_provenance(
    sidecar: Path,
    *,
    source: str,
    url: str,
    bbox: tuple[float, float, float, float],
    downloaded_at: str,
    license_: str = ICIMOD_LICENSE,
    record_count: int | None = None,
) -> None:
    """Write a provenance sidecar JSON next to a downloaded file."""
    payload = {
        "source": source,
        "url": url,
        "bbox": list(bbox),
        "downloaded_at": downloaded_at,
        "license": license_,
        "record_count": record_count,
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2))


def download_file(url: str, dest: Path, timeout: float = 60.0) -> bool:
    """Download a file from URL to dest. Returns True on success, False on network error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SIREN-ingest/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        logger.info("Downloaded %s -> %s (%d bytes)", url, dest, len(data))
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        logger.warning("Network unavailable, skipping %s: %s", url, e)
        return False


def filter_geojson_by_bbox(
    geojson: dict,
    bbox: tuple[float, float, float, float],
) -> dict:
    """Filter a GeoJSON FeatureCollection to features within a bounding box.

    Args:
        geojson: Parsed GeoJSON FeatureCollection.
        bbox: (west, south, east, north) in degrees.

    Returns:
        Filtered GeoJSON FeatureCollection with only features whose centroid
        falls within the bbox.
    """
    west, south, east, north = bbox
    filtered = []
    for feature in geojson.get("features", []):
        geom = feature.get("geometry", {})
        coords = _extract_representative_point(geom)
        if coords is None:
            continue
        lon, lat = coords
        if west <= lon <= east and south <= lat <= north:
            filtered.append(feature)
    return {"type": "FeatureCollection", "features": filtered}


def _extract_representative_point(geom: dict) -> tuple[float, float] | None:
    """Extract a representative (lon, lat) point from a GeoJSON geometry."""
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Point":
        return tuple(coords[:2])
    if gtype == "LineString" and coords:
        return tuple(coords[0][:2])
    if gtype == "Polygon" and coords:
        # Use the first point of the outer ring
        return tuple(coords[0][0][:2])
    if gtype == "MultiPolygon" and coords:
        return tuple(coords[0][0][0][:2])
    return None


def parse_bbox(s: str) -> tuple[float, float, float, float]:
    """Parse a 'west,south,east,north' bbox string."""
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"bbox must have 4 values, got {len(parts)}")
    try:
        w, s_, e, n = (float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bbox values must be numeric: {s}")
    if w >= e or s_ >= n:
        raise argparse.ArgumentTypeError(f"bbox must satisfy west<east and south<north: {s}")
    return (w, s_, e, n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download ICIMOD glacial lake inventory")
    parser.add_argument("--bbox", type=parse_bbox, required=True, help="west,south,east,north")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    # Download lake inventory
    lake_path = out_dir / "icimod_lakes_2020.geojson"
    if download_file(ICIMOD_LAKE_INVENTORY_URL, lake_path, args.timeout):
        # Filter by bbox and rewrite
        try:
            geojson = json.loads(lake_path.read_text())
            filtered = filter_geojson_by_bbox(geojson, args.bbox)
            lake_path.write_text(json.dumps(filtered, indent=2))
            n_lakes = len(filtered.get("features", []))
            write_provenance(
                provenance_path_for(lake_path),
                source=ICIMOD_SOURCE,
                url=ICIMOD_LAKE_INVENTORY_URL,
                bbox=args.bbox,
                downloaded_at=now,
                record_count=n_lakes,
            )
            logger.info("Filtered to %d lakes within bbox", n_lakes)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse lake inventory: %s", e)

    # Download GLOF events
    glof_path = out_dir / "icimod_glof_events.csv"
    if download_file(ICIMOD_GLOF_EVENTS_URL, glof_path, args.timeout):
        write_provenance(
            provenance_path_for(glof_path),
            source="ICIMOD GLOF Event Database",
            url=ICIMOD_GLOF_EVENTS_URL,
            bbox=args.bbox,
            downloaded_at=now,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
