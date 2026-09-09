"""High Mountain Asia (HMA) Glacial Lake Inventory ingest (V3 §3.5).

Downloads the HMA glacial lake inventory from the NASA MEaSUREs /
NSIDC DAAC dataset, which covers the Hindu Kush-Karakoram-Himalaya,
Pamir, Tien Shan, and Tibetan Plateau regions. This complements the
ICIMOD inventory with:

  - Higher-resolution lake boundaries (30 m vs 90 m)
  - Multi-temporal inventories (2000, 2010, 2018, 2020)
  - Lake area change trajectories (critical for breach susceptibility)

The data is used by the breach susceptibility model (Sprint 2.6) for
temporal trend features: rapidly expanding lakes have higher breach
probability than stable lakes.

Usage:
    python -m siren.ingest.hma --bbox 86.0,27.0,87.5,28.5 --out data/raw/hma/

Offline-safe: if the network is unavailable, prints a message and exits 0.

Data sources:
    - HMA Glacial Lake Inventory (Chen et al., 2021): GeoJSON
    - Lake area change time series: CSV

Provenance: a sidecar JSON is written beside every downloaded file,
recording the source URL, bbox, download timestamp, and record count.
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

# HMA glacial lake inventory (Chen et al., 2021, hosted via NSIDC / Zenodo).
HMA_LAKE_INVENTORY_URL = (
    "https://zenodo.org/record/5532194/files/hma_glacial_lakes_2020.geojson"
)
HMA_LAKE_CHANGES_URL = (
    "https://zenodo.org/record/5532194/files/hma_lake_area_changes.csv"
)

# Data license: CC-BY 4.0 (open access, commercial use permitted).
HMA_LICENSE = "CC-BY 4.0"
HMA_SOURCE = "HMA Glacial Lake Inventory (Chen et al., 2021)"

# Reuse the provenance + download helpers from icimod (shared pattern).
from siren.ingest.icimod import (  # noqa: E402, F401
    provenance_path_for,
    write_provenance,
    download_file,
    filter_geojson_by_bbox,
    parse_bbox,
)


def write_hma_provenance(
    sidecar: Path,
    *,
    url: str,
    bbox: tuple[float, float, float, float],
    downloaded_at: str,
    record_count: int | None = None,
) -> None:
    """Write a provenance sidecar for an HMA dataset file."""
    write_provenance(
        sidecar,
        source=HMA_SOURCE,
        url=url,
        bbox=bbox,
        downloaded_at=downloaded_at,
        license_=HMA_LICENSE,
        record_count=record_count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download HMA glacial lake inventory")
    parser.add_argument("--bbox", type=parse_bbox, required=True, help="west,south,east,north")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    # Download lake inventory
    lake_path = out_dir / "hma_lakes_2020.geojson"
    if download_file(HMA_LAKE_INVENTORY_URL, lake_path, args.timeout):
        try:
            geojson = json.loads(lake_path.read_text())
            filtered = filter_geojson_by_bbox(geojson, args.bbox)
            lake_path.write_text(json.dumps(filtered, indent=2))
            n_lakes = len(filtered.get("features", []))
            write_hma_provenance(
                provenance_path_for(lake_path),
                url=HMA_LAKE_INVENTORY_URL,
                bbox=args.bbox,
                downloaded_at=now,
                record_count=n_lakes,
            )
            logger.info("Filtered to %d HMA lakes within bbox", n_lakes)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse HMA lake inventory: %s", e)

    # Download lake area changes
    changes_path = out_dir / "hma_lake_area_changes.csv"
    if download_file(HMA_LAKE_CHANGES_URL, changes_path, args.timeout):
        write_hma_provenance(
            provenance_path_for(changes_path),
            url=HMA_LAKE_CHANGES_URL,
            bbox=args.bbox,
            downloaded_at=now,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
