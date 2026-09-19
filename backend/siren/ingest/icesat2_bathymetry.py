"""Download the ICESat-2 supraglacial lake bathymetry corpus (Zenodo).

Record: Data for "A Framework for Automated Supraglacial Lake Detection
and Depth Retrieval in ICESat-2 Photon Data Across the Greenland and
Antarctic Ice Sheets" — 1,249 lakes, HDF5 files containing lakebed photon
fits (``depth_data`` group: lat/lon, water_depth_meters,
lakebed_fit_elevation_meters).

Files are enumerated via the Zenodo REST API — record id is resolved
from the concept DOI so we always get the latest version's file list.
~1 GB total; use --limit for a partial corpus (pretraining needs only a
subset). Files resume: existing files on disk are skipped.

Domain caveat (recorded in provenance): supraglacial lakes are
morphologically different from moraine-dammed GLOF lakes — this corpus
is PRETRAINING data for the bathymetry model, not evaluation data.

Offline-safe (ADR-004): exits cleanly without network.

Usage:
    python -m siren.ingest.icesat2_bathymetry --limit 50
    python -m siren.ingest.icesat2_bathymetry   # full corpus (~1 GB)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from siren.ingest.imja_bathymetry import (
    ZENODO_API,
    _download,
    _fetch_json,
)

logger = logging.getLogger(__name__)

ICESAT2_RECORD = "10901738"
DEFAULT_OUT = Path("data/datasets/icesat2_bathymetry")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--record", default=ICESAT2_RECORD,
                   help="Zenodo record id (files version)")
    p.add_argument("--limit", type=int, default=None,
                   help="max number of HDF5 files to download")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        meta = _fetch_json(f"{ZENODO_API}/{args.record}", args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"Zenodo record {args.record} unreachable: {exc} — "
              "re-run on a networked host.")
        return 1 if args.strict else 0

    files = [f for f in meta.get("files", [])
             if f.get("key", "").endswith((".h5", ".hdf5", ".nc", ".zip"))]
    if not files:
        # Fall back to all files if extension filter found nothing
        files = meta.get("files", [])
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"Zenodo record {args.record} lists no files.")
        return 1 if args.strict else 0

    print(f"Downloading {len(files)} file(s) from Zenodo record "
          f"{args.record} ...")
    downloaded, skipped = [], 0
    for i, f in enumerate(files, 1):
        name = f.get("key") or f.get("filename") or f"file_{i}"
        url = f.get("links", {}).get("self") or f.get("links", {}).get(
            "download")
        if not url:
            continue
        dest = args.out / name
        if dest.exists():
            skipped += 1
            continue
        if _download(url, dest, args.timeout):
            downloaded.append(dest)
        if i % 25 == 0:
            print(f"  {i}/{len(files)} processed...")

    manifest = args.out / "manifest.json"
    args.out.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "zenodo_record": args.record,
        "doi": meta.get("doi"),
        "title": meta.get("metadata", {}).get("title"),
        "n_downloaded": len(downloaded),
        "n_skipped_existing": skipped,
        "domain_caveat": (
            "supraglacial lakes — pretraining data only, not "
            "moraine-dammed GLOF evaluation data"
        ),
    }, indent=2))

    print(f"✓ {len(downloaded)} downloaded, {skipped} skipped "
          f"(existing) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
