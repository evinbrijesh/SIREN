"""SRTM DEM download CLI.

Usage:
    python -m siren.ingest.srtm --bbox 86.65,27.65,87.00,27.98 --out data/raw/

Downloads SRTM 1-arc-second (~30 m) tiles covering the bbox from NASA Earthdata
(https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/). A provenance sidecar
`<file>.json` is written beside every downloaded file:
    {source, bbox, acquired_at, retries, scene_id, download_url}

Retry logic with exponential backoff on transient failures (max 3 retries).

Offline-safe (ADR-004): if the network is unavailable, prints a clear message
and exits with code 0 — never crashes. Prep-time acquisition script only.

Auth: set EARTHDATA_USERNAME + EARTHDATA_PASSWORD (free account at
urs.earthdata.nasa.gov). NASA Earthdata accepts HTTP Basic auth.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SRTM_BASE = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"
SRTM_ACQUIRED = "2000-02-11"
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


# --------------------------------------------------------------------------- #
# SRTM operations
# --------------------------------------------------------------------------- #
def tile_names(bbox: tuple[float, float, float, float]) -> list[str]:
    """Return SRTMGL1 tile filenames covering the bbox (1deg x 1deg tiles)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lons = range(int(math.floor(lon_min)), int(math.floor(lon_max)) + 1)
    lats = range(int(math.floor(lat_min)), int(math.floor(lat_max)) + 1)
    names: list[str] = []
    for lat in lats:
        for lon in lons:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            names.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.SRTMGL1.hgt.zip")
    return names


def _auth_headers() -> dict:
    user = os.environ.get("EARTHDATA_USERNAME")
    pw = os.environ.get("EARTHDATA_PASSWORD")
    if not (user and pw):
        return {}
    token = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _http_get_bytes(url: str, timeout: int = 600, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_tile(name: str, out_dir: Path, bbox) -> tuple[Path, int, str]:
    """Download one SRTM tile. Returns (path, retries, url)."""
    url = SRTM_BASE + name
    out_path = out_dir / name
    headers = _auth_headers()

    def _fetch() -> Path:
        out_path.write_bytes(_http_get_bytes(url, headers=headers))
        return out_path

    path, retries = retry(_fetch)
    return path, retries, url


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m siren.ingest.srtm",
        description="Download SRTM 1-arc-second DEM tiles from NASA Earthdata.",
    )
    p.add_argument("--bbox", required=True, type=parse_bbox,
                   help="bbox 'lon_min,lat_min,lon_max,lat_max'")
    p.add_argument("--out", default="data/raw", help="output directory (default: data/raw)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = tile_names(args.bbox)
    print(f"Need {len(names)} SRTM tile(s): {', '.join(names)}")
    n = 0
    try:
        for name in names:
            out_path = out_dir / name
            if out_path.exists():
                print(f"  ✓ already present: {name}")
                write_provenance(
                    provenance_path_for(out_path),
                    source="nasa-earthdata-srtm",
                    bbox=args.bbox,
                    acquired_at=SRTM_ACQUIRED,
                    retries=0,
                    scene_id=name,
                    download_url=SRTM_BASE + name,
                )
                n += 1
                continue
            path, retries, url = download_tile(name, out_dir, args.bbox)
            write_provenance(
                provenance_path_for(path),
                source="nasa-earthdata-srtm",
                bbox=args.bbox,
                acquired_at=SRTM_ACQUIRED,
                retries=retries,
                scene_id=name,
                download_url=url,
            )
            print(f"  ✓ {name} (retries={retries})")
            n += 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"✗ Network unavailable or Earthdata error: {exc}", file=sys.stderr)
        print("  Offline-safe exit (partial files may remain).", file=sys.stderr)
        return 0
    print(f"✓ Downloaded {n}/{len(names)} tile(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
