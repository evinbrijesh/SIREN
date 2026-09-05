"""GPM IMERG rainfall download CLI.

Usage:
    python -m siren.ingest.imerg --bbox 86.65,27.65,87.00,27.98 --date 2026-07-01:2026-08-31 --out data/raw/

Downloads GPM IMERG Late daily L3 rainfall (GPM_3IMERDL.07) NetCDF files from
NASA GES DISC for each day in the date range. A provenance sidecar `<file>.json`
is written beside every downloaded file:
    {source, bbox, acquired_at, retries, scene_id, download_url}

Retry logic with exponential backoff on transient failures (max 3 retries).

Offline-safe (ADR-004): if the network is unavailable, prints a clear message
and exits with code 0 — never crashes. Prep-time acquisition script only.

Auth: set EARTHDATA_USERNAME + EARTHDATA_PASSWORD (free account at
urs.earthdata.nasa.gov). Subsetting to the bbox is a post-processing step
(performed by the pipeline); this script fetches the daily granules.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

IMERG_BASE = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERDL.07"
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


def parse_date_range(s: str) -> tuple[str, str]:
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--date must be 'YYYY-MM-DD:YYYY-MM-DD', got {s!r}"
        )
    return parts[0], parts[1]


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
# IMERG operations
# --------------------------------------------------------------------------- #
def file_for_date(d: date) -> tuple[str, str]:
    """Return (filename, url) for the IMERG Late daily granule of date d."""
    doy = d.timetuple().tm_yday
    fname = f"3IMERDL.{d.strftime('%Y%m%d')}.nc4"
    url = f"{IMERG_BASE}/{d.year}/{doy:03d}/{fname}"
    return fname, url


def date_range(start: str, end: str):
    """Yield date objects for each day in [start, end] inclusive."""
    d = date.fromisoformat(start)
    e = date.fromisoformat(end)
    if d > e:
        return
    while d <= e:
        yield d
        d += timedelta(days=1)


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


def download_day(d: date, out_dir: Path, bbox) -> tuple[Path, int, str, str]:
    """Download one IMERG daily file. Returns (path, retries, url, filename)."""
    fname, url = file_for_date(d)
    out_path = out_dir / fname
    headers = _auth_headers()

    def _fetch() -> Path:
        out_path.write_bytes(_http_get_bytes(url, headers=headers))
        return out_path

    path, retries = retry(_fetch)
    return path, retries, url, fname


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m siren.ingest.imerg",
        description="Download GPM IMERG Late daily rainfall from NASA GES DISC.",
    )
    p.add_argument("--bbox", required=True, type=parse_bbox,
                   help="bbox 'lon_min,lat_min,lon_max,lat_max'")
    p.add_argument("--date", required=True, type=parse_date_range,
                   help="date range 'YYYY-MM-DD:YYYY-MM-DD'")
    p.add_argument("--out", default="data/raw", help="output directory (default: data/raw)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    days = list(date_range(*args.date))
    if not days:
        print(f"✗ Invalid date range {args.date}.", file=sys.stderr)
        return 0
    print(f"Fetching {len(days)} IMERG daily file(s) for {args.date[0]}..{args.date[1]}")
    n = 0
    try:
        for d in days:
            path, retries, url, fname = download_day(d, out_dir, args.bbox)
            write_provenance(
                provenance_path_for(path),
                source="nasa-gesdisc-imerg-late-daily",
                bbox=args.bbox,
                acquired_at=d.isoformat(),
                retries=retries,
                scene_id=fname,
                download_url=url,
            )
            print(f"  ✓ {fname} (retries={retries})")
            n += 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"✗ Network unavailable or GES DISC error: {exc}", file=sys.stderr)
        print("  Offline-safe exit (partial files may remain).", file=sys.stderr)
        return 0
    print(f"✓ Downloaded {n}/{len(days)} file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
