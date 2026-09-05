"""CDSE STAC search and download CLI.

Usage:
    python -m siren.ingest.cdse --bbox 86.65,27.65,87.00,27.98 --sensor s1 --date 2026-07-01:2026-08-31 --out data/raw/
    python -m siren.ingest.cdse --bbox 86.65,27.65,87.00,27.98 --sensor s2 --date 2025-11-01:2025-12-01 --out data/raw/

Searches the Copernicus Data Space Ecosystem STAC API for Sentinel-1 GRD
(`sentinel-1-grd`) or Sentinel-2 L2A (`sentinel-2-l2a`) scenes overlapping the
bbox, then downloads the primary data asset for each scene into --out. A
provenance sidecar `<file>.json` is written beside every downloaded file:
    {source, bbox, acquired_at, retries, scene_id, download_url}

Retry logic with exponential backoff on transient failures (max 3 retries).

Offline-safe (ADR-004): if the network is unavailable, prints a clear message
to stderr and exits with code 0 — never crashes. This is a prep-time
acquisition script, never a runtime dependency.

Auth (optional): set CDSE_TOKEN, or CDSE_USERNAME + CDSE_PASSWORD (free account
at dataspace.copernicus.eu). Public STAC search works without auth; downloads
of some assets require it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
COLLECTIONS = {"s1": "sentinel-1-grd", "s2": "sentinel-2-l2a"}
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds; doubled each retry


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def parse_bbox(s: str) -> tuple[float, float, float, float]:
    """Parse 'lon_min,lat_min,lon_max,lat_max' into a 4-tuple of floats."""
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
    """Parse 'YYYY-MM-DD:YYYY-MM-DD' into a (start, end) tuple."""
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--date must be 'YYYY-MM-DD:YYYY-MM-DD', got {s!r}"
        )
    return parts[0], parts[1]


def provenance_path_for(path: Path) -> Path:
    """Sidecar path: <file><suffix>.json (e.g. scene.zip -> scene.zip.json)."""
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
    """Write a provenance sidecar JSON next to a downloaded file."""
    payload = {
        "source": source,
        "bbox": list(bbox),
        "acquired_at": acquired_at,
        "retries": retries,
        "scene_id": scene_id,
        "download_url": download_url,
    }
    sidecar.write_text(json.dumps(payload, indent=2))


def retry(fn, *args, max_retries: int = MAX_RETRIES, base_delay: float = BASE_DELAY, **kwargs):
    """Call fn with exponential backoff on transient failures.

    Retries on urllib.error.URLError and HTTPError 5xx / 429.
    Returns (result, retries_used). Raises the last exception if all fail.
    """
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
# HTTP primitives (single choke-point — tests mock urllib.request.urlopen)
# --------------------------------------------------------------------------- #
def _http_post_json(url: str, payload: dict, timeout: int = 60, headers: dict | None = None):
    data = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_get_bytes(url: str, timeout: int = 3600, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# --------------------------------------------------------------------------- #
# CDSE operations
# --------------------------------------------------------------------------- #
def _auth_token() -> str | None:
    """Best-effort CDSE bearer token from env. None if unconfigured."""
    token = os.environ.get("CDSE_TOKEN")
    if token:
        return token
    user = os.environ.get("CDSE_USERNAME")
    pw = os.environ.get("CDSE_PASSWORD")
    if not (user and pw):
        return None
    data = urllib.parse.urlencode(
        {"grant_type": "password", "username": user, "password": pw, "client_id": "cdse-public"}
    ).encode()
    req = urllib.request.Request(TOKEN_URL, data=data)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())["access_token"]


def search_stac(
    bbox: tuple[float, float, float, float],
    sensor: str,
    date_range: tuple[str, str],
    limit: int = 10,
    token: str | None = None,
) -> list[dict]:
    """Search CDSE STAC. Returns a list of STAC item (feature) dicts."""
    collection = COLLECTIONS[sensor]
    start, end = date_range
    body = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": limit,
    }
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = _http_post_json(STAC_URL, body, headers=headers)
    return resp.get("features", [])


def _pick_asset(item: dict) -> tuple[str | None, dict | None]:
    """Pick the primary downloadable asset from a STAC item."""
    assets = item.get("assets", {})
    for key in ("data", "product", "manifest", "metadata"):
        a = assets.get(key)
        if a and "href" in a:
            return key, a
    for key, a in assets.items():
        if "href" in a:
            return key, a
    return None, None


def _acquired_at(item: dict) -> str:
    props = item.get("properties", {})
    return props.get("datetime") or props.get("acquired") or ""


def download_scene(
    item: dict, out_dir: Path, token: str | None = None
) -> tuple[Path, int, str]:
    """Download one scene's primary asset. Returns (path, retries, url)."""
    scene_id = item.get("id", "scene")
    key, asset = _pick_asset(item)
    if asset is None:
        raise RuntimeError(f"no downloadable asset for {scene_id}")
    url = asset["href"]
    ext = Path(url).suffix or ".bin"
    if key == "data" and ext.lower() not in (".zip", ".tif", ".tiff", ".nc"):
        ext = ".zip"
    out_path = out_dir / f"{scene_id}{ext}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

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
        prog="python -m siren.ingest.cdse",
        description="Search and download Sentinel-1/2 scenes from Copernicus CDSE STAC.",
    )
    p.add_argument("--bbox", required=True, type=parse_bbox,
                   help="bbox 'lon_min,lat_min,lon_max,lat_max'")
    p.add_argument("--sensor", required=True, choices=("s1", "s2"),
                   help="s1 = Sentinel-1 GRD, s2 = Sentinel-2 L2A")
    p.add_argument("--date", required=True, type=parse_date_range,
                   help="date range 'YYYY-MM-DD:YYYY-MM-DD'")
    p.add_argument("--out", default="data/raw", help="output directory (default: data/raw)")
    p.add_argument("--limit", type=int, default=10, help="max scenes to download")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = "copernicus-cdse-stac"
    try:
        token = _auth_token()
        features, _ = retry(
            search_stac, args.bbox, args.sensor, args.date, limit=args.limit, token=token
        )
    except (urllib.error.URLError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"✗ Network unavailable or CDSE error: {exc}", file=sys.stderr)
        print("  Offline-safe exit (no files written).", file=sys.stderr)
        return 0

    if not features:
        print(f"✗ No {args.sensor} scenes found over bbox in {args.date}.", file=sys.stderr)
        return 0

    print(f"Found {len(features)} scene(s); downloading to {out_dir}")
    n = 0
    for item in features:
        scene_id = item.get("id", f"scene-{n}")
        try:
            path, retries, url = download_scene(item, out_dir, token=token)
        except Exception as exc:  # noqa: BLE001 — one bad scene shouldn't kill the batch
            print(f"  ✗ failed {scene_id}: {exc}", file=sys.stderr)
            continue
        write_provenance(
            provenance_path_for(path),
            source=source,
            bbox=args.bbox,
            acquired_at=_acquired_at(item),
            retries=retries,
            scene_id=scene_id,
            download_url=url,
        )
        print(f"  ✓ {path.name} (retries={retries})")
        n += 1
    print(f"✓ Downloaded {n} scene(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
