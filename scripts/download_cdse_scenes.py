#!/usr/bin/env python3
"""Download Sentinel-1/2 scenes from CDSE for ML training pairs.

Searches the CDSE STAC API for scenes matching target dates and downloads
the full SAFE archives with resume support.

Usage:
    python scripts/download_cdse_scenes.py --group south_lhonak_2023
    python scripts/download_cdse_scenes.py --group imja_2025
    python scripts/download_cdse_scenes.py --group imja_2026_pre
    python scripts/download_cdse_scenes.py --all

Credentials are read from the repo-root .env file (CDSE_USERNAME/CDSE_PASSWORD).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"

# Bounding boxes (west, south, east, north)
BBOX_IMJA = [86.85, 27.85, 87.05, 28.05]
BBOX_LHONAK = [88.50, 27.60, 88.80, 27.95]

# Download groups: (collection, target_dates, bbox, output_subdir)
GROUPS = {
    "south_lhonak_2023": {
        "bbox": BBOX_LHONAK,
        "out": RAW_DIR / "south_lhonak_2023",
        "s1": ["2023-09-25", "2023-09-28", "2023-10-02", "2023-10-07",
               "2023-10-10", "2023-10-14", "2023-10-26"],
        "s2": ["2023-09-26", "2023-10-06", "2023-10-16", "2023-10-21",
               "2023-10-26"],
    },
    "imja_2025": {
        "bbox": BBOX_IMJA,
        "out": RAW_DIR / "imja_2025",
        "s1": ["2025-10-28", "2025-11-09", "2025-11-13", "2025-12-03",
               "2025-12-15"],
        "s2": ["2025-10-25", "2025-11-07", "2025-11-12", "2025-12-02",
               "2025-12-17"],
    },
    "imja_2026_pre": {
        "bbox": BBOX_IMJA,
        "out": RAW_DIR / "imja_2026_pre",
        "s1": ["2026-06-13", "2026-06-17"],
        "s2": ["2026-06-15", "2026-06-20"],
    },
}

COLLECTIONS = {
    "s1": "sentinel-1-grd",
    "s2": "sentinel-2-l2a",
}


def load_env() -> None:
    """Load credentials from repo-root .env if not already in environment.

    Handles the common dotenv quoting styles: KEY=value, KEY='value',
    KEY="value".
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def get_token() -> str:
    user = os.environ.get("CDSE_USERNAME")
    pw = os.environ.get("CDSE_PASSWORD")
    if not user or not pw:
        sys.exit("CDSE_USERNAME/CDSE_PASSWORD not set (check repo-root .env)")
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "password",
            "username": user,
            "password": pw,
            "client_id": "cdse-public",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_scene(
    token: str, collection: str, bbox: list[float], date: str, tolerance_days: int = 1
) -> dict | None:
    """Find the best scene for a target date (lowest cloud for S2)."""
    dt = datetime.strptime(date, "%Y-%m-%d")
    start = (dt - timedelta(days=tolerance_days)).strftime("%Y-%m-%dT00:00:00Z")
    end = (dt + timedelta(days=tolerance_days)).strftime("%Y-%m-%dT23:59:59Z")
    body = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{start}/{end}",
        "limit": 50,
    }
    resp = requests.post(
        STAC_URL,
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=90,
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        return None
    if collection == "sentinel-2-l2a":
        features.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    else:
        # Prefer the pass closest to the target date
        features.sort(
            key=lambda f: abs(
                (
                    datetime.fromisoformat(
                        f["properties"]["datetime"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    - dt
                ).total_seconds()
            )
        )
    return features[0]


def download_scene(token: str, scene: dict, out_dir: Path) -> bool:
    """Download a scene with resume support. Returns True if complete."""
    scene_id = scene["id"]
    filename = f"{scene_id}.zip"
    dest = out_dir / filename

    asset = scene.get("assets", {}).get("Product")
    if asset is None:
        for key in ("data", "product", "download"):
            if key in scene.get("assets", {}):
                asset = scene["assets"][key]
                break
    if asset is None or not asset.get("href"):
        print(f"    [error] no downloadable Product asset for {scene_id}")
        return False
    url = asset["href"]
    expected_size = asset.get("file:size")

    if dest.exists() and dest.stat().st_size > 0:
        if expected_size and dest.stat().st_size == expected_size:
            print(f"    [skip] {filename} already complete "
                  f"({expected_size / 1e9:.2f} GB)")
            return True
        if _zip_valid(dest):
            print(f"    [skip] {filename} already complete (zip valid)")
            return True
        print(f"    [resume] {filename} truncated "
              f"({dest.stat().st_size / 1e9:.2f} GB of "
              f"{(expected_size or 0) / 1e9:.2f} GB)")

    # Refresh the token per scene: CDSE tokens expire (~10 min) and large
    # archives can take longer than that on slow links.
    try:
        token = get_token()
    except Exception as exc:  # noqa: BLE001 — keep the old token as fallback
        print(f"    [warn] token refresh failed ({exc}); using existing token")

    print(f"    [download] {filename}"
          + (f" ({expected_size / 1e9:.2f} GB)" if expected_size else ""))
    cmd = [
        "curl", "-f", "-L", "--retry", "5", "--retry-delay", "10",
        "-C", "-",                       # resume support
        "-H", f"Authorization: Bearer {token}",
        "-o", str(dest),
        "--progress-bar",
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"    [fail] curl exit {result.returncode} for {filename}")
        return False
    if expected_size and dest.stat().st_size != expected_size:
        print(f"    [fail] {filename} size mismatch "
              f"({dest.stat().st_size} != {expected_size})")
        return False
    if not _zip_valid(dest):
        print(f"    [fail] {filename} failed zip integrity check")
        return False
    print(f"    [ok] {filename} ({dest.stat().st_size / 1e9:.2f} GB)")
    return True


def _zip_valid(path: Path) -> bool:
    """Check zip integrity (central directory readable)."""
    result = subprocess.run(
        ["unzip", "-t", "-q", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def run_group(token: str, name: str, group: dict, dry_run: bool = False) -> tuple[int, int]:
    out_dir = group["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for kind in ("s1", "s2"):
        collection = COLLECTIONS[kind]
        for date in group[kind]:
            print(f"  {kind.upper()} {date} ({collection})")
            scene = search_scene(token, collection, group["bbox"], date)
            if scene is None:
                print(f"    [missing] no scene found for {date}")
                fail += 1
                continue
            actual = scene["properties"]["datetime"][:10]
            if actual != date:
                print(f"    [note] using {actual} (closest to {date})")
            if dry_run:
                asset = scene.get("assets", {}).get("Product", {})
                size = asset.get("file:size", 0)
                cc = scene["properties"].get("eo:cloud_cover")
                cloud = f", cloud={cc:.1f}%" if cc is not None else ""
                print(f"    [dry-run] {scene['id']} "
                      f"({size / 1e9:.2f} GB{cloud})")
                ok += 1
                continue
            if download_scene(token, scene, out_dir):
                ok += 1
            else:
                fail += 1
    return ok, fail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=sorted(GROUPS),
        help="Download group to run",
    )
    parser.add_argument("--all", action="store_true", help="Run all groups")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve scenes and print sizes without downloading",
    )
    args = parser.parse_args()

    if not args.group and not args.all:
        parser.error("specify --group or --all")

    load_env()
    token = get_token()
    print("CDSE authentication OK\n")

    groups = sorted(GROUPS) if args.all else [args.group]
    total_ok = total_fail = 0
    for name in groups:
        print(f"=== {name} ===")
        ok, fail = run_group(token, name, GROUPS[name], dry_run=args.dry_run)
        total_ok += ok
        total_fail += fail
        print(f"  -> {ok} ok, {fail} failed\n")

    print(f"DONE: {total_ok} resolved/downloaded, {total_fail} failed")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
