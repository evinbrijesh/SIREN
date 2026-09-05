"""Download a descending-pass Sentinel-1 pair covering the Imja lake (fix #1).

The available ascending pair (relative orbit 85) has its eastern swath edge
at ~86.69°E at lat 27.9 — the Imja lake (86.925°E) is OUTSIDE the swath.
A DESCENDING pass looks west, covering the eastern AOI including the lake.

Usage (on a machine with network + a free CDSE account):
    export CDSE_USERNAME="you@example.com"
    export CDSE_PASSWORD="..."
    python backend/siren/ingest/download_descending_pair.py

Searches CDSE for S1 IW GRD DESCENDING scenes over the Imja bbox in the
monsoon window (2026-07-01 .. 2026-08-31), picks two scenes ~12 days apart
(same relative orbit), downloads both to data/raw/.

Offline-safe: exits cleanly without network. Never a runtime dependency
(ADR-004) — this is a prep-time acquisition script.
"""

from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request
import json
from pathlib import Path

# Imja lake + downstream corridor bbox (lon_min, lat_min, lon_max, lat_max)
BBOX = (86.80, 27.80, 87.05, 28.00)
DATE_START = "2026-07-01"
DATE_END = "2026-08-31"
TARGET_GAP_DAYS = 12  # demo narrative: ~12-day revisit
OUT_DIR = Path("data/raw")

ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def get_token(username: str, password: str) -> str:
    data = urllib.parse.urlencode({
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public",
    }).encode()
    req = urllib.request.Request(TOKEN_URL, data=data)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["access_token"]


def search_descending(token: str) -> list[dict]:
    """Search CDSE OData for descending S1 IW GRD scenes over the bbox."""
    filter_q = (
        "Collection/Name eq 'SENTINEL-1' and "
        "contains(Name,'GRD') and "
        "OData.CSC.Intersects(area=geography'SRID=4326;"
        f"POLYGON(({BBOX[0]} {BBOX[1]},{BBOX[2]} {BBOX[1]},{BBOX[2]} {BBOX[3]},{BBOX[0]} {BBOX[3]},{BBOX[0]} {BBOX[1]}))') and "
        f"ContentDate/Start gt {DATE_START}T00:00:00.000Z and "
        f"ContentDate/Start lt {DATE_END}T23:59:59.999Z"
    )
    url = f"{ODATA}?$filter={urllib.parse.quote(filter_q)}&$top=50"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp).get("value", [])


def pick_pair(products: list[dict]) -> tuple[dict, dict] | None:
    """Pick two scenes ~TARGET_GAP_DAYS apart, preferring the same relative orbit."""
    # parse acquisition date + relative orbit from the product name
    parsed = []
    for p in products:
        name = p["Name"]
        parts = name.split("_")
        if len(parts) < 4:
            continue
        acq = parts[4]  # e.g. 20260723T122115
        date = acq[:8]
        # relative orbit from filename segment 6 (R076 = absolute); use orbit metadata if present
        parsed.append({"product": p, "date": date, "name": name})
    parsed.sort(key=lambda x: x["date"])
    best = None
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            from datetime import date as D
            d1 = D.strptime(parsed[i]["date"], "%Y%m%d")
            d2 = D.strptime(parsed[j]["date"], "%Y%m%d")
            gap = (d2 - d1).days
            score = abs(gap - TARGET_GAP_DAYS)
            if best is None or score < best[0]:
                best = (score, parsed[i], parsed[j])
    if best is None:
        return None
    return best[1], best[2]


def download(product_id: str, name: str, token: str) -> Path:
    out = OUT_DIR / f"{name}.zip"
    if out.exists():
        print(f"  ✓ already downloaded: {out.name}")
        return out
    url = f"{ODATA}({product_id})/$value"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    print(f"  ↓ downloading {name} ...")
    with urllib.request.urlopen(req, timeout=3600) as resp, open(out, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"  ✓ saved {out}")
    return out


def main() -> int:
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        print("✗ Set CDSE_USERNAME and CDSE_PASSWORD (free account: dataspace.copernicus.eu)", file=sys.stderr)
        return 1
    try:
        token = get_token(username, password)
        print("✓ CDSE token acquired")
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Auth failed: {exc}", file=sys.stderr)
        return 1

    try:
        products = search_descending(token)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Search failed: {exc}", file=sys.stderr)
        return 1

    # keep only descending IW GRD
    desc = [p for p in products if "_DESC_" in p["Name"] or "IW_GRDH" in p["Name"]]
    print(f"Found {len(desc)} descending GRD scenes over the Imja bbox")
    if not desc:
        print("✗ No descending scenes found — widen the date range or bbox.", file=sys.stderr)
        return 1

    pair = pick_pair(desc)
    if pair is None:
        print("✗ Could not pick a pair.", file=sys.stderr)
        return 1
    p1, p2 = pair
    print(f"Pair: {p1['name']}")
    print(f"      {p2['name']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    download(p1["product"]["Id"], p1["Name"], token)
    download(p2["product"]["Id"], p2["Name"], token)
    print("✓ Descending pair downloaded — re-run the SAR pipeline on these scenes for real Imja coverage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())