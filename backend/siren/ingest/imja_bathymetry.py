"""Download Imja Tsho bathymetric surveys (Zenodo + UT Austin repository).

Sources:
    1. Zenodo record 18251249 — April 2002 survey (Sakai/Fujita):
       sounding-line depths through ~70 cm lake ice, GNSS positions,
       WGS84/UTM 45N. Downloaded via the Zenodo REST API (files are
       enumerated from the record metadata — no hardcoded filenames).
    2. UT Austin repository (hdl.handle.net/2152/19754) — September 2012
       sonar survey (Somos-Valenzuela et al. 2014): fetches the item page
       and downloads linked bitstreams matching bathymetry CSV/zip.

Offline-safe (ADR-004): exits cleanly without network, writes nothing.
Provenance sidecars are written beside every downloaded file.

Usage:
    python -m siren.ingest.imja_bathymetry
    python -m siren.ingest.imja_bathymetry --out data/datasets/imja_bathymetry
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api/records"
IMJA_2002_RECORD = "18251249"
UT_2012_ITEM = "https://repositories.lib.utexas.edu/items/2152/19754"
DEFAULT_OUT = Path("data/datasets/imja_bathymetry")
UA = {"User-Agent": "SIREN-ingest/1.0"}


def _fetch_json(url: str, timeout: float = 60.0) -> dict:
    req = Request(url, headers=UA)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _download(url: str, dest: Path, timeout: float = 60.0) -> bool:
    try:
        req = Request(url, headers=UA)
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        dest.with_suffix(dest.suffix + ".provenance.json").write_text(
            json.dumps({
                "source_url": url,
                "downloaded_bytes": len(data),
            }, indent=2)
        )
        logger.info("Downloaded %s -> %s (%d bytes)", url, dest, len(data))
        return True
    except (URLError, OSError, TimeoutError) as exc:
        logger.warning("Fetch failed %s: %s", url, exc)
        return False


def fetch_zenodo_record(record_id: str, out_dir: Path,
                        timeout: float = 60.0) -> list[Path]:
    """Download all files attached to a Zenodo record."""
    try:
        meta = _fetch_json(f"{ZENODO_API}/{record_id}", timeout)
    except (URLError, OSError, TimeoutError) as exc:
        logger.warning("Zenodo record %s unreachable: %s", record_id, exc)
        return []

    files = meta.get("files", [])
    if not files:
        logger.warning("Zenodo record %s lists no files", record_id)
        return []

    downloaded = []
    for f in files:
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        name = f.get("key") or f.get("filename") or "file"
        if not url:
            continue
        dest = out_dir / name
        if _download(url, dest, timeout):
            downloaded.append(dest)
    return downloaded


def fetch_ut_2012(out_dir: Path, timeout: float = 60.0) -> list[Path]:
    """Best-effort fetch of the 2012 Imja survey bitstreams.

    The DSpace item page links bitstreams; we download any whose name
    looks like bathymetry data (csv/zip/txt). If the page layout has
    changed this simply returns what it found — never fabricates.
    """
    try:
        req = Request(UT_2012_ITEM, headers=UA)
        with urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (URLError, OSError, TimeoutError) as exc:
        logger.warning("UT repository unreachable: %s", exc)
        return []

    links = re.findall(r'href="([^"]*bitstream[^"]*)"', html)
    data_links = [
        l for l in links
        if re.search(r"\.(csv|zip|txt|xlsx?|dat)(\?|$)", l, re.IGNORECASE)
        and not re.search(r"license|readme", l, re.IGNORECASE)
    ]
    downloaded = []
    for l in dict.fromkeys(data_links):   # dedupe, keep order
        url = l if l.startswith("http") else (
            "https://repositories.lib.utexas.edu" + l
        )
        name = url.split("/")[-1].split("?")[0] or "imja2012.bin"
        dest = out_dir / f"imja_2012_{name}"
        if _download(url, dest, timeout):
            downloaded.append(dest)
    return downloaded


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    got = fetch_zenodo_record(IMJA_2002_RECORD, args.out / "imja_2002",
                              args.timeout)
    got += fetch_ut_2012(args.out / "imja_2012", args.timeout)

    if not got:
        print("Nothing downloaded (network unavailable or no files found). "
              "Re-run on a networked host.")
        return 1 if args.strict else 0

    manifest = args.out / "manifest.json"
    manifest.write_text(json.dumps({
        "sources": {
            "imja_2002_zenodo": f"{ZENODO_API}/{IMJA_2002_RECORD}",
            "imja_2012_ut": UT_2012_ITEM,
        },
        "files": [str(f) for f in got],
    }, indent=2))
    print(f"✓ {len(got)} file(s) downloaded to {args.out}")
    for f in got:
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
