"""STAC-based discovery of clear Sentinel-2 L2A scenes for Imja gold labels.

Searches the Copernicus Data Space Ecosystem (CDSE) STAC catalogue for the
clearest S2 L2A acquisition within ±7 days of each SAR date that still lacks
a gold label. Produces a JSON manifest of candidate scenes to download.

This module does **not** download anything — downloads require a free CDSE
account and credentials (CDSE_USERNAME/CDSE_PASSWORD or CDSE_TOKEN). The
manifest can be fed to ``python -m siren.ingest.cdse`` or any HTTP client.

Usage:
    python -m siren.ml.stac_label_candidates \
        [--pairs shoulder,winter] \
        [--window 7] \
        [--max-cloud 30] \
        [--out data/processed/gold_label_candidates/stac_manifest.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from siren.ml.label_registry import PAIRS, list_labels

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
IMJA_BBOX = (86.65, 27.65, 87.00, 27.98)  # ~Imja AOI


def _date_from_yyyymmdd(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))


def _fmt(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _scene_id_to_safe(scene_id: str) -> str:
    """Heuristic STAC scene id -> expected SAFE zip filename."""
    # Typical id: S2C_MSIL2A_20251117T044959_N0511_R076_T45RVM_20251117T065411
    return f"{scene_id}.SAFE.zip"


def discover_candidates(
    pair_names: list[str] | None = None,
    window_days: int = 7,
    max_cloud_fraction: float = 30.0,
) -> dict[str, Any]:
    """Return a manifest of clear S2 L2A scenes for missing gold dates.

    The manifest is keyed by SAR date and contains the best (clearest,
    nearest) STAC feature for each date, plus a list of fallback features
    sorted by cloud fraction then temporal offset.
    """
    from siren.ingest.cdse import search_all_scenes

    labels = list_labels()
    manifest: dict[str, Any] = {
        "imja_bbox": IMJA_BBOX,
        "window_days": window_days,
        "max_cloud_fraction": max_cloud_fraction,
        "download_auth": "Set CDSE_USERNAME/CDSE_PASSWORD or CDSE_TOKEN",
        "candidates": {},
    }

    considered: set[str] = set()
    for name, meta in PAIRS.items():
        if pair_names and name not in pair_names:
            continue
        for key in ("t0", "t1"):
            sar_date = meta[key]
            if sar_date in considered:
                continue
            considered.add(sar_date)
            rec = labels.get(sar_date)
            if rec and rec["tier"] == "gold" and rec["exists"]:
                continue  # already have gold label

            sar_d = _date_from_yyyymmdd(sar_date)
            start = _fmt(sar_d - timedelta(days=window_days))
            end = _fmt(sar_d + timedelta(days=window_days))
            try:
                feats = search_all_scenes(IMJA_BBOX, "s2", (start, end), max_scenes=100)
            except Exception as exc:
                logger.warning("STAC search failed for %s: %s", sar_date, exc)
                manifest["candidates"][sar_date] = {
                    "sar_date": sar_date,
                    "pair": name,
                    "status": "stac_error",
                    "error": str(exc),
                }
                continue

            if not feats:
                manifest["candidates"][sar_date] = {
                    "sar_date": sar_date,
                    "pair": name,
                    "status": "no_scenes",
                }
                continue

            # Sort by cloud cover (ascending) then temporal offset
            def _score(f: dict) -> tuple[float, int]:
                p = f.get("properties", {})
                cloud = float(
                    p.get("eo:cloud_cover") or p.get("cloud_cover") or 100.0
                )
                acq = p.get("datetime", "")[:10]
                try:
                    acq_d = date.fromisoformat(acq)
                    offset = abs((acq_d - sar_d).days)
                except Exception:
                    offset = 999
                return (cloud, offset)

            ranked = sorted(feats, key=_score)
            best = ranked[0]
            best_props = best.get("properties", {})
            best_cloud = float(
                best_props.get("eo:cloud_cover")
                or best_props.get("cloud_cover")
                or 100.0
            )
            best_acq = best_props.get("datetime", "")[:10]
            best_offset = abs(
                (date.fromisoformat(best_acq) - sar_d).days
            ) if best_acq else None

            # Extract OData product URL if available (requires auth to download)
            assets = best.get("assets", {})
            download_url = None
            for k in ("Product", "product", "download", "data"):
                if k in assets and "href" in assets[k]:
                    download_url = assets[k]["href"]
                    break

            manifest["candidates"][sar_date] = {
                "sar_date": sar_date,
                "pair": name,
                "status": "found",
                "acquisition_date": best_acq,
                "offset_days": best_offset,
                "cloud_fraction": best_cloud,
                "below_threshold": best_cloud <= max_cloud_fraction,
                "scene_id": best.get("id"),
                "expected_filename": _scene_id_to_safe(best.get("id", "")),
                "download_url": download_url,
                "stac_feature": best,
                "alternatives": [
                    {
                        "scene_id": f.get("id"),
                        "acquisition_date": f.get("properties", {}).get("datetime", "")[:10],
                        "cloud_fraction": float(
                            f.get("properties", {}).get("eo:cloud_cover")
                            or f.get("properties", {}).get("cloud_cover")
                            or 100.0
                        ),
                    }
                    for f in ranked[1:6]
                ],
            }

    return manifest


def write_manifest(
    path: Path | str | None = None,
    pair_names: list[str] | None = None,
    window_days: int = 7,
    max_cloud_fraction: float = 30.0,
) -> Path:
    """Discover candidates and write the manifest JSON to disk."""
    out = Path(path) if path else (
        REPO_ROOT / "data" / "processed" / "gold_label_candidates" / "stac_manifest.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = discover_candidates(pair_names, window_days, max_cloud_fraction)
    out.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("wrote STAC manifest to %s", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pairs", default=None,
        help="comma-separated pair names (default: all eval-eligible missing pairs)",
    )
    ap.add_argument("--window", type=int, default=7, help="±days search window")
    ap.add_argument("--max-cloud", type=float, default=30.0, help="cloud fraction threshold")
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "data" / "processed" / "gold_label_candidates" / "stac_manifest.json",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pair_names = args.pairs.split(",") if args.pairs else None
    out = write_manifest(args.out, pair_names, args.window, args.max_cloud)
    print(out)


if __name__ == "__main__":
    main()
