"""Pre-download Copernicus DEM tiles for all Sen1Floods11 chips (Level 2).

Scans all 431 Sen1Floods11 chips across train/val/test splits, determines
which Copernicus GLO-30 DEM tiles are needed, and downloads them to
``data/raw/dem/copernicus_glo30/`` with caching.

This pre-download step avoids network calls during training and ensures
the 4-channel dataset can be built deterministically from cached tiles.

Usage:
    python -m siren.ml.prefetch_dem
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from siren.ml.dem_fetch import (
    DEM_CACHE_DIR,
    tiles_for_bounds,
    download_tile,
)
from siren.ml.dataset import SEN1FLOODS11_ROOT

logger = logging.getLogger(__name__)


def _all_chip_bounds() -> list[tuple[str, float, float, float, float]]:
    """Return (chip_id, west, south, east, north) for all 431 chips."""
    import rasterio

    bounds_list = []
    for split_dir in ["train", "valid", "test"]:
        s1_dir = SEN1FLOODS11_ROOT / split_dir / "S1"
        if not s1_dir.exists():
            continue
        for tif in sorted(s1_dir.glob("*_S1Hand.tif")):
            with rasterio.open(str(tif)) as src:
                b = src.bounds
                bounds_list.append((tif.name, b.left, b.bottom, b.right, b.top))
    return bounds_list


def prefetch_all_dem_tiles() -> dict:
    """Download all Copernicus DEM tiles needed for the Sen1Floods11 dataset.

    Returns:
        Dict with summary stats: total_chips, tiles_needed, tiles_downloaded,
        tiles_failed, cache_dir.
    """
    all_bounds = _all_chip_bounds()
    logger.info("Scanning %d chips for DEM tile requirements...", len(all_bounds))

    # Collect all unique tile coordinates
    all_tile_coords: set[tuple[int, int]] = set()
    for chip_id, w, s, e, n in all_bounds:
        for lat, lon in tiles_for_bounds(w, s, e, n):
            all_tile_coords.add((lat, lon))

    logger.info("Need %d unique DEM tiles for %d chips", len(all_tile_coords), len(all_bounds))

    downloaded = 0
    failed = 0
    failed_tiles: list[str] = []
    for lat, lon in sorted(all_tile_coords):
        try:
            download_tile(lat, lon)
            downloaded += 1
        except Exception as e:
            logger.warning("Failed to download tile (lat=%d, lon=%d): %s", lat, lon, e)
            failed += 1
            failed_tiles.append(f"lat={lat},lon={lon}: {e}")

    logger.info("Downloaded %d tiles, %d failed", downloaded, failed)
    if failed > 0:
        logger.warning("Failed tiles: %s", failed_tiles[:10])

    return {
        "total_chips": len(all_bounds),
        "tiles_needed": len(all_tile_coords),
        "tiles_downloaded": downloaded,
        "tiles_failed": failed,
        "failed_tiles": failed_tiles[:20],
        "cache_dir": str(DEM_CACHE_DIR),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stats = prefetch_all_dem_tiles()
    print()
    print("=" * 60)
    print("Copernicus DEM Prefetch Summary")
    print("=" * 60)
    for k, v in stats.items():
        if k == "failed_tiles" and isinstance(v, list) and len(v) > 5:
            print(f"  {k}: {len(v)} tiles (showing first 5)")
            for t in v[:5]:
                print(f"    {t}")
        else:
            print(f"  {k}: {v}")
    if stats["tiles_failed"] > 0:
        sys.exit(1)
