"""Label-refined Himalayan lake chips — per-date NDWI + contracted inventory.

Follow-up to the 2026-09-18 gate failure (ADR-014): the stratified
adapter over-segments ~1–2 SAR px (~90–180 m) beyond the true waterline
because it learned the *median* 2022–2024 inventory outline, which is
~19% wider than any given date's edge (gold ⊂ inventory, P 0.99 /
R 0.81). This module rebuilds the chip labels from two sources:

  (a) per-date NDWI targets — where any downloaded S2 L2A scene within
      ±21 days of t1 (all tiles covering the swath) has usable SCL
      coverage, a SAR pixel is water iff >50% of its ~90 m footprint is
      NDWI-water (NDWI > 0.15 on SCL-valid pixels, the gold-label
      criterion). Per pixel the scene nearest to t1 wins. Teaches
      "majority water", not "any water".
  (b) contracted inventory targets — where no S2 scene covers the pixel,
      the inventory polygon eroded by ~1 SAR px, matching the measured
      median-vs-date bias. Works on all chips.

Both are applied per-pixel on the full SAR grid, then cropped to the
existing chip windows — a chip can mix sources along its coverage seams.

Honesty notes:
  * Monsoon optical coverage is partial even across tiles — (a) covers
    what is clear; (b) carries the rest.
  * Merged scenes carry ±1–11 day offsets from t1 — monsoon shorelines
    can drift; the >50%-footprint rule makes the label robust to
    sub-footprint drift but not to real expansion events.
  * Gold labels (``imja_gold_label_*.tif``) are eval-only — they sit on
    the held-out eval dates and are never consumed here.
  * Eroded fallback can erase micro-tarn positives entirely (lake < ~2
    px); those chips become all-negative — honest at ~90 m pitch.

Usage:
    python -m siren.ml.lake_label_refine [--chips DIR] [--out DIR]
                                         [--s2 ZIP [ZIP ...]]
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform as warp_transform

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

SAR_T1 = PROCESSED_DIR / "imja_desc_20260714_sar_vv_vh_db.tif"
S2_T1_DATE = "20260714"
# Every downloaded S2 L2A scene contributes per-date labels where its
# tile + clear SCL cover a SAR pixel; per pixel the scene nearest to t1
# wins. Resolved at runtime from data/raw (``s2_scenes()``) so newly
# acquired tiles are picked up automatically.
S2_SCENE_GLOB = "S2*_MSIL2A_*.zip"
CHIPS_DIR = DATA_DIR / "datasets" / "himalayan_lake_chips"
OUT_DIR = DATA_DIR / "datasets" / "himalayan_lake_chips_refined"

CHIP = 96
NDWI_THRESHOLD = 0.15        # same criterion as the gold labels
SCL_VALID = {4, 5, 6, 7, 11}  # veg / bare / water / unclassified / snow-ice
LABEL_CELL_DEG = 0.0002      # ~20–22 m — near SCL native pitch
FOOTPRINT_HALF_CELLS = 2     # 5×5 cells ≈ 100 m ≈ SAR footprint
MIN_VALID_FRAC = 0.5         # need ≥50% usable SCL in the footprint
WATER_FRAC = 0.5             # >50% of footprint water => positive
ERODE_PX = 1                 # inventory contraction (~90 m)

SRC_S2_NDWI = 1
SRC_INVENTORY_ERODED = 2
S2_WINDOW_DAYS = 21           # label relevance window around the SAR pair


def _scene_date(path: Path) -> str | None:
    import re
    m = re.search(r"_(\d{8})T", path.name)
    return m.group(1) if m else None


def s2_scenes(
    raw_dir: Path | str | None = None,
    t1_date: str = S2_T1_DATE,
    window_days: int = S2_WINDOW_DAYS,
) -> list[Path]:
    """S2 scenes within ``window_days`` of t1, nearest date first.

    Per-date labels are only meaningful near the SAR pair — a stale
    scene's shoreline is weaker evidence than the contracted inventory,
    so scenes outside the window are excluded rather than merged.
    """
    import datetime as _dt

    raw = Path(raw_dir) if raw_dir else DATA_DIR / "raw"
    t1 = _dt.date(int(t1_date[:4]), int(t1_date[4:6]), int(t1_date[6:]))
    picked = []
    for p in sorted(raw.glob(S2_SCENE_GLOB)):
        d = _scene_date(p)
        if not d:
            continue
        dd = _dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))
        dist = abs((dd - t1).days)
        if dist <= window_days:
            picked.append((dist, d, p))
    picked.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in picked]


def _s2_bounds_lonlat(s2_zip: Path) -> tuple[float, float, float, float]:
    """S2 tile footprint (west, south, east, north) in EPSG:4326."""
    from siren.preprocess.s2_optical import _find_band_path

    with zipfile.ZipFile(str(s2_zip)) as zf:
        with zf.open(_find_band_path(zf, "B03", "20m")) as f:
            with rasterio.open(f) as src:
                b = src.bounds
                xs, ys = warp_transform(
                    src.crs, "EPSG:4326",
                    [b.left, b.right, b.left, b.right],
                    [b.bottom, b.bottom, b.top, b.top],
                )
    return min(xs), min(ys), max(xs), max(ys)


def _s2_water_valid_grid(
    s2_zip: Path, bounds: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NDWI-water + SCL-valid masks on a regular EPSG:4326 ~20 m grid.

    Returns (water, valid, transform): uint8 masks over ``bounds``.
    """
    from siren.preprocess.s2_optical import _find_band_path

    west, south, east, north = bounds
    w = int(np.ceil((east - west) / LABEL_CELL_DEG))
    h = int(np.ceil((north - south) / LABEL_CELL_DEG))
    dst_transform = from_bounds(west, south, east, north, w, h)

    b03 = np.zeros((h, w), dtype=np.float32)
    b08 = np.zeros((h, w), dtype=np.float32)
    scl = np.zeros((h, w), dtype=np.uint8)
    with zipfile.ZipFile(str(s2_zip)) as zf:
        # B08 exists only at 10 m (the 20 m NIR is B8A); warping it to
        # the ~20 m label grid handles the resampling.
        for name, res, out, resampling in (
            ("B03", "20m", b03, Resampling.bilinear),
            ("B08", "10m", b08, Resampling.bilinear),
            ("SCL", "20m", scl, Resampling.nearest),
        ):
            with zf.open(_find_band_path(zf, name, res)) as f:
                with rasterio.open(f) as src:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=out,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs="EPSG:4326",
                        resampling=resampling,
                    )

    valid = np.isin(scl, list(SCL_VALID))
    ndwi = (b03 - b08) / (b03 + b08 + 1e-6)
    water = valid & (ndwi > NDWI_THRESHOLD)
    return water.astype(np.uint8), valid.astype(np.uint8), dst_transform


def _windowed_fraction(
    water: np.ndarray,
    valid: np.ndarray,
    transform,
    lon: np.ndarray,
    lat: np.ndarray,
    half_cells: int = FOOTPRINT_HALF_CELLS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-SAR-pixel water fraction + usable-SCL fraction over the ~90 m
    footprint, via summed-area tables. Returns (water_frac, valid_frac)
    float32 arrays on the SAR grid; both 0 where the window has no
    overlap with the label grid."""
    h, w = water.shape
    inv = ~transform
    cols = inv.a * lon + inv.b * lat + inv.c
    rows = inv.d * lon + inv.e * lat + inv.f

    iw = np.zeros((h + 1, w + 1), dtype=np.int64)
    iw[1:, 1:] = np.cumsum(np.cumsum(water, axis=0), axis=1)
    iv = np.zeros((h + 1, w + 1), dtype=np.int64)
    iv[1:, 1:] = np.cumsum(np.cumsum(valid, axis=0), axis=1)

    r1 = np.clip(np.floor(rows).astype(np.int64) - half_cells, 0, h)
    r2 = np.clip(np.floor(rows).astype(np.int64) + half_cells + 1, 0, h)
    c1 = np.clip(np.floor(cols).astype(np.int64) - half_cells, 0, w)
    c2 = np.clip(np.floor(cols).astype(np.int64) + half_cells + 1, 0, w)
    win_n = (r2 - r1) * (c2 - c1)

    wsum = iw[r2, c2] - iw[r1, c2] - iw[r2, c1] + iw[r1, c1]
    vsum = iv[r2, c2] - iv[r1, c2] - iv[r2, c1] + iv[r1, c1]

    water_frac = np.where(vsum > 0, wsum / np.maximum(vsum, 1), 0.0)
    valid_frac = np.where(win_n > 0, vsum / np.maximum(win_n, 1), 0.0)
    return water_frac.astype(np.float32), valid_frac.astype(np.float32)


def refine_labels(
    chips_dir: Path | str = CHIPS_DIR,
    s2_zips: list[Path | str] | Path | str | None = None,
    sar_path: Path | str = SAR_T1,
    out_dir: Path | str = OUT_DIR,
    erode_px: int = ERODE_PX,
) -> dict:
    """Rebuild chip labels: per-date NDWI where any S2 scene covers the
    pixel (nearest-to-t1 wins), eroded inventory elsewhere. Writes
    chips.npz + manifest.json + report.json to ``out_dir``."""
    from scipy.ndimage import binary_erosion

    from siren.detect.sar import sar_grid_lonlat
    from siren.ml.himalayan_lake_dataset import (
        lake_grid_positions,
        load_lake_inventory,
        rasterize_lake_labels,
    )

    chips_dir, out_dir = Path(chips_dir), Path(out_dir)
    if s2_zips is None:
        scenes = s2_scenes()
    elif isinstance(s2_zips, (str, Path)):
        scenes = [Path(s2_zips)]
    else:
        scenes = [Path(p) for p in s2_zips]
    data = np.load(chips_dir / "chips.npz")
    x = data["x"]
    manifest = json.loads((chips_dir / "manifest.json").read_text())

    ll = sar_grid_lonlat(str(sar_path))
    if ll is None:
        raise RuntimeError(f"no GCP geolocation in {sar_path}")
    lon, lat = ll

    # (b) fallback: inventory rasterised on the SAR grid, eroded ~1 px.
    lakes = load_lake_inventory(min_elev_m=4000, min_area_km2=0.02)
    positions, _, _ = lake_grid_positions(lakes, str(sar_path))
    inventory = rasterize_lake_labels(lakes, positions, str(sar_path)) > 0
    fallback = binary_erosion(inventory, iterations=erode_px)

    # (a) per-date NDWI merged across scenes: each SAR pixel takes the
    # label of the first scene (nearest to t1) whose ~90 m footprint is
    # ≥MIN_VALID_FRAC usable SCL; pixels no scene covers fall back.
    refined = fallback.astype(np.uint8)
    src_map = np.full(lon.shape, SRC_INVENTORY_ERODED, dtype=np.uint8)
    scene_stats = []
    covered = np.zeros(lon.shape, dtype=bool)
    for idx, s2_zip in enumerate(scenes):
        if not s2_zip.exists():
            logger.warning("S2 scene missing: %s", s2_zip)
            continue
        tb = _s2_bounds_lonlat(s2_zip)
        west = max(tb[0], float(np.nanmin(lon)) - 0.002)
        south = max(tb[1], float(np.nanmin(lat)) - 0.002)
        east = min(tb[2], float(np.nanmax(lon)) + 0.002)
        north = min(tb[3], float(np.nanmax(lat)) + 0.002)
        if not (east > west and north > south):
            logger.warning("%s does not intersect the SAR grid", s2_zip.name)
            continue
        water, valid, transform = _s2_water_valid_grid(
            s2_zip, (west, south, east, north)
        )
        wfrac, vfrac = _windowed_fraction(water, valid, transform, lon, lat)
        s2_ok = vfrac >= MIN_VALID_FRAC
        take = s2_ok & ~covered
        refined[take] = (wfrac > WATER_FRAC)[take].astype(np.uint8)
        src_map[take] = SRC_S2_NDWI
        covered |= s2_ok
        scene_stats.append({
            "scene": s2_zip.name,
            "date": _scene_date(s2_zip),
            "grid_valid_frac": round(float(s2_ok.mean()), 4),
            "px_labeled": int(take.sum()),
        })
        logger.info(
            "%s: %.1f%% grid footprint-valid, %d px newly labeled",
            s2_zip.name, 100.0 * float(s2_ok.mean()), int(take.sum()),
        )

    s2_ok = covered

    half = CHIP // 2
    y_out = np.empty((len(manifest), CHIP, CHIP), dtype=np.uint8)
    src_out = np.empty((len(manifest), CHIP, CHIP), dtype=np.uint8)
    for i, m in enumerate(manifest):
        r0, c0 = m["row"] - half, m["col"] - half
        y_out[i] = refined[r0:r0 + CHIP, c0:c0 + CHIP]
        src_out[i] = src_map[r0:r0 + CHIP, c0:c0 + CHIP]
        m["label_src_frac"] = round(float((src_out[i] == SRC_S2_NDWI).mean()), 4)
        m["pos_px_refined"] = int(y_out[i].sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "chips.npz", x=x, y=y_out, src=src_out)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    lake = np.array([m["kind"] == "lake" for m in manifest])
    pos = np.array([m["pos_px_refined"] for m in manifest])
    orig = np.array([m["pos_px"] for m in manifest])
    s2_chip = np.array([m["label_src_frac"] for m in manifest])
    report = {
        "experiment": "label-refined chip rebuild (per-date NDWI + eroded inventory)",
        "label_semantics": {
            "s2_ndwi": (
                f">50% of ~90 m footprint NDWI>{NDWI_THRESHOLD} on SCL-valid "
                f"px, nearest-to-t1 scene wins; ≥{MIN_VALID_FRAC:.0%} usable "
                "SCL in footprint required"
            ),
            "inventory_eroded": (
                f"median inventory polygon eroded {erode_px} px (~90 m) — "
                "fallback where S2 coverage absent/cloudy"
            ),
        },
        "scenes": {
            "sar_t1": Path(sar_path).name,
            "s2": [s.name for s in scenes if s.exists()],
            "s2_window_days": S2_WINDOW_DAYS,
            "s2_per_scene": scene_stats,
        },
        "chips_total": int(len(manifest)),
        "grid_s2_valid_frac": round(float(s2_ok.mean()), 4),
        "lake_chips": {
            "total": int(lake.sum()),
            "fully_s2": int(((s2_chip > 0.99) & lake).sum()),
            "partial_s2": int(((s2_chip > 0.01) & (s2_chip <= 0.99) & lake).sum()),
            "no_s2": int(((s2_chip <= 0.01) & lake).sum()),
            "pos_px_median_before": float(np.median(orig[lake])),
            "pos_px_median_after": float(np.median(pos[lake])),
            "chips_eroded_to_zero": int(((pos == 0) & (orig > 0) & lake).sum()),
        },
        "pos_px_total_before": int(orig.sum()),
        "pos_px_total_after": int(pos.sum()),
        "limitations": [
            "Monsoon optical coverage is partial — per-date labels cover "
            "wherever any in-window scene has usable SCL; eroded-"
            "inventory fallback carries the rest (label seams inside "
            "mixed chips).",
            "Merged scenes carry ±1–11 day offsets from t1; monsoon "
            "shoreline drift inside a lag is absorbed into the "
            ">50%-footprint rule but real expansion would be "
            "mislabelled.",
            "Eroded fallback erases micro-tarn positives (<~2 px); "
            "those chips become all-negative.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    logger.info("wrote refined chips to %s", out_dir)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chips", default=str(CHIPS_DIR))
    ap.add_argument("--s2", nargs="*", default=None,
                    help="S2 SAFE zips to merge (default: all raw scenes "
                         "within ±%d days of t1)" % S2_WINDOW_DAYS)
    ap.add_argument("--sar", default=str(SAR_T1))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--erode-px", type=int, default=ERODE_PX)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(refine_labels(
        args.chips, args.s2, args.sar, args.out, args.erode_px
    ), indent=1))


if __name__ == "__main__":
    main()
