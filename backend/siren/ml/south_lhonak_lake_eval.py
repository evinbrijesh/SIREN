"""South Lhonak pre/post Pleiades DEM lake diagnostic.

Measures what a 1 m DSM can honestly measure about the October 2023
South Lhonak GLOF and compares the load-bearing Huggel volume estimate
against the real event:

    - Pre-event lake surface area and elevation (flat-surface detection).
    - Post-event residual surface area and elevation -> surface drawdown.
    - Visible dewatered volume: sum of positive (pre - post) elevation
      change over the pre-event lake footprint. This is a LOWER BOUND on
      the released water volume — a DSM cannot see the still-submerged
      part of the emptied bowl, so drawdown below the post-event
      waterline is invisible.
    - Huggel (2002) area-volume estimate for the measured area vs the
      documented release of ~40-50 x 10^6 m^3.

Ground truth (Sattar et al. 2025, Science; ICIMOD/DOe reports):
    - Event: moraine failure + lake outburst, 2023-10-03 ~22:30 IST
    - Released volume: ~40-50 MCM (documented range)
    - Lake still exists post-event (partial drawdown, not full emptying)

Data: data/datasets/south_lhonak_pleiades_dem/ (CC BY 4.0, Gascoin &
Cook, Zenodo 13124662). 1 m Pleiades DEMs, UTM 45N, pre 2022-10-18 and
post 2023-10-29 (26 days after the event).

Usage:
    python -m siren.ml.south_lhonak_lake_eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds
from scipy import ndimage

from siren.risk.breach_volume import HUGGEL_ALPHA, HUGGEL_GAMMA

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEM_DIR = REPO_ROOT / "data" / "datasets" / "south_lhonak_pleiades_dem"
PRE_DEM = DEM_DIR / "20221018.tif"
POST_DEM = DEM_DIR / "20231029.tif"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"

# Lake-basin window in UTM 45N (EPSG:32645), enclosing South Lhonak lake
# plus the moraine outlet. Verified against the pre-event DEM: the flat
# water surface clusters at E 620.8k-621.8k, N 3084.9k-3086.3k.
LAKE_BOX_UTM45N = (620500.0, 3084200.0, 622200.0, 3087000.0)

# Documented ground truth for the 2023-10-03 event
GROUND_TRUTH = {
    "event": "South Lhonak GLOF, 2023-10-03",
    "release_volume_mcm": (40.0, 50.0),
    "lake_area_km2_published": (0.9, 1.2),  # pre-event literature range
    "drawdown_m_published": (10.0, 20.0),
}


def _largest_flat_region(
    dem: np.ndarray,
    elev_lo: float,
    elev_hi: float,
    std_window: int = 9,
    std_max: float = 1.5,
    open_size: int = 5,
) -> np.ndarray:
    """Return a boolean mask of the largest connected flat region in an
    elevation band — the water-surface signature in a DSM.

    A water surface is locally flat over scales much larger than the
    surrounding moraine/ice terrain. The mask is cleaned by morphological
    opening so isolated flat speckles cannot win the connected-component
    vote. Returns all-False when nothing qualifies.
    """
    filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    local_std = ndimage.generic_filter(filled, np.std, size=std_window)
    flat = (local_std < std_max) & (dem > elev_lo) & (dem < elev_hi)
    flat = ndimage.binary_opening(flat, structure=np.ones((open_size, open_size)))
    labels, n = ndimage.label(flat)
    if n == 0:
        return np.zeros(dem.shape, dtype=bool)
    sizes = ndimage.sum_labels(flat, labels, range(1, n + 1))
    return labels == int(np.argmax(sizes)) + 1


def measure_lake(
    pre_path: Path = PRE_DEM,
    post_path: Path = POST_DEM,
    box: tuple = LAKE_BOX_UTM45N,
    elev_band_pre: tuple = (5160.0, 5225.0),
    elev_band_post: tuple = (5100.0, 5225.0),
) -> dict:
    """Measure lake area, drawdown, and visible dewatered volume.

    Returns a dict of raw measurements (SI units) plus a provenance note.
    """
    with rasterio.open(pre_path) as pre:
        win = from_bounds(*box, transform=pre.transform)
        win = win.round_offsets().round_lengths()
        pre_arr = pre.read(1, window=win, masked=True).filled(np.nan)
        win_transform = pre.window_transform(win)
        cell_m2 = abs(win_transform.a * win_transform.e)

    post_arr = np.full(pre_arr.shape, np.nan, dtype=np.float64)
    with rasterio.open(post_path) as post:
        reproject(
            source=rasterio.band(post, 1),
            destination=post_arr,
            src_transform=post.transform,
            src_crs=post.crs,
            src_nodata=post.nodata,
            dst_transform=win_transform,
            dst_crs=pre.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    lake_pre = _largest_flat_region(pre_arr, *elev_band_pre)
    lake_post = _largest_flat_region(post_arr, *elev_band_post)

    area_pre_m2 = float(lake_pre.sum() * cell_m2)
    area_post_m2 = float(lake_post.sum() * cell_m2)
    surf_pre_m = float(np.nanmedian(pre_arr[lake_pre])) if lake_pre.any() else np.nan
    surf_post_m = (
        float(np.nanmedian(post_arr[lake_post])) if lake_post.any() else np.nan
    )

    diff = pre_arr - post_arr  # positive = surface lowered (emptying)
    dewatered = lake_pre & (diff > 1.0)
    visible_emptied_m3 = float(np.nansum(np.where(dewatered, diff, 0.0)) * cell_m2)

    huggel_m3 = HUGGEL_ALPHA * area_pre_m2**HUGGEL_GAMMA

    return {
        "cell_m2": cell_m2,
        "lake_area_pre_m2": area_pre_m2,
        "lake_area_post_m2": area_post_m2,
        "surface_elev_pre_m": surf_pre_m,
        "surface_elev_post_m": surf_post_m,
        "surface_drawdown_m": surf_pre_m - surf_post_m,
        "dewatered_px": int(dewatered.sum()),
        "visible_emptied_m3": visible_emptied_m3,
        "huggel_volume_m3": huggel_m3,
        "provenance": {
            "pre_dem": pre_path.name,
            "post_dem": post_path.name,
            "lake_box_utm45n": box,
            "method": (
                "largest connected flat region (local std < 1.5 over 9x9) "
                "within the lake elevation band"
            ),
        },
    }


def evaluate(meas: dict) -> dict:
    """Compare measurements against documented ground truth."""
    rel_lo, rel_hi = GROUND_TRUTH["release_volume_mcm"]
    area_pub_lo, area_pub_hi = GROUND_TRUTH["lake_area_km2_published"]

    huggel_mcm = meas["huggel_volume_m3"] / 1e6
    area_km2 = meas["lake_area_pre_m2"] / 1e6
    visible_mcm = meas["visible_emptied_m3"] / 1e6

    # Huggel predicts TOTAL lake volume; the documented figure is the
    # RELEASED fraction (the lake did not fully empty). A defensible
    # comparison is order-of-magnitude: does the formula land near the
    # true total (release + residual)?
    huggel_vs_release_pct = 100.0 * (huggel_mcm - (rel_lo + rel_hi) / 2) / (
        (rel_lo + rel_hi) / 2
    )

    # Sensitivity: Huggel across the published area range, since the
    # flat-surface area estimate biases low at lake fringes.
    huggel_area_sweep_mcm = [
        round(HUGGEL_ALPHA * (a * 1e6) ** HUGGEL_GAMMA / 1e6, 1)
        for a in (area_pub_lo, area_km2, area_pub_hi)
    ]

    return {
        "lake_area_km2": round(area_km2, 3),
        "lake_area_within_published_range": bool(area_pub_lo <= area_km2 <= area_pub_hi),
        "surface_drawdown_m": round(meas["surface_drawdown_m"], 1),
        "visible_emptied_mcm": round(visible_mcm, 2),
        "huggel_total_volume_mcm": round(huggel_mcm, 1),
        "documented_release_mcm": list(GROUND_TRUTH["release_volume_mcm"]),
        "huggel_vs_documented_release_pct": round(huggel_vs_release_pct, 1),
        "huggel_area_sweep_mcm": {
            "areas_km2": [area_pub_lo, round(area_km2, 3), area_pub_hi],
            "volumes_mcm": huggel_area_sweep_mcm,
        },
        "limitations": [
            "DSM differencing only captures drawdown above the post-event "
            "waterline; the still-submerged emptied volume is invisible, "
            "so visible_emptied_mcm is a strict lower bound.",
            "Flat-surface detection may under-cover lake fringes "
            "(shadowed/icy margins), biasing the area estimate low.",
            "Post-event 'flat' pixels include exposed lakebed/mudflat, "
            "so lake_area_post overstates residual water extent.",
            "Huggel predicts total lake volume; the documented 40-50 MCM "
            "is the released fraction of a partially-emptied lake.",
        ],
    }


def run_evaluation(
    pre_path: Path = PRE_DEM,
    post_path: Path = POST_DEM,
) -> dict:
    meas = measure_lake(pre_path, post_path)
    results = {
        "event": GROUND_TRUTH["event"],
        "measurements": {k: v for k, v in meas.items() if k != "provenance"},
        "provenance": meas["provenance"],
    }
    results.update(evaluate(meas))

    print()
    print("=" * 70)
    print("South Lhonak Pleiades DEM Lake Diagnostic")
    print("=" * 70)
    print(f"  Pre-event lake area:   {results['lake_area_km2']:.3f} km2 "
          f"(published range {area_range_str()})")
    print(f"  Surface elevation:     {meas['surface_elev_pre_m']:.0f} m -> "
          f"{meas['surface_elev_post_m']:.0f} m "
          f"(drawdown {results['surface_drawdown_m']} m)")
    print(f"  Visible emptied vol:   {results['visible_emptied_mcm']:.2f} MCM "
          f"(lower bound — submerged bowl invisible)")
    print(f"  Huggel total volume:   {results['huggel_total_volume_mcm']} MCM")
    print(f"  Documented release:    {rel_range_str()} MCM")
    print(f"  Huggel vs release:     {results['huggel_vs_documented_release_pct']:+.1f}%")
    print("=" * 70)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINT_DIR / "south_lhonak_lake_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved: %s", out_path)
    return results


def area_range_str() -> str:
    lo, hi = GROUND_TRUTH["lake_area_km2_published"]
    return f"{lo}-{hi} km2"


def rel_range_str() -> str:
    lo, hi = GROUND_TRUTH["release_volume_mcm"]
    return f"{lo:.0f}-{hi:.0f}"


def main():
    parser = argparse.ArgumentParser(
        description="South Lhonak Pleiades DEM lake diagnostic"
    )
    parser.add_argument("--pre-dem", type=Path, default=PRE_DEM)
    parser.add_argument("--post-dem", type=Path, default=POST_DEM)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_evaluation(args.pre_dem, args.post_dem)


if __name__ == "__main__":
    main()
