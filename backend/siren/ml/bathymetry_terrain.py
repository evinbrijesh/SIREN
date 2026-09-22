"""Terrain/morphology feature regression for bathymetry (E2, Level 3.2).

The metadata-only corrector (``bathymetry_residual.py``: log area +
lake_type + region one-hots) reaches 68.0% overall / 36.6% Himalaya
grouped-LOO MAPE vs Huggel's 76.6% / 33.5% — no in-domain win. This
module adds the missing information source the roadmap calls for:
per-lake TERRAIN and MORPHOLOGY features derived from the Copernicus
DEM GLO30 tiles and RGI glacier outlines, which are deployable (no
survey data required):

    - lake geometry: area, perimeter, compactness, elongation
    - dam/rim steepness: slope statistics in the ring around the lake
      edge (moraine dam face — the feature the area-only formulas miss)
    - terrain context: window relief and mean slope
    - surface elevation: rim-min DEM (hypsometric convention)
    - glacier context: distance to nearest RGI polygon and glacier
      fraction of the buffered window

Model protocol mirrors ``bathymetry_residual.py`` exactly so numbers
are comparable: grouped leave-one-lake-out over the 20 dense-surveyed
lakes, residual over the Huggel baseline in log space,
``V_pred = V_huggel * exp(f(X))``, ridge + GPR with analytic 95%
interval. Survey-only quantities (max depth, mean depth, point count)
are excluded — they are unavailable at deployment.

The global metadata compilation cannot be used here: its entries carry
no lake polygon and our DEM coverage is six 1° tiles — terrain features
would require manufacturing geometry, which the project rules forbid.

Usage:
    python -m siren.ml.bathymetry_terrain          # extract + benchmark + report
    python -m siren.ml.bathymetry_terrain --eval   # metrics only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box

from siren.ml.bathymetry_benchmark import (
    build_volume_estimates,
    huggel_volume_m3,
)
from siren.ml.bathymetry_dataset import (
    LakeRecord,
    load_all_surveyed_lakes,
)
from siren.ml.bathymetry_training_data import (
    _build_lake_mask,
    _estimate_z_surface,
    _find_dem_tile,
    BUFFER_FACTOR,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
REPORT_PATH = (
    REPO_ROOT / "models" / "checkpoints" / "bathymetry_terrain_loo.json"
)
FEATURES_CACHE = (
    REPO_ROOT / "models" / "checkpoints" / "bathymetry_terrain_features.json"
)

RGI_ZIPS = [
    DATA_DIR / "datasets" / f"RGI2000-v7.0-G-{r}_{n}.zip"
    for r, n in [
        ("13", "central_asia"),
        ("14", "south_asia_west"),
        ("15", "south_asia_east"),
    ]
]

GATE_TARGET_MAPE = 0.15
RIM_RING_PX = 3            # ring half-width for dam-face slope stats
M_PER_DEG_LAT = 110_540.0  # metres per degree latitude

FEATURE_NAMES = [
    "log_area_km2",
    "perimeter_km",
    "compactness",
    "elongation",
    "z_surface_m",
    "rim_slope_mean_deg",
    "rim_slope_p90_deg",
    "relief_window_m",
    "slope_window_mean_deg",
    "glacier_dist_km",
    "glacier_frac_window",
]


# ---------------------------------------------------------------------------
# Glacier context (RGI polygons, loaded lazily once)
# ---------------------------------------------------------------------------

_RGI_GDF: gpd.GeoDataFrame | None = None


def _load_rgi() -> gpd.GeoDataFrame:
    """Load the three Himalayan RGI v7 region shapefiles (vsizip)."""
    global _RGI_GDF
    if _RGI_GDF is None:
        frames = []
        for z in RGI_ZIPS:
            if not z.exists():
                logger.warning("RGI region missing: %s", z)
                continue
            shp = z.with_suffix("").name + ".shp"
            frames.append(gpd.read_file(f"/vsizip/{z}/{shp}"))
        if not frames:
            raise FileNotFoundError("no RGI region shapefiles found")
        _RGI_GDF = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), crs=frames[0].crs
        )
        logger.info("RGI loaded: %d glaciers", len(_RGI_GDF))
    return _RGI_GDF


def _utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:326{zone:02d}" if lat >= 0 else f"EPSG:327{zone:02d}"


def _lake_polygon_wgs84(record: LakeRecord):
    """Best available lake geometry in EPSG:4326 (outline else hull)."""
    if record.outline is not None:
        return (
            gpd.GeoDataFrame({"geometry": [record.outline]}, crs=record.crs)
            .to_crs("EPSG:4326")
            .geometry.iloc[0]
        )
    return (
        record.points.to_crs("EPSG:4326").geometry.union_all().convex_hull
    )


def _glacier_features(
    lake_geom_wgs84, window_bounds: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Distance (km) to the nearest RGI polygon and glacier fraction of
    the buffered window. ``nan`` if RGI is unavailable."""
    try:
        rgi = _load_rgi()
    except FileNotFoundError:
        return float("nan"), float("nan")
    cx, cy = lake_geom_wgs84.centroid.x, lake_geom_wgs84.centroid.y
    utm = _utm_epsg(cx, cy)
    lake_utm = (
        gpd.GeoDataFrame({"geometry": [lake_geom_wgs84]}, crs="EPSG:4326")
        .to_crs(utm)
        .geometry.iloc[0]
    )
    # Prefilter glaciers to the window bbox (deg) expanded by ~10 km
    l, b, r, t = window_bounds
    pad = 0.1
    cand_idx = rgi.sindex.query(
        box(l - pad, b - pad, r + pad, t + pad), predicate="intersects"
    )
    if len(cand_idx) == 0:
        return float("nan"), 0.0
    cand = rgi.iloc[cand_idx].to_crs(utm)
    dist_km = float(cand.distance(lake_utm).min()) / 1000.0
    win_utm = (
        gpd.GeoDataFrame({"geometry": [box(l, b, r, t)]}, crs="EPSG:4326")
        .to_crs(utm)
        .geometry.iloc[0]
    )
    win_area_km2 = float(win_utm.area) / 1e6
    inter_km2 = float(cand.intersection(win_utm).area.sum()) / 1e6
    return dist_km, min(inter_km2 / max(win_area_km2, 1e-9), 1.0)


# ---------------------------------------------------------------------------
# Per-lake terrain feature extraction
# ---------------------------------------------------------------------------


def extract_lake_features(record: LakeRecord) -> dict[str, Any] | None:
    """Extract deployable terrain/morphology features for one lake.

    Returns None if the lake has no DEM tile coverage or no usable
    surface-elevation estimate.
    """
    from scipy.ndimage import binary_dilation, sobel

    lake_geom = _lake_polygon_wgs84(record)
    cx, cy = lake_geom.centroid.x, lake_geom.centroid.y

    tile_path = _find_dem_tile(cx, cy)
    if tile_path is None:
        logger.warning("%s: no DEM tile — skipped", record.lake_name)
        return None

    minx, miny, maxx, maxy = lake_geom.bounds
    wl = minx - (maxx - minx) * BUFFER_FACTOR
    wr = maxx + (maxx - minx) * BUFFER_FACTOR
    wb = miny - (maxy - miny) * BUFFER_FACTOR
    wt = maxy + (maxy - miny) * BUFFER_FACTOR

    with rasterio.open(tile_path) as ds:
        win = from_bounds(wl, wb, wr, wt, ds.transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        dem = ds.read(1, window=win, fill_value=np.nan).astype(np.float64)
        win_transform = ds.window_transform(win)
        win_bounds = rasterio.windows.bounds(win, ds.transform)

    dem[dem < -1000] = np.nan

    lake_mask = _build_lake_mask(
        record, win_bounds, win_transform, dem.shape[1], dem.shape[0]
    ).astype(bool)
    if not lake_mask.any():
        logger.warning("%s: empty lake mask on DEM grid", record.lake_name)
        return None

    try:
        # NaN rim cells are excluded by the helper's >-1000 filter.
        z_surface = _estimate_z_surface(dem, lake_mask)
    except ValueError as exc:
        logger.warning("%s: %s", record.lake_name, exc)
        return None

    # Slope in degrees — convert pixel size to metres at scene latitude
    # (Copernicus GLO30 is EPSG:4326, ~1 arcsec).
    dx_deg = abs(win_transform.a)
    dy_deg = abs(win_transform.e)
    dx_m = dx_deg * M_PER_DEG_LAT * np.cos(np.radians(cy))
    dy_m = dy_deg * M_PER_DEG_LAT
    # Sobel is smoother than central differences on noisy DEMs; fill
    # nodata with the median first so NaNs don't smear into the field.
    dem_filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    slope = np.degrees(
        np.arctan(np.hypot(sobel(dem_filled, axis=1) / (8 * dx_m),
                           sobel(dem_filled, axis=0) / (8 * dy_m)))
    )
    slope[~np.isfinite(dem)] = np.nan

    ring = binary_dilation(lake_mask, iterations=RIM_RING_PX) & ~lake_mask
    rim_slope = slope[ring]
    rim_slope = rim_slope[np.isfinite(rim_slope)]
    land = ~lake_mask & np.isfinite(slope)

    # Geometry features in a metric CRS
    utm = _utm_epsg(cx, cy)
    geom_utm = (
        gpd.GeoDataFrame({"geometry": [lake_geom]}, crs="EPSG:4326")
        .to_crs(utm)
        .geometry.iloc[0]
    )
    area_m2 = float(geom_utm.area)
    perim_m = float(geom_utm.length)
    mrr = geom_utm.minimum_rotated_rectangle
    coords = np.asarray(mrr.exterior.coords)
    edges = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
    elongation = float(edges.max() / max(edges.min(), 1e-9))

    glacier_dist_km, glacier_frac = _glacier_features(
        lake_geom, win_bounds
    )

    area_km2 = area_m2 / 1e6
    feats = {
        "lake_id": record.lake_id,
        "lake_name": record.lake_name,
        "log_area_km2": float(np.log(max(area_km2, 1e-9))),
        "perimeter_km": perim_m / 1000.0,
        "compactness": float(
            4 * np.pi * area_m2 / max(perim_m**2, 1.0)
        ),
        "elongation": elongation,
        "z_surface_m": float(z_surface),
        "rim_slope_mean_deg": float(np.mean(rim_slope)) if rim_slope.size else float("nan"),
        "rim_slope_p90_deg": (
            float(np.percentile(rim_slope, 90)) if rim_slope.size else float("nan")
        ),
        "relief_window_m": float(
            np.nanmax(dem) - np.nanmin(dem)
        ),
        "slope_window_mean_deg": float(np.nanmean(slope[land])),
        "glacier_dist_km": glacier_dist_km,
        "glacier_frac_window": glacier_frac,
    }
    logger.info(
        "%s: area=%.3f km2  rim_slope=%.1f deg  glac_dist=%.2f km",
        record.lake_name, area_km2, feats["rim_slope_mean_deg"],
        glacier_dist_km,
    )
    return feats


def extract_all_features(
    records: list[LakeRecord] | None = None,
    cache_path: Path = FEATURES_CACHE,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Extract features for all surveyed lakes, with a JSON cache.

    DEM window reads + RGI queries take ~seconds per lake; the cache
    makes benchmark iteration cheap and the feature table inspectable.
    """
    if use_cache and cache_path.exists():
        feats = json.loads(cache_path.read_text())
        logger.info("loaded %d cached feature rows", len(feats))
        return feats

    if records is None:
        records = load_all_surveyed_lakes()
    rows = []
    for rec in sorted(records, key=lambda r: r.lake_id):
        f = extract_lake_features(rec)
        if f is not None:
            rows.append(f)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows, indent=1))
    logger.info("extracted features for %d/%d lakes", len(rows), len(records))
    return rows


# ---------------------------------------------------------------------------
# Grouped-LOO benchmark (protocol matches bathymetry_residual.py)
# ---------------------------------------------------------------------------


def run_terrain_loo_benchmark(
    feature_rows: list[dict[str, Any]] | None = None,
    gate_target_mape: float = GATE_TARGET_MAPE,
) -> dict[str, Any]:
    """Grouped leave-one-lake-out over the dense-surveyed lakes.

    Per fold: StandardScaler on train features, ridge + GPR on the
    log-residual over Huggel; the held-out lake's volume is predicted as
    ``V_huggel * exp(f(X))``. Reports MAPE + GPR 95% interval coverage.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        WhiteKernel,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if feature_rows is None:
        feature_rows = extract_all_features()

    estimates = {
        e.lake_id: e for e in build_volume_estimates()
    }
    rows = [
        (r, estimates[r["lake_id"]])
        for r in feature_rows
        if r["lake_id"] in estimates
        and np.isfinite(r.get("glacier_dist_km", float("nan")))
    ]
    if len(rows) < 5:
        raise RuntimeError(
            f"too few lakes with features+volume ({len(rows)})"
        )

    X_all = np.array(
        [[r[k] for k in FEATURE_NAMES] for r, _ in rows], dtype=np.float64
    )
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3)

    folds: list[dict[str, Any]] = []
    for i in range(len(rows)):
        test_row, test_est = rows[i]
        tr_idx = [j for j in range(len(rows)) if j != i]
        train = [rows[j][0] for j in tr_idx]
        train_ests = [rows[j][1] for j in tr_idx]

        imp = SimpleImputer(strategy="median").fit(X_all[tr_idx])
        scaler = StandardScaler().fit(imp.transform(X_all[tr_idx]))
        Xs = scaler.transform(imp.transform(X_all[tr_idx]))

        huggel_tr = np.array(
            [huggel_volume_m3(e.area_km2) for e in train_ests]
        )
        y_tr = np.array([e.ground_truth_volume_m3 for e in train_ests])
        resid_tr = np.log(np.maximum(y_tr, 1.0)) - np.log(
            np.maximum(huggel_tr, 1.0)
        )

        ridge = Ridge(alpha=1.0).fit(Xs, resid_tr)
        gpr = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=42,
            n_restarts_optimizer=1,
        )
        try:
            gpr.fit(Xs, resid_tr)
            gpr_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("GPR failed holding out %s: %s",
                           test_row["lake_name"], exc)
            gpr_ok = False

        gt = test_est.ground_truth_volume_m3
        hv = huggel_volume_m3(test_est.area_km2)
        xt = scaler.transform(imp.transform(X_all[i : i + 1]))
        ridge_v = hv * float(np.exp(ridge.predict(xt)[0]))
        if gpr_ok:
            res, std = gpr.predict(xt, return_std=True)
            gpr_v = hv * float(np.exp(res[0]))
            lo = hv * float(np.exp(res[0] - 1.96 * std[0]))
            hi = hv * float(np.exp(res[0] + 1.96 * std[0]))
            covered = bool(lo <= gt <= hi)
        else:
            gpr_v, covered = np.nan, None

        folds.append({
            "lake_id": test_row["lake_id"],
            "lake_name": test_row["lake_name"],
            "area_km2": test_est.area_km2,
            "ground_truth_volume_m3": round(gt, 1),
            "huggel_ape": abs(hv - gt) / gt,
            "ridge_ape": abs(ridge_v - gt) / gt,
            "gpr_ape": abs(gpr_v - gt) / gt if gpr_ok else None,
            "gpr_interval_covers": covered,
        })

    def _mape(key):
        vals = [f[key] for f in folds if f[key] is not None]
        return float(np.mean(vals)) if vals else None

    def _median(key):
        vals = [f[key] for f in folds if f[key] is not None]
        return float(np.median(vals)) if vals else None

    cov = [f["gpr_interval_covers"] for f in folds
           if f["gpr_interval_covers"] is not None]
    summary = {
        "n_lakes": len(folds),
        "huggel_mape": _mape("huggel_ape"),
        "huggel_median_ape": _median("huggel_ape"),
        "ridge_mape": _mape("ridge_ape"),
        "ridge_median_ape": _median("ridge_ape"),
        "gpr_mape": _mape("gpr_ape"),
        "gpr_median_ape": _median("gpr_ape"),
        "gpr_interval_coverage_95": float(np.mean(cov)) if cov else None,
    }
    gpr_mape = summary["gpr_mape"]
    return {
        "status": "benchmarked",
        "model": "terrain_feature_residual (ridge + gpr over Huggel)",
        "target": "log(V) - log(V_huggel); V_pred = V_huggel * exp(f(X))",
        "features": FEATURE_NAMES,
        "excluded_features": [
            "max_depth_m, mean_depth_m, n_points (survey-only)",
            "lake polygon manufactured for metadata-compilation lakes "
            "(forbidden — dense lakes only)",
        ],
        "n_lakes_evaluated": len(folds),
        "gate_target_mape": gate_target_mape,
        "summary": summary,
        "gpr_passes_gate": bool(
            gpr_mape is not None and gpr_mape < gate_target_mape
        ),
        "ridge_passes_gate": bool(
            summary["ridge_mape"] is not None
            and summary["ridge_mape"] < gate_target_mape
        ),
        "folds": folds,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=REPORT_PATH)
    p.add_argument("--eval", action="store_true",
                   help="print metrics without writing the report")
    p.add_argument("--no-cache", action="store_true",
                   help="recompute terrain features (ignore JSON cache)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = extract_all_features(use_cache=not args.no_cache)
    if not rows:
        logger.error("no terrain features extracted")
        return 1

    result = run_terrain_loo_benchmark(rows)
    s = result["summary"]
    print("\n" + "=" * 64)
    print("Terrain-feature bathymetry regression — grouped LOO (dense lakes)")
    print("=" * 64)
    print(f"  n={s['n_lakes']} lakes | features={len(FEATURE_NAMES)}")
    print(f"  Huggel  MAPE {s['huggel_mape']:.3f}  (median {s['huggel_median_ape']:.3f})")
    print(f"  ridge   MAPE {s['ridge_mape']:.3f}  (median {s['ridge_median_ape']:.3f})")
    print(f"  GPR     MAPE {s['gpr_mape']:.3f}  (median {s['gpr_median_ape']:.3f}) "
          f"| 95%-coverage {s['gpr_interval_coverage_95']:.2f}")
    print(f"  Gate <{GATE_TARGET_MAPE:.2f}: "
          f"ridge {'PASS' if result['ridge_passes_gate'] else 'FAIL'} | "
          f"gpr {'PASS' if result['gpr_passes_gate'] else 'FAIL'}")
    print("=" * 64)

    if not args.eval:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        logger.info("Report written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
