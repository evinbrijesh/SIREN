"""E3 feasibility diagnostic — does Sentinel-2 optically separate the
Imja terminus lake from the surrounding glacier?

Before training MultiModalFusionNet (ADR-013 §9.7.3), this script answers
the cheaper question: is the glacier-vs-water boundary even spectrally
resolvable in this AOI on this date? If MNDWI already separates them, the
fusion branch has a learnable signal; if not, SAR dielectric confusion has
no optical rescue.

Method:
    1. Extract NDWI / MNDWI / SCL cloud mask from the S2 L2A archive,
       reprojected to a ~10 m EPSG:4326 grid over the Dudh Koshi AOI.
    2. Sample four populations on that grid:
         - LAKE: trusted baseline water mask (resampled, UTM -> grid)
         - GLACIER: RGI polygon minus baseline-water vicinity
         - SAR_DET: the terrain-gated SAR shadow detections (mapped
           through GCP geolocation)
         - BACKGROUND: everything else in the AOI
    3. Report per-class NDWI/MNDWI distributions, SCL class fractions,
       and separability (Fisher ratio + best-threshold accuracy/F1 for
       LAKE vs GLACIER).

Usage:
    python -m siren.ml.s2_spectral_eval
    python -m siren.ml.s2_spectral_eval --s2 data/raw/S2B_...zip
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

DEFAULT_S2 = DATA_DIR / "raw" / "S2B_MSIL2A_20260705T044659_N0512_R076_T45RVL_20260705T083506.zip"
AOI_PATH = DATA_DIR / "assets" / "dudh_koshi_aoi.geojson"
# Lake reference: union of the per-observation scenario masks (EPSG:4326,
# georeferenced to the Imja lake area). NOTE: baseline_water_mask.tif
# covers the Rolwaling valley ~30 km west of the AOI and cannot serve as
# the Imja lake reference.
EXPANSION_MASK_GLOB = "obs-*_expansion_mask.tif"
RGI_SHP_PATH = (
    "/vsizip/"
    + str(
        DATA_DIR / "datasets" / "RGI2000-v7.0-G-15_south_asia_east.zip"
        / "RGI2000-v7.0-G-15_south_asia_east.shp"
    )
)
SAR_CACHE_PATH = DATA_DIR / "processed" / "imja_desc_20260714_sar_vv_vh_db.tif"
RULE_MASK_PATH = DATA_DIR / "processed" / "obs-002_expansion_mask.tif"
DEM_PATH = DATA_DIR / "raw" / "srtm_30m.tif"

# SCL class names (L2A Scene Classification Layer)
SCL_NAMES = {
    0: "nodata", 1: "saturated", 2: "dark_pixels", 3: "cloud_shadow",
    4: "vegetation", 5: "bare_soil", 6: "water", 7: "unclassified",
    8: "cloud_med", 9: "cloud_high", 10: "cirrus", 11: "snow_ice",
}
# Clear-view classes for glacier-lake terrain (no veg assumption)
CLEAR_SCL = {4, 5, 6, 7, 11}


def _aoi_bounds() -> tuple[float, float, float, float]:
    import geopandas as gpd

    gdf = gpd.read_file(str(AOI_PATH))
    w, s, e, n = gdf.total_bounds
    return float(w), float(s), float(e), float(n)


def _grid_lonlat(shape: tuple[int, int], bounds: tuple, ) -> tuple[np.ndarray, np.ndarray]:
    """lon/lat center coordinates of a regular north-up EPSG:4326 grid."""
    h, w = shape
    west, south, east, north = bounds
    cols = (np.arange(w) + 0.5) / w
    rows = (np.arange(h) + 0.5) / h
    lon = west + cols * (east - west)
    lat = north - rows * (north - south)
    return np.meshgrid(lon, lat)[0], np.meshgrid(lon, lat)[1]


def _sample_raster(path: str, lon: np.ndarray, lat: np.ndarray, fill=0.0) -> np.ndarray:
    """Nearest-sample a georeferenced raster at (lon, lat) points."""
    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(path) as ds:
        arr = ds.read(1)
        if ds.crs is not None and ds.crs.to_epsg() != 4326:
            xs, ys = warp_transform(
                "EPSG:4326", ds.crs,
                lon.ravel().tolist(), lat.ravel().tolist(),
            )
        else:
            xs, ys = lon.ravel().tolist(), lat.ravel().tolist()
        inv = ~ds.transform
    bx = np.asarray(xs).reshape(lon.shape)
    by = np.asarray(ys).reshape(lat.shape)
    cols = np.rint(inv.a * bx + inv.b * by + inv.c).astype(np.int64)
    rows = np.rint(inv.d * bx + inv.e * by + inv.f).astype(np.int64)
    valid = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    out = np.full(lon.shape, fill, dtype=np.float32)
    out[valid] = arr[rows[valid], cols[valid]].astype(np.float32)
    return out


def _best_threshold(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Scan thresholds in both directions; return the best-F1 operating point.

    Accuracy is meaningless under class imbalance (a glacier population
    dwarfs the lake), so the scan maximises F1 for "pos is on side `dir`
    of threshold t".
    """
    cands = np.unique(np.concatenate([pos, neg]))
    best = {"threshold": 0.0, "direction": "gt", "f1": 0.0,
            "precision": 0.0, "recall": 0.0}
    for t in cands:
        for direction, pmask, nmask in [
            ("gt", pos > t, neg > t),
            ("lt", pos < t, neg < t),
        ]:
            tp = float(pmask.sum())
            fp = float(nmask.sum())
            fn = float((~pmask).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
            if f1 > best["f1"]:
                best = {
                    "threshold": float(t),
                    "direction": direction,
                    "f1": f1,
                    "precision": tp / max(tp + fp, 1e-9),
                    "recall": tp / max(tp + fn, 1e-9),
                }
    best["precision"] = round(best["precision"], 4)
    best["recall"] = round(best["recall"], 4)
    best["f1"] = round(best["f1"], 4)
    return best


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(random pos value > random neg value).

    0.5 = no separation; near 1 or 0 = cleanly separable (direction in
    the sign). Computed via rank statistics on the concatenated sample;
    ties contribute 0.5 via midranks.
    """
    from scipy.stats import rankdata

    combined = np.concatenate([pos, neg])
    ranks = rankdata(combined)  # midranks handle ties
    rank_sum_pos = ranks[: pos.size].sum()
    u = rank_sum_pos - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def _stats(arr: np.ndarray) -> dict:
    return {
        "n": int(arr.size),
        "mean": round(float(np.nanmean(arr)), 4),
        "std": round(float(np.nanstd(arr)), 4),
        "p05": round(float(np.nanpercentile(arr, 5)), 4),
        "p50": round(float(np.nanpercentile(arr, 50)), 4),
        "p95": round(float(np.nanpercentile(arr, 95)), 4),
    }


def run_eval(s2_path: Path, px_m: float = 10.0) -> dict:
    import geopandas as gpd

    from siren.preprocess.s2_optical import extract_optical_features
    from siren.detect.sar import (
        sar_grid_dem_slope,
        sar_grid_lonlat,
        sar_grid_polygon_mask,
        sar_grid_sample,
    )
    from siren.ml.engine import ChangeDetectionEngine

    west, south, east, north = _aoi_bounds()
    # ~10 m pixels in degrees at lat ~28
    dlat = px_m / 110_540.0
    dlon = px_m / (111_320.0 * np.cos(np.deg2rad((south + north) / 2)))
    shape = (int(np.ceil((north - south) / dlat)), int(np.ceil((east - west) / dlon)))
    logger.info("AOI grid %s at ~%dm over bounds W%.3f S%.3f E%.3f N%.3f",
                shape, px_m, west, south, east, north)

    feats = extract_optical_features(
        s2_path,
        target_crs="EPSG:4326",
        target_bounds=(west, south, east, north),
        target_shape=shape,
    )
    ndwi, mndwi, cloud = feats["ndwi"], feats["mndwi"], feats["cloud_mask"]
    lon, lat = _grid_lonlat(shape, (west, south, east, north))

    # Populations
    lake = np.zeros(shape, dtype=bool)
    for obs_mask_path in sorted(
        (DATA_DIR / "processed").glob(EXPANSION_MASK_GLOB)
    ):
        lake |= _sample_raster(str(obs_mask_path), lon, lat) > 0
    glacier_all = sar_grid_polygon_mask(RGI_SHP_PATH, lon, lat)
    glacier = glacier_all & ~lake
    aoi_mask = sar_grid_polygon_mask(str(AOI_PATH), lon, lat)
    background = aoi_mask & ~lake & ~glacier_all

    # Gated SAR detections mapped onto the optical grid
    sar_det = np.zeros(shape, dtype=bool)
    if SAR_CACHE_PATH.exists() and RULE_MASK_PATH.exists() and torch_ok():
        try:
            gated = _gated_sar_mask()
            if gated is not None:
                sar_ll = sar_grid_lonlat(str(SAR_CACHE_PATH))
                ys, xs = np.where(gated > 0)
                det_lon = sar_ll[0][ys, xs]
                det_lat = sar_ll[1][ys, xs]
                cols = ((det_lon - west) / (east - west) * shape[1]).astype(int)
                rows = ((north - det_lat) / (north - south) * shape[0]).astype(int)
                ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
                sar_det[rows[ok], cols[ok]] = True
        except Exception as exc:
            logger.warning("SAR detection mapping skipped: %s", exc)

    pops = {"lake": lake, "glacier": glacier, "sar_det": sar_det, "background": background}

    # SCL classes over the AOI and per population (nearest 20 m SCL read)
    scl_hist = {}
    scl = None
    try:
        scl = _read_scl_on_grid(s2_path, shape, (west, south, east, north))
        in_aoi = scl[aoi_mask]
        for code, name in SCL_NAMES.items():
            cnt = int((in_aoi == code).sum())
            if cnt:
                scl_hist[name] = {
                    "px": cnt, "frac": round(cnt / max(in_aoi.size, 1), 4)
                }
    except Exception as exc:
        logger.warning("SCL histogram skipped: %s", exc)

    # Distributions
    dist = {}
    for name, m in pops.items():
        if m.any():
            pop_scl = {}
            if scl is not None:
                in_pop = scl[m]
                for code, scl_name in SCL_NAMES.items():
                    cnt = int((in_pop == code).sum())
                    if cnt:
                        pop_scl[scl_name] = round(cnt / in_pop.size, 4)
            dist[name] = {
                "n_px": int(m.sum()),
                "ndwi": _stats(ndwi[m]),
                "mndwi": _stats(mndwi[m]),
                "cloud_frac": round(float(cloud[m].mean()), 4),
                "scl_hist": pop_scl,
            }

    # Separability: lake vs glacier (clear pixels only)
    clear = cloud < 0.5
    lake_v = lake & clear
    glac_v = glacier & clear
    sep = {}
    for idx_name, idx in [("ndwi", ndwi), ("mndwi", mndwi)]:
        lp, gp = idx[lake_v], idx[glac_v]
        if lp.size and gp.size:
            mu_l, mu_g = float(lp.mean()), float(gp.mean())
            var_l, var_g = float(lp.var()), float(gp.var())
            fisher = (mu_l - mu_g) ** 2 / max(var_l + var_g, 1e-9)
            sep[idx_name] = {
                "fisher_ratio": round(fisher, 3),
                "auc_lake_gt_glacier": round(_auc(lp, gp), 4),
                "lake_mean": round(mu_l, 4),
                "glacier_mean": round(mu_g, 4),
                "best_threshold": _best_threshold(lp, gp),
                "frac_lake_gt_0": round(float((lp > 0).mean()), 3),
                "frac_glacier_gt_0": round(float((gp > 0).mean()), 3),
            }

    report = {
        "s2_scene": s2_path.name,
        "aoi_bounds": [west, south, east, north],
        "grid_shape": list(shape),
        "populations": dist,
        "scl_aoi_histogram": scl_hist,
        "clear_frac_aoi": round(
            float((cloud[aoi_mask] < 0.5).mean()), 4
        ) if aoi_mask.any() else None,
        "lake_vs_glacier_separability": sep,
        "note": (
            "MNDWI>0 is the classic open-water test; on debris-covered "
            "glacier ice it typically fails — that's exactly what this "
            "diagnostic measures."
        ),
    }
    return report


def _gated_sar_mask() -> np.ndarray | None:
    """Reproduce the pipeline's terrain-gated shadow mask on the SAR grid."""
    import rasterio

    from siren.detect.sar import (
        sar_grid_dem_slope, sar_grid_lonlat, sar_grid_polygon_mask,
        sar_grid_sample,
    )
    from siren.ml.engine import ChangeDetectionEngine
    from siren.preprocess.sar_calibrate import (
        extract_and_cache_vv_vh_db, find_imja_descending_pair,
    )
    from scipy.ndimage import binary_dilation

    pair = find_imja_descending_pair(DATA_DIR / "raw")
    if pair is None:
        return None
    t0_safe, t1_safe = pair
    t0 = extract_and_cache_vv_vh_db(
        t0_safe, DATA_DIR / "processed" / "imja_desc_20260702_sar_vv_vh_db.tif"
    )
    t1 = extract_and_cache_vv_vh_db(t1_safe, SAR_CACHE_PATH)
    engine = ChangeDetectionEngine(device="cpu")
    ml = engine.predict_change_mask(t0, t1)

    ll = sar_grid_lonlat(str(SAR_CACHE_PATH))
    if ll is None:
        return ml
    lon_g, lat_g = ll
    exclusion = np.zeros(ml.shape, dtype=bool)
    aoi_g = sar_grid_polygon_mask(str(AOI_PATH), lon_g, lat_g)
    exclusion |= ~aoi_g
    if DEM_PATH.exists():
        slope_g = sar_grid_dem_slope(str(SAR_CACHE_PATH), str(DEM_PATH))
        exclusion |= (~np.isnan(slope_g)) & (slope_g > 15.0)
    glac_g = sar_grid_polygon_mask(RGI_SHP_PATH, lon_g, lat_g)
    lake_union = np.zeros(ml.shape, dtype=bool)
    for obs_mask_path in sorted(
        (DATA_DIR / "processed").glob(EXPANSION_MASK_GLOB)
    ):
        lake_union |= sar_grid_sample(str(obs_mask_path), lon_g, lat_g) > 0
    lake_vic = binary_dilation(lake_union, iterations=3)
    exclusion |= glac_g & ~lake_vic
    return np.where(exclusion, 0, ml).astype(np.uint8)


def _read_scl_on_grid(s2_path, shape, bounds):
    """Read the 20 m SCL band, resampled nearest onto the AOI grid."""
    import zipfile

    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import from_bounds
    from siren.preprocess.s2_optical import _find_band_path

    west, south, east, north = bounds
    h, w = shape
    dst_transform = from_bounds(west, south, east, north, w, h)
    out = np.zeros((h, w), dtype=np.uint8)
    with zipfile.ZipFile(str(s2_path)) as zf:
        scl_path = _find_band_path(zf, "SCL", "20m")
        with zf.open(scl_path) as f:
            with rasterio.open(f) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=out,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest,
                )
    return out


def torch_ok() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="S2 spectral separability eval (E3)")
    parser.add_argument("--s2", type=Path, default=DEFAULT_S2)
    parser.add_argument("--px-m", type=float, default=30.0,
                        help="AOI grid pixel size in metres (30 m default — "
                             "keeps memory small; the indices are broad-"
                             "scale anyway)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_eval(args.s2, px_m=args.px_m)
    out = args.out or (REPO_ROOT / "models" / "checkpoints" / "s2_spectral_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
