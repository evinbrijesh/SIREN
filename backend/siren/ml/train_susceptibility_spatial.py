"""Leakage-free spatial-block cross-validation for GLOF breach susceptibility.

Replaces the disqualified feature-generation path in
``real_glof_dataset.py`` / ``train_susceptibility.py`` (PRD v4.7 §17.3).
The v1/v2 checkpoints scored CV ROC-AUC 0.9948 / Brier 0.0253 because the
loader sampled dam geometry, expansion rate and rain anomaly from
label-conditioned distributions (breached lakes got narrow dams, high
expansion, positive rain anomaly; stable lakes the opposite). This script
contains **no label-dependent feature generation** — every feature is a
measured quantity derived identically for both classes.

Data sources (all on disk, offline):

    Positives (breached=1):
        data/datasets/HMAGLOFDB/Database/GLOFs/HMAGLOFDB.csv
        773 documented GLOF events → deduplicated to unique lakes
        (repeat outbursts of the same lake share location; 1 km clustering).
    Negatives (breached=0):
        data/datasets/glacial_lake_2022-2024/Glacial_Lake_2022.shp
        ICIMOD HMA inventory (31,698 lakes) sampled >= 1 km from every
        breach centroid, stratified by spatial block.
    Glacier context:
        data/datasets/RGI2000-v7.0-G-{13,14,15}_*.zip
        Randolph Glacier Inventory v7 polygons for all three HMA regions.

Measured feature set (identical derivation for both classes):
    lake_elev_m              elevation (HMAGLOFDB Elev_lake / ICIMOD Lake_Elev)
    log_lake_area_km2        log1p area; positives prefer ICIMOD spatial match
                             (<=1 km), fall back to HMAGLOFDB Area
    log_dist_glacier_m       log1p distance to nearest RGI glacier polygon
    log_glacier_area_10km    log1p total RGI glacier area within 10 km
                             (upstream ice-avalanche trigger potential)

Deliberately excluded (leakage / non-physical channels):
    lake_type                not available for ICIMOD negatives; subsumed by
                             dist_to_glacier (supraglacial ~ 0 m)
    dam geometry             only the leaked proxy existed; no measured source
    expansion / rain anomaly post-event or label-biased in the old loader
    lat / lon / region       spatial proxies — geography is handled by the
                             blocking scheme, not memorised as a feature
    breach year              reporting-era bias, not a physical attribute

Spatial blocking:
    KMeans on spherical (Cartesian) coordinates fit on the breach lakes,
    negatives assigned via predict() — each CV fold holds out an entire
    geographically contiguous block, so test lakes share no climate/geology
    neighbourhood with train lakes (spatial autocorrelation elimination).

Known limitations (recorded in the report):
    - ICIMOD 2022 area for pre-2022 events is a post-event measurement;
      partially-drained lakes may be systematically smaller.
    - Negative labels assume unrecorded breaches are rare.
    - No DEM-derived slope: GLO-30 tiles on disk cover only fragments of HMA.

Usage:
    python -m siren.ml.train_susceptibility_spatial --eval
    python -m siren.ml.train_susceptibility_spatial --n-blocks 5 --n-nonbreach 2000
    python -m siren.ml.train_susceptibility_spatial --save-model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
HMAGLOFDB_CSV = (
    REPO_ROOT / "data" / "datasets" / "HMAGLOFDB" / "Database" / "GLOFs" / "HMAGLOFDB.csv"
)
ICIMOD_SHP = (
    REPO_ROOT / "data" / "datasets" / "glacial_lake_2022-2024" / "Glacial_Lake_2022.shp"
)
RGI_ZIPS = [
    REPO_ROOT / "data" / "datasets" / "RGI2000-v7.0-G-13_central_asia.zip",
    REPO_ROOT / "data" / "datasets" / "RGI2000-v7.0-G-14_south_asia_west.zip",
    REPO_ROOT / "data" / "datasets" / "RGI2000-v7.0-G-15_south_asia_east.zip",
]
DEFAULT_REPORT_PATH = (
    REPO_ROOT / "models" / "checkpoints" / "xgboost_spatial_cv_report.json"
)

# Albers Equal-Area over HMA — metre-accurate distances/areas for the domain
HMA_AEA = "+proj=aea +lat_1=25 +lat_2=45 +lat_0=35 +lon_0=85 +datum=WGS84 +units=m"

DEDUP_RADIUS_M = 1_000.0        # repeat outbursts of one lake
BREACH_EXCLUSION_M = 1_000.0    # negatives must clear every breach centroid
ICIMOD_MATCH_M = 1_000.0        # positive -> measured inventory polygon
GLACIER_CONTEXT_M = 10_000.0    # upstream ice-avalanche context window

FEATURE_NAMES = [
    "lake_elev_m",
    "log_lake_area_km2",
    "log_dist_glacier_m",
    "log_glacier_area_10km",
]

XGBOOST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _load_rgi_glaciers() -> object:
    """Load RGI v7 polygons for RGI-13/14/15, merged in the HMA AEA CRS."""
    import geopandas as gpd

    frames = []
    for zpath in RGI_ZIPS:
        inner = next(
            n for n in zipfile.ZipFile(zpath).namelist() if n.endswith(".shp")
        )
        gdf = gpd.read_file(f"zip://{zpath}!{inner}")
        frames.append(gdf[["geometry"]].to_crs(HMA_AEA))
    glaciers = pd.concat(frames, ignore_index=True)
    glaciers["geometry"] = glaciers.geometry.make_valid()
    glaciers["glacier_area_m2"] = glaciers.geometry.area
    logger.info("RGI glaciers loaded: %d polygons", len(glaciers))
    return gpd.GeoDataFrame(glaciers, crs=HMA_AEA)


def _dedupe_breach_lakes(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeat GLOF events of the same lake (<=1 km) to one row.

    HMAGLOFDB has 773 event rows but only ~347 unique source glaciers;
    without dedup the same physical lake lands in both train and test,
    inflating CV metrics (pseudo-replication). Keeps the earliest event.
    """
    df = df.sort_values("Year_approx", na_position="last").reset_index(drop=True)
    coords = np.radians(df[["lat", "lon"]].values)
    R = 6_371_000.0
    keep_idx: list[int] = []
    cluster_centres: list[np.ndarray] = []
    for i, (lat_r, lon_r) in enumerate(coords):
        p = np.array([
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r),
        ])
        dup = False
        for c in cluster_centres:
            chord = np.linalg.norm(p - c)
            if 2 * R * np.arcsin(min(chord / 2, 1.0)) <= DEDUP_RADIUS_M:
                dup = True
                break
        if not dup:
            keep_idx.append(i)
            cluster_centres.append(p)
    out = df.iloc[keep_idx].reset_index(drop=True)
    logger.info("Dedup: %d events -> %d unique lakes", len(df), len(out))
    return out


def _load_positives() -> pd.DataFrame:
    """HMAGLOFDB breach events -> unique lake rows with measured fields."""
    df = pd.read_csv(str(HMAGLOFDB_CSV), encoding="latin-1")
    df["Lat_lake"] = pd.to_numeric(df["Lat_lake"], errors="coerce")
    df["Lon_lake"] = pd.to_numeric(df["Lon_lake"], errors="coerce")
    df["Elev_lake"] = pd.to_numeric(df["Elev_lake"], errors="coerce")
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    df["Year_approx"] = pd.to_numeric(df["Year_approx"], errors="coerce")
    df = df[df["Lat_lake"].notna() & df["Lon_lake"].notna()].copy()

    df = df.rename(columns={"Lat_lake": "lat", "Lon_lake": "lon"})
    df["hmaglofdb_area_km2"] = df["Area"] / 1e6
    df.loc[df["hmaglofdb_area_km2"] <= 0, "hmaglofdb_area_km2"] = np.nan
    df["breached"] = 1
    df = _dedupe_breach_lakes(df)
    return df[["lat", "lon", "Elev_lake", "hmaglofdb_area_km2", "breached",
               "Lake_type", "Year_approx"]]


def _load_icimod() -> object:
    """ICIMOD 2022 inventory (measured elev/area for match + negatives)."""
    import geopandas as gpd

    gdf = gpd.read_file(str(ICIMOD_SHP)).to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["Lake_Elev"] = pd.to_numeric(gdf["Lake_Elev"], errors="coerce")
    gdf["Area"] = pd.to_numeric(gdf["Area"], errors="coerce")
    return gdf


def _match_icimod(pos: pd.DataFrame, icimod: object) -> pd.DataFrame:
    """Attach measured ICIMOD area/elev to breach lakes within 1 km."""
    import geopandas as gpd
    from shapely.geometry import Point

    pos = pos.copy()
    pos["icimod_area_km2"] = np.nan
    pos["icimod_elev_m"] = np.nan

    pts = gpd.GeoDataFrame(
        pos[["lat", "lon"]],
        geometry=[Point(lo, la) for la, lo in zip(pos["lat"], pos["lon"])],
        crs="EPSG:4326",
    ).to_crs(HMA_AEA)
    ic_proj = icimod.to_crs(HMA_AEA)

    matches = ic_proj.sindex.nearest(
        pts.geometry, return_all=False, max_distance=ICIMOD_MATCH_M
    )
    for p_idx, i_idx in zip(matches[0], matches[1]):
        row = ic_proj.iloc[i_idx]
        pos.at[p_idx, "icimod_area_km2"] = (
            row["Area"] / 1e6 if pd.notna(row["Area"]) else np.nan
        )
        pos.at[p_idx, "icimod_elev_m"] = row["Lake_Elev"]

    n_matched = int(pos["icimod_area_km2"].notna().sum())
    logger.info("ICIMOD match: %d/%d breach lakes within %.0f m",
                n_matched, len(pos), ICIMOD_MATCH_M)
    return pos


def _to_cartesian(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """lon/lat degrees -> unit-sphere Cartesian (for spherical KMeans)."""
    coords = np.radians(np.column_stack([lats, lons]))
    lat_r, lon_r = coords[:, 0], coords[:, 1]
    return np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ])


def _fit_blocks(lats: np.ndarray, lons: np.ndarray, n_blocks: int, seed: int):
    """Fit spherical KMeans on coordinates -> fitted KMeans object."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    km.fit(_to_cartesian(lats, lons))
    return km


def _sample_negatives(
    pos: pd.DataFrame,
    icimod: object,
    kmeans,
    n_nonbreach: int,
    seed: int,
) -> pd.DataFrame:
    """Sample stable lakes >=1 km from every breach, stratified by block."""
    import geopandas as gpd
    from shapely.geometry import Point

    rng = np.random.default_rng(seed)

    breach_pts = gpd.GeoDataFrame(
        geometry=[Point(lo, la) for la, lo in zip(pos["lat"], pos["lon"])],
        crs="EPSG:4326",
    ).to_crs(HMA_AEA)
    exclusion = breach_pts.geometry.buffer(BREACH_EXCLUSION_M).union_all()

    cand = icimod[~icimod.to_crs(HMA_AEA).geometry.intersects(exclusion)].copy()
    cand = cand[cand["Lake_Elev"].notna() & cand["Area"].notna()
                & (cand["Area"] > 0)]
    logger.info("Negative candidates after exclusion: %d", len(cand))

    # Assign each candidate to the spatial block of its location
    cand["block"] = kmeans.predict(
        _to_cartesian(cand["Latitude"].values, cand["Longitude"].values)
    )

    frac = pd.Series(kmeans.labels_).value_counts(normalize=True).sort_index()
    sampled_idx: list[int] = []
    for blk, f in frac.items():
        pool = cand[cand["block"] == blk]
        n = min(round(n_nonbreach * f), len(pool))
        if n > 0:
            sampled_idx.extend(
                pool.index[rng.choice(len(pool), size=n, replace=False)].tolist()
            )
    # Top-up from the full pool if a block was short
    remaining = n_nonbreach - len(sampled_idx)
    if remaining > 0:
        avail = cand[~cand.index.isin(sampled_idx)]
        n = min(remaining, len(avail))
        sampled_idx.extend(
            avail.index[rng.choice(len(avail), size=n, replace=False)].tolist()
        )

    neg = cand.loc[sampled_idx].copy()
    return pd.DataFrame({
        "lat": neg["Latitude"].astype(float),
        "lon": neg["Longitude"].astype(float),
        "icimod_area_km2": neg["Area"].astype(float) / 1e6,
        "lake_elev_m": neg["Lake_Elev"].astype(float),
        "breached": 0,
    })


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

def _glacier_features(df: pd.DataFrame, glaciers: object) -> pd.DataFrame:
    """Measured glacier-context features from RGI polygons (both classes)."""
    import geopandas as gpd
    from shapely.geometry import Point

    pts = gpd.GeoDataFrame(
        df.index,
        geometry=[Point(lo, la) for la, lo in zip(df["lat"], df["lon"])],
        crs="EPSG:4326",
    ).to_crs(HMA_AEA)

    dist_m = np.full(len(pts), np.nan)
    pairs = glaciers.sindex.nearest(pts.geometry, return_all=False)
    for p_idx, g_idx in zip(pairs[0], pairs[1]):
        dist_m[p_idx] = pts.geometry.iloc[p_idx].distance(
            glaciers.geometry.iloc[g_idx]
        )

    area_10km = np.zeros(len(pts))
    bufs = pts.geometry.buffer(GLACIER_CONTEXT_M)
    hits = glaciers.sindex.query(bufs, predicate="intersects")
    for p_idx, g_idx in zip(hits[0], hits[1]):
        area_10km[p_idx] += bufs.iloc[p_idx].intersection(
            glaciers.geometry.iloc[g_idx]
        ).area

    df = df.copy()
    df["dist_glacier_m"] = dist_m
    df["glacier_area_10km_m2"] = area_10km
    return df


def load_leakage_free_dataset(
    n_nonbreach: int = 2000,
    n_blocks: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the measured-only susceptibility dataset with spatial blocks."""
    pos = _load_positives()
    icimod = _load_icimod()
    pos = _match_icimod(pos, icimod)

    # Resolve measured area: ICIMOD match preferred (uniform source with
    # negatives), HMAGLOFDB Area as fallback; drop lakes with neither.
    pos["lake_area_km2"] = pos["icimod_area_km2"].fillna(pos["hmaglofdb_area_km2"])
    pos["lake_elev_m"] = pos["icimod_elev_m"].fillna(pos["Elev_lake"])
    pos = pos[pos["lake_area_km2"].notna() & pos["lake_elev_m"].notna()]
    pos = pos[pos["lake_area_km2"] > 0]
    logger.info("Positives with measured area+elev: %d", len(pos))

    # Spatial blocks are defined by the breach-lake geography (fit on
    # positives, predict for negatives) — unsupervised, coordinate-only.
    kmeans = _fit_blocks(pos["lat"].values, pos["lon"].values, n_blocks, seed)

    neg = _sample_negatives(pos, icimod, kmeans, n_nonbreach, seed)
    neg["lake_area_km2"] = neg["icimod_area_km2"]

    df = pd.concat([pos, neg], ignore_index=True)

    glaciers = _load_rgi_glaciers()
    df = _glacier_features(df, glaciers)

    df["log_lake_area_km2"] = np.log1p(df["lake_area_km2"])
    df["log_dist_glacier_m"] = np.log1p(df["dist_glacier_m"].clip(lower=0))
    df["log_glacier_area_10km"] = np.log1p(df["glacier_area_10km_m2"] / 1e6)

    df = df.dropna(subset=FEATURE_NAMES).reset_index(drop=True)
    df["block"] = kmeans.predict(
        _to_cartesian(df["lat"].values, df["lon"].values)
    )

    logger.info(
        "Dataset: %d lakes (%d breached / %d stable), %d blocks",
        len(df), int(df["breached"].sum()), int((df["breached"] == 0).sum()),
        df["block"].nunique(),
    )
    return df, FEATURE_NAMES


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def train_spatial_block_cv(
    df: pd.DataFrame,
    features: list[str],
    n_blocks: int,
    seed: int = 42,
) -> dict:
    """GroupKFold over spatial blocks — each fold tests an unseen region."""
    import xgboost as xgb
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold

    X = df[features].values.astype(np.float32)
    y = df["breached"].values.astype(int)
    groups = df["block"].values

    gkf = GroupKFold(n_splits=n_blocks)
    fold_metrics, importances = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        y_tr, y_te = y[tr], y[te]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            logger.warning("Fold %d: single-class split — skipped", fold)
            continue

        scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = xgb.XGBClassifier(
            **{**XGBOOST_PARAMS, "random_state": seed + fold},
            scale_pos_weight=scale_pos,
        )
        model.fit(X[tr], y_tr)
        p = model.predict_proba(X[te])[:, 1]

        # In-fold calibration diagnostic (cross-fitted): split the training
        # blocks internally, fit a same-config booster on half, fit an
        # isotonic mapping on the OTHER half's out-of-sample predictions,
        # then apply the frozen mapping to the held-out test block. The
        # mapping never sees test data; the booster half-split keeps the
        # isotonic inputs honest (not in-sample for the calibrator).
        brier_cal = float("nan")
        inner = list(GroupKFold(n_splits=2).split(X[tr], y_tr, groups[tr]))
        if inner:
            a_idx, b_idx = inner[0]
            tr_a, tr_b = tr[a_idx], tr[b_idx]
            if len(np.unique(y[tr_a])) == 2 and len(np.unique(y[tr_b])) == 2:
                from sklearn.isotonic import IsotonicRegression

                sp_a = (y[tr_a] == 0).sum() / max((y[tr_a] == 1).sum(), 1)
                mdl_a = xgb.XGBClassifier(
                    **{**XGBOOST_PARAMS, "random_state": seed + fold},
                    scale_pos_weight=sp_a,
                )
                mdl_a.fit(X[tr_a], y[tr_a])
                p_b = mdl_a.predict_proba(X[tr_b])[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_b, y[tr_b])
                p_cal = iso.predict(p)
                brier_cal = float(brier_score_loss(y_te, p_cal))

        blk = int(groups[te][0])
        m = {
            "fold": fold,
            "heldout_block": blk,
            "n_train": len(tr),
            "n_test": len(te),
            "n_breached_test": int(y_te.sum()),
            "roc_auc": float(roc_auc_score(y_te, p)),
            "pr_auc": float(average_precision_score(y_te, p)),
            "brier_score": float(brier_score_loss(y_te, p)),
            "brier_score_calibrated": (
                brier_cal if not np.isnan(brier_cal) else None
            ),
        }
        fold_metrics.append(m)
        importances.append(model.feature_importances_)
        logger.info(
            "Fold %d [block %d]: AUC=%.3f  PR-AUC=%.3f  Brier=%.4f  "
            "Brier-cal=%.4f  (%d test, %d breached)",
            fold, blk, m["roc_auc"], m["pr_auc"], m["brier_score"],
            brier_cal if not np.isnan(brier_cal) else -1,
            m["n_test"], m["n_breached_test"],
        )

    aucs = [m["roc_auc"] for m in fold_metrics]
    briers = [m["brier_score"] for m in fold_metrics]
    briers_cal = [m["brier_score_calibrated"] for m in fold_metrics
                  if m["brier_score_calibrated"] is not None]
    mean_brier = float(np.mean(briers)) if briers else float("nan")
    mean_brier_cal = float(np.mean(briers_cal)) if briers_cal else float("nan")

    return {
        "mean_roc_auc": float(np.mean(aucs)) if aucs else None,
        "std_roc_auc": float(np.std(aucs)) if aucs else None,
        "mean_brier": mean_brier if not np.isnan(mean_brier) else None,
        "std_brier": float(np.std(briers)) if briers else None,
        "mean_brier_calibrated": (
            mean_brier_cal if not np.isnan(mean_brier_cal) else None
        ),
        "calibration_protocol": (
            "cross-fitted isotonic: internal 2-block split of each training "
            "fold, mapping fitted on held-out half, frozen, applied to test"
        ),
        "mean_pr_auc": float(np.mean([m["pr_auc"] for m in fold_metrics]))
        if fold_metrics else None,
        "fold_metrics": fold_metrics,
        "feature_importances": {
            f: float(v) for f, v in
            zip(features, np.mean(importances, axis=0))
        } if importances else {},
        "passes_brier_gate": bool(
            not np.isnan(mean_brier) and mean_brier < 0.15
        ),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-free spatial-block CV for GLOF susceptibility"
    )
    parser.add_argument("--output", "-o", type=Path,
                        default=DEFAULT_REPORT_PATH,
                        help="Evaluation report path")
    parser.add_argument("--n-nonbreach", type=int, default=2000)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", action="store_true",
                        help="Print metrics without writing the report")
    parser.add_argument("--save-model", action="store_true",
                        help="Also persist a booster trained on all data")
    parser.add_argument("--model-output", type=Path,
                        default=REPO_ROOT / "models" / "checkpoints"
                        / "xgboost_susceptibility_spatial.json",
                        help="Booster path when --save-model is set")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df, features = load_leakage_free_dataset(
        n_nonbreach=args.n_nonbreach,
        n_blocks=args.n_blocks,
        seed=args.seed,
    )
    results = train_spatial_block_cv(df, features, args.n_blocks, args.seed)

    report = {
        "status": "gate_evaluated",
        "evaluation_valid": True,
        "inference_allowed": False,
        "validation_protocol": f"{args.n_blocks}-fold spatial block GroupKFold",
        "leakage_controls": [
            "measured features only — no label-conditioned generation",
            "repeat events deduplicated to unique lakes (1 km)",
            "negatives excluded within 1 km of every breach centroid",
            "test folds are whole contiguous spatial blocks",
            "lat/lon/region/lake_type/breach_year excluded as features",
        ],
        "features": features,
        "n_samples": len(df),
        "n_breached": int(df["breached"].sum()),
        "n_stable": int((df["breached"] == 0).sum()),
        "brier_gate": 0.15,
        "gate_metric": (
            "mean_brier (uncalibrated). scale_pos_weight shifts the logit "
            "intercept to balance training loss, inflating raw probabilities "
            "above the empirical base rate; mean_brier_calibrated is the "
            "cross-fitted in-fold isotonic diagnostic."
        ),
        "data_provenance": {
            "positives": str(HMAGLOFDB_CSV.relative_to(REPO_ROOT)),
            "negatives": str(ICIMOD_SHP.relative_to(REPO_ROOT)),
            "glaciers": [z.name for z in RGI_ZIPS],
        },
        "known_limitations": [
            "ICIMOD 2022 area is post-event for pre-2022 breaches",
            "unrecorded breaches may contaminate negatives",
            "no DEM-derived slope (GLO-30 tiles cover fragments of HMA)",
            (
                "passing the Brier gate does not itself promote the model — "
                "promotion is a separate PRD §9.8 decision"
            ),
        ],
        **results,
    }

    print("\n" + "=" * 64)
    print("Leakage-free spatial-block CV — susceptibility")
    print("=" * 64)
    print(f"  Samples: {report['n_samples']} "
          f"({report['n_breached']} breached / {report['n_stable']} stable)")
    print(f"  Mean ROC-AUC:  {results['mean_roc_auc']:.4f} "
          f"+/- {results['std_roc_auc']:.4f}")
    print(f"  Mean Brier:    {results['mean_brier']:.4f} "
          f"+/- {results['std_brier']:.4f}")
    if results["mean_brier_calibrated"] is not None:
        print(f"  Mean Brier (in-fold isotonic): "
              f"{results['mean_brier_calibrated']:.4f}")
    print(f"  Mean PR-AUC:   {results['mean_pr_auc']:.4f}")
    print(f"  Brier gate (<0.15): {'PASS' if results['passes_brier_gate'] else 'FAIL'}"
          "  [uncalibrated]")
    for m in results["fold_metrics"]:
        cal = (f" Brier-cal={m['brier_score_calibrated']:.4f}"
               if m["brier_score_calibrated"] is not None else "")
        print(f"    fold {m['fold']} block {m['heldout_block']}: "
              f"AUC={m['roc_auc']:.3f} Brier={m['brier_score']:.4f}{cal} "
              f"({m['n_breached_test']}/{m['n_test']} breached)")
    print("=" * 64)

    if not args.eval:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        logger.info("Report written to %s", args.output)

    if args.save_model:
        import xgboost as xgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.model_selection import GroupKFold

        X_all = df[features].values.astype(np.float32)
        y_all = df["breached"].values.astype(int)
        groups_all = df["block"].values
        scale_pos = (y_all == 0).sum() / max((y_all == 1).sum(), 1)

        # Out-of-fold predictions across all spatial blocks -> honest inputs
        # for the final calibrator (never in-sample for the model scored).
        oof = np.full(len(df), np.nan)
        for tr_idx, te_idx in GroupKFold(n_splits=args.n_blocks).split(
            X_all, y_all, groups_all
        ):
            sp = (y_all[tr_idx] == 0).sum() / max((y_all[tr_idx] == 1).sum(), 1)
            m = xgb.XGBClassifier(**XGBOOST_PARAMS, scale_pos_weight=sp)
            m.fit(X_all[tr_idx], y_all[tr_idx])
            oof[te_idx] = m.predict_proba(X_all[te_idx])[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, y_all)

        model = xgb.XGBClassifier(**XGBOOST_PARAMS, scale_pos_weight=scale_pos)
        model.fit(X_all, y_all)
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(args.model_output))
        logger.info("Booster saved to %s", args.model_output)

        # Persist the isotonic mapping as thresholds — reconstructable at
        # inference via np.interp(p_raw, x_thresholds, y_thresholds).
        cal_path = args.model_output.with_suffix(".calibration.json")
        cal_path.write_text(json.dumps({
            "method": "isotonic_crossfit_oof",
            "x_thresholds": iso.X_thresholds_.tolist(),
            "y_thresholds": iso.y_thresholds_.tolist(),
            "apply": "p_cal = np.interp(p_raw, x_thresholds, y_thresholds)",
            "note": (
                "raw booster probabilities are intercept-inflated by "
                "scale_pos_weight; apply this mapping before using p as a "
                "calibrated probability"
            ),
        }, indent=2))
        logger.info("Calibrator saved to %s", cal_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
