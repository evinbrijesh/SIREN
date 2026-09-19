"""External validation: score published potentially-dangerous lakes.

Scores a curated list of lakes flagged as dangerous in the literature
(ICIMOD PDGL inventories, Veh et al. 2020, Rounce et al. 2016, Shrestha
et al. 2023) plus two lakes that actually breached (South Lhonak 2023,
Dig Tsho 1985) against the spatially-validated susceptibility booster +
isotonic calibrator.

Each literature coordinate is matched to the nearest ICIMOD inventory
lake (records match distance; >3 km is flagged). A random sample of
inventory lakes provides the population baseline for percentile ranks.

Honesty caveats recorded in the report:
    - Lakes that breached are measured post-event in ICIMOD-2022 —
      drained/refilled geometries bias their score downward.
    - Lakes within 1 km of an HMAGLOFDB breach centroid are in the
      training positives — marked in_training_positives; their score is
      not out-of-sample evidence.

Usage:
    python -m siren.ml.score_known_lakes --n-baseline 1500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from siren.ml import train_susceptibility_spatial as sus

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = (
    REPO_ROOT / "models" / "checkpoints" / "xgboost_susceptibility_spatial.json"
)
CAL_PATH = MODEL_PATH.with_suffix(".calibration.json")
REPORT_PATH = (
    REPO_ROOT / "models" / "checkpoints" / "pdgl_external_validation.json"
)
MATCH_TOLERANCE_KM = 3.0

# Published coordinates (approximate centroids, decimal degrees).
# Sources: ICIMOD PDGL inventories; Veh et al. 2020; Rounce et al. 2016;
# Shrestha et al. 2023 (South Lhonak post-event); WECS/ICIMOD Dig Tsho.
KNOWN_LAKES = [
    {"name": "Imja Tsho", "lat": 27.8980, "lon": 86.9250,
     "note": "canonical PDGL; 2016 controlled lowering"},
    {"name": "Tsho Rolpa", "lat": 27.8680, "lon": 86.4830,
     "note": "largest Nepal PDGL; 2000 siphon lowering"},
    {"name": "Lower Barun", "lat": 27.7630, "lon": 87.1040,
     "note": "rapid expansion, thinning moraine"},
    {"name": "Thulagi", "lat": 28.4340, "lon": 84.4860,
     "note": "Manaslu; PDGL (Rounce 2016)"},
    {"name": "Lumding Tsho", "lat": 27.8820, "lon": 86.6280,
     "note": "Rolwaling; PDGL"},
    {"name": "Dudh Pokhari (Gokyo 3rd)", "lat": 27.9340, "lon": 86.6940,
     "note": "Gokyo valley; settlement-adjacent"},
    {"name": "Thorthormi Tsho", "lat": 28.0660, "lon": 90.2760,
     "note": "Bhutan; highest-risk PDGL (Veh 2020)"},
    {"name": "Raphstreng Tsho", "lat": 28.0630, "lon": 90.3120,
     "note": "Bhutan; upstream of Thorthormi"},
    {"name": "Chamlang South Tsho", "lat": 27.7560, "lon": 86.9750,
     "note": "PDGL, Hongu valley"},
    {"name": "South Lhonak (breached 2023)", "lat": 27.9120, "lon": 88.1900,
     "note": "BREACHED Oct 2023 — post-event geometry biases score down"},
    {"name": "Dig Tsho (breached 1985)", "lat": 27.8760, "lon": 86.6000,
     "note": "BREACHED Aug 1985 — post-event geometry biases score down"},
]

FEATURES = ["lake_elev_m", "log_lake_area_km2", "log_dist_glacier_m",
            "log_glacier_area_10km"]


def _match_known_lakes(known: list[dict], icimod) -> pd.DataFrame:
    """Match literature coordinates to nearest ICIMOD lake polygon."""
    import geopandas as gpd

    ic_proj = icimod.to_crs(3857)
    rows = []
    for k in known:
        pt = gpd.GeoSeries.from_xy(
            [k["lon"]], [k["lat"]], crs=4326).to_crs(3857)
        # sindex.nearest returns (2, n) index pairs; column 1 is the tree
        # (ICIMOD) positional index. Compute distance explicitly — the
        # return_distance layout differs across geopandas versions.
        pairs = ic_proj.sindex.nearest(pt.geometry, return_all=False)
        tree_idx = int(pairs[0]) if pairs.ndim == 1 else int(pairs[1][0])
        cand = icimod.iloc[tree_idx]
        dist_m = float(pt.iloc[0].distance(ic_proj.geometry.iloc[tree_idx]))
        rows.append({
            **k,
            "matched_icimod_id": str(cand["ID"]),
            "match_km": dist_m / 1000.0,
            "lat": float(cand["Latitude"]),
            "lon": float(cand["Longitude"]),
            "lake_elev_m": float(cand["Lake_Elev"]),
            "lake_area_km2": float(cand["Area"]) / 1e6,
            "matched": bool(dist_m <= MATCH_TOLERANCE_KM * 1000),
        })
    return pd.DataFrame(rows)


def _score(df: pd.DataFrame, model, cal) -> pd.Series:
    X = df[FEATURES].values.astype(np.float32)
    p_raw = model.predict_proba(X)[:, 1]
    if cal is not None:
        p = np.interp(p_raw, cal["x_thresholds"], cal["y_thresholds"])
    else:
        p = p_raw
    return pd.Series(p, index=df.index), pd.Series(p_raw, index=df.index)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-baseline", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=REPORT_PATH)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import xgboost as xgb

    if not MODEL_PATH.exists():
        logger.error("Booster not found: %s", MODEL_PATH)
        return 1
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    cal = json.loads(CAL_PATH.read_text()) if CAL_PATH.exists() else None

    icimod = sus._load_icimod()
    glaciers = sus._load_rgi_glaciers()

    # Training-positive lookup: lakes within 1 km of an HMAGLOFDB breach
    # centroid are in-sample (score is not out-of-sample evidence).
    events = pd.read_csv(str(sus.HMAGLOFDB_CSV), encoding="latin-1")
    events["lat"] = pd.to_numeric(events["Lat_lake"], errors="coerce")
    events["lon"] = pd.to_numeric(events["Lon_lake"], errors="coerce")
    breach_pts = events.dropna(subset=["lat", "lon"])[["lat", "lon"]]

    known = _match_known_lakes(KNOWN_LAKES, icimod)
    known = sus._glacier_features(known, glaciers)
    known["log_lake_area_km2"] = np.log1p(known["lake_area_km2"])
    known["log_dist_glacier_m"] = np.log1p(
        known["dist_glacier_m"].clip(lower=0))
    known["log_glacier_area_10km"] = np.log1p(
        known["glacier_area_10km_m2"] / 1e6)

    def _in_training(lat, lon):
        d = np.sqrt(((breach_pts["lat"] - lat) * 111.32) ** 2
                    + ((breach_pts["lon"] - lon) * 111.32
                       * np.cos(np.radians(lat))) ** 2)
        return bool((d <= 1.0).any())

    known["in_training_positives"] = [
        _in_training(r["lat"], r["lon"]) for _, r in known.iterrows()
    ]

    # Population baseline for percentile ranks
    base = icimod.sample(n=min(args.n_baseline, len(icimod)),
                         random_state=args.seed)
    base_df = pd.DataFrame({
        "lat": base["Latitude"].astype(float).values,
        "lon": base["Longitude"].astype(float).values,
        "lake_elev_m": base["Lake_Elev"].astype(float).values,
        "lake_area_km2": base["Area"].astype(float).values / 1e6,
    })
    base_df = sus._glacier_features(base_df, glaciers)
    base_df["log_lake_area_km2"] = np.log1p(base_df["lake_area_km2"])
    base_df["log_dist_glacier_m"] = np.log1p(
        base_df["dist_glacier_m"].clip(lower=0))
    base_df["log_glacier_area_10km"] = np.log1p(
        base_df["glacier_area_10km_m2"] / 1e6)

    known["p_breach"], known["p_breach_raw"] = _score(known, model, cal)
    base_p, _ = _score(base_df, model, cal)
    known["percentile"] = [
        float((base_p < p).mean() * 100) for p in known["p_breach"]
    ]

    report = {
        "status": "external_validation",
        "model": MODEL_PATH.name,
        "calibrated": cal is not None,
        "n_baseline_lakes": len(base_df),
        "baseline_median_p": float(base_p.median()),
        "baseline_p90": float(base_p.quantile(0.9)),
        "caveats": [
            (
                "breached lakes measured post-event in ICIMOD-2022 "
                "(drained geometry biases score down)"
            ),
            (
                "in_training_positives=True means the lake is within "
                "1 km of an HMAGLOFDB breach centroid — in-sample, not "
                "validation"
            ),
            (
                "literature coordinates matched to nearest ICIMOD "
                "polygon; match_km > 3 flags failed matches"
            ),
        ],
        "lakes": known[
            ["name", "matched_icimod_id", "match_km", "matched",
             "lake_elev_m", "lake_area_km2", "dist_glacier_m",
             "p_breach", "p_breach_raw", "percentile",
             "in_training_positives", "note"]
        ].round(4).to_dict("records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))

    print(f"\n{'Lake':<34}{'p_cal':>7}{'p_raw':>7}{'pct':>6}"
          f"{'in-train':>10}")
    for _, r in known.iterrows():
        print(f"{r['name']:<34}{r['p_breach']:>7.3f}"
              f"{r['p_breach_raw']:>7.3f}{r['percentile']:>6.1f}"
              f"{r['in_training_positives']!s:>10}")
    print(f"\nBaseline: median {report['baseline_median_p']:.3f}, "
          f"p90 {report['baseline_p90']:.3f} "
          f"({report['n_baseline_lakes']} lakes)")
    logger.info("Report written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
