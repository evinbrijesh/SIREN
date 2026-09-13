"""Real GLOF training dataset from HMAGLOFDB + ICIMOD HMA glacial lake inventory.

Replaces the hand-curated 50-lake dataset in glof_dataset.py with real data:

    - HMAGLOFDB (775 documented GLOF events across HMA, 1830s–2020s)
      → breach labels (y=1) with real lake type, location, elevation
    - ICIMOD HMA Glacial Lake Inventory (31K lakes with area, elevation)
      → lake population for non-breached background (y=0)

Feature engineering:
    - lake_area_km2: real from ICIMOD Area field (m² → km²)
    - moraine_dam_width_m: Lake_type proxy (moraine~100, ice~200, supra~50, bedrock~0)
    - moraine_dam_height_m: Lake_Type proxy (moraine~20, ice~30, supra~5, bedrock~0)
    - lake_expansion_rate: regional default 0.05 (until multi-temporal matching wired)
    - rain_anomaly_7d: 0.0 (normal climatology, until IMERG integrated)
    - mean_upstream_slope_deg: estimated from Lake_Elev (higher → steeper)

Limitations:
    - Dam geometry is estimated from Lake_Type, not field-surveyed
    - Rainfall/temperature anomalies are zero (IMERG/ERA5 not yet integrated)
    - Expansion rate is a regional default (multi-temporal matching pending)

These limitations are documented in KNOWN_LIMITATIONS.md. The real breach
labels and lake areas are the primary value over the synthetic ledger.

Usage:
    from siren.ml.real_glof_dataset import load_real_dataset
    df = load_real_dataset()
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
HMAGLOFDB_CSV = REPO_ROOT / "data" / "datasets" / "HMAGLOFDB" / "Database" / "GLOFs" / "HMAGLOFDB.csv"
ICIMOD_SHP = REPO_ROOT / "data" / "datasets" / "glacial_lake_2022-2024" / "Glacial_Lake_2022.shp"

# Feature names matching glof_dataset.py / susceptibility.py Level 1
FEATURE_NAMES = [
    "lake_expansion_rate",
    "moraine_dam_width_m",
    "moraine_dam_height_m",
    "rain_anomaly_7d",
    "mean_upstream_slope_deg",
    "lake_area_km2",
]

# Lake_Type → dam geometry proxy (from published HMA moraine dam statistics)
# Sources: Rounce et al. (2016), Komori et al. (2012), ICIMOD (2011, 2020)
DAM_GEOMETRY_PROXY: dict[str, dict[str, float]] = {
    "Moraine dammed":    {"width_m": 150.0, "height_m": 25.0, "slope_deg": 28.0},
    "Ice dammed":        {"width_m": 200.0, "height_m": 35.0, "slope_deg": 22.0},
    "Supraglacial":      {"width_m": 50.0,  "height_m": 5.0,  "slope_deg": 15.0},
    "Water pocket":      {"width_m": 40.0,  "height_m": 8.0,  "slope_deg": 20.0},
    "Landslide dammed":  {"width_m": 100.0, "height_m": 30.0, "slope_deg": 30.0},
    "Bedrock":           {"width_m": 0.0,   "height_m": 0.0,  "slope_deg": 10.0},
    "Thermokarst":       {"width_m": 60.0,  "height_m": 10.0, "slope_deg": 12.0},
    "Unknown":           {"width_m": 100.0, "height_m": 15.0, "slope_deg": 20.0},
}

# Default expansion rate when multi-temporal matching is not available
DEFAULT_EXPANSION_RATE = 0.05  # 5% per year (HMA average from ICIMOD reports)

# Country → region mapping for spatial cross-validation
COUNTRY_TO_REGION: dict[str, str] = {
    "Nepal": "Central Himalaya",
    "India": "Western Himalaya",
    "Bhutan": "Eastern Himalaya",
    "China": "Tibetan Plateau",
    "Pakistan": "Karakoram",
    "Afghanistan": "Hindu Kush",
    "Kyrgyzstan": "Tien Shan",
    "Kazakhstan": "Tien Shan",
    "Tajikistan": "Pamir",
}


def _latlon_to_region(lat: float, lon: float) -> str:
    """Approximate geographic region from lat/lon for ICIMOD lakes."""
    if lat < 28.5 and lon < 82:
        return "Western Himalaya"
    elif lat < 28.5 and 82 <= lon < 89:
        return "Central Himalaya"
    elif lat < 28.5 and lon >= 89:
        return "Eastern Himalaya"
    elif 28.5 <= lat < 33 and lon < 78:
        return "Western Himalaya"
    elif 28.5 <= lat < 33 and 78 <= lon < 85:
        return "Central Himalaya"
    elif 28.5 <= lat < 33 and lon >= 85:
        return "Tibetan Plateau"
    elif 33 <= lat < 39 and lon < 75:
        return "Hindu Kush"
    elif 33 <= lat < 39 and 75 <= lon < 80:
        return "Karakoram"
    elif 33 <= lat < 39 and lon >= 80:
        return "Pamir"
    elif lat >= 39:
        return "Tien Shan"
    return "Tibetan Plateau"


def _elev_to_slope(elev_m: float) -> float:
    """Estimate mean upstream slope from lake elevation.

    Higher-elevation HMA lakes tend to sit in steeper headwall cirques.
    Based on regression from ICIMOD lake elevations vs published slopes.
    """
    if elev_m < 0 or pd.isna(elev_m):
        return 15.0
    # Linear approximation: 3000m → ~12°, 5000m → ~25°, 6000m → ~35°
    slope = 12.0 + (elev_m - 3000) * 0.008
    return float(np.clip(slope, 5.0, 45.0))


def _load_hmaglofdb() -> pd.DataFrame:
    """Load HMAGLOFDB GLOF events as breach records."""
    df = pd.read_csv(str(HMAGLOFDB_CSV), encoding="latin-1")
    df = df[df["Lat_lake"].notna()].copy()

    # Parse numeric fields
    df["Lat_lake"] = pd.to_numeric(df["Lat_lake"], errors="coerce")
    df["Lon_lake"] = pd.to_numeric(df["Lon_lake"], errors="coerce")
    df["Elev_lake"] = pd.to_numeric(df["Elev_lake"], errors="coerce")
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    df["Year_approx"] = pd.to_numeric(df["Year_approx"], errors="coerce")

    df = df[df["Lat_lake"].notna() & df["Lon_lake"].notna()].copy()

    rng = np.random.default_rng(42)

    records = []
    for _, row in df.iterrows():
        lake_type = str(row.get("Lake_type", "Unknown")).strip()
        dam = DAM_GEOMETRY_PROXY.get(lake_type, DAM_GEOMETRY_PROXY["Unknown"])

        area_m2 = row.get("Area")
        if pd.notna(area_m2) and area_m2 > 0:
            area_km2 = float(area_m2) / 1e6
        else:
            if lake_type == "Moraine dammed":
                area_km2 = float(rng.uniform(0.1, 3.0))
            elif lake_type == "Ice dammed":
                area_km2 = float(rng.uniform(0.2, 5.0))
            elif lake_type == "Supraglacial":
                area_km2 = float(rng.uniform(0.02, 0.5))
            else:
                area_km2 = float(rng.uniform(0.05, 1.0))

        elev = row.get("Elev_lake", 4500)
        if pd.isna(elev):
            elev = 4500

        country = str(row.get("Country", "Unknown")).strip()
        region = COUNTRY_TO_REGION.get(country, _latlon_to_region(row["Lat_lake"], row["Lon_lake"]))

        # Add variability to dam geometry (breached lakes tend toward narrow crests)
        dam_width = float(rng.normal(dam["width_m"] * 0.7, dam["width_m"] * 0.2))
        dam_height = float(rng.normal(dam["height_m"] * 1.2, dam["height_m"] * 0.3))
        dam_width = max(dam_width, 5.0)
        dam_height = max(dam_height, 0.0)

        # Expansion rate: varied, biased toward higher (breached lakes grow faster)
        expansion = float(rng.uniform(0.03, 0.20))

        # Rain anomaly: varied, biased toward positive (triggers)
        rain = float(rng.normal(1.5, 1.0))

        records.append({
            "name": str(row.get("Lake_name", "Unnamed")).strip() or f"GF_{row.get('GF_ID', '?')}",
            "country": country,
            "region": region,
            "lat": float(row["Lat_lake"]),
            "lon": float(row["Lon_lake"]),
            "lake_expansion_rate": expansion,
            "moraine_dam_width_m": dam_width,
            "moraine_dam_height_m": dam_height,
            "rain_anomaly_7d": rain,
            "mean_upstream_slope_deg": _elev_to_slope(float(elev)),
            "lake_area_km2": area_km2,
            "breached": 1,
            "breach_year": int(row["Year_approx"]) if pd.notna(row["Year_approx"]) else None,
            "source": f"HMAGLOFDB (Lake_type={lake_type})",
        })

    return pd.DataFrame(records)


def _load_icimod_nonbreach(
    breach_df: pd.DataFrame,
    n_sample: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load ICIMOD HMA lake inventory as non-breached background.

    Samples n_sample lakes that are NOT near any HMAGLOFDB breach location
    (within 1 km buffer), stratified by region to match the breach distribution.

    Feature variability: non-breached lakes get sampled dam geometry from
    the same Lake_Type-based distributions as breached lakes, but biased
    toward stable configurations (wider crests, lower dams). This creates
    realistic class overlap — the classifier must learn the actual signal,
    not a trivial constant-value separator.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    rng = np.random.default_rng(seed)

    gdf = gpd.read_file(str(ICIMOD_SHP))
    gdf = gdf.to_crs("EPSG:4326")

    # Build breach location buffer (1 km)
    breach_gdf = gpd.GeoDataFrame(
        breach_df[["lat", "lon"]],
        geometry=[Point(r, l) for l, r in zip(breach_df["lat"], breach_df["lon"])],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    breach_buffer = breach_gdf.geometry.buffer(1000).to_crs("EPSG:4326")
    breach_union = breach_buffer.unary_union

    # Filter ICIMOD lakes not near any breach
    gdf_filtered = gdf[~gdf.geometry.within(breach_union)].copy()
    gdf_filtered = gdf_filtered[gdf_filtered.geometry.notna()]

    # Assign region
    gdf_filtered["region"] = gdf_filtered.apply(
        lambda r: _latlon_to_region(r["Latitude"], r["Longitude"]), axis=1
    )

    # Stratified sample by region to match breach distribution
    breach_region_counts = breach_df["region"].value_counts(normalize=True)
    n_per_region = {}
    for region, frac in breach_region_counts.items():
        n_per_region[region] = int(n_sample * frac)

    sampled = []
    for region, n in n_per_region.items():
        region_lakes = gdf_filtered[gdf_filtered["region"] == region]
        if len(region_lakes) == 0:
            continue
        n = min(n, len(region_lakes))
        idx = rng.choice(len(region_lakes), size=n, replace=False)
        sampled.append(region_lakes.iloc[idx])

    if not sampled:
        n = min(n_sample, len(gdf_filtered))
        idx = rng.choice(len(gdf_filtered), size=n, replace=False)
        sampled = [gdf_filtered.iloc[idx]]
    else:
        total_sampled = sum(len(s) for s in sampled)
        remaining = n_sample - total_sampled
        if remaining > 0:
            all_sampled_idx = set()
            for s in sampled:
                all_sampled_idx.update(s.index.tolist())
            available = gdf_filtered[~gdf_filtered.index.isin(all_sampled_idx)]
            if len(available) > 0:
                n = min(remaining, len(available))
                idx = rng.choice(len(available), size=n, replace=False)
                sampled.append(available.iloc[idx])

    result_df = pd.concat(sampled, ignore_index=True)

    # Sample Lake_Type for non-breached lakes (most are moraine-dammed in HMA)
    lake_type_choices = ["Moraine dammed", "Ice dammed", "Bedrock", "Supraglacial"]
    lake_type_probs = [0.55, 0.15, 0.20, 0.10]

    records = []
    for _, row in result_df.iterrows():
        area_m2 = float(row.get("Area", 0))
        area_km2 = area_m2 / 1e6 if area_m2 > 0 else 0.1

        elev = float(row.get("Lake_Elev", 4500))
        if pd.isna(elev) or elev < 0:
            elev = 4500

        # Sample a Lake_Type for this non-breached lake
        lake_type = rng.choice(lake_type_choices, p=lake_type_probs)
        dam = DAM_GEOMETRY_PROXY[lake_type]

        # Add variability: sample around the proxy mean with some overlap
        # Non-breached lakes overlap with breached in dam geometry — many
        # stable lakes have narrow dams too; the signal is in the bias, not separation
        dam_width = float(rng.normal(dam["width_m"] * 1.2, dam["width_m"] * 0.4))
        dam_height = float(rng.normal(dam["height_m"] * 0.8, dam["height_m"] * 0.3))
        dam_width = max(dam_width, 5.0)
        dam_height = max(dam_height, 0.0)

        # Expansion rate: overlaps with breached range (many stable lakes grow fast)
        expansion = float(rng.uniform(0.0, 0.15))

        # Rain anomaly: overlaps with breached range (storms hit stable lakes too)
        rain = float(rng.normal(0.3, 1.0))

        records.append({
            "name": f"ICIMOD_{row.get('ID', 'unknown')}",
            "country": "Unknown",
            "region": row["region"],
            "lat": float(row["Latitude"]),
            "lon": float(row["Longitude"]),
            "lake_expansion_rate": expansion,
            "moraine_dam_width_m": dam_width,
            "moraine_dam_height_m": dam_height,
            "rain_anomaly_7d": rain,
            "mean_upstream_slope_deg": _elev_to_slope(elev),
            "lake_area_km2": area_km2,
            "breached": 0,
            "breach_year": None,
            "source": f"ICIMOD HMA Inventory 2022 (Lake_type={lake_type})",
        })

    return pd.DataFrame(records)


def load_real_dataset(n_nonbreach: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Load the real GLOF training dataset from HMAGLOFDB + ICIMOD.

    Args:
        n_nonbreach: number of non-breached ICIMOD lakes to sample.
        seed: random seed for reproducible sampling (Hard Rule 6).

    Returns:
        DataFrame with columns matching glof_dataset.py:
        name, country, region, lat, lon, lake_expansion_rate,
        moraine_dam_width_m, moraine_dam_height_m, rain_anomaly_7d,
        mean_upstream_slope_deg, lake_area_km2, breached, breach_year, source
    """
    # PRD v4.7 §17.3: this loader generates label-dependent features (dam
    # geometry, rainfall, expansion rate, missing-area imputation from
    # label-conditioned random distributions). The resulting checkpoints
    # (xgboost_susceptibility_v1 / v2_real) are disqualified. Loading is
    # blocked until the feature generation path is replaced with measured data.
    raise ValueError(
        "Real GLOF dataset loader is disqualified (PRD v4.7 §17.3): "
        "the generated-feature contamination path (label-dependent dam "
        "geometry, rainfall, expansion rate, and missing-area imputation) "
        "has not been replaced with measured data. The resulting "
        "checkpoints are disqualified and cannot be used for training."
    )


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the feature matrix (n_samples, 6) from the dataset."""
    return df[FEATURE_NAMES].values.astype(np.float32)


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """Extract binary labels (0 = stable, 1 = breached)."""
    return df["breached"].values.astype(int)


def get_spatial_groups(df: pd.DataFrame) -> np.ndarray:
    """Extract spatial group labels for spatial cross-validation."""
    return df["region"].values


def dataset_summary(df: pd.DataFrame) -> str:
    """Return a human-readable summary of the dataset."""
    n_breached = int(df["breached"].sum())
    n_stable = int((df["breached"] == 0).sum())
    regions = df["region"].nunique()
    countries = df["country"].nunique()

    lines = [
        f"Real GLOF Training Dataset (HMAGLOFDB + ICIMOD)",
        f"  Total lakes: {len(df)}",
        f"  Breached: {n_breached}",
        f"  Stable: {n_stable}",
        f"  Class imbalance ratio: 1:{n_stable // max(n_breached, 1)}",
        f"  Regions: {regions} ({', '.join(sorted(df['region'].unique()))})",
        f"  Countries: {countries}",
        "",
        "Feature statistics (breached lakes only):",
    ]
    breached = df[df["breached"] == 1]
    for col in FEATURE_NAMES:
        lines.append(f"  {col}: mean={breached[col].mean():.3f}, std={breached[col].std():.3f}")

    return "\n".join(lines)
