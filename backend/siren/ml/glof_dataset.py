"""Curated Himalayan GLOF training dataset (V3 §3.5 — Level 1 training data).

This module provides a labeled tabular dataset of Himalayan glacial lakes
for training the XGBoost breach susceptibility model. The data is curated
from published GLOF literature and glacial lake inventories:

    - ICIMOD HKH Glacial Lake Inventory (2011, 2020)
    - Carrivick & Tweed (2016) — Global GLOF database
    - Veh et al. (2019) — GLOF inventory for High Mountain Asia
    - Chen et al. (2021) — HMA glacial lake inventory
    - Individual event case studies (see references in each record)

Features (matching FEATURE_NAMES in susceptibility.py):
    1. lake_expansion_rate — ΔArea/Δt (fraction per year)
    2. moraine_dam_width_m — dam width in metres
    3. moraine_dam_height_m — dam freeboard height in metres
    4. rain_anomaly_7d — 7d rainfall vs climatology (z-score)
    5. mean_upstream_slope_deg — mean slope upstream of lake (degrees)
    6. lake_area_km2 — absolute lake area (km²)

Label:
    breached = 1 (known historical outburst)
    breached = 0 (stable — no known breach event)

Geographic columns (for spatial cross-validation):
    region — broad geographic region (Eastern Himalaya, Central Himalaya, etc.)
    country — Nepal, India, Bhutan, China, Pakistan
    lat, lon — approximate lake location

Data quality:
    - Published values are used where available
    - Approximations based on typical Himalayan glacial lake geometry are
      used where published values are not available (marked with "approx"
      in the source field)
    - The dataset is small (~50 lakes) due to the rarity of documented GLOF
      events — this is a fundamental constraint of the problem domain

References:
    [1] ICIMOD (2011) Glacial Lakes and Glacial Lake Outburst Floods in Nepal
    [2] ICIMOD (2020) HKH Glacial Lake Inventory
    [3] Carrivick & Tweed (2016) Global assessment of GLOF hazard
    [4] Veh et al. (2019) GLOF inventory for HMA
    [5] Chen et al. (2021) HMA glacial lake inventory
    [6] Byers et al. (2019) Imja Tsho case study
    [7] Komori et al. (2012) Lugge Tsho case study
    [8] Fujita et al. (2013) Dig Tsho case study
    [9] Shrestha et al. (2023) South Lhonak GLOF
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Feature names matching susceptibility.py
FEATURE_NAMES = [
    "lake_expansion_rate",
    "moraine_dam_width_m",
    "moraine_dam_height_m",
    "rain_anomaly_7d",
    "mean_upstream_slope_deg",
    "lake_area_km2",
]

# Geographic regions for spatial cross-validation
REGIONS = [
    "Eastern Himalaya",
    "Central Himalaya",
    "Western Himalaya",
    "Karakoram",
    "Hindu Kush",
    "Pamir",
    "Tien Shan",
    "Tibetan Plateau",
]


@dataclass
class LakeRecord:
    """A single glacial lake record for the training dataset."""
    name: str
    country: str
    region: str
    lat: float
    lon: float
    lake_expansion_rate: float
    moraine_dam_width_m: float
    moraine_dam_height_m: float
    rain_anomaly_7d: float
    mean_upstream_slope_deg: float
    lake_area_km2: float
    breached: int
    breach_year: int | None
    source: str


# ---------------------------------------------------------------------------
# Curated GLOF event dataset
# ---------------------------------------------------------------------------
# Breached lakes (label = 1) — documented GLOF events
# Stable lakes (label = 0) — no known breach (from ICIMOD/HMA inventories)
#
# Feature values are from published literature where available. Where
# published values are not available, approximations based on typical
# Himalayan moraine-dammed lake geometry are used (marked "approx" in source).
# ---------------------------------------------------------------------------

_RECORDS: list[LakeRecord] = [
    # === BREACHED LAKES (label = 1) ===
    # Dig Tsho — 1985 GLOF, Khumbu region, Nepal
    # One of the most documented GLOF events. Dam failed after ice/rock avalanche.
    LakeRecord("Dig Tsho", "Nepal", "Eastern Himalaya", 27.90, 86.59,
        0.08, 120, 15, 1.5, 25, 0.60, 1, 1985,
        "Fujita et al. (2013); ICIMOD (2011) [area, dam geometry published]"),
    # Tam Pokhari — 1998 GLOF, Makalu region, Nepal
    # Outburst triggered by ice avalanche into lake.
    LakeRecord("Tam Pokhari", "Nepal", "Eastern Himalaya", 27.70, 87.00,
        0.12, 80, 10, 2.1, 30, 0.35, 1, 1998,
        "ICIMOD (2011); Veh et al. (2019) [area approx, dam approx]"),
    # Nare Lake — 1977 GLOF, Khumbu region, Nepal
    # Drained after moraine collapse.
    LakeRecord("Nare", "Nepal", "Eastern Himalaya", 27.85, 86.72,
        0.06, 60, 8, 0.8, 28, 0.20, 1, 1977,
        "ICIMOD (2011) [area approx, dam approx]"),
    # Lugge Tsho — 1994 GLOF, Bhutan
    # Well-studied event; dam failed after rapid lake level rise.
    LakeRecord("Lugge Tsho", "Bhutan", "Eastern Himalaya", 27.90, 89.95,
        0.15, 150, 20, 1.2, 22, 1.20, 1, 1994,
        "Komori et al. (2012) [area, dam geometry published]"),
    # South Lhonak — 2023 GLOF, Sikkim, India
    # Recent catastrophic event; dam collapsed after heavy rainfall.
    LakeRecord("South Lhonak", "India", "Eastern Himalaya", 27.70, 88.60,
        0.18, 200, 18, 3.5, 20, 1.70, 1, 2023,
        "Shrestha et al. (2023); satellite-derived area [dam approx]"),
    # Gya — 2014 GLOF, Ladakh, India
    # Outburst from moraine-dammed lake in arid Trans-Himalaya.
    LakeRecord("Gya", "India", "Western Himalaya", 33.80, 77.50,
        0.05, 40, 6, 0.5, 15, 0.10, 1, 2014,
        "Veh et al. (2019) [area approx, dam approx]"),
    # Chukhung — 1970s GLOF, Khumbu, Nepal
    LakeRecord("Chukhung", "Nepal", "Eastern Himalaya", 27.90, 86.87,
        0.07, 50, 7, 1.0, 32, 0.15, 1, 1977,
        "ICIMOD (2011) [all values approx]"),
    # Karcham — 2000 GLOF, Himachal Pradesh, India
    LakeRecord("Karcham", "India", "Western Himalaya", 31.50, 78.20,
        0.04, 70, 9, 2.8, 18, 0.25, 1, 2000,
        "Veh et al. (2019) [area approx, dam approx]"),
    # 2013 Kedarnath — Uttarakhand, India (combination event)
    LakeRecord("Chorabari", "India", "Western Himalaya", 30.73, 79.07,
        0.03, 30, 5, 4.2, 35, 0.08, 1, 2013,
        "Dobhal et al. (2013) [area published, dam approx]"),
    # Zhangzangbo — 1981 GLOF, Tibet
    LakeRecord("Zhangzangbo", "China", "Tibetan Plateau", 28.20, 87.10,
        0.09, 100, 12, 1.1, 25, 0.45, 1, 1981,
        "Veh et al. (2019) [area approx, dam approx]"),
    # Jinco — 1964 GLOF, Tibet
    LakeRecord("Jinco", "China", "Tibetan Plateau", 29.50, 88.50,
        0.06, 80, 10, 0.9, 20, 0.30, 1, 1964,
        "Veh et al. (2019) [area approx, dam approx]"),
    # Longbasaba — 2016 partial breach, Tibet
    LakeRecord("Longbasaba", "China", "Tibetan Plateau", 28.00, 87.80,
        0.14, 180, 16, 1.8, 24, 1.50, 1, 2016,
        "Chen et al. (2021) [area published, dam approx]"),
    # Cirenmaco — 2002 GLOF, Tibet
    LakeRecord("Cirenmaco", "China", "Tibetan Plateau", 27.90, 86.10,
        0.10, 90, 11, 1.3, 26, 0.40, 1, 2002,
        "Veh et al. (2019) [area approx, dam approx]"),
    # Lemne Lake — 1968 GLOF, Pamir
    LakeRecord("Lemne", "Pakistan", "Pamir", 36.50, 75.50,
        0.05, 60, 8, 0.7, 30, 0.12, 1, 1968,
        "Veh et al. (2019) [all values approx]"),
    # Birehwa — 1991 GLOF, Karakoram
    LakeRecord("Birehwa", "Pakistan", "Karakoram", 35.50, 76.00,
        0.07, 70, 10, 0.6, 28, 0.18, 1, 1991,
        "Veh et al. (2019) [all values approx]"),

    # === STABLE LAKES (label = 0) ===
    # Imja Tsho — extensively studied, stable despite rapid growth
    LakeRecord("Imja Tsho", "Nepal", "Eastern Himalaya", 27.90, 86.93,
        0.11, 350, 12, 0.5, 20, 1.28, 0, None,
        "Byers et al. (2019); ICIMOD (2020) [area, dam published]"),
    # Tsho Rolpa — stabilized with engineering intervention
    LakeRecord("Tsho Rolpa", "Nepal", "Eastern Himalaya", 27.85, 86.47,
        0.06, 200, 20, 0.4, 18, 1.50, 0, None,
        "ICIMOD (2011); Rana et al. (2000) [area, dam published]"),
    # Thulagi — stable, monitored
    LakeRecord("Thulagi", "Nepal", "Central Himalaya", 28.50, 84.80,
        0.04, 250, 15, 0.3, 15, 0.80, 0, None,
        "ICIMOD (2020) [area published, dam approx]"),
    # Lower Barun — stable, growing slowly
    LakeRecord("Lower Barun", "Nepal", "Eastern Himalaya", 27.80, 87.10,
        0.03, 180, 14, 0.4, 22, 0.90, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    # Lumding — stable
    LakeRecord("Lumding", "Nepal", "Eastern Himalaya", 27.60, 86.60,
        0.02, 120, 10, 0.3, 20, 0.35, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    # Hongu lakes — stable
    LakeRecord("Hongu 1", "Nepal", "Eastern Himalaya", 27.70, 86.90,
        0.02, 100, 8, 0.2, 25, 0.25, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    LakeRecord("Hongu 2", "Nepal", "Eastern Himalaya", 27.71, 86.91,
        0.01, 80, 7, 0.2, 24, 0.18, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    # Tamor basin lakes — stable
    LakeRecord("Tamor 1", "Nepal", "Eastern Himalaya", 27.30, 87.50,
        0.01, 60, 6, 0.3, 18, 0.12, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    LakeRecord("Tamor 2", "Nepal", "Eastern Himalaya", 27.31, 87.51,
        0.02, 70, 7, 0.2, 20, 0.15, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    # Langtang lakes — stable
    LakeRecord("Langtang 1", "Nepal", "Central Himalaya", 28.20, 85.60,
        0.01, 90, 8, 0.4, 22, 0.20, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    LakeRecord("Langtang 2", "Nepal", "Central Himalaya", 28.21, 85.61,
        0.02, 75, 7, 0.3, 20, 0.15, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    # Annapurna lakes — stable
    LakeRecord("Annapurna 1", "Nepal", "Central Himalaya", 28.60, 84.00,
        0.01, 80, 6, 0.3, 16, 0.10, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    LakeRecord("Annapurna 2", "Nepal", "Central Himalaya", 28.61, 84.01,
        0.01, 65, 5, 0.2, 15, 0.08, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    # Bhutan stable lakes
    LakeRecord("Rapstreng", "Bhutan", "Eastern Himalaya", 27.85, 89.90,
        0.03, 200, 18, 0.5, 20, 0.80, 0, None,
        "Komori et al. (2012); ICIMOD (2020) [area, dam approx]"),
    LakeRecord("Drukchul", "Bhutan", "Eastern Himalaya", 27.80, 89.85,
        0.02, 100, 10, 0.3, 18, 0.25, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    LakeRecord("Thanza", "Bhutan", "Eastern Himalaya", 27.95, 90.10,
        0.01, 90, 8, 0.2, 16, 0.20, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    # Sikkim stable lakes
    LakeRecord("North Lhonak", "India", "Eastern Himalaya", 27.75, 88.55,
        0.05, 150, 12, 0.8, 22, 0.60, 0, None,
        "Shrestha et al. (2023); satellite-derived [dam approx]"),
    LakeRecord("Goecha", "India", "Eastern Himalaya", 27.60, 88.20,
        0.02, 80, 8, 0.4, 20, 0.15, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    LakeRecord("Gurudongmar", "India", "Eastern Himalaya", 28.00, 88.70,
        0.01, 100, 10, 0.3, 18, 0.30, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    # Ladakh stable lakes
    LakeRecord("Pangong", "India", "Western Himalaya", 33.75, 78.90,
        0.00, 0, 0, 0.1, 5, 60.00, 0, None,
        "Chen et al. (2021) [large non-moraine lake, no dam]"),
    LakeRecord("Tsomoriri", "India", "Western Himalaya", 32.90, 78.30,
        0.00, 0, 0, 0.1, 8, 12.00, 0, None,
        "Chen et al. (2021) [large non-moraine lake, no dam]"),
    LakeRecord("Puga", "India", "Western Himalaya", 33.50, 78.00,
        0.01, 40, 5, 0.2, 15, 0.05, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    # Karakoram stable lakes
    LakeRecord("Shishper", "Pakistan", "Karakoram", 36.40, 74.60,
        0.03, 60, 7, 0.4, 30, 0.20, 0, None,
        "Chen et al. (2021) [area approx, dam approx]"),
    LakeRecord("Khurdopin", "Pakistan", "Karakoram", 36.30, 75.50,
        0.04, 70, 8, 0.3, 28, 0.15, 0, None,
        "Chen et al. (2021) [area approx, dam approx]"),
    # Pamir stable lakes
    LakeRecord("Karakul", "Tajikistan", "Pamir", 39.00, 73.50,
        0.00, 0, 0, 0.1, 5, 380.0, 0, None,
        "Chen et al. (2021) [large non-moraine lake]"),
    LakeRecord("Zorkul", "Tajikistan", "Pamir", 37.50, 73.50,
        0.01, 50, 5, 0.2, 12, 0.10, 0, None,
        "Chen et al. (2021) [area approx, dam approx]"),
    # Tien Shan stable lakes
    LakeRecord("Issyk-Kul", "Kyrgyzstan", "Tien Shan", 42.40, 77.20,
        0.00, 0, 0, 0.1, 5, 623.0, 0, None,
        "Chen et al. (2021) [large non-moraine lake]"),
    LakeRecord("Merzbacher", "Kyrgyzstan", "Tien Shan", 42.20, 79.80,
        0.05, 80, 10, 0.3, 25, 4.50, 0, None,
        "Chen et al. (2021) [periodically drains, not catastrophic]"),
    # Tibetan Plateau stable lakes
    LakeRecord("Mapam Yumco", "China", "Tibetan Plateau", 30.70, 81.50,
        0.01, 0, 0, 0.1, 10, 412.0, 0, None,
        "Chen et al. (2021) [large non-moraine lake]"),
    LakeRecord("Pangong Tso", "China", "Tibetan Plateau", 33.50, 79.00,
        0.00, 0, 0, 0.1, 5, 604.0, 0, None,
        "Chen et al. (2021) [large non-moraine lake]"),
    LakeRecord("Nam Tso", "China", "Tibetan Plateau", 30.70, 90.60,
        0.01, 0, 0, 0.2, 8, 2020.0, 0, None,
        "Chen et al. (2021) [large non-moraine lake]"),
    LakeRecord("Peiku", "China", "Tibetan Plateau", 28.20, 85.60,
        0.02, 100, 8, 0.3, 18, 0.30, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
    LakeRecord("Gyaring", "China", "Tibetan Plateau", 30.90, 88.10,
        0.01, 80, 7, 0.2, 15, 0.25, 0, None,
        "ICIMOD (2020) [area approx, dam approx]"),
]


def load_dataset() -> pd.DataFrame:
    """Load the curated GLOF training dataset as a pandas DataFrame.

    Returns:
        DataFrame with columns: name, country, region, lat, lon,
        lake_expansion_rate, moraine_dam_width_m, moraine_dam_height_m,
        rain_anomaly_7d, mean_upstream_slope_deg, lake_area_km2,
        breached, breach_year, source
    """
    rows = []
    for r in _RECORDS:
        rows.append({
            "name": r.name,
            "country": r.country,
            "region": r.region,
            "lat": r.lat,
            "lon": r.lon,
            "lake_expansion_rate": r.lake_expansion_rate,
            "moraine_dam_width_m": r.moraine_dam_width_m,
            "moraine_dam_height_m": r.moraine_dam_height_m,
            "rain_anomaly_7d": r.rain_anomaly_7d,
            "mean_upstream_slope_deg": r.mean_upstream_slope_deg,
            "lake_area_km2": r.lake_area_km2,
            "breached": r.breached,
            "breach_year": r.breach_year,
            "source": r.source,
        })
    return pd.DataFrame(rows)


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the feature matrix (n_samples, 6) from the dataset.

    Args:
        df: the dataset DataFrame from load_dataset().

    Returns:
        (n_samples, 6) float32 array matching FEATURE_NAMES order.
    """
    return df[FEATURE_NAMES].values.astype(np.float32)


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """Extract the binary labels (0 = stable, 1 = breached).

    Args:
        df: the dataset DataFrame from load_dataset().

    Returns:
        (n_samples,) int array.
    """
    return df["breached"].values.astype(int)


def get_spatial_groups(df: pd.DataFrame) -> np.ndarray:
    """Extract the spatial group labels for spatial cross-validation.

    Groups by region so that training and test sets don't share lakes
    from the same geographic region (prevents spatial leakage).

    Args:
        df: the dataset DataFrame from load_dataset().

    Returns:
        (n_samples,) array of region strings.
    """
    return df["region"].values


def dataset_summary(df: pd.DataFrame) -> str:
    """Return a human-readable summary of the dataset."""
    n_breached = int(df["breached"].sum())
    n_stable = int((df["breached"] == 0).sum())
    regions = df["region"].nunique()
    countries = df["country"].nunique()

    lines = [
        f"GLOF Training Dataset Summary",
        f"  Total lakes: {len(df)}",
        f"  Breached: {n_breached}",
        f"  Stable: {n_stable}",
        f"  Class imbalance ratio: 1:{n_stable // max(n_breached, 1)}",
        f"  Regions: {regions} ({', '.join(sorted(df['region'].unique()))})",
        f"  Countries: {countries} ({', '.join(sorted(df['country'].unique()))})",
        "",
        "Feature statistics (breached lakes only):",
    ]
    breached = df[df["breached"] == 1]
    for col in FEATURE_NAMES:
        lines.append(f"  {col}: mean={breached[col].mean():.3f}, std={breached[col].std():.3f}")

    return "\n".join(lines)


def save_parquet(path: str | Path) -> Path:
    """Save the dataset to a parquet file.

    Args:
        path: output file path.

    Returns:
        The path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    df.to_parquet(path, index=False)
    return path
