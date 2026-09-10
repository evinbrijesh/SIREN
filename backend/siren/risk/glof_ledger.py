"""Expanded GLOF ground-truth ledger for susceptibility model training (V3 §3.2).

Generates a curated dataset of 250+ glacial lake attributes paired with
historical breach / non-breach labels, based on the physical characteristics
of real GLOF events from the High Mountain Asia region.

Sources (V3 §3.2):
    - Veh et al. (2022): ~250+ documented historical outburst events across
      Tibet, Bhutan, India, and Nepal, paired against non-breached background lakes.
    - Carrivick & Tweed (2016): global GLOF database with dam geometry and
      trigger mechanisms.
    - ICIMOD glacial lake inventory: lake area, dam type, and expansion rates.

The generator produces physically realistic feature values based on
published distributions from these sources. It is NOT a substitute for
the real curated database — it is a deterministic synthetic generator
that produces the correct statistical properties for model training when
the full database is not yet digitised.

Feature set (V3 §3.2 Level 2 — 9 physics-grounded features):
    1. lake_expansion_rate       — ΔArea/Δt (fraction per year)
    2. dam_width_height_ratio    — W_dam / H_dam (geotechnical piping ratio)
    3. moraine_dam_height_m      — dam freeboard height (m)
    4. rain_anomaly_7d           — 7d rainfall vs climatology (z-score)
    5. mean_upstream_slope_deg   — mean slope upstream (degrees)
    6. lake_area_km2             — absolute lake area (km²)
    7. ice_core_contact_ratio    — L_contact / L_perimeter (calving shockwave)
    8. temp_anomaly_0c_isotherm  — freezing-level height anomaly (m)
    9. dam_width_height_ratio_sq — (W/H)² (non-linear piping threshold)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from siren.risk.susceptibility import FEATURE_NAMES_V2

logger = logging.getLogger(__name__)

# Number of lakes in the expanded ledger (Veh et al. 2022 scale)
DEFAULT_N_LAKES: int = 280
# Breach fraction (~15% of HMA glacial lakes have historically breached)
DEFAULT_BREACH_FRACTION: float = 0.15
# Random seed for reproducibility (Hard Rule 6)
DEFAULT_SEED: int = 42


@dataclass
class GLOFLake:
    """A single glacial lake record in the expanded ledger."""
    lake_id: str
    region: str  # "nepal", "bhutan", "tibet", "india", "pakistan"
    breached: bool
    lake_expansion_rate: float
    dam_width_height_ratio: float
    moraine_dam_height_m: float
    rain_anomaly_7d: float
    mean_upstream_slope_deg: float
    lake_area_km2: float
    ice_core_contact_ratio: float
    temp_anomaly_0c_isotherm: float

    @property
    def dam_width_height_ratio_sq(self) -> float:
        return self.dam_width_height_ratio ** 2

    def to_feature_vector(self) -> np.ndarray:
        """Convert to a 9-feature vector in FEATURE_NAMES_V2 order."""
        return np.array([[
            self.lake_expansion_rate,
            self.dam_width_height_ratio,
            self.moraine_dam_height_m,
            self.rain_anomaly_7d,
            self.mean_upstream_slope_deg,
            self.lake_area_km2,
            self.ice_core_contact_ratio,
            self.temp_anomaly_0c_isotherm,
            self.dam_width_height_ratio_sq,
        ]], dtype=np.float32)

    @property
    def label(self) -> int:
        return 1 if self.breached else 0


def _sample_breach_features(rng: np.random.Generator) -> dict:
    """Sample physically realistic features for a BREACHED lake.

    Breached lakes tend to have:
        - High expansion rate (rapidly growing lakes are unstable)
        - Low W/H ratio (narrow moraine crests undercut faster during piping)
        - High ice-core contact (calving ice creates displacement shockwaves)
        - Positive temperature anomaly (rapid thermal melt destabilises ice cores)
        - High rain anomaly (extreme precipitation triggers overtopping)
    """
    return {
        "lake_expansion_rate": rng.uniform(0.05, 0.35),  # 5-35% per year
        "dam_width_height_ratio": rng.uniform(2.0, 8.0),  # narrow crests
        "moraine_dam_height_m": rng.uniform(20, 80),  # tall dams
        "rain_anomaly_7d": rng.uniform(1.5, 5.0),  # extreme rainfall
        "mean_upstream_slope_deg": rng.uniform(25, 45),  # steep catchments
        "lake_area_km2": rng.uniform(0.3, 5.0),  # moderate to large
        "ice_core_contact_ratio": rng.uniform(0.15, 0.60),  # significant calving
        "temp_anomaly_0c_isotherm": rng.uniform(100, 500),  # 100-500m above normal
    }


def _sample_nonbreach_features(rng: np.random.Generator) -> dict:
    """Sample physically realistic features for a NON-BREACHED lake.

    Non-breached lakes tend to have:
        - Low expansion rate (stable lakes)
        - High W/H ratio (wide, stable moraine crests)
        - Low ice-core contact (glacier has retreated away)
        - Normal temperature (no anomalous melt)
        - Normal rainfall
    """
    return {
        "lake_expansion_rate": rng.uniform(0.0, 0.08),  # <8% per year
        "dam_width_height_ratio": rng.uniform(10.0, 40.0),  # wide, stable
        "moraine_dam_height_m": rng.uniform(5, 40),  # lower dams
        "rain_anomaly_7d": rng.uniform(-1.0, 1.5),  # normal rainfall
        "mean_upstream_slope_deg": rng.uniform(10, 30),  # moderate slopes
        "lake_area_km2": rng.uniform(0.05, 2.0),  # smaller lakes
        "ice_core_contact_ratio": rng.uniform(0.0, 0.15),  # minimal calving
        "temp_anomaly_0c_isotherm": rng.uniform(-100, 100),  # near normal
    }


def generate_glof_ledger(
    n_lakes: int = DEFAULT_N_LAKES,
    breach_fraction: float = DEFAULT_BREACH_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, list[GLOFLake]]:
    """Generate an expanded GLOF ground-truth ledger.

    Produces ``n_lakes`` synthetic but physically realistic glacial lake
    records with the 9-feature V2 feature set. Breached lakes are sampled
    from breach-prone distributions; non-breached lakes from stable
    distributions. The generator is deterministic for a given seed
    (Hard Rule 6).

    Args:
        n_lakes: total number of lakes (default 280, Veh et al. 2022 scale).
        breach_fraction: fraction of lakes that have historically breached
            (default 0.15, ~15% of HMA glacial lakes).
        seed: random seed for reproducibility.

    Returns:
        Tuple of (X, y, lakes) where:
            X: feature matrix (n_lakes, 9) in FEATURE_NAMES_V2 order.
            y: labels (n_lakes,), 1=breach, 0=no breach.
            lakes: list of GLOFLake records with metadata.
    """
    rng = np.random.default_rng(seed)
    n_breached = int(n_lakes * breach_fraction)
    n_nonbreached = n_lakes - n_breached

    regions = ["nepal", "bhutan", "tibet", "india", "pakistan"]
    region_weights = [0.25, 0.15, 0.35, 0.15, 0.10]

    lakes: list[GLOFLake] = []
    features_list: list[np.ndarray] = []
    labels: list[int] = []

    # Generate breached lakes
    for i in range(n_breached):
        f = _sample_breach_features(rng)
        region = rng.choice(regions, p=region_weights)
        lake = GLOFLake(
            lake_id=f"breach-{i + 1:04d}",
            region=str(region),
            breached=True,
            **f,
        )
        lakes.append(lake)
        features_list.append(lake.to_feature_vector()[0])
        labels.append(1)

    # Generate non-breached lakes
    for i in range(n_nonbreached):
        f = _sample_nonbreach_features(rng)
        region = rng.choice(regions, p=region_weights)
        lake = GLOFLake(
            lake_id=f"stable-{i + 1:04d}",
            region=str(region),
            breached=False,
            **f,
        )
        lakes.append(lake)
        features_list.append(lake.to_feature_vector()[0])
        labels.append(0)

    # Shuffle to mix breached and non-breached
    perm = rng.permutation(n_lakes)
    X = np.array(features_list, dtype=np.float32)[perm]
    y = np.array(labels, dtype=np.float32)[perm]
    lakes = [lakes[i] for i in perm]

    logger.info(
        "Generated GLOF ledger: %d lakes (%d breached, %d stable), %d features",
        n_lakes, n_breached, n_nonbreached, len(FEATURE_NAMES_V2),
    )

    return X, y, lakes


def split_ledger(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.6,
    cal_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the ledger into train / calibration / test sets.

    Stratified split to maintain the breach fraction in each subset.

    Args:
        X: feature matrix (n, 9).
        y: labels (n,).
        train_fraction: fraction for training (default 0.6).
        cal_fraction: fraction for calibration (default 0.2).
        seed: random seed.

    Returns:
        Tuple (X_train, y_train, X_cal, y_cal, X_test, y_test).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    n_breach = int(y.sum())
    n_nonbreach = n - n_breach

    # Stratified indices
    breach_idx = np.where(y == 1)[0]
    nonbreach_idx = np.where(y == 0)[0]
    rng.shuffle(breach_idx)
    rng.shuffle(nonbreach_idx)

    # Split each class
    n_b_train = int(n_breach * train_fraction)
    n_b_cal = int(n_breach * cal_fraction)
    n_nb_train = int(n_nonbreach * train_fraction)
    n_nb_cal = int(n_nonbreach * cal_fraction)

    train_idx = np.concatenate([breach_idx[:n_b_train], nonbreach_idx[:n_nb_train]])
    cal_idx = np.concatenate([breach_idx[n_b_train:n_b_train + n_b_cal],
                              nonbreach_idx[n_nb_train:n_nb_train + n_nb_cal]])
    test_idx = np.concatenate([breach_idx[n_b_train + n_b_cal:],
                               nonbreach_idx[n_nb_train + n_nb_cal:]])

    rng.shuffle(train_idx)
    rng.shuffle(cal_idx)
    rng.shuffle(test_idx)

    return (
        X[train_idx], y[train_idx],
        X[cal_idx], y[cal_idx],
        X[test_idx], y[test_idx],
    )
