"""Shadow evidence integration for the production pipeline (V3 §3.6, §4.3, §6).

Wires the Sprint 2/3 ML modules into the pipeline as **shadow evidence** —
the deterministic 5-factor hazard score remains authoritative (ADR-010 §3).
The shadow evidence is attached to change_stats for the review card UI and
for shadow-mode evaluation against the rules-only baseline.

Shadow components:
    1. XGBoost breach susceptibility (V3 §3.2) — P_breach with conformal intervals
    2. HAND exposure (V3 §3.2) — vertical clearance vs planar buffer
    3. FNO hydrodynamic surrogate (V3 §4) — h_water + T_arrival (triggered at P_breach ≥ 0.70)
    4. Dual-path dispatch (V3 §4.4) — SMS + satellite (simulated in demo)
    5. RFC 3161 timestamp anchoring (V3 §4.5) — audit chain root anchoring

All shadow evidence is:
    - Attached to change_stats as "shadow_evidence" metadata
    - Never injected into the canonical hazard score (ADR-010 §3)
    - Gated on the ML evaluation gate (IoU > 0.65 / Brier < 0.15)
    - Offline-safe (simulated when hardware/network unavailable)

The pipeline calls `attach_shadow_evidence()` after step 7 (risk scoring).
The dispatch and timestamp anchoring are called from the API layer when a
human confirms a dispatch.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default moraine dam geometry for the demo basin (Imja Tsho)
# These are approximate values from ICIMOD literature — replaced by
# inventory data in production.
DEFAULT_MORAINE_DAM_WIDTH_M: float = 350.0
DEFAULT_MORAINE_DAM_HEIGHT_M: float = 12.0
DEFAULT_LAKE_AREA_KM2: float = 1.28  # Imja Tsho approximate area


def attach_shadow_evidence(
    change_stats: dict[str, Any],
    obs_config: dict[str, Any],
    rainfall_24h: float,
    rainfall_7d: float,
    dem_path: str | None = None,
) -> dict[str, Any]:
    """Attach shadow evidence to change_stats (V3 §3.6, §6).

    This is the main integration point. Called by the pipeline after step 7
    (risk scoring). The shadow evidence is attached as
    ``change_stats["shadow_evidence"]`` and never modifies the canonical
    hazard score.

    Args:
        change_stats: the change statistics dict (mutated in-place).
        obs_config: the observation configuration (expansion_pct, mean_slope, etc.).
        rainfall_24h: 24-hour rainfall in mm.
        rainfall_7d: 7-day cumulative rainfall in mm.
        dem_path: optional path to the DEM GeoTIFF for HAND computation.

    Returns:
        The shadow evidence dict (also attached to change_stats).
    """
    shadow: dict[str, Any] = {}

    # 1. XGBoost breach susceptibility (V3 §3.2)
    try:
        shadow["susceptibility"] = _compute_shadow_susceptibility(
            change_stats, obs_config, rainfall_7d
        )
    except Exception as exc:
        logger.warning("Shadow susceptibility failed: %s", exc)
        shadow["susceptibility"] = {"error": str(exc)}

    # 2. HAND exposure (V3 §3.2) — only if DEM available
    if dem_path:
        try:
            shadow["hand"] = _compute_shadow_hand(
                dem_path, obs_config, change_stats
            )
        except Exception as exc:
            logger.warning("Shadow HAND failed: %s", exc)
            shadow["hand"] = {"error": str(exc)}

    # 3. FNO hydrodynamic surrogate (V3 §4) — triggered at P_breach ≥ 0.70
    sus = shadow.get("susceptibility", {})
    p_breach = sus.get("p_breach", 0.0) if isinstance(sus, dict) else 0.0
    if isinstance(p_breach, (int, float)) and p_breach >= 0.70:
        try:
            shadow["hydro_surrogate"] = _compute_shadow_hydro(
                obs_config, p_breach, dem_path
            )
        except Exception as exc:
            logger.warning("Shadow FNO failed: %s", exc)
            shadow["hydro_surrogate"] = {"error": str(exc)}
    else:
        shadow["hydro_surrogate"] = {
            "is_triggered": False,
            "reason": f"P_breach={p_breach:.3f} < 0.70 gate",
        }

    # Mark as shadow evidence (ADR-010 §3: not load-bearing)
    shadow["is_shadow"] = True
    shadow["gate_status"] = "shadow_only"
    shadow["note"] = (
        "ML evidence is shadow-only (ADR-010 §3). The deterministic 5-factor "
        "hazard score remains authoritative until the ML evaluation gate "
        "(IoU > 0.65 / Brier < 0.15) is passed."
    )

    change_stats["shadow_evidence"] = shadow
    return shadow


def _compute_shadow_susceptibility(
    change_stats: dict[str, Any],
    obs_config: dict[str, Any],
    rainfall_7d: float,
) -> dict[str, Any]:
    """Compute XGBoost breach susceptibility as shadow evidence.

    Constructs the 6-feature vector from pipeline data and runs the
    susceptibility scorer. The scorer is trained on synthetic data if no
    trained model is available (demo mode).
    """
    from siren.risk.susceptibility import SusceptibilityScorer, FEATURE_NAMES

    # Construct feature vector from pipeline data
    expansion_pct = obs_config.get("expansion_pct", 0.0)
    # Lake expansion rate: convert % to fraction per year (demo assumption:
    # observations are ~12 days apart → annualize)
    lake_expansion_rate = expansion_pct / 100.0 * 30.0  # rough annualization

    mean_slope = obs_config.get("mean_slope_degrees", 20.0)
    lake_area = change_stats.get("water_area_km2", DEFAULT_LAKE_AREA_KM2)

    # Rain anomaly: 7d rainfall vs climatology (demo: use 7d as z-score proxy)
    # In production, this would compare against ERA5 climatology
    rain_anomaly = max(0.0, (rainfall_7d - 20.0) / 10.0)  # rough z-score

    features = np.array([[
        lake_expansion_rate,
        DEFAULT_MORAINE_DAM_WIDTH_M,
        DEFAULT_MORAINE_DAM_HEIGHT_M,
        rain_anomaly,
        mean_slope,
        lake_area,
    ]], dtype=np.float32)

    # Train a scorer on synthetic data (demo mode — production uses a
    # pre-trained model loaded from disk)
    scorer = SusceptibilityScorer(random_state=42)
    _train_synthetic_scorer(scorer)

    result = scorer.predict(features)
    return result.to_dict()


def _train_synthetic_scorer(scorer) -> None:
    """Train a susceptibility scorer on synthetic data (demo fallback).

    In production, a pre-trained model would be loaded from disk. This
    synthetic training produces a functional scorer for shadow-mode
    evaluation.
    """
    from siren.risk.susceptibility import SusceptibilityScorer
    rng = np.random.RandomState(42)

    # Generate synthetic training data: 200 samples with 6 features
    n = 200
    X = rng.uniform(
        low=[0.0, 50.0, 2.0, -2.0, 5.0, 0.1],
        high=[2.0, 500.0, 30.0, 5.0, 45.0, 5.0],
        size=(n, 6),
    ).astype(np.float32)

    # Synthetic labels: breach if expansion rate + rain anomaly are high
    y = (
        X[:, 0] * 0.3 + X[:, 3] * 0.2 + X[:, 4] * 0.01 > 0.5
    ).astype(int)

    # Split into train + calibration
    idx = rng.permutation(n)
    n_cal = n // 3
    scorer.train(
        X[idx[n_cal:]], y[idx[n_cal:]],
        X[idx[:n_cal]], y[idx[:n_cal]],
    )


def _compute_shadow_hand(
    dem_path: str,
    obs_config: dict[str, Any],
    change_stats: dict[str, Any],
) -> dict[str, Any]:
    """Compute HAND-based exposure as shadow evidence.

    In demo mode, this returns a summary of what HAND would produce.
    Full HAND computation requires the pysheds pipeline (may not be
    available on all Python versions).
    """
    from siren.geo.hand import water_stage_for_severity, DEFAULT_WATER_STAGE_M

    # Determine severity from change_stats (the deterministic pipeline's output)
    severity = change_stats.get("severity", "watch")

    try:
        h_water_stage = water_stage_for_severity(severity)
    except ValueError:
        h_water_stage = DEFAULT_WATER_STAGE_M["watch"]

    return {
        "is_available": False,  # pysheds may not be available
        "h_water_stage_m": h_water_stage,
        "severity": severity,
        "note": (
            "HAND computation requires the pysheds pipeline. In demo mode, "
            "the policy default water stage height is used as a placeholder. "
            "Production would compute the full HAND raster and intersect "
            "exposures against HAND ≤ h_water_stage."
        ),
    }


def _compute_shadow_hydro(
    obs_config: dict[str, Any],
    p_breach: float,
    dem_path: str | None,
) -> dict[str, Any]:
    """Compute FNO hydrodynamic surrogate as shadow evidence.

    Triggered only when P_breach ≥ 0.70 (V3 §4.3). In demo mode, returns
    a simulated result.
    """
    from siren.geo.hydro_surrogate import HydroSurrogate, FNO_TRIGGER_GATE

    # In demo mode, we don't have a trained FNO model — return the trigger
    # status and what would be computed
    return {
        "is_triggered": True,
        "p_breach": round(p_breach, 4),
        "trigger_gate": FNO_TRIGGER_GATE,
        "note": (
            "FNO hydrodynamic surrogate triggered (P_breach >= 0.70). "
            "In demo mode, h_water and T_arrival are not computed — "
            "production would run the trained FNO model to produce the "
            "dynamic water depth grid and arrival times at named points."
        ),
    }


def shadow_dispatch(
    dispatch_id: str,
    payload: str,
    recipient_group: str = "default",
) -> dict[str, Any]:
    """Dispatch an alert via the dual-path hardware engine (shadow mode).

    Called by the API layer when a human confirms a dispatch. In demo mode,
    both paths return SIMULATED receipts.

    Args:
        dispatch_id: unique dispatch identifier.
        payload: the ≤250-byte compressed alert payload.
        recipient_group: recipient group name.

    Returns:
        DispatchResult as a dict.
    """
    from siren.alerting.dispatch import DispatchEngine

    engine = DispatchEngine(simulate=True)
    result = engine.dispatch(dispatch_id, payload, recipient_group)
    return result.to_dict()


def shadow_anchor_audit_chain(chain_root: str) -> dict[str, Any]:
    """Anchor the audit chain root to a timestamp authority (shadow mode).

    Called by the API/audit layer after appending a new audit entry. In demo
    mode, produces a simulated timestamp anchor.

    Args:
        chain_root: the SHA-256 hash of the latest audit chain entry.

    Returns:
        TimestampAnchor as a dict.
    """
    from siren.audit.timestamp import anchor_chain_root

    anchor = anchor_chain_root(chain_root, simulate=True)
    return anchor.to_dict()
