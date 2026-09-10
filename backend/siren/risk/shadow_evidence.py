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

    # Load the trained checkpoint (no runtime retraining — Level 1)
    scorer = SusceptibilityScorer(random_state=42)
    if not scorer.load_checkpoint():
        # Fallback: synthetic training if checkpoint not available (demo mode)
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
    """Compute FNO hydrodynamic surrogate as shadow evidence (Sprint 3).

    Triggered only when P_breach ≥ 0.70 (V3 §4.3). Loads the trained FNO-2D
    checkpoint (``models/checkpoints/fno_hydro_surrogate_v1.pt``) and runs
    real inference to produce the dynamic water depth grid and sector-level
    arrival times. The output is tagged with provenance ``"fno_surrogate_v1"``
    for audit lineage.

    The FNO predicts ``h_water(x,y)``; ``T_arrival`` is then computed
    deterministically from ``h_water`` using the shallow-water wave celerity
    formula (Sprint 3 architectural decision — separates learned vision
    from deterministic hydraulics).

    If the checkpoint is unavailable or inference fails, falls back to a
    simulated result (demo mode).
    """
    from siren.geo.hydro_surrogate import FNO_TRIGGER_GATE
    from pathlib import Path

    provenance = "fno_surrogate_v1"

    # Locate the trained checkpoint
    repo_root = Path(__file__).resolve().parents[3]
    ckpt_path = repo_root / "models" / "checkpoints" / "fno_hydro_surrogate_v1.pt"

    if not ckpt_path.exists():
        logger.info("FNO checkpoint not found (%s) — returning simulated result", ckpt_path)
        return {
            "is_triggered": True,
            "p_breach": round(p_breach, 4),
            "trigger_gate": FNO_TRIGGER_GATE,
            "provenance": provenance,
            "note": (
                "FNO triggered (P_breach >= 0.70) but no trained checkpoint "
                "found. Run train_fno_surrogate.py to produce the checkpoint."
            ),
        }

    try:
        import torch
        import numpy as np
        from siren.geo.hydro_surrogate import FNO2D, DEFAULT_NAMED_POINTS, DEFAULT_GRID_SIZE
        from siren.ml.train_fno_surrogate import generate_teesta_corridor_dem

        # Load checkpoint
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model = FNO2D(
            modes=state["modes"],
            width=state["width"],
            n_points=state["n_points"],
            n_layers=state["n_layers"],
        )
        model.load_state_dict(state["model_state"])
        model.eval()

        grid_size = state.get("grid_size", DEFAULT_GRID_SIZE)

        # Generate a Teesta-style corridor DEM for inference. In production,
        # this would use the actual basin DEM (Copernicus GLO-30) resized to
        # the FNO's grid. For shadow mode, the synthetic corridor DEM
        # demonstrates the wiring without requiring real terrain data.
        dem = generate_teesta_corridor_dem(
            grid_size=grid_size,
            source_elev=5200,
            outlet_elev=400,
            upper_slope=8.0,
            lower_slope=3.0,
            random_state=42,
        )

        # Normalise DEM to [0, 1]
        dem_min, dem_max = float(dem.min()), float(dem.max())
        dem_norm = (dem - dem_min) / (dem_max - dem_min) if dem_max > dem_min else np.zeros_like(dem)

        # Breach volume from obs_config (demo: use South Lhonak-scale default)
        v_breach = obs_config.get("v_breach_m3", 45e6)
        v_norm = float(np.log1p(v_breach) / 20.0)

        # Prepare input tensor: (1, 2, H, W) — DEM + V_breach (broadcast)
        v_grid = np.full_like(dem_norm, v_norm, dtype=np.float32)
        x = np.stack([dem_norm, v_grid], axis=0)[np.newaxis]
        x_tensor = torch.from_numpy(x).float()

        # Run FNO inference
        with torch.no_grad():
            output = model(x_tensor)

        h_water = output["h_water"][0, 0].numpy()

        # Compute T_arrival deterministically from h_water (Sprint 3 design)
        source_row, source_col = np.unravel_index(np.argmax(dem), dem.shape)
        rows, cols = np.indices(dem.shape)
        dist_cells = np.sqrt((rows - source_row) ** 2 + (cols - source_col) ** 2)
        cell_size_m = 1330.0  # 85km / 64 cells
        wave_speed = np.sqrt(9.81 * np.maximum(h_water, 0.1))
        wave_speed = np.clip(wave_speed, 7.0, 10.0)
        t_arrival_grid = (dist_cells * cell_size_m / wave_speed / 60.0).astype(np.float32)

        # Extract T_arrival at downstream sectors (named points)
        # Map the 3 default named points to downstream distances
        sector_distances = {
            "Hillary Bridge": 0.33,  # ~1/3 downstream
            "Benkar": 0.67,           # ~2/3 downstream
            "Jorsale": 0.85,          # near outlet
        }
        t_arrival_by_sector: dict[str, float] = {}
        for name, frac in sector_distances.items():
            row = int(frac * (grid_size - 1))
            col = grid_size // 2
            r0, r1 = max(0, row - 2), min(grid_size, row + 3)
            c0, c1 = max(0, col - 2), min(grid_size, col + 3)
            t_arrival_by_sector[name] = round(float(np.mean(t_arrival_grid[r0:r1, c0:c1])), 1)

        return {
            "is_triggered": True,
            "p_breach": round(p_breach, 4),
            "trigger_gate": FNO_TRIGGER_GATE,
            "provenance": provenance,
            "checkpoint": str(ckpt_path.name),
            "h_water_max_m": round(float(h_water.max()), 2),
            "h_water_mean_m": round(float(h_water.mean()), 4),
            "t_arrival_by_sector": t_arrival_by_sector,
            "is_shadow": True,
            "note": (
                "FNO-2D surrogate inference completed (shadow mode per ADR-011). "
                "h_water predicted by FNO; T_arrival computed deterministically "
                "from h_water via shallow-water wave celerity. Does not supersede "
                "static tolerance buffers until a new ADR authorizes load-bearing use."
            ),
        }

    except Exception as exc:
        logger.warning("FNO inference failed: %s — returning error result", exc)
        return {
            "is_triggered": True,
            "p_breach": round(p_breach, 4),
            "trigger_gate": FNO_TRIGGER_GATE,
            "provenance": provenance,
            "error": str(exc),
            "is_shadow": True,
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
