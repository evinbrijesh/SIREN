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
        shadow["susceptibility"] = {"is_available": False, "error": str(exc)}

    # 2. HAND exposure (V3 §3.2) — only if DEM available
    if dem_path:
        try:
            shadow["hand"] = _compute_shadow_hand(
                dem_path, obs_config, change_stats
            )
        except Exception as exc:
            logger.warning("Shadow HAND failed: %s", exc)
            shadow["hand"] = {"error": str(exc)}

    # 3. FNO hydrodynamic surrogate (V3 §4) — triggered at P_breach ≥ 0.70.
    # When susceptibility is unavailable (no valid checkpoint), the FNO must
    # not be triggered and must not invent a P_breach value (PRD v4.7 §17.3).
    sus = shadow.get("susceptibility", {})
    p_breach = sus.get("p_breach") if isinstance(sus, dict) else None
    sus_available = isinstance(sus, dict) and sus.get("is_available", False) is True

    if not sus_available or p_breach is None:
        shadow["hydro_surrogate"] = {
            "is_triggered": False,
            "is_available": False,
            "reason": "susceptibility unavailable — FNO trigger gate requires a valid P_breach",
        }
    elif isinstance(p_breach, (int, float)) and p_breach >= 0.70:
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
    susceptibility scorer. The scorer must load a pre-trained, valid
    checkpoint — runtime training on synthetic or generated features is
    prohibited (PRD v4.7 §17.3). When no valid checkpoint is available,
    returns an explicit ``is_available=False`` result without a ``p_breach``.
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

    # Load the trained checkpoint. Runtime training is prohibited — if the
    # checkpoint is missing or disqualified, return an explicit unavailable
    # result rather than inventing a probability (PRD v4.7 §17.3).
    scorer = SusceptibilityScorer(random_state=42)
    if not scorer.load_checkpoint():
        return {
            "is_available": False,
            "reason": (
                "No valid susceptibility checkpoint loaded. The XGBoost "
                "breach susceptibility model is disqualified pending a "
                "real-data evaluation (PRD v4.7 §17.3). Runtime training "
                "on synthetic or generated features is prohibited."
            ),
        }

    result = scorer.predict(features)
    out = result.to_dict()
    out["is_available"] = True
    return out


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

    Triggered only when P_breach ≥ 0.70 (V3 §4.3). Per PRD v4.7 §17.3, this
    function fails closed:

      - If no real basin DEM is supplied (``dem_path`` is None), it returns
        an explicit ``is_available=False`` / ``status="disqualified"`` result
        rather than fabricating a Teesta-shaped synthetic corridor DEM.
      - If the FNO checkpoint metadata sidecar is marked disqualified, it
        returns the same unavailable result rather than running inference
        on an unvalidated surrogate.

    The deterministic corridor and tolerance buffers remain authoritative.
    """
    from siren.geo.hydro_surrogate import FNO_TRIGGER_GATE
    from pathlib import Path

    provenance = "fno_surrogate_v1"

    # Fail closed: no real terrain → no fabricated hydrodynamics.
    if dem_path is None:
        return {
            "is_triggered": True,
            "is_available": False,
            "status": "disqualified",
            "p_breach": round(p_breach, 4),
            "trigger_gate": FNO_TRIGGER_GATE,
            "provenance": provenance,
            "reason": (
                "FNO triggered (P_breach >= 0.70) but no real basin DEM was "
                "supplied. Fabricating a synthetic Teesta-shaped corridor DEM "
                "is prohibited (PRD v4.7 §17.3). Supply a real basin DEM to "
                "enable FNO inference."
            ),
        }

    # Locate the trained checkpoint and its metadata sidecar.
    repo_root = Path(__file__).resolve().parents[3]
    ckpt_path = repo_root / "models" / "checkpoints" / "fno_hydro_surrogate_v1.pt"
    meta_path = repo_root / "models" / "checkpoints" / "fno_hydro_surrogate_v1.meta.json"

    if not ckpt_path.exists():
        return {
            "is_triggered": True,
            "is_available": False,
            "status": "unavailable",
            "p_breach": round(p_breach, 4),
            "trigger_gate": FNO_TRIGGER_GATE,
            "provenance": provenance,
            "reason": (
                "FNO triggered (P_breach >= 0.70) but no trained checkpoint "
                "found. Run train_fno_surrogate.py to produce the checkpoint."
            ),
        }

    # Reject disqualified checkpoints via the metadata sidecar.
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("evaluation_valid") is False
            or meta.get("status") == "disqualified"
            or meta.get("inference_allowed") is False
        ):
            return {
                "is_triggered": True,
                "is_available": False,
                "status": "disqualified",
                "p_breach": round(p_breach, 4),
                "trigger_gate": FNO_TRIGGER_GATE,
                "provenance": provenance,
                "reason": (
                    "FNO checkpoint is disqualified (synthetic training data, "
                    "no real multi-basin hydraulic validation). Inference is "
                    "blocked until a valid checkpoint is produced (PRD v4.7 §17.3)."
                ),
            }

    try:
        import torch
        import numpy as np
        from siren.geo.hydro_surrogate import FNO2D, DEFAULT_NAMED_POINTS, DEFAULT_GRID_SIZE

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

        # Read the supplied real basin DEM and resize it to the FNO grid.
        # Fabricating a synthetic corridor DEM is prohibited (PRD v4.7 §17.3).
        import rasterio
        from skimage.transform import resize

        with rasterio.open(str(dem_path)) as src:
            dem = src.read(1).astype(np.float32)

        dem = resize(
            dem, (grid_size, grid_size),
            order=1, mode="edge", anti_aliasing=True, preserve_range=True,
        ).astype(np.float32)

        # Normalise DEM to [0, 1]
        dem_min, dem_max = float(dem.min()), float(dem.max())
        dem_norm = (dem - dem_min) / (dem_max - dem_min) if dem_max > dem_min else np.zeros_like(dem)

        # Breach volume resolution (ADR-012 frozen 2-channel contract).
        # Priority: explicit override → hypsometric estimation → raise.
        # No silent fallback to a hardcoded default (CLAUDE.md).
        v_breach_override = obs_config.get("v_breach_m3")
        breach_volume_info: dict[str, Any] | None = None

        if v_breach_override is not None:
            v_breach = float(v_breach_override)
        else:
            # Estimate from segmentation delta + DEM hypsometry
            from siren.risk.breach_volume import estimate_breach_volume

            pre_mask = obs_config.get("pre_water_mask")
            post_mask = obs_config.get("post_water_mask")
            pixel_area = obs_config.get("pixel_area_m2")

            if pre_mask is None or post_mask is None or pixel_area is None:
                raise ValueError(
                    "v_breach_m3 not provided in obs_config and cannot be "
                    "estimated: pre_water_mask, post_water_mask, and "
                    "pixel_area_m2 are required for hypsometric estimation. "
                    "Either supply v_breach_m3 explicitly or pass the "
                    "segmentation masks + pixel area."
                )

            bv_result = estimate_breach_volume(
                pre_water_mask=np.asarray(pre_mask),
                post_water_mask=np.asarray(post_mask),
                dem=dem.astype(np.float64),
                pixel_area_m2=float(pixel_area),
            )
            v_breach = bv_result.v_breach_m3
            breach_volume_info = bv_result.to_dict()

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
        sector_distances = {
            "Hillary Bridge": 0.33,
            "Benkar": 0.67,
            "Jorsale": 0.85,
        }
        t_arrival_by_sector: dict[str, float] = {}
        for name, frac in sector_distances.items():
            row = int(frac * (grid_size - 1))
            col = grid_size // 2
            r0, r1 = max(0, row - 2), min(grid_size, row + 3)
            c0, c1 = max(0, col - 2), min(grid_size, col + 3)
            t_arrival_by_sector[name] = round(float(np.mean(t_arrival_grid[r0:r1, c0:c1])), 1)

        result = {
            "is_triggered": True,
            "is_available": True,
            "p_breach": round(p_breach, 4),
            "trigger_gate": FNO_TRIGGER_GATE,
            "provenance": provenance,
            "checkpoint": str(ckpt_path.name),
            "v_breach_m3": round(v_breach, 1),
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
        if breach_volume_info is not None:
            result["breach_volume"] = breach_volume_info
        return result

    except Exception as exc:
        logger.warning("FNO inference failed: %s — returning error result", exc)
        return {
            "is_triggered": True,
            "is_available": False,
            "status": "error",
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
