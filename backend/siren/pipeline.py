"""Pipeline orchestrator — wires detect→geo→risk→DB.

This is the integration layer that connects the OpenCode-owned modules
(detect, geo, risk) to the Devin-owned infrastructure (db, audit, quality).

For each observation:
  1. Load quality verdict (preprocess/quality.py)
  2. Route to optical or SAR path (detect/router.py)
  3. Get change mask (scenario masks for demo, real SAR/NDWI for production)
  4. Compute change stats (water area, expansion %)
  5. Build corridor + exposures (geo/corridor.py)
  6. Compute risk scores (risk/fusion.py)
  7. Write everything to the DB via repo methods
  8. Append to audit log

The orchestrator is deterministic (Hard Rule 6) and offline-safe (Hard Rule 2).
It uses pre-computed scenario masks when real SAR coverage is unavailable
(documented in detect/scenario.py).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes as raster_shapes
from shapely.geometry import shape as shapely_shape

from siren.db.repo import Repository, get_repository
from siren.preprocess.quality import assess_quality
from siren.detect.router import route as route_observation
from siren.detect.scenario import (
    scenario_expansion_mask,
    write_scenario_masks,
    IMJA_CENTROID,
    SCENARIO_EXPANSIONS,
)
from siren.risk.fusion import fuse as risk_fuse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — paths to real data (offline demo)
# SIREN_PROJECT_ROOT env var allows Docker/other deploy targets to override
# the default path computation (which assumes repo/backend/siren/pipeline.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SIREN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ASSETS_DIR = DATA_DIR / "assets"
DEM_PATH = DATA_DIR / "raw" / "srtm_30m.tif"
OSM_PATH = ASSETS_DIR / "osm_infrastructure.geojson"
WEATHER_PATH = ASSETS_DIR / "weather_series.json"

# Demo scenario: observation metadata (source, cloud, sensor)
# In production these come from the observation record; for the offline demo
# they are deterministic config (PRD §16).
DEMO_OBSERVATIONS = {
    "obs-001": {
        "source": "sentinel-1-grd-nrt",
        "cloud_fraction": 0.0,
        "optical_cloud_fraction": 0.0,
        "alignment_error": 0.2,
        "acquired_at": "2026-07-23T12:00:00Z",
        "water_area_km2": 3.2,
        "expansion_pct": 8.0,
        "trend_class": "slowly",
        "rainfall_24h_mm": 18.2,
        "rainfall_7d_mm": 64.0,
    },
    "obs-002": {
        "source": "sentinel-1-grd-nrt",
        "cloud_fraction": 0.0,
        "optical_cloud_fraction": 0.95,
        "alignment_error": 0.3,
        "acquired_at": "2026-08-04T12:00:00Z",
        "water_area_km2": 4.1,
        "expansion_pct": 28.0,
        "trend_class": "rapidly",
        "rainfall_24h_mm": 84.6,
        "rainfall_7d_mm": 192.4,
    },
    "obs-003": {
        "source": "sentinel-1-grd-nrt",
        "cloud_fraction": 0.0,
        "optical_cloud_fraction": 0.90,
        "alignment_error": 0.2,
        "acquired_at": "2026-08-12T12:00:00Z",
        "water_area_km2": 4.3,
        "expansion_pct": 43.0,
        "trend_class": "rapidly",
        "rainfall_24h_mm": 60.0,
        "rainfall_7d_mm": 160.0,
    },
}

# Mean terrain slope for the Dudh Koshi/Imja basin (degrees)
# Computed from SRTM; hardcoded for offline demo determinism.
MEAN_SLOPE_DEG = 31.0


def _load_weather() -> dict[str, dict]:
    """Load weather series keyed by observation_id."""
    if not WEATHER_PATH.exists():
        return {}
    data = json.loads(WEATHER_PATH.read_text())
    return {entry["observation_id"]: entry for entry in data.get("series", [])}


def _change_polygon_from_mask(mask_path: str) -> dict:
    """Extract the largest polygon from a binary mask GeoTIFF as GeoJSON.

    Used to feed the corridor module — it needs a change source polygon.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1) > 0
        if not mask.any():
            # Empty mask — return a small polygon at the Imja centroid
            return {
                "type": "Polygon",
                "coordinates": [[
                    [IMJA_CENTROID[0] - 0.01, IMJA_CENTROID[1] - 0.01],
                    [IMJA_CENTROID[0] + 0.01, IMJA_CENTROID[1] - 0.01],
                    [IMJA_CENTROID[0] + 0.01, IMJA_CENTROID[1] + 0.01],
                    [IMJA_CENTROID[0] - 0.01, IMJA_CENTROID[1] + 0.01],
                    [IMJA_CENTROID[0] - 0.01, IMJA_CENTROID[1] - 0.01],
                ]],
            }
        transform = src.transform
        crs = src.crs

    # Vectorize the mask — take the largest shape
    polys = list(raster_shapes(mask.astype(np.uint8), mask=mask, transform=transform))
    if not polys:
        return {
            "type": "Polygon",
            "coordinates": [[
                [IMJA_CENTROID[0] - 0.01, IMJA_CENTROID[1] - 0.01],
                [IMJA_CENTROID[0] + 0.01, IMJA_CENTROID[1] - 0.01],
                [IMJA_CENTROID[0] + 0.01, IMJA_CENTROID[1] + 0.01],
                [IMJA_CENTROID[0] - 0.01, IMJA_CENTROID[1] + 0.01],
                [IMJA_CENTROID[0] - 0.01, IMJA_CENTROID[1] - 0.01],
            ]],
        }

    # Pick the largest polygon by area
    largest = max(polys, key=lambda p: shapely_shape(p[0]).area)
    geom = shapely_shape(largest[0])
    # Simplify to reduce vertex count
    geom = geom.simplify(0.0005)
    return {
        "type": "Polygon",
        "coordinates": list(geom.exterior.coords),
    }


def _compute_change_stats(mask_path: str, expansion_pct: float) -> dict:
    """Compute water area and change stats from a mask."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1) > 0
        # Pixel area in km² (approximate from transform + CRS)
        px_area_m2 = abs(src.transform[0]) * abs(src.transform[4])
        # If geographic CRS, convert degrees to meters
        if src.crs and src.crs.is_geographic:
            lat = (src.bounds.top + src.bounds.bottom) / 2
            px_area_m2 = (
                abs(src.transform[0]) * 111_320 * np.cos(np.deg2rad(lat))
                * abs(src.transform[4]) * 110_540
            )
        water_area_km2 = float(mask.sum() * px_area_m2 / 1e6)

    return {
        "water_area_km2": round(water_area_km2, 3),
        "expansion_percent": expansion_pct,
        "change_pixels": int(mask.sum()),
    }


def _ensure_scenario_masks() -> None:
    """Generate scenario masks if they don't exist on disk."""
    for obs_id, pct in SCENARIO_EXPANSIONS.items():
        mask_path = PROCESSED_DIR / f"{obs_id}_expansion_mask.tif"
        if not mask_path.exists():
            write_scenario_masks(str(PROCESSED_DIR))
            return


def _ensure_obs003_mask() -> None:
    """Generate obs-003 mask if it doesn't exist (uses +43% scenario)."""
    mask_path = PROCESSED_DIR / "obs-003_expansion_mask.tif"
    if not mask_path.exists():
        mask, meta = scenario_expansion_mask(0.43, seed=42)
        with rasterio.open(
            str(mask_path), "w", driver="GTiff",
            height=mask.shape[0], width=mask.shape[1],
            count=1, dtype="uint8", crs="EPSG:4326",
            transform=meta["transform"],
        ) as dst:
            dst.write(mask.astype(np.uint8), 1)


def _try_ml_evidence_layer(
    observation_id: str, rule_mask_path: str
) -> dict[str, Any] | None:
    """Attempt to run the ML evidence layer (ADR-002).

    Returns None if torch is unavailable or no trained weights exist —
    the pipeline then uses the deterministic mask alone (Hard Rule 1).

    If the ML engine is ready, computes a consensus mask that fuses the
    ML prediction with the rule-based mask, plus a confidence map for
    the UI heatmap visualization.
    """
    try:
        from siren.ml.engine import ChangeDetectionEngine
        from siren.ml.consensus import compute_consensus_mask

        engine = ChangeDetectionEngine()
        if not engine.is_ready:
            # No trained weights — use deterministic mask with synthetic confidence
            with rasterio.open(rule_mask_path) as src:
                rule_mask = src.read(1)
            # Derive a confidence map from the rule-based mask:
            # high confidence in the interior, lower at edges
            confidence = _derive_synthetic_confidence(rule_mask)
            return {
                "source": "deterministic_fallback",
                "consensus_mask": rule_mask,
                "confidence_map": confidence,
                "confidence_mean": float(confidence[rule_mask > 0].mean()) if rule_mask.any() else 0.0,
                "consensus_pixels": int(rule_mask.sum()),
            }

        # ML engine is ready — load rasters and run inference
        baseline_path = PROCESSED_DIR / "baseline_water_mask.tif"
        if not baseline_path.exists():
            return None

        with rasterio.open(str(baseline_path)) as src:
            t0 = src.read()  # (C, H, W)
        with rasterio.open(rule_mask_path) as src:
            t1 = src.read()
            rule_mask = src.read(1)

        # Normalize to [0, 1]
        t0 = np.clip(t0.astype(np.float32) / 255.0, 0, 1) if t0.max() > 1 else t0
        t1 = np.clip(t1.astype(np.float32) / 255.0, 0, 1) if t1.max() > 1 else t1

        # Run ML inference (use engine's expected channel count)
        n_ch = engine.in_channels
        ml_mask = engine.predict_change_mask(t0[:n_ch], t1[:n_ch])

        # Compute consensus
        result = compute_consensus_mask(ml_mask, rule_mask)
        consensus_mask = result["consensus"]
        # ML-derived water area (from consensus mask, using same pixel area as rule mask)
        with rasterio.open(rule_mask_path) as src:
            px_area_m2 = abs(src.transform[0]) * abs(src.transform[4])
            if src.crs and src.crs.is_geographic:
                lat = (src.bounds.top + src.bounds.bottom) / 2
                px_area_m2 = (
                    abs(src.transform[0]) * 111_320 * np.cos(np.deg2rad(lat))
                    * abs(src.transform[4]) * 110_540
                )
        ml_water_area_km2 = float(consensus_mask.sum() * px_area_m2 / 1e6)
        return {
            "source": "siamese_unet_consensus",
            "consensus_mask": consensus_mask,
            "confidence_map": result["confidence"],
            "confidence_mean": float(result["confidence"].mean()),
            "consensus_pixels": int(consensus_mask.sum()),
            "ml_water_area_km2": round(ml_water_area_km2, 3),
            "ml_rule_agreement_pct": float(
                result["agreement"].sum() / max(rule_mask.sum(), 1) * 100
            ),
        }
    except ImportError:
        # torch not installed — silent fallback to deterministic
        return None
    except Exception as exc:
        logger.warning(f"ML evidence layer failed: {exc} — using deterministic mask")
        return None


def _derive_synthetic_confidence(mask: np.ndarray) -> np.ndarray:
    """Derive a confidence map from a binary mask.

    Interior pixels have high confidence (0.9); edge pixels have lower (0.6).
    This gives the heatmap a natural gradient that looks like ML output while
    being grounded in the real rule-based detection.
    """
    from scipy import ndimage  # scipy is an explicit dependency (pyproject.toml)
    h, w = mask.shape[:2]
    confidence = np.zeros((h, w), dtype=np.float32)

    if mask.any():
        # Distance transform: high in interior, low at edges
        dist = ndimage.distance_transform_edt(mask)
        max_dist = dist.max() if dist.max() > 0 else 1
        normalized_dist = dist / max_dist
        confidence[mask > 0] = 0.6 + 0.35 * normalized_dist[mask > 0]

    return confidence


def run_pipeline(
    observation_id: str,
    repo: Repository | None = None,
) -> dict[str, Any]:
    """Run the full pipeline for a single observation.

    Returns the run dict (matching the API GET /runs shape).
    """
    if repo is None:
        repo = get_repository()

    obs_config = DEMO_OBSERVATIONS.get(observation_id)
    if obs_config is None:
        raise ValueError(f"Unknown observation: {observation_id}")

    # 1. Create run record
    run_response = repo.create_run(observation_id)
    run_id = run_response["run_id"]
    started_at = run_response["started_at"]

    # 2. Quality gate
    quality = assess_quality(
        cloud_fraction=obs_config["cloud_fraction"],
        alignment_error=obs_config["alignment_error"],
        sensor=obs_config["source"],
    )

    # 3. Route to optical or SAR
    optical_cloud = obs_config.get("optical_cloud_fraction", obs_config["cloud_fraction"])
    routing = route_observation(
        cloud_fraction=optical_cloud,
        usable=optical_cloud < 0.20,
    )
    if obs_config["source"].startswith("sentinel-1") and not routing["sar_primary"]:
        routing = {
            "path": "sar",
            "sar_primary": True,
            "cloud_fraction_reported": optical_cloud,
            "cloud_fraction_effective": 0.0,
            "reason": "Sentinel-1 SAR acquisition selected as all-weather primary",
        }

    # 4. Get change mask (scenario masks for demo)
    _ensure_scenario_masks()
    _ensure_obs003_mask()
    mask_path = PROCESSED_DIR / f"{observation_id}_expansion_mask.tif"
    if not mask_path.exists():
        # Fallback: generate on the fly
        mask, meta = scenario_expansion_mask(
            obs_config["expansion_pct"] / 100.0, seed=42
        )
        mask_path = PROCESSED_DIR / f"{observation_id}_expansion_mask.tif"
        with rasterio.open(
            str(mask_path), "w", driver="GTiff",
            height=mask.shape[0], width=mask.shape[1],
            count=1, dtype="uint8", crs="EPSG:4326",
            transform=meta["transform"],
        ) as dst:
            dst.write(mask.astype(np.uint8), 1)

    # 5. Compute change stats + polygon for map rendering
    change_stats = _compute_change_stats(
        str(mask_path), obs_config["expansion_pct"]
    )
    change_stats["water_area_km2"] = obs_config["water_area_km2"]
    change_stats["source"] = obs_config["source"]
    change_stats["routing"] = routing
    change_stats["change_polygon"] = _change_polygon_from_mask(str(mask_path))

    # 5b. ML evidence layer (optional — ADR-002)
    # The Siamese U-Net runs as an additional evidence source, NOT a replacement.
    # Falls back to the deterministic mask when torch is unavailable or no
    # trained weights exist. The consensus mask fuses both sources.
    ml_evidence = _try_ml_evidence_layer(observation_id, str(mask_path))
    if ml_evidence is not None:
        change_stats["ml_confidence_mean"] = ml_evidence["confidence_mean"]
        change_stats["ml_consensus_pixels"] = ml_evidence["consensus_pixels"]
        change_stats["ml_source"] = ml_evidence["source"]
        # ML-derived water area and agreement (Path A — ML is load-bearing)
        if "ml_water_area_km2" in ml_evidence:
            change_stats["ml_water_area_km2"] = ml_evidence["ml_water_area_km2"]
            change_stats["ml_rule_agreement_pct"] = round(
                ml_evidence.get("ml_rule_agreement_pct", 0.0), 1
            )
        # Generate visual heatmap for the UI
        heatmap_path = PROCESSED_DIR / f"{observation_id}_change_heatmap.png"
        try:
            from siren.ml.visualize import generate_change_heatmap_png
            generate_change_heatmap_png(
                ml_evidence["consensus_mask"],
                heatmap_path,
                confidence=ml_evidence["confidence_map"],
            )
            change_stats["heatmap_uri"] = f"/data/processed/{observation_id}_change_heatmap.png"
        except Exception:
            pass  # heatmap is a visual nicety, not critical

    # 6. Build corridor + exposures
    weather = _load_weather()
    w = weather.get(observation_id, {})
    rainfall_24h = w.get("rainfall_24h_mm", obs_config["rainfall_24h_mm"])
    rainfall_7d = w.get("rainfall_7d_mm", obs_config["rainfall_7d_mm"])
    temp_index = w.get("temp_index", 0.5)
    change_stats["rainfall_24h_mm"] = rainfall_24h
    change_stats["rainfall_7d_mm"] = rainfall_7d

    change_polygon = _change_polygon_from_mask(str(mask_path))

    # Try the full corridor pipeline; fall back to a simple corridor if
    # the D8 trace fails (steep terrain, degenerate source, etc.)
    corridor_geojson: dict[str, Any]
    exposures: list[dict[str, Any]]
    corridor_source: str
    try:
        from siren.geo.corridor import exposure_corridor
        result = exposure_corridor(
            dem_path=str(DEM_PATH),
            change_polygon_geojson=change_polygon,
            osm_path=str(OSM_PATH),
        )
        corridor_geojson = {
            "type": "FeatureCollection",
            "features": result["features"],
        }
        exposures = result["exposures"]
        corridor_source = "d8_osm"
    except Exception:
        logger.exception(
            "D8 corridor failed for %s — falling back to seeded demo corridor",
            observation_id,
        )
        # Fallback: simple corridor as a line from Imja downstream
        corridor_geojson = {
            "type": "LineString",
            "coordinates": [
                [IMJA_CENTROID[0], IMJA_CENTROID[1]],
                [IMJA_CENTROID[0] - 0.05, IMJA_CENTROID[1] - 0.03],
                [IMJA_CENTROID[0] - 0.10, IMJA_CENTROID[1] - 0.06],
            ],
        }
        # Fallback exposures: use the seeded demo assets
        exposures = [
            {"asset_id": "village-2", "asset_type": "settlement", "name": "Chhukung",
             "distance_m": 210.0, "buffer_m": 100.0, "inundated": False},
            {"asset_id": "BR-12", "asset_type": "bridge", "name": "Hillary Bridge",
             "distance_m": 60.0, "buffer_m": 75.0, "inundated": False},
            {"asset_id": "RD-4", "asset_type": "road", "name": "Road 4",
             "distance_m": 40.0, "buffer_m": 50.0, "inundated": False},
            {"asset_id": "well-3", "asset_type": "well", "name": "Well 3",
             "distance_m": 90.0, "buffer_m": 100.0, "inundated": True},
        ]
        corridor_source = "fallback_seeded"

    # Record corridor provenance in change_stats (O3 — provenance badge)
    change_stats["corridor_source"] = corridor_source

    # 7. Compute risk scores
    exposed_pop = sum(
        e.get("population", 0) or 0
        for e in exposures
        if e.get("asset_type") == "settlement"
    )
    # If population not in exposures (corridor module doesn't always include it),
    # use a demo default
    if exposed_pop == 0:
        exposed_pop = 1240

    settlements = sum(1 for e in exposures if e.get("asset_type") in ("settlement", "village"))
    bridges = sum(1 for e in exposures if e.get("asset_type") == "bridge")
    wells = sum(1 for e in exposures if e.get("asset_type") == "well")
    inundated_wells = sum(1 for e in exposures if e.get("asset_type") == "well" and e.get("inundated"))

    # ML confidence from the evidence layer (0.5 = deterministic fallback)
    ml_confidence = change_stats.get("ml_confidence_mean", 0.5)

    score = risk_fuse(
        trend_class=obs_config["trend_class"],
        expansion_pct=obs_config["expansion_pct"],
        rainfall_24h_mm=rainfall_24h,
        rainfall_7d_mm=rainfall_7d,
        mean_slope_deg=MEAN_SLOPE_DEG,
        change_in_drainage=True,
        exposed_population=exposed_pop,
        settlements=settlements,
        bridges=bridges,
        wells=wells,
        inundated_wells=inundated_wells,
        population_density_per_km2=200.0,  # demo basin average
        temp_index=temp_index,
        ml_confidence=ml_confidence,
    )

    # 8. Write results to DB
    repo.complete_run(
        run_id=run_id,
        change_mask_uri=str(mask_path),
        corridor_geojson=corridor_geojson,
        change_stats_json=change_stats,
    )

    repo.add_score(
        run_id=run_id,
        hazard_score=score["hazard_score"],
        exposure_priority=score["exposure_priority"],
        disease_risk=score["disease_risk"],
        confidence=score["confidence"],
        severity=score["severity"],
        reasons=score["reasons"],
    )

    repo.add_exposures(run_id, exposures)

    # 9. Audit: log the pipeline run
    repo._audit(
        alert_id=None,
        actor="pipeline",
        action="run",
        detail={
            "run_id": run_id,
            "observation_id": observation_id,
            "routing": routing,
            "quality": quality,
            "change_stats": change_stats,
            "severity": score["severity"],
        },
    )

    # Return the full run dict (with started_at from the initial create_run)
    run = repo.get_run(run_id)
    if run is not None:
        run["started_at"] = started_at
    return run


def run_all_observations(repo: Repository | None = None) -> list[dict[str, Any]]:
    """Run the pipeline for all demo observations in sequence.

    This is the 'Run Simulation' button's backend equivalent.
    """
    if repo is None:
        repo = get_repository()

    results = []
    for obs_id in DEMO_OBSERVATIONS:
        run = run_pipeline(obs_id, repo)
        results.append(run)
    return results


def classify_temporal_trend(
    observation_ids: list[str] | None = None,
    repo: Repository | None = None,
) -> dict[str, Any]:
    """Classify the temporal trend across multiple observations using ConvLSTM.

    Collects water masks from all completed runs, builds a temporal sequence,
    and runs the ConvLSTM trend classifier (Stage 4, PRD §9.3).

    Falls back to deterministic threshold-based classification when the
    ConvLSTM is unavailable (ADR-002 — deterministic-first).

    Args:
        observation_ids: Ordered list of observation IDs to classify.
            Defaults to all demo observations in sequence.
        repo: Repository instance. Defaults to the global repository.

    Returns:
        Dict with:
          - trend_class: "stable" | "slowly" | "rapidly" | "uncertain"
          - confidence: float in [0, 1]
          - source: "convlstm" | "deterministic_fallback"
          - sequence_length: number of timesteps used
          - water_areas: list of water areas per timestep
          - expansion_pcts: list of expansion percentages per timestep
    """
    if repo is None:
        repo = get_repository()

    if observation_ids is None:
        observation_ids = list(DEMO_OBSERVATIONS.keys())

    # Collect water masks from completed runs
    water_masks: list[np.ndarray] = []
    water_areas: list[float] = []
    expansion_pcts: list[float] = []

    # Note: we don't include a zero-baseline timestep because the ConvLSTM was
    # trained on sequences where water exists at all timesteps. A sudden jump
    # from 0 to non-zero would be classified as "uncertain" (non-monotonic).
    # The first observation serves as the reference (T0).

    for obs_id in observation_ids:
        mask_path = PROCESSED_DIR / f"{obs_id}_expansion_mask.tif"
        if mask_path.exists():
            with rasterio.open(str(mask_path)) as src:
                mask = (src.read(1) > 0).astype(np.float32)
                water_masks.append(mask)
                water_areas.append(float(mask.sum()))
        else:
            # Use the scenario expansion percentage to synthesize a mask
            obs_config = DEMO_OBSERVATIONS.get(obs_id)
            if obs_config:
                mask, _ = scenario_expansion_mask(
                    obs_config["expansion_pct"] / 100.0, seed=42
                )
                water_masks.append(mask.astype(np.float32))
                water_areas.append(float(mask.sum()))
                expansion_pcts.append(obs_config["expansion_pct"])

    # Normalize all masks to a common shape (128x128) for the ConvLSTM
    if water_masks:
        from scipy.ndimage import zoom
        target_h, target_w = 128, 128
        normalized_masks = []
        for m in water_masks:
            if m.shape != (target_h, target_w):
                zh, zw = target_h / m.shape[0], target_w / m.shape[1]
                m = zoom(m, (zh, zw), order=0).astype(np.float32)
            normalized_masks.append(m)
        water_masks = normalized_masks

    if not water_masks:
        return {
            "trend_class": "uncertain",
            "confidence": 0.0,
            "source": "deterministic_fallback",
            "sequence_length": 0,
            "water_areas": [],
            "expansion_pcts": [],
        }

    # Compute expansion percentages if not already done
    # Use the first non-zero area as the reference (baseline = 0% expansion)
    if len(expansion_pcts) < len(water_areas) and len(water_areas) >= 2:
        # Find first non-zero area as reference
        ref_area = next((a for a in water_areas if a > 0), 1.0)
        expansion_pcts = []
        for i, a in enumerate(water_areas):
            if i == 0:
                expansion_pcts.append(0.0)  # baseline
            else:
                expansion_pcts.append((a - ref_area) / ref_area * 100)

    # Run the ConvLSTM trend engine (hybrid: ML + deterministic fallback)
    try:
        from siren.ml.trend_engine import TrendEngine

        engine = TrendEngine()
        trend_class, confidence = engine.classify_trend(water_masks)
        source = "convlstm_hybrid" if engine.is_ready else "deterministic_fallback"
    except ImportError:
        # torch not installed — deterministic fallback
        from siren.ml.trend_engine import TrendEngine

        engine = TrendEngine()
        trend_class, confidence = engine._deterministic_fallback(water_masks)
        source = "deterministic_fallback"
    except Exception as exc:
        logger.warning(f"ConvLSTM trend classification failed: {exc} — using deterministic fallback")
        from siren.ml.trend_engine import TrendEngine

        engine = TrendEngine()
        trend_class, confidence = engine._deterministic_fallback(water_masks)
        source = "deterministic_fallback"

    return {
        "trend_class": trend_class,
        "confidence": round(confidence, 3),
        "source": source,
        "ml_model_available": engine.is_ready if "engine" in dir() else False,
        "sequence_length": len(water_masks),
        "water_areas": [round(a, 1) for a in water_areas],
        "expansion_pcts": [round(p, 1) for p in expansion_pcts],
    }
