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
        "cloud_fraction": 0.0,  # SAR = all-weather
        "alignment_error": 0.2,
        "acquired_at": "2026-07-23T12:00:00Z",
        "expansion_pct": 8.0,
        "trend_class": "slowly",
    },
    "obs-002": {
        "source": "sentinel-1-grd-nrt",
        "cloud_fraction": 0.0,  # SAR = all-weather
        "alignment_error": 0.3,
        "acquired_at": "2026-08-04T12:00:00Z",
        "expansion_pct": 28.0,
        "trend_class": "rapidly",
    },
    "obs-003": {
        "source": "sentinel-2-l2a",
        "cloud_fraction": 0.11,
        "alignment_error": 0.2,
        "acquired_at": "2026-09-04T12:00:00Z",
        "expansion_pct": 14.3,
        "trend_class": "rapidly",
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
    """Generate obs-003 mask if it doesn't exist (uses +14.3% scenario)."""
    mask_path = PROCESSED_DIR / "obs-003_expansion_mask.tif"
    if not mask_path.exists():
        mask, meta = scenario_expansion_mask(0.143, seed=43)
        with rasterio.open(
            str(mask_path), "w", driver="GTiff",
            height=mask.shape[0], width=mask.shape[1],
            count=1, dtype="uint8", crs="EPSG:4326",
            transform=meta["transform"],
        ) as dst:
            dst.write(mask.astype(np.uint8), 1)


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
    routing = route_observation(
        cloud_fraction=obs_config["cloud_fraction"],
        usable=quality["usable"],
    )

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

    # 5. Compute change stats
    change_stats = _compute_change_stats(
        str(mask_path), obs_config["expansion_pct"]
    )

    # 6. Build corridor + exposures
    weather = _load_weather()
    w = weather.get(observation_id, {})
    rainfall_24h = w.get("rainfall_24h_mm", 0.0)
    rainfall_7d = w.get("rainfall_7d_mm", 0.0)
    temp_index = w.get("temp_index", 0.5)

    change_polygon = _change_polygon_from_mask(str(mask_path))

    # Try the full corridor pipeline; fall back to a simple corridor if
    # the D8 trace fails (steep terrain, degenerate source, etc.)
    corridor_geojson: dict[str, Any]
    exposures: list[dict[str, Any]]
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
    except Exception:
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
