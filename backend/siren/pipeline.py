"""Pipeline orchestrator — wires detect→geo→risk→DB.

This is the integration layer that connects the analysis modules
(detect, geo, risk) to the persistence and audit infrastructure
(db, audit, quality).

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
from siren.ml.promotion import DEMOTE_ENV_VAR, is_demoted, is_promoted, promotion_record

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
AOI_GEOJSON_PATH = ASSETS_DIR / "dudh_koshi_aoi.geojson"
RGI_GLACIER_SHP_PATH = (
    "/vsizip/"
    + str(
        DATA_DIR
        / "datasets"
        / "RGI2000-v7.0-G-15_south_asia_east.zip"
        / "RGI2000-v7.0-G-15_south_asia_east.shp"
    )
)
# NOTE: baseline_water_mask.tif was regenerated 2026-09-17 with a
# corrected window transform; it now covers the AOI (was previously
# stamped ~65 km west in the Rolwaling valley by a full-tile profile bug).
# Terrain gate for the ML shadow mask (ADR-010 display hygiene — shapes
# what the reviewer sees, never the load-bearing rule mask or the score).
# Water surfaces are flat: >15° at ~90 m pitch cannot be lake. RGI glacier
# outlines include terminus lakes (Imja sits on the glacier tongue), so
# glacier gating is disabled in the vicinity of mapped baseline water.
ML_SLOPE_GATE_DEG = 15.0
ML_LAKE_VICINITY_DILATION_PX = 3  # ~270 m at ~90 m SAR pixel pitch

# Demo scenario: observation IDs + scenario-only overrides.
#
# Production Roadmap §1.2 / Sprint 1 Step 7: observation metadata (source,
# cloud, acquired_at, water area, rainfall, raster_uri, ...) is no longer
# hardcoded here — it lives in the observations table, seeded by
# repo._seed() for the demo and registered via repo.register_observation()
# (or the STAC daemon) for live acquisitions. The pipeline reads every
# observation from the DB via repo.get_observation().
#
# What remains here is the small set of *scenario-modeling* parameters
# that are not raw observation metadata and have no DB column: the
# deterministic trend-class fallback and the mask-provenance tag. These
# drive the frozen demo scenario masks (PRD §9.2/§16) and are applied
# only for the three demo observation IDs.
DEMO_OBS_IDS: tuple[str, ...] = ("obs-001", "obs-002", "obs-003")

DEMO_SCENARIO_OVERRIDES: dict[str, dict[str, Any]] = {
    "obs-001": {"trend_class": "slowly"},
    "obs-002": {"trend_class": "rapidly"},
    "obs-003": {
        "trend_class": "rapidly",
        # All 3 demo observations have real Sentinel-1 SAFE archives.
        # The rule-based mask is a deterministic scenario per PRD §9.2/§16;
        # the ML shadow layer uses real calibrated VV/VH sigma0 dB from the
        # SAFE archive (downloaded 2026-09-08 from CDSE).
        "mask_provenance": "sentinel-1-grd",
    },
}

# Mean terrain slope for the Dudh Koshi/Imja basin (degrees)
# Computed from SRTM; hardcoded for offline demo determinism.
MEAN_SLOPE_DEG = 31.0


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
        "coordinates": [list(geom.exterior.coords)],
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


# Module-level cache for ML engines (avoids reloading weights per observation)
_ml_engine_cache: dict[str, Any] = {}

# MC Dropout rate for the runtime engine (E1, ADR-013 §9.7.4). Dropout2d is
# parameter-free, so this is checkpoint-compatible; in eval mode dropout is
# identity, so the deterministic path is unchanged.
ML_MC_DROPOUT_RATE = 0.1


def _get_ml_engine() -> Any:
    """Get or create a cached ChangeDetectionEngine.

    Constructed with dropout>0 so the MC Dropout uncertainty path (E1) is
    available, and prefers CUDA when present — MC sampling over a full
    swath is ~30× faster on GPU. The deterministic path is unchanged
    (dropout is inert in eval mode).
    """
    if "engine" not in _ml_engine_cache:
        from siren.ml.engine import ChangeDetectionEngine
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass
        _ml_engine_cache["engine"] = ChangeDetectionEngine(
            device=device, dropout=ML_MC_DROPOUT_RATE
        )
    return _ml_engine_cache["engine"]


def _try_ml_evidence_layer(
    observation_id: str, rule_mask_path: str
) -> dict[str, Any] | None:
    """Attempt to run the ML evidence layer in SHADOW MODE (ADR-010).

    Returns None if torch is unavailable or no trained weights exist —
    the pipeline then uses the deterministic mask alone (Hard Rule 1).

    ADR-010 safety boundaries enforced here:
      - The ML output is recorded as SEPARATE EVIDENCE only. It does NOT
        replace the rule-based change mask, does NOT filter rule-detected
        pixels, and does NOT enter the hazard score (see risk/fusion.py —
        the 5-factor formula has no ML term).
      - The SegFormer classifier (archived/disqualified) previously
        REPLACED the consensus mask via seg_result["filtered_mask"]. That
        was a load-bearing ML violation ("do not let a classifier remove
        rule-detected pixels"). Its output is now retained strictly as
        metadata under "classification_breakdown" and never modifies the
        mask used downstream.
      - Source labels: "ml-shadow" when WaterUNet weights are loaded and
        producing supplementary evidence; "deterministic-fallback" when
        the ML engine is not ready (no torch or no weights).

    The "consensus_mask" returned here is ALWAYS the rule-based mask.
    The ML prediction is stored separately under "ml_shadow_mask" for
    the UI to display as supplementary evidence, never as the
    load-bearing mask for corridor/exposure/scoring.
    """
    try:
        from siren.ml.consensus import compute_consensus_mask

        engine = _get_ml_engine()

        # Always load the rule-based mask — it is the load-bearing mask
        # regardless of whether ML is available.
        with rasterio.open(rule_mask_path) as src:
            rule_mask = src.read(1)

        if not engine.is_ready:
            # No trained weights — deterministic fallback with synthetic confidence
            confidence = _derive_synthetic_confidence(rule_mask)
            return {
                "source": "deterministic-fallback",
                "consensus_mask": rule_mask,
                "confidence_map": confidence,
                "confidence_mean": float(confidence[rule_mask > 0].mean()) if rule_mask.any() else 0.0,
                "consensus_pixels": int(rule_mask.sum()),
            }

        # ML engine is ready — run inference in SHADOW MODE.
        # The ML mask is supplementary evidence, NOT the load-bearing mask.
        #
        # TENSOR CONTRACT (2026-09-14): ChangeDetectionEngine auto-detects the
        # checkpoint architecture and input contract from the state_dict.
        # The preferred checkpoint is the gate-passed 6-channel Kuro Siwo
        # WaterResUNet (ADR-011.1), whose contract is
        # (VV_post, VH_post, VV_pre, VH_pre, dVV, dVH) — see
        # ml/contract.py::build_kuro_siwo_tensor. The engine builds that
        # multi-temporal tensor internally from the pre/post calibrated
        # VV/VH dB pair. The legacy 2-channel path is retained as a fallback.
        # Safety is preserved because the model is shadow-only and cannot
        # affect hazard scoring, corridor routing, exposure, or dispatch.
        from siren.preprocess.sar_calibrate import (
            extract_and_cache_vv_vh_db,
            find_imja_descending_pair,
            find_safe_for_observation,
        )

        raw_dir = DATA_DIR / "raw"
        sar_pair_provenance: dict[str, Any] | None = None

        # Multi-temporal contract (6-ch Kuro Siwo): prefer the verified
        # descending-orbit pair that covers Imja Tsho (86.925°E). The demo
        # observations' own ascending scenes (relative orbit 85) only cover
        # the western AOI and miss the lake, so shadow evidence computed
        # from them cannot see the feature of interest. The Δσ⁰ channels
        # also require both dates on the same orbit — the descending pair
        # (2026-07-02 / 2026-07-14) is the only same-orbit pre/post pair on
        # disk that covers the lake. Same pair for every observation: it is
        # the only real lake-covering evidence available, and its scene
        # dates are recorded in provenance.
        imja_pair = (
            find_imja_descending_pair(raw_dir) if engine.is_multitemporal else None
        )
        if imja_pair is not None:
            t0_safe, t1_safe = imja_pair
            t1_cache = PROCESSED_DIR / "imja_desc_20260714_sar_vv_vh_db.tif"
            t0_db = extract_and_cache_vv_vh_db(
                t0_safe, PROCESSED_DIR / "imja_desc_20260702_sar_vv_vh_db.tif"
            )
            t1_db = extract_and_cache_vv_vh_db(t1_safe, t1_cache)
            sar_pair_provenance = {
                "t0_scene": Path(t0_safe).name,
                "t1_scene": Path(t1_safe).name,
                "orbit": "descending",
                "covers_imja": True,
            }
        else:
            # Per-observation ascending scenes (fallback when the descending
            # pair is not on disk — e.g. fresh offline checkouts where
            # data/raw only carries the demo archives).
            safe_path = find_safe_for_observation(observation_id, raw_dir)
            if safe_path is None:
                logger.info(
                    f"No SAFE archive for {observation_id} — "
                    "ML shadow mask skipped (no calibrated SAR input)"
                )
                confidence = _derive_synthetic_confidence(rule_mask)
                return {
                    "source": "deterministic-fallback",
                    "consensus_mask": rule_mask,
                    "confidence_map": confidence,
                    "confidence_mean": float(confidence[rule_mask > 0].mean()) if rule_mask.any() else 0.0,
                    "consensus_pixels": int(rule_mask.sum()),
                }

            # Extract and cache calibrated VV/VH dB for the current observation
            t1_cache = PROCESSED_DIR / f"{observation_id}_sar_vv_vh_db.tif"
            t1_db = extract_and_cache_vv_vh_db(safe_path, t1_cache)

            # For the baseline (t0), use obs-001's calibrated SAR if available,
            # otherwise fall back to the current scene (self-comparison → no change).
            baseline_safe = find_safe_for_observation("obs-001", raw_dir)
            if baseline_safe is not None and observation_id != "obs-001":
                t0_cache = PROCESSED_DIR / "obs-001_sar_vv_vh_db.tif"
                t0_db = extract_and_cache_vv_vh_db(baseline_safe, t0_cache)
            else:
                # obs-001 is the baseline — compare against itself (no change expected)
                t0_db = t1_db.copy()

            sar_pair_provenance = {
                "t0_scene": Path(baseline_safe).name if baseline_safe and observation_id != "obs-001" else Path(safe_path).name,
                "t1_scene": Path(safe_path).name,
                "orbit": "ascending",
                "covers_imja": False,
            }

        # Resize t0 to match t1's spatial dimensions if they differ
        if t0_db.shape[1:] != t1_db.shape[1:]:
            from scipy.ndimage import zoom
            target_h, target_w = t1_db.shape[1], t1_db.shape[2]
            zh, zw = target_h / t0_db.shape[1], target_w / t0_db.shape[2]
            t0_resized = np.zeros((t0_db.shape[0], target_h, target_w), dtype=np.float32)
            for c_idx in range(t0_db.shape[0]):
                zoomed = zoom(t0_db[c_idx], (zh, zw), order=1)
                zh_act, zw_act = zoomed.shape
                if zh_act > target_h:
                    zoomed = zoomed[:target_h]
                elif zh_act < target_h:
                    zoomed = np.pad(zoomed, ((0, target_h - zh_act), (0, 0)), mode="edge")
                if zw_act > target_w:
                    zoomed = zoomed[:, :target_w]
                elif zw_act < target_w:
                    zoomed = np.pad(zoomed, ((0, 0), (0, target_w - zw_act)), mode="edge")
                t0_resized[c_idx] = zoomed
            t0_db = t0_resized

        # Run ML inference with calibrated VV/VH dB input.
        # engine.predict_state_and_change() calls normalize_sar() internally
        # to clamp [-30, 0] dB → [0, 1] per the frozen contract. The
        # state/change split matters for persistent lakes: a correct
        # segmenter sees Imja at BOTH dates, so expansion alone is ~empty
        # — the review card needs water_t1 (extent) alongside the delta.
        ml_state = engine.predict_state_and_change(t0_db, t1_db)
        # ADR-014-am1 promotion: when the expansion component is
        # promoted, the primary evidence mask is the Δp expansion —
        # the gate-evaluated contract. The binary extent-difference
        # mask stays recorded alongside for the audit trail.
        _exp_promoted = is_promoted("sar_segmentation_expansion")
        ml_mask = (
            ml_state["expansion_dp"] if _exp_promoted
            else ml_state["expansion"]
        )
        ml_water_t1 = ml_state["water_t1"]
        ml_drainage = ml_state["drainage"]

        # Terrain / evidence-quality gating of the shadow mask (ADR-010):
        # the Kuro Siwo model is badly out-of-distribution on high-Himalaya
        # terrain — it fires on glacier surfaces and across the full
        # Sentinel-1 swath far outside the modelled basin. The displayed
        # evidence is gated by (1) the AOI polygon — evidence outside the
        # basin cannot be evaluated, (2) DEM slope > 15° within DEM
        # coverage — water surfaces are flat, and (3) RGI glacier outlines
        # EXCEPT in the vicinity of mapped baseline water — RGI outlines
        # include terminus lakes like Imja, so a hard glacier gate would
        # erase the very feature being monitored. The rule-based mask and
        # the hazard score are never touched (Hard Rule 1).
        gate_stats: dict[str, Any] = {}
        # Hoisted for the MC-Dropout conformal block below — both are
        # populated inside the gating try and stay None when the SAR grid
        # has no geolocation.
        lake_vic: np.ndarray | None = None
        px_area_m2: float | None = None
        try:
            from siren.detect.sar import (
                sar_grid_dem_slope,
                sar_grid_lonlat,
                sar_grid_polygon_mask,
                sar_grid_sample,
            )

            ll = sar_grid_lonlat(str(t1_cache))
            if ll is not None:
                lon_g, lat_g = ll
                exclusion = np.zeros(ml_mask.shape, dtype=bool)

                if AOI_GEOJSON_PATH.exists():
                    aoi_g = sar_grid_polygon_mask(
                        str(AOI_GEOJSON_PATH), lon_g, lat_g
                    )
                    gate_stats["ml_shadow_px_outside_aoi"] = int(
                        (ml_mask & ~aoi_g).sum()
                    )
                    exclusion |= ~aoi_g

                if DEM_PATH.exists():
                    slope_g = sar_grid_dem_slope(str(t1_cache), str(DEM_PATH))
                    steep_g = (~np.isnan(slope_g)) & (
                        slope_g > ML_SLOPE_GATE_DEG
                    )
                    gate_stats["ml_shadow_px_steep"] = int(
                        (ml_mask & steep_g).sum()
                    )
                    exclusion |= steep_g

                if Path(
                    DATA_DIR / "datasets" / "RGI2000-v7.0-G-15_south_asia_east.zip"
                ).exists():
                    glac_g = sar_grid_polygon_mask(
                        RGI_GLACIER_SHP_PATH, lon_g, lat_g
                    )
                    # Known-water vicinity: union of the per-observation
                    # scenario masks (georeferenced to the Imja lake area).
                    lake_union = np.zeros(ml_mask.shape, dtype=bool)
                    for obs_mask_path in sorted(
                        PROCESSED_DIR.glob("obs-*_expansion_mask.tif")
                    ):
                        lake_union |= (
                            sar_grid_sample(str(obs_mask_path), lon_g, lat_g)
                            > 0
                        )
                    if lake_union.any():
                        from scipy.ndimage import binary_dilation

                        lake_vic = binary_dilation(
                            lake_union, iterations=ML_LAKE_VICINITY_DILATION_PX
                        )
                    else:
                        lake_vic = np.zeros(ml_mask.shape, dtype=bool)
                    glac_gate = glac_g & ~lake_vic
                    gate_stats["ml_shadow_px_glacier"] = int(
                        (ml_mask & glac_gate).sum()
                    )
                    exclusion |= glac_gate

                gate_stats["ml_shadow_px_raw"] = int(ml_mask.sum())
                ml_mask = np.where(exclusion, 0, ml_mask).astype(np.uint8)
                gate_stats["ml_shadow_px_gated"] = int(ml_mask.sum())

                # The same display-hygiene gate applies to the state layers
                # — glacier/slope OOD noise pollutes the extent and drainage
                # maps identically.
                ml_water_t1 = np.where(exclusion, 0, ml_water_t1).astype(np.uint8)
                ml_drainage = np.where(exclusion, 0, ml_drainage).astype(np.uint8)

                # Per-pixel area and map bounds from the GCP lon/lat grid
                # (the SAR cache has no affine — GCPs are the geolocation).
                dlat_m = float(np.abs(np.diff(lat_g, axis=0)).mean()) * 110_540
                dlon_m = (
                    float(np.abs(np.diff(lon_g, axis=1)).mean())
                    * 111_320
                    * np.cos(np.deg2rad(float(np.nanmean(lat_g))))
                )
                px_area_m2 = dlat_m * dlon_m
                gate_stats["ml_water_extent_px"] = int(ml_water_t1.sum())
                gate_stats["ml_water_extent_km2"] = round(
                    ml_water_t1.sum() * px_area_m2 / 1e6, 3
                )
                gate_stats["ml_expansion_km2"] = round(
                    ml_mask.sum() * px_area_m2 / 1e6, 3
                )
                # Promotion-scoped expansion: the ADR-014-am1 contract is
                # Δp expansion inside monitorable-lake vicinity, so the
                # load-bearing measurement restricts the gated mask to
                # lake_vic — expansion at other lakes must not inflate
                # the monitored lake's water_area_change_percent.
                if lake_vic is not None and lake_vic.any():
                    vic_exp_px = int((ml_mask & lake_vic).sum())
                    gate_stats["ml_expansion_lake_vicinity_px"] = vic_exp_px
                    gate_stats["ml_expansion_lake_vicinity_km2"] = round(
                        vic_exp_px * px_area_m2 / 1e6, 3
                    )
                gate_stats["ml_drainage_px"] = int(ml_drainage.sum())
                gate_stats["ml_drainage_km2"] = round(
                    ml_drainage.sum() * px_area_m2 / 1e6, 3
                )
                gate_stats["ml_sar_grid_bounds"] = [
                    [float(lon_g[0, 0]), float(lat_g[0, 0])],
                    [float(lon_g[0, -1]), float(lat_g[0, -1])],
                    [float(lon_g[-1, -1]), float(lat_g[-1, -1])],
                    [float(lon_g[-1, 0]), float(lat_g[-1, 0])],
                ]
                gate_stats["ml_terrain_gate"] = {
                    "slope_deg": ML_SLOPE_GATE_DEG,
                    "glacier_exempt_lake_vicinity": True,
                    "aoi_restricted": AOI_GEOJSON_PATH.exists(),
                }

                # Geographic agreement (replaces the index-resize overlap,
                # which assumes identical extents and is meaningless when
                # the full-scene SAR mask meets the small AOI rule mask):
                # the rule mask is sampled onto the SAR grid through GCP
                # geolocation.
                rule_on_sar = (
                    sar_grid_sample(rule_mask_path, lon_g, lat_g) > 0
                )
                rule_px = int(rule_on_sar.sum())
                overlap_px = int((ml_mask & rule_on_sar).sum())
                gate_stats["ml_rule_overlap_px"] = overlap_px
                gate_stats["ml_rule_overlap_pct"] = (
                    round(overlap_px / rule_px * 100, 1) if rule_px else 0.0
                )

                # §9.8.3 cross-check: neural-vs-deterministic agreement
                # verdict — runs every run in shadow mode too so
                # disagreement evidence accumulates before promotion.
                from siren.ml.cross_check import evaluate_overlap

                gate_stats["cross_check"] = evaluate_overlap(
                    ml_mask, rule_on_sar
                )

                # Union-policy expansion evidence (ADR-014-am1): the
                # promoted neural Δp mask and the deterministic mask are
                # complementary detectors — the union records the combined
                # evidence and per-source contributions; disagreement in
                # either direction stays visible via cross_check above.
                if _exp_promoted:
                    union = ml_mask | rule_on_sar
                    gate_stats["expansion_union_px"] = int(union.sum())
                    gate_stats["expansion_union_km2"] = round(
                        union.sum() * px_area_m2 / 1e6, 3
                    )
                    gate_stats["expansion_neural_only_px"] = int(
                        (ml_mask & ~rule_on_sar).sum()
                    )
                    gate_stats["expansion_rule_only_px"] = int(
                        (rule_on_sar & ~ml_mask).sum()
                    )
                    gate_stats["ml_expansion_binary_px"] = int(
                        ml_state["expansion"].sum()
                    )
                    gate_stats["promotion"] = {
                        "component": "sar_segmentation_expansion",
                        "method": "neural_primary_union_deterministic",
                        "evidence": promotion_record(
                            "sar_segmentation_expansion"
                        ),
                    }
        except Exception as exc:
            logger.warning(f"Shadow-mask terrain gating failed: {exc}")

        # E1: MC Dropout uncertainty (ADR-013 §9.7.4) — a per-pixel variance
        # map σ²(x, y) alongside the shadow mask. Informational only until
        # the conformal gate is evaluated on the held-out calibration split.
        # Only runs when the engine was constructed with dropout > 0 —
        # without dropout layers every MC pass is identical and the map is
        # degenerate zeros.
        uncertainty_map: np.ndarray | None = None
        uncertainty_stats: dict[str, Any] = {}
        if engine.has_dropout_layers():
            try:
                unc = engine.predict_change_uncertainty(t0_db, t1_db, n_samples=20)
                uncertainty_map = np.sqrt(unc["change_variance"]).astype(np.float32)
                uncertainty_stats = {
                    "uncertainty_method": unc["method"],
                    "uncertainty_mean_variance": round(
                        float(unc["change_variance"].mean()), 6
                    ),
                    "uncertainty_max_std": round(
                        float(np.sqrt(unc["change_variance"]).max()), 4
                    ),
                    "uncertainty_conformal_quantile": unc.get("conformal_quantile"),
                    "uncertainty_conformal_gate_passed": unc.get(
                        "conformal_gate_passed", False
                    ),
                }
                # Level 2.4 — conformal interval on the neural expansion
                # measurement. With a calibrated q* the per-pixel water
                # interval is [p̂ − q*, p̂ + q*]; the lower-bound expansion
                # counts pixels that are water at t1 even at the lower
                # bound and non-water at t0 even at the upper bound, and
                # the upper bound counts pixels that could be expansion.
                # Restricted to the monitorable-lake vicinity like the
                # load-bearing measurement it bounds.
                q_star = unc.get("conformal_quantile")
                if (
                    q_star is not None
                    and lake_vic is not None and lake_vic.any()
                    and px_area_m2 is not None
                ):
                    tau_u = engine.default_threshold
                    p1, p0 = unc["water_prob_t1"], unc["water_prob_t0"]
                    lo_px = int(
                        (
                            ((p1 - q_star) >= tau_u)
                            & ((p0 + q_star) < tau_u)
                            & lake_vic
                        ).sum()
                    )
                    hi_px = int(
                        (
                            ((p1 + q_star) >= tau_u)
                            & ((p0 - q_star) < tau_u)
                            & lake_vic
                        ).sum()
                    )
                    uncertainty_stats["ml_expansion_vicinity_ci90_px"] = [
                        lo_px, hi_px,
                    ]
                    uncertainty_stats["ml_expansion_vicinity_lo_km2"] = round(
                        lo_px * px_area_m2 / 1e6, 3
                    )
                    uncertainty_stats["ml_expansion_vicinity_hi_km2"] = round(
                        hi_px * px_area_m2 / 1e6, 3
                    )
            except Exception as exc:
                logger.warning(f"MC Dropout uncertainty failed: {exc}")

        # Compute DEM slope for physical consensus gating (metadata only)
        dem_slope_arr = None
        if DEM_PATH.exists():
            try:
                from scipy.ndimage import zoom
                from siren.detect.sar import dem_slope as compute_dem_slope
                slope_full, ds = compute_dem_slope(str(DEM_PATH))
                ds.close()
                h, w = rule_mask.shape
                if slope_full.shape != (h, w):
                    zh, zw = h / slope_full.shape[0], w / slope_full.shape[1]
                    dem_slope_arr = zoom(slope_full, (zh, zw), order=0).astype(np.float32)
                else:
                    dem_slope_arr = slope_full.astype(np.float32)
            except Exception as exc:
                logger.warning(f"DEM slope computation failed: {exc} — consensus without slope gating")

        # Compute consensus for DISPLAY/CONFIDENCE only.
        # The rule_mask remains the load-bearing mask returned as "consensus_mask".
        result = compute_consensus_mask(ml_mask, rule_mask, dem_slope=dem_slope_arr)

        # SegFormer classification — ARCHIVED / DISQUALIFIED (ADR-010).
        # The SegFormer classifier was disqualified in the 2026-09-07 DL audit
        # (it was not the SegFormer architecture and could remove real flood
        # pixels from the mask). It is no longer instantiated or run. The
        # classification_breakdown is retained as an empty stub so downstream
        # consumers do not break, but no disqualified model is loaded.
        classification_breakdown: dict[str, Any] = {
            "source": "archived_disqualified",
            "class_distribution": {},
            "false_alarm_count": 0,
            "classifications": [],
            "note": "SegFormer archived 2026-09-07 — not instantiated per ADR-010",
        }

        # Water area is always computed from the RULE-BASED mask, not from
        # any ML-filtered mask.
        with rasterio.open(rule_mask_path) as src:
            px_area_m2 = abs(src.transform[0]) * abs(src.transform[4])
            if src.crs and src.crs.is_geographic:
                lat = (src.bounds.top + src.bounds.bottom) / 2
                px_area_m2 = (
                    abs(src.transform[0]) * 111_320 * np.cos(np.deg2rad(lat))
                    * abs(src.transform[4]) * 110_540
                )
        ml_water_area_km2 = float(rule_mask.sum() * px_area_m2 / 1e6)
        return {
            "source": "ml-shadow",
            "consensus_mask": rule_mask,  # ALWAYS the rule-based mask
            "ml_shadow_mask": ml_mask,     # supplementary evidence, NOT load-bearing
            "ml_water_extent_mask": ml_water_t1,  # persistent extent at t1 (shadow)
            "ml_drainage_mask": ml_drainage,      # receded water (shadow)
            "confidence_map": result["confidence"],
            "confidence_mean": float(result["confidence"].mean()),
            "consensus_pixels": int(rule_mask.sum()),
            "ml_water_area_km2": round(ml_water_area_km2, 3),
            "ml_rule_agreement_pct": float(
                result["agreement"].sum() / max(rule_mask.sum(), 1) * 100
            ),
            "classification_breakdown": classification_breakdown,
            # Model provenance (ADR-013: every neural component records its
            # method and checkpoint so the audit trail shows what produced
            # this evidence).
            "model_architecture": engine.architecture,
            "model_checkpoint": (
                engine.weights_path.name if engine.weights_path else None
            ),
            "model_in_channels": engine.in_channels,
            "model_contract": (
                "kuro_siwo_6ch" if engine.is_multitemporal else "single_date"
            ),
            # SAR pair provenance — which real scenes fed the shadow layer
            # and whether they cover the lake (descending pair does;
            # per-observation ascending scenes do not).
            "sar_pair": sar_pair_provenance,
            # E1 uncertainty (ADR-013 §9.7.4): scalar stats are JSON-safe and
            # flow into change_stats; the map array is consumed below for
            # the PNG layer and must NOT be stored in the JSON column.
            "uncertainty_map": uncertainty_map,
            **uncertainty_stats,
            **gate_stats,
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


def _load_observation_config(
    observation_id: str, repo: Repository
) -> dict[str, Any] | None:
    """Load observation metadata from the database (Sprint 1 Step 7).

    Every observation — demo (seeded by repo._seed) or live (registered via
    repo.register_observation / the STAC daemon) — is read from the
    observations table. This is the dynamic DB-backed registry: the
    pipeline no longer carries a parallel hardcoded metadata dict.

    For the three demo observation IDs, scenario-modeling overrides
    (trend-class fallback, mask-provenance tag) are layered on top from
    ``DEMO_SCENARIO_OVERRIDES`` — these are scenario parameters, not
    observation metadata, and have no DB column.

    Returns None if the observation is not registered in the DB.
    """
    obs = repo.get_observation(observation_id)
    if obs is None:
        return None

    is_demo = observation_id in DEMO_SCENARIO_OVERRIDES
    # Build a config dict in the shape the pipeline expects.
    config: dict[str, Any] = {
        "source": obs["source"],
        "cloud_fraction": obs["cloud_fraction"] or 0.0,
        "optical_cloud_fraction": obs.get("optical_cloud_fraction")
            or obs["cloud_fraction"] or 0.0,
        "alignment_error": 0.2 if obs.get("alignment_ok", True) else 1.0,
        "acquired_at": obs["acquired_at"],
        "water_area_km2": obs.get("water_area_km2") or 0.0,
        "expansion_pct": obs.get("water_area_change_percent") or 0.0,
        "rainfall_24h_mm": obs.get("rainfall_24h_mm") or 0.0,
        "rainfall_7d_mm": obs.get("rainfall_7d_mm") or 0.0,
        "temp_index": obs.get("temp_index") or 0.5,  # disease driver; 0.5 neutral fallback
        "_is_live": not is_demo,
        "raster_uri": obs.get("raster_uri"),
        "basin_id": obs.get("basin_id"),
        "mean_slope_degrees": obs.get("mean_slope_degrees"),
    }
    if is_demo:
        # trend_class fallback + mask_provenance (scenario params, not DB cols)
        config["trend_class"] = DEMO_SCENARIO_OVERRIDES[observation_id].get(
            "trend_class", "uncertain"
        )
        if "mask_provenance" in DEMO_SCENARIO_OVERRIDES[observation_id]:
            config["mask_provenance"] = DEMO_SCENARIO_OVERRIDES[observation_id][
                "mask_provenance"
            ]
    else:
        config["trend_class"] = "uncertain"  # live obs have no preset trend
    return config


def run_pipeline(
    observation_id: str,
    repo: Repository | None = None,
) -> dict[str, Any]:
    """Run the full pipeline for a single observation.

    Accepts both demo observations (obs-001/002/003, using hardcoded config and
    scenario masks) and live observations registered in the database via
    repo.register_observation(). This unblocks the Live Phase 4 observation-
    acceptance blocker while keeping the frozen deterministic pipeline unchanged.

    Returns the run dict (matching the API GET /runs shape).
    """
    if repo is None:
        repo = get_repository()

    obs_config = _load_observation_config(observation_id, repo)
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

    # 4. Get change mask
    #    Demo observations: use scenario masks (frozen behavior)
    #    Live observations: use a pre-computed mask at raster_uri or a
    #      processed mask at data/processed/{obs_id}_expansion_mask.tif
    is_live = obs_config.get("_is_live", False)
    if not is_live:
        _ensure_scenario_masks()
        _ensure_obs003_mask()
    mask_path = PROCESSED_DIR / f"{observation_id}_expansion_mask.tif"
    if not mask_path.exists():
        if is_live:
            # For live observations, check if a mask was provided at raster_uri
            raster_uri = obs_config.get("raster_uri")
            if raster_uri:
                provided = PROJECT_ROOT / raster_uri
                if provided.exists():
                    mask_path = provided
                else:
                    raise ValueError(
                        f"Live observation {observation_id}: mask not found at "
                        f"{provided}. A change mask must be pre-computed and "
                        f"registered via raster_uri before running the pipeline."
                    )
            else:
                raise ValueError(
                    f"Live observation {observation_id}: no raster_uri registered. "
                    f"Register the observation with a mask path via "
                    f"repo.register_observation() first."
                )
        else:
            # Demo fallback: generate scenario mask on the fly
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

    # 5b. ML evidence layer (ADR-010 shadow + ADR-014-am1 promotion).
    # WaterUNet evidence does NOT replace the rule-based mask, does NOT
    # filter rule-detected pixels, and does NOT feed the corridor or
    # exposure layers. Exception: the promoted sar_segmentation_expansion
    # component supplies the load-bearing water_area_change_percent for
    # the hazard score within its certified scope (see 7a). Falls back
    # to deterministic when torch is unavailable or no weights exist.
    ml_evidence = _try_ml_evidence_layer(observation_id, str(mask_path))
    if ml_evidence is not None:
        change_stats["ml_confidence_mean"] = ml_evidence["confidence_mean"]
        change_stats["ml_consensus_pixels"] = ml_evidence["consensus_pixels"]
        change_stats["ml_source"] = ml_evidence["source"]
        # Water area is always from the rule-based mask (not ML-filtered)
        if "ml_water_area_km2" in ml_evidence:
            change_stats["ml_water_area_km2"] = ml_evidence["ml_water_area_km2"]
            change_stats["ml_rule_agreement_pct"] = round(
                ml_evidence.get("ml_rule_agreement_pct", 0.0), 1
            )
        # Model + input provenance (ADR-013: the audit trail must show which
        # checkpoint and which real scenes produced the shadow evidence).
        for key in (
            "model_architecture", "model_checkpoint", "model_in_channels",
            "model_contract", "sar_pair",
            "uncertainty_method", "uncertainty_mean_variance",
            "uncertainty_max_std", "uncertainty_map_uri",
            "uncertainty_conformal_quantile",
            "uncertainty_conformal_gate_passed",
            "ml_shadow_px_raw", "ml_shadow_px_gated",
            "ml_shadow_px_outside_aoi", "ml_shadow_px_steep",
            "ml_shadow_px_glacier", "ml_terrain_gate",
            "ml_rule_overlap_px", "ml_rule_overlap_pct", "cross_check",
            "ml_water_extent_px", "ml_water_extent_km2",
            "ml_expansion_km2", "ml_drainage_px", "ml_drainage_km2",
            "ml_expansion_lake_vicinity_px", "ml_expansion_lake_vicinity_km2",
            "ml_expansion_vicinity_ci90_px",
            "ml_expansion_vicinity_lo_km2", "ml_expansion_vicinity_hi_km2",
            "ml_sar_grid_bounds",
            "expansion_union_px", "expansion_union_km2",
            "expansion_neural_only_px", "expansion_rule_only_px",
            "ml_expansion_binary_px", "promotion",
        ):
            if key in ml_evidence and ml_evidence[key] is not None:
                change_stats[key] = ml_evidence[key]
        # Generate visual heatmap for the UI (from the rule-based mask)
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

        # Save the WaterUNet shadow mask as a separate PNG for the UI toggle.
        # This is supplementary evidence only — never used for scoring or dispatch.
        if "ml_shadow_mask" in ml_evidence:
            shadow_mask_path = PROCESSED_DIR / f"{observation_id}_ml_shadow_mask.png"
            try:
                from siren.ml.visualize import generate_change_heatmap_png
                generate_change_heatmap_png(
                    ml_evidence["ml_shadow_mask"],
                    shadow_mask_path,
                )
                change_stats["ml_shadow_mask_uri"] = f"/data/processed/{observation_id}_ml_shadow_mask.png"
            except Exception:
                pass  # shadow mask visualization is optional

        # State/change layers (state-vs-change split): the persistent water
        # extent at t1 keeps a stable lake visible even when expansion is
        # ~zero; drainage shows receded water. Both are shadow evidence —
        # display only, never scoring.
        for key, suffix, rgb in (
            ("ml_water_extent_mask", "ml_water_extent", (30, 136, 229)),   # blue
            ("ml_drainage_mask", "ml_drainage", (251, 140, 0)),            # amber
        ):
            layer = ml_evidence.get(key)
            if layer is not None:
                try:
                    from siren.ml.visualize import generate_binary_mask_png
                    generate_binary_mask_png(
                        layer,
                        PROCESSED_DIR / f"{observation_id}_{suffix}.png",
                        rgb=rgb,
                    )
                    change_stats[f"{suffix}_uri"] = (
                        f"/data/processed/{observation_id}_{suffix}.png"
                    )
                except Exception:
                    pass  # state layer visualization is optional

        # Save the MC Dropout uncertainty (std) map as a PNG layer (E1).
        # Informational only — never enters scoring (ADR-010 / ADR-013).
        unc_map = ml_evidence.get("uncertainty_map")
        if unc_map is not None:
            unc_png_path = PROCESSED_DIR / f"{observation_id}_ml_uncertainty.png"
            try:
                from siren.ml.visualize import generate_confidence_heatmap_png
                vmax = float(unc_map.max())
                normalized = unc_map / vmax if vmax > 0 else unc_map
                generate_confidence_heatmap_png(normalized, unc_png_path)
                change_stats["uncertainty_map_uri"] = (
                    f"/data/processed/{observation_id}_ml_uncertainty.png"
                )
            except Exception:
                pass  # uncertainty visualization is optional

    # 5c. SegFormer classification breakdown — METADATA ONLY (ADR-010).
    # Previously this replaced the consensus mask (load-bearing ML violation).
    # Now it is reported as auxiliary metadata for the UI and never modifies
    # the mask used downstream.
    if ml_evidence is not None:
        breakdown = ml_evidence.get("classification_breakdown", {})
        change_stats["segformer_source"] = breakdown.get("source", "unavailable")
        change_stats["segformer_class_distribution"] = breakdown.get("class_distribution", {})
        change_stats["segformer_false_alarms"] = breakdown.get("false_alarm_count", 0)
        change_stats["segformer_classifications"] = breakdown.get("classifications", [])

    # 6. Build corridor + exposures
    # Weather (rainfall, temp_index) now comes from the DB observation record
    # (Sprint 1 Step 8 — weather_series.json retired). The STAC daemon /
    # repo.register_observation() populates these at ingest time; the
    # offline demo seeds them via repo._seed().
    rainfall_24h = obs_config["rainfall_24h_mm"]
    rainfall_7d = obs_config["rainfall_7d_mm"]
    temp_index = obs_config["temp_index"]
    change_stats["rainfall_24h_mm"] = rainfall_24h
    change_stats["rainfall_7d_mm"] = rainfall_7d

    # 6a. Lake thermal state (deterministic seasonal gate). A frozen lake
    # surface is not liquid water at C-band — SAR change stats over a
    # frozen lake measure ice, so the state annotates interpretation and
    # suppresses the shadow hydro/volume trigger (attach_shadow_evidence
    # reads lake_thermal_state from change_stats).
    try:
        from siren.detect.thermal_state import estimate_thermal_state
        thermal = estimate_thermal_state(obs_config["acquired_at"][:10])
    except Exception as exc:  # noqa: BLE001 — gate must never break the run
        logger.warning(f"Thermal-state estimation failed: {exc}")
        thermal = {"state": "unknown", "lake_temp_c": None}
    change_stats["lake_thermal_state"] = thermal["state"]
    change_stats["lake_temp_c_7d_mean"] = thermal.get("lake_temp_c")
    change_stats["thermal_method"] = thermal.get("method")
    frozen = thermal["state"] == "frozen_surface"
    if frozen:
        change_stats["sar_change_reliable"] = False
        change_stats["ml_drainage_not_hydrological"] = True
        change_stats["frozen_state_note"] = (
            "Lake surface frozen (7-day lake temp "
            f"{thermal.get('lake_temp_c')}°C) — SAR change layers measure "
            "ice, not water; drainage stats are not hydrological."
        )

    # 6a-bis. Operational scope (vulnerable-season window). The neural
    # water/change evidence is certified for the monsoon window only
    # (Jun-Sep: 82.6% of dated GLOF events, ~80% of annual rainfall —
    # see siren/scope.py). Outside the window the deterministic baseline
    # stays authoritative and the neural evidence is annotated as
    # out-of-scope; the reason surfaces on the review card.
    try:
        from siren.scope import scope_for_date
        change_stats["operational_scope"] = scope_for_date(
            obs_config["acquired_at"][:10]
        )
    except Exception as exc:  # noqa: BLE001 — scope annotation must never break the run
        logger.warning(f"Operational-scope annotation failed: {exc}")
        change_stats["operational_scope"] = {
            "in_scope": None,
            "reason": "scope annotation unavailable",
        }

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

    # ADR-010: ML confidence is NOT passed to the hazard score. The 5-factor
    # formula (PRD §9.5) uses physical/deterministic inputs; the sole
    # promoted exception is the expansion factor, which is neural-sourced
    # only within the certified scope (see 7a) and otherwise stays on the
    # deterministic registry value. ML evidence remains in change_stats
    # as metadata for the UI.

    # Trend classification: deterministic area-history (ADR-010).
    # The ConvLSTM trend model was archived/disqualified in the 2026-09-07 DL
    # audit (trained on synthetic mask progressions, not real satellite
    # sequences). S_trend is now computed exclusively from deterministic
    # area-history rules per PRD §9.5.
    trend_class = obs_config["trend_class"]  # deterministic fallback
    trend_source = "deterministic"
    trend_confidence = 0.0
    try:
        # Build the observation sequence up to and including this one.
        # For demo observations, use the demo sequence order.
        # For live observations, use all observations for the same basin
        # ordered by acquired_at.
        if not is_live:
            obs_sequence = []
            for oid in DEMO_OBS_IDS:
                obs_sequence.append(oid)
                if oid == observation_id:
                    break
        else:
            # Query all observations for this basin, ordered by date
            all_obs = repo.list_observations()
            basin_id = obs_config.get("basin_id", "dudh-koshi-demo-01")
            obs_sequence = [
                o["observation_id"] for o in all_obs
                if o.get("basin_id") == basin_id
                and o["observation_id"] <= observation_id  # only up to this one
            ]
            if not obs_sequence:
                obs_sequence = [observation_id]
        trend_result = classify_temporal_trend(
            observation_ids=obs_sequence, repo=repo
        )
        trend_class = trend_result["trend_class"]
        trend_source = trend_result["source"]
        trend_confidence = trend_result["confidence"]
    except Exception as exc:
        logger.warning(f"Deterministic trend classification failed: {exc} — using config fallback")
    change_stats["trend_source"] = trend_source
    change_stats["trend_confidence"] = trend_confidence

    # 7a. Expansion-source resolution (ADR-014-am1, operational-primary).
    # When the promoted sar_segmentation_expansion component produced
    # in-scope neural evidence this run, the gated Δp expansion measured
    # inside the monitorable-lake vicinity is the load-bearing
    # water_area_change_percent for the hazard score. The registry value
    # stays recorded as the labeled deterministic cross-check; any scope
    # violation (out-of-window, frozen surface, non-descending orbit, no
    # Imja coverage, runtime demotion) keeps it authoritative.
    expansion_pct = obs_config["expansion_pct"]
    expansion_source = "deterministic_fallback"
    _sar_pair = (
        ml_evidence.get("sar_pair") if isinstance(ml_evidence, dict) else None
    ) or {}
    _scope_in = (
        isinstance(change_stats.get("operational_scope"), dict)
        and change_stats["operational_scope"].get("in_scope") is True
    )
    _exp_km2: float | None = None
    _exp_metric: str | None = None
    if isinstance(ml_evidence, dict):
        _exp_km2 = ml_evidence.get("ml_expansion_lake_vicinity_km2")
        _exp_metric = "lake_vicinity" if _exp_km2 is not None else None
        if _exp_km2 is None:
            _exp_km2 = ml_evidence.get("ml_expansion_km2")
            _exp_metric = "gated_grid" if _exp_km2 is not None else None
    _baseline_km2 = float(obs_config.get("water_area_km2") or 0.0)
    if (
        is_promoted("sar_segmentation_expansion")
        and _exp_km2 is not None
        and _sar_pair.get("orbit") == "descending"
        and bool(_sar_pair.get("covers_imja"))
        and _scope_in
        and not frozen
        and _baseline_km2 > 0.0
    ):
        expansion_pct = round(float(_exp_km2) / _baseline_km2 * 100.0, 2)
        expansion_source = "neural_primary"
    change_stats["expansion_pct_source"] = expansion_source
    if _exp_metric is not None:
        change_stats["expansion_pct_neural_metric"] = _exp_metric
    change_stats["expansion_pct_deterministic"] = obs_config["expansion_pct"]
    if expansion_source == "neural_primary":
        change_stats["expansion_pct_neural"] = expansion_pct
        # The effective expansion is what the score consumed — keep the
        # scalar honest for downstream consumers (dynamic_escalation
        # reads change_stats["expansion_percent"]) while the registry
        # value stays recorded above as the labeled cross-check.
        change_stats["expansion_percent"] = expansion_pct
    elif is_demoted("sar_segmentation_expansion"):
        change_stats["expansion_pct_demoted"] = True

    score = risk_fuse(
        trend_class=trend_class,
        expansion_pct=expansion_pct,
        rainfall_24h_mm=rainfall_24h,
        rainfall_7d_mm=rainfall_7d,
        mean_slope_deg=obs_config.get("mean_slope_degrees") or MEAN_SLOPE_DEG,
        change_in_drainage=True,
        exposed_population=exposed_pop,
        settlements=settlements,
        bridges=bridges,
        wells=wells,
        inundated_wells=inundated_wells,
        population_density_per_km2=200.0,  # demo basin average
        temp_index=temp_index,
    )

    if frozen:
        score["reasons"].append(
            "Lake thermal state: FROZEN_SURFACE — SAR change layers "
            "measure ice, not water; breach-volume trigger suppressed"
        )

    # §9.8.3 neural/deterministic cross-check: material disagreement is
    # surfaced as a review reason — never silently resolved. Fires in
    # shadow mode too (accumulates disagreement evidence pre-promotion).
    _cc = change_stats.get("cross_check")
    if (
        isinstance(_cc, dict)
        and _cc.get("material_disagreement")
        and _cc.get("reason")
    ):
        score["reasons"].append(_cc["reason"])

    # 7a-bis. Out-of-scope observations carry the scope note as a review
    # reason so the analyst sees why the neural evidence is advisory here
    # (deterministic baseline stays authoritative — never silently dropped).
    _scope = change_stats.get("operational_scope", {})
    if isinstance(_scope, dict) and _scope.get("in_scope") is False:
        score["reasons"].append(_scope["reason"])

    # 7a-ter. Expansion provenance — the review card must show which
    # source drove the expansion factor. A neural-primary measurement
    # makes the score "mixed" (the other four factors stay
    # deterministic); a runtime demotion is never silent.
    if expansion_source == "neural_primary":
        score["method"] = "mixed"
        score["reasons"].append(
            f"expansion evidence is neural-primary (promoted Δp mask, "
            f"{expansion_pct:+.1f}% of registered baseline "
            f"{_baseline_km2:.2f} km²); deterministic registry value "
            f"{obs_config['expansion_pct']:+.1f}% kept as labeled "
            f"cross-check"
        )
        # Level 2.4 — conformal interval on the expansion measurement.
        # When the calibrated 90% interval reaches zero the measured
        # expansion may be noise: flag it on the review card rather than
        # silently downgrading (the human gate stays authoritative).
        _lo_km2 = ml_evidence.get("ml_expansion_vicinity_lo_km2")
        _hi_km2 = ml_evidence.get("ml_expansion_vicinity_hi_km2")
        if _lo_km2 is not None and _hi_km2 is not None and _baseline_km2 > 0:
            _lo_pct = round(float(_lo_km2) / _baseline_km2 * 100.0, 2)
            _hi_pct = round(float(_hi_km2) / _baseline_km2 * 100.0, 2)
            change_stats["expansion_pct_ci90"] = [_lo_pct, _hi_pct]
            _uncertain = _lo_pct <= 0.0
            change_stats["expansion_trend_uncertain"] = _uncertain
            if _uncertain:
                score["reasons"].append(
                    f"trend uncertain: 90% conformal interval for neural "
                    f"expansion is [{_lo_pct:+.1f}%, {_hi_pct:+.1f}%] — "
                    f"includes zero"
                )
    elif change_stats.get("expansion_pct_demoted"):
        score["reasons"].append(
            f"sar_segmentation_expansion runtime-demoted via "
            f"{DEMOTE_ENV_VAR} — deterministic registry expansion "
            f"is load-bearing"
        )

    # 7b. Attach shadow evidence (V3 §3.6, §6 — ADR-010 §3: not load-bearing)
    # The deterministic 5-factor hazard score remains authoritative. Shadow
    # evidence (susceptibility, HAND, FNO) is attached to change_stats for
    # the review card UI and shadow-mode evaluation.
    try:
        from siren.risk.shadow_evidence import attach_shadow_evidence
        attach_shadow_evidence(
            change_stats=change_stats,
            obs_config=obs_config,
            rainfall_24h=rainfall_24h,
            rainfall_7d=rainfall_7d,
            dem_path=str(DEM_PATH) if DEM_PATH else None,
        )
    except Exception as exc:
        logger.warning(f"Shadow evidence attachment failed: {exc}")
        change_stats["shadow_evidence"] = {"error": str(exc), "is_shadow": True}

    shadow = change_stats.get("shadow_evidence", {})

    # 7b.1 Promoted ML advisory: when dynamic_escalation is promoted and
    # its pre-breach warning fires (expansion + elevated P_escalation),
    # the warning surfaces as a review reason — advisory only, the
    # deterministic severity and the human gate are unchanged.
    _esc = shadow.get("dynamic_escalation", {}) if isinstance(
        shadow, dict) else {}
    if (
        isinstance(_esc, dict)
        and _esc.get("promoted")
        and _esc.get("pre_breach_warning")
    ):
        score["reasons"].append(
            f"ML escalation advisory (promoted): P_escalation="
            f"{_esc.get('p_dynamic', 0):.2f} >= "
            f"{_esc.get('warning_threshold', 0.65)} with detected "
            f"expansion {_esc.get('expansion_pct', 0):.0f}% — "
            f"pre-breach warning"
        )

    # 7b.2 Learned risk fusion advisory (Level 6): when the fused scorer
    # is promoted and its calibrated event probability clears the
    # advisory threshold, the probability surfaces as a review reason —
    # advisory only; the deterministic severity and human gate are
    # unchanged.
    _fus = shadow.get("learned_risk_fusion", {}) if isinstance(
        shadow, dict) else {}
    if (
        isinstance(_fus, dict)
        and _fus.get("promoted")
        and _fus.get("elevated_event_probability")
    ):
        score["reasons"].append(
            f"ML risk-fusion advisory (promoted): P_event_fused="
            f"{_fus.get('p_fused', 0):.2f} >= "
            f"{_fus.get('advisory_threshold', 0.65)} — learned "
            f"severity evidence concurs with elevated risk; "
            f"deterministic severity stays authoritative"
        )

    # 7c. Wire FNO arrival horizons to the corridor (Sprint 3, shadow-only).
    # If the FNO surrogate was triggered (P_breach ≥ 0.70) and produced
    # sector arrival times, attach them to the corridor exposures. This
    # does NOT modify the deterministic corridor or hazard score — it
    # enriches the corridor result with shadow telemetry for the review
    # card UI and audit lineage. Provenance is tagged "fno_surrogate_v1".
    hydro = shadow.get("hydro_surrogate", {}) if isinstance(shadow, dict) else {}
    if isinstance(hydro, dict) and hydro.get("t_arrival_by_sector"):
        try:
            from siren.geo.corridor import attach_arrival_horizons
            attach_arrival_horizons(
                corridor_result=corridor_geojson,
                t_arrival_by_sector=hydro["t_arrival_by_sector"],
                provenance=hydro.get("provenance", "fno_surrogate_v1"),
            )
            # Also tag exposures with arrival times for the DB
            for exp in exposures:
                name = exp.get("name", "") or exp.get("asset_id", "")
                for sector, t in hydro["t_arrival_by_sector"].items():
                    if sector.lower() in name.lower() or name.lower() in sector.lower():
                        exp["t_arrival_min"] = round(float(t), 1)
                        exp["fno_provenance"] = hydro.get("provenance", "fno_surrogate_v1")
                        break
        except Exception as exc:
            logger.warning(f"FNO arrival horizon attachment failed: {exc}")

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
        method=score.get("method", "deterministic_fallback"),
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
    for obs_id in DEMO_OBS_IDS:
        run = run_pipeline(obs_id, repo)
        results.append(run)
    return results


def classify_temporal_trend(
    observation_ids: list[str] | None = None,
    repo: Repository | None = None,
) -> dict[str, Any]:
    """Classify the temporal trend across multiple observations.

    Uses deterministic area-history rules per PRD §9.5 and ADR-010.
    The ConvLSTM trend model was archived/disqualified in the 2026-09-07 DL
    audit (trained on synthetic mask progressions, not real satellite
    sequences). S_trend is now computed exclusively from deterministic
    area deltas.

    Args:
        observation_ids: Ordered list of observation IDs to classify.
            Defaults to all demo observations in sequence.
        repo: Repository instance. Defaults to the global repository.

    Returns:
        Dict with:
          - trend_class: "stable" | "slowly" | "rapidly" | "uncertain"
          - confidence: float in [0, 1]
          - source: "deterministic"
          - sequence_length: number of timesteps used
          - water_areas: list of water areas per timestep
          - expansion_pcts: list of expansion percentages per timestep
    """
    if repo is None:
        repo = get_repository()

    if observation_ids is None:
        observation_ids = list(DEMO_OBS_IDS)

    # Collect water masks from completed runs
    water_masks: list[np.ndarray] = []
    water_areas: list[float] = []
    expansion_pcts: list[float] = []

    # The first observation serves as the reference (T0).

    for obs_id in observation_ids:
        mask_path = PROCESSED_DIR / f"{obs_id}_expansion_mask.tif"
        if mask_path.exists():
            with rasterio.open(str(mask_path)) as src:
                mask = (src.read(1) > 0).astype(np.float32)
                water_masks.append(mask)
                water_areas.append(float(mask.sum()))
        else:
            # Use the observation's expansion percentage (from the DB
            # registry) to synthesize a scenario mask.
            obs = repo.get_observation(obs_id)
            expansion_pct = obs.get("water_area_change_percent") if obs else None
            if expansion_pct is not None:
                mask, _ = scenario_expansion_mask(
                    expansion_pct / 100.0, seed=42
                )
                water_masks.append(mask.astype(np.float32))
                water_areas.append(float(mask.sum()))
                expansion_pcts.append(expansion_pct)

    if not water_masks:
        return {
            "trend_class": "uncertain",
            "confidence": 0.0,
            "source": "deterministic",
            "sequence_length": 0,
            "water_areas": [],
            "expansion_pcts": [],
        }

    # Compute expansion percentages if not already done
    # Use the first non-zero area as the reference (baseline = 0% expansion)
    if len(expansion_pcts) < len(water_areas) and len(water_areas) >= 2:
        ref_area = next((a for a in water_areas if a > 0), 1.0)
        expansion_pcts = []
        for i, a in enumerate(water_areas):
            if i == 0:
                expansion_pcts.append(0.0)  # baseline
            else:
                expansion_pcts.append((a - ref_area) / ref_area * 100)

    # Deterministic trend classification from area history (ADR-010).
    # No ConvLSTM — the model is archived/disqualified.
    # Rules (PRD §9.5):
    #   - stable:    expansion < 5%
    #   - slowly:    5% <= expansion < 20%
    #   - rapidly:   expansion >= 20%
    #   - uncertain: non-monotonic or insufficient data
    if len(expansion_pcts) >= 2:
        latest_exp = expansion_pcts[-1]
        # Check monotonicity (allow small noise)
        areas = water_areas
        is_monotonic = all(
            areas[i + 1] >= areas[i] * 0.95 for i in range(len(areas) - 1)
        )
        if not is_monotonic:
            trend_class = "uncertain"
            confidence = 0.3
        elif latest_exp >= 20.0:
            trend_class = "rapidly"
            confidence = min(0.9, 0.5 + latest_exp / 100.0)
        elif latest_exp >= 5.0:
            trend_class = "slowly"
            confidence = 0.5
        else:
            trend_class = "stable"
            confidence = 0.9
    else:
        trend_class = "uncertain"
        confidence = 0.3

    return {
        "trend_class": trend_class,
        "confidence": round(confidence, 3),
        "source": "deterministic",
        "ml_model_available": False,  # ConvLSTM archived — no ML model
        "sequence_length": len(water_masks),
        "water_areas": [round(a, 1) for a in water_areas],
        "expansion_pcts": [round(p, 1) for p in expansion_pcts],
    }
