# SIREN — Complete Project Explanation

> **Purpose:** This document is the single reference that answers any question about the SIREN project — what it is, why it was built, how it works, what the code does, what the UI shows, and what the limitations are. It consolidates every document in `docs/` into one navigable, self-contained explanation.

---

## Table of Contents

1. [What is SIREN?](#1-what-is-siren)
2. [Why does it exist? (Problem & Motivation)](#2-why-does-it-exist)
3. [Track 7 Alignment](#3-track-7-alignment)
4. [Architecture Overview](#4-architecture-overview)
5. [Tech Stack](#5-tech-stack)
6. [Repository Structure](#6-repository-structure)
7. [Data Sources](#7-data-sources)
8. [The Pipeline — Step by Step](#8-the-pipeline--step-by-step)
9. [Quality Gate](#9-quality-gate)
10. [Change Detection — Two Paths](#10-change-detection--two-paths)
11. [Weather-Adaptive Router](#11-weather-adaptive-router)
12. [Corridor & Exposure Mapping](#12-corridor--exposure-mapping)
13. [Risk Fusion — How Scores Are Calculated](#13-risk-fusion--how-scores-are-calculated)
14. [Disease Prevention Layer (Track 7.iii)](#14-disease-prevention-layer-track-7iii)
15. [Human-in-the-Loop Review](#15-human-in-the-loop-review)
16. [Resilient Dispatch — The ≤250-Byte Payload](#16-resilient-dispatch--the-250-byte-payload)
17. [Audit Log & Hash Chain](#17-audit-log--hash-chain)
18. [ML Evidence Layer (Optional)](#18-ml-evidence-layer-optional)
19. [Search & Rescue Priority Layer](#19-search--rescue-priority-layer)
20. [Frontend — The Four Views](#20-frontend--the-four-views)
21. [Demo Scenario — The 36-Hour Script](#21-demo-scenario--the-36-hour-script)
22. [API Endpoints — Complete Reference](#22-api-endpoints--complete-reference)
23. [Data Contracts — Exact Schemas](#23-data-contracts--exact-schemas)
24. [Architecture Decision Records (ADRs)](#24-architecture-decision-records-adrs)
25. [Build Roadmap — Phases 0–7](#25-build-roadmap--phases-07)
26. [Testing — 104 Tests Explained](#26-testing--104-tests-explained)
27. [Definition of Done](#27-definition-of-done)
28. [Known Limitations](#28-known-limitations)
29. [Future Roadmap (V1–V4)](#29-future-roadmap-v1v4)
30. [Quick Start — Running the Demo](#30-quick-start--running-the-demo)
31. [Glossary of Terms](#31-glossary-of-terms)

---

## 1. What is SIREN?

**SIREN** stands for **Satellite-Informed Risk & Emergency Network**. It is a human-in-the-loop, satellite-assisted early-warning and disaster-response platform designed for vulnerable Himalayan basins.

### In one sentence

SIREN fuses satellite imagery (Sentinel-1 SAR + Sentinel-2 optical) with rainfall, terrain, river, population, and infrastructure data to detect hazards, model downstream exposure, and — only after human confirmation — dispatch a bandwidth-light alert with disease-prevention guidance.

### What it does (the 4 questions)

SIREN answers four operational questions that emergency coordinators need:

1. **What changed?** — Detects water expansion, surface scouring, or other change between satellite observations.
2. **How serious is it?** — Computes a hazard score (H), exposure priority (E), and disease risk (D_risk) with explainable reasons.
3. **Who and what are in the path?** — Maps a hydrological corridor downstream and intersects it with critical assets (villages, bridges, wells, roads, clinics).
4. **What should responders do right now?** — Generates a disease-prevention action sheet and dispatches a compressed alert.

### What it does NOT do

- Does NOT predict the exact time of a glacial-lake outburst.
- Does NOT issue autonomous evacuation orders.
- Does NOT diagnose diseases from satellite imagery.
- Does NOT identify individuals or locate trapped persons.
- Does NOT replace government early-warning systems.

---

## 2. Why does it exist?

### The core problem

Mountain communities in the Himalayas face cascading hazards: extreme rainfall, flash floods, landslides, glacial-lake outburst floods (GLOFs), debris flows, and communication blackouts. The same event cascades through a chain of systems:

```
High-altitude lake/slope changes first
    → River corridor becomes dangerous next
    → Downstream settlements, roads, bridges, hospitals, water supplies affected afterward
```

Two operational failure modes compound this:

**Failure Mode 1 — Communication collapse (Track 7, Area ii):** Ground networks collapse when roads, bridges, and cell towers are washed out — often exactly when coordination matters most. SIREN's alert payload is designed for constrained links (compressed SMS, LoRa mesh, satellite messengers).

**Failure Mode 2 — Disease prevention (Track 7, Area iii):** Waterborne disease is one of the largest secondary killers after flooding — contaminated wells, submerged sanitation, severed clinic access. SIREN intersects detected inundation with water points and health facilities to generate an immediate contamination-priority list.

### The hypothesis

If SIREN automatically compares new satellite observations against a historical baseline, fuses the resulting change signal with weather, terrain, and infrastructure exposure, and routes the evidence through a human-review workflow, emergency teams can identify priority zones and trigger disease-prevention response measurably faster than through manual, disconnected analysis — particularly during cloud-covered monsoon windows when optical-only systems go blind.

---

## 3. Track 7 Alignment

SIREN was built for **Track 7: Living with Uncertainties, Building with Resilience** at the `>.hack();'26` hackathon.

| Track Area | SIREN's Response |
|---|---|
| **Area ii: Communication Systems During Disasters** | ≤250-byte compressed payload for LoRa mesh / satellite messenger / low-bandwidth SMS. Geofenced dispatch to recipient groups. Channel simulator (SMS/LoRa/Satellite) in AuditView. |
| **Area iii: Curbing Diseases That Arise During Disasters** | Disease Prevention Action Sheet: inundated water points × population density × temperature index. Boil-water advisories, chlorine dispatch quotas, per-zone water-purification recommendations. |
| **Area i: Personnel Identification** | Served indirectly through rescue-prioritization: exposure corridors, road-cut analysis, settlement-level population estimates. Explicitly NOT in MVP scope (V4 roadmap). |

---

## 4. Architecture Overview

### High-level pipeline

```
Sentinel-1 SAR / Sentinel-2 Optical / SRTM DEM / GPM IMERG / OSM
                         ↓
         Preprocessing, co-registration & quality gate
                         ↓
    Weather-adaptive router (cloud ≥20% → SAR path)
                         ↓
    SAR backscatter differencing  ⇄  Optical NDWI differencing
                         ↓
         D8 + OSM hydrological corridor & exposure analysis
                         ↓
     Risk fusion (H + E + D_risk) + disease-prevention actions
                         ↓
                   Human-in-the-loop review
                         ↓
    Resilient geofenced dispatch (≤250 bytes)  +  audit log
```

### Hybrid AI & Computational Architecture

SIREN is deliberately a **hybrid pipeline** — deterministic physical modeling plus optional deep-learning vision — so every score stays explainable.

```
Data Layer:
  Sentinel-1 SAR GRD → Quality Gate → Weather-Adaptive Router
  Sentinel-2 Optical → Quality Gate → Weather-Adaptive Router
  SRTM 30m DEM       ──────────────► D8 Flow Accumulation
  GPM IMERG Rainfall ──────────────► Risk Fusion Engine
  OSM Infrastructure ──────────────► Exposure Corridor

Processing Layer:
  Quality Gate → Cloud Check → Route to SAR or Optical
  SAR Path: Backscatter differencing (VV/VH ratio thresholding)
  Optical Path: NDWI differencing (when clear)
  SegFormer: Land-cover classification (optional, V3)

Risk Layer:
  Temporal Trend: Persistence / Regression
  Risk Fusion Engine: H + E + D_risk scoring
  Disease Engine: Waterborne disease risk index

Review Layer:
  Coordinator Review Console → Human Decision
  Audit Log (immutable) → Dispatch Engine (SMS/LoRa/Satellite)
```

---

## 5. Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Backend** | Python 3.11+, FastAPI, Pydantic | Geospatial ecosystem (rasterio, geopandas, xarray) is unmatched; FastAPI gives typed endpoints for free |
| **Raster ops** | rasterio, numpy, xarray | COG read/write, reprojection, NDWI/backscatter math |
| **Hydrology (D8)** | pysheds (fallback: WhiteboxTools) | Battle-tested D8 flow accumulation; pysheds is pure-Python fallback |
| **Vector ops** | geopandas + shapely | Buffer/intersect against OSM layers |
| **Database** | SQLite (JSON columns) + GeoJSON files | Zero-ops, offline-safe; PostGIS is the V2 migration path (ADR-001) |
| **Frontend** | React + Vite + TypeScript | Fast scaffolding, typed API contracts |
| **Map** | MapLibre GL JS | Free, no token, raster+vector overlays, swipe-compare support |
| **State/data** | TanStack Query | Polling for pipeline run status |
| **ML (optional)** | PyTorch + pretrained weights | Only if hours 0–16 go well; deterministic baseline is the deliverable (ADR-002) |
| **Storage** | SQLite + GeoJSON + GeoTIFF on disk | No PostGIS, no Redis |
| **Deployment** | Docker Compose | One-command: `./start.sh` |

### Dependency whitelist (Hard Rule 8)

Allowed: rasterio, geopandas, shapely, numpy, xarray, pysheds, fastapi, pydantic, pytest.

Exception: torch/torchvision are allowed as an optional `[ml]` extra for the evidence layer only (ADR-002 addendum). The deterministic fallback runs without them.

Anything else: stop and ask.

---

## 6. Repository Structure

```
project/
├── backend/
│   ├── siren/
│   │   ├── api/              # FastAPI routes + Pydantic models + map asset endpoints
│   │   │   ├── app.py        # FastAPI application entry point
│   │   │   ├── models.py     # Pydantic models (Observation, Score, Alert, etc.)
│   │   │   └── map_assets.py # Pre-rendered map tile overlays (DEM hillshade, SAR, baseline)
│   │   ├── ingest/           # CDSE STAC, Earthdata SRTM, IMERG, Overpass downloaders
│   │   │   ├── cdse.py       # Sentinel-1/2 STAC search + download
│   │   │   ├── srtm.py       # SRTM 1-arc-second from NASA Earthdata
│   │   │   ├── imerg.py      # GPM IMERG rainfall pull
│   │   │   └── openmeteo.py  # Open-Meteo weather context
│   │   ├── preprocess/       # Clip, reproject, co-register, cloud mask, quality gate
│   │   │   ├── quality.py    # Quality gate: cloud fraction, alignment, confidence multiplier
│   │   │   └── ...           # clip, reproject, co_register, cloud_mask
│   │   ├── detect/           # NDWI diff, SAR backscatter ratio, weather-adaptive router, scenario masks
│   │   │   ├── ndwi.py       # Optical NDWI differencing
│   │   │   ├── sar.py        # SAR backscatter log-ratio thresholding
│   │   │   ├── router.py     # Weather-adaptive routing (cloud ≥20% → SAR)
│   │   │   └── scenario.py   # Deterministic demo masks near Imja lake
│   │   ├── geo/              # D8 corridor, tolerance buffers, exposure intersections
│   │   │   └── corridor.py   # Combined D8 + OSM river buffering (295 lines)
│   │   ├── risk/             # Hazard H, exposure E, disease D_risk, SAR priority + reasons
│   │   │   ├── fusion.py     # Risk fusion engine (H + E + D_risk + reasons)
│   │   │   └── sar_priority.py  # Search & Rescue priority ranking
│   │   ├── ml/               # Optional ML evidence layer (deterministic fallback, torch-gated)
│   │   │   ├── consensus.py  # Deterministic consensus mask (NDWI + SAR agreement)
│   │   │   ├── engine.py     # Siamese U-Net / ChangeFormer inference (torch-gated)
│   │   │   ├── model.py      # Model definitions
│   │   │   ├── train.py      # Training entry point
│   │   │   └── visualize.py  # Heatmap and preview generation
│   │   ├── alerting/         # ≤250-byte payload codec, validator
│   │   │   ├── codec.py      # encode/decode alert to/from bytes
│   │   │   └── validate.py   # Size validation (≤250 bytes)
│   │   ├── audit/            # Append-only log writer + SHA-256 hash chain
│   │   │   ├── writer.py     # Append-only audit log writer
│   │   │   └── hash_chain.py # SHA-256 hash chain for tamper-evidence
│   │   ├── db/               # SQLite schema + repositories
│   │   │   ├── schema.sql    # Database schema (observations, runs, scores, reviews, dispatches, audit_log)
│   │   │   └── repos.py      # Repository layer (CRUD for non-audit tables)
│   │   └── pipeline.py       # Orchestrator: detect→geo→risk→DB→audit
│   └── tests/
│       ├── fixtures/         # Synthetic rasters + fake OSM GeoJSON
│       │   ├── rasters/      # baseline.tif, expanded_water.tif, cloudy_optical.tif
│       │   └── osm/          # fake_assets.geojson (2 villages, 1 bridge, 3 wells)
│       ├── test_quality.py   # 11 tests — Quality gate (PRD §9.1)
│       ├── test_codec.py     # 12 tests — Payload codec ≤250 bytes, round-trip
│       ├── test_audit.py     # 11 tests — Append-only enforcement, hash chain
│       ├── test_api.py       # 12 tests — All API endpoints, human gate
│       ├── test_preprocess.py# 6 tests — Clip, reproject, co-register on synthetic rasters
│       ├── test_ingest.py    # 25 tests — CLI argument parsing, provenance sidecars
│       ├── test_pipeline.py  # 5 tests — Full orchestrator: detect→geo→risk→DB→audit
│       ├── test_ml.py        # 13 tests — ML evidence layer (deterministic fallback, torch-gated)
│       └── test_sar_priority.py # 9 tests — SAR priority ranking (PRD §15)
├── frontend/
│   └── src/
│       ├── views/            # MapView, TimelineView, ReviewView, AuditView
│       ├── api/              # Typed API client + offline mock fallback
│       ├── simulation/       # SimulationContext (shared demo state)
│       ├── components/       # OfflineBadge (online/offline event listener)
│       ├── theme/            # ThemeProvider + ThemeToggle (Ops Dark / Light / Satellite)
│       ├── utils/            # ntfy.ts — shared ntfy.sh live alert utility
│       └── index.css         # Tailwind base + design tokens
├── data/
│   ├── raw/                  # Downloaded scenes (gitignored, never hand-edited)
│   ├── processed/            # Aligned rasters, masks (gitignored, pipeline-written)
│   └── assets/               # Basin GeoJSON, OSM extracts, weather series (committed)
├── docs/
│   ├── PRD.md                # Product Requirements Document (v4.3)
│   ├── BUILD_ROADMAP.md      # 36-hour build plan with phase checkpoints
│   ├── API_CONTRACT.md       # HTTP API surface
│   ├── UI_DESIGN.md          # Coordinator console design spec
│   ├── DEVIN_BRIEFS.md       # Devin task dispatch briefs (D1-D8, archived)
│   ├── ADR-001..005          # Architecture decision records
│   ├── KNOWN_LIMITATIONS.md  # One-page limitations reference
│   └── explanation.md        # This document
├── Dockerfile.backend        # Backend image (Python + GDAL + geospatial stack)
├── Dockerfile.frontend       # Frontend image (Node build → nginx serve)
├── docker-compose.yml        # One-command orchestration
├── start.sh                  # Port-conflict-aware launcher
├── README.md                 # Quick-start guide
└── AGENTS.md                 # Agent routing and hard rules
```

---

## 7. Data Sources

### Primary datasets

| Dataset | Type / Source | Role in SIREN | Storage |
|---|---|---|---|
| **Sentinel-1 C-SAR** | Copernicus CDSE, 10 m GRD | All-weather backscatter differencing for water/debris tracking | GeoTIFF/COG + metadata |
| **Sentinel-2 MSI** | Copernicus CDSE, 10–20 m multispectral | Cloud-free optical NDWI and Siamese U-Net segmentation | GeoTIFF/COG + metadata |
| **SRTM 1 Arc-Second** | NASA Earthdata, 30 m DEM | Elevation, slope angle, D8 downstream hydrological flow | Raster DEM |
| **GPM IMERG** | NASA GES DISC, 0.1° NRT | Basin-wide antecedent rainfall and storm-intensity metrics | NetCDF/GeoTIFF/feature table |
| **Open-Meteo** | HTTP API | Development weather and soil-moisture context | JSON time series |
| **OpenStreetMap (HOT)** | Humanitarian OSM Team vectors | Roads, bridges, clinics, schools, and municipal water sources | GeoJSON/GeoPackage |
| **Sen1Floods11** | Public benchmark dataset | Pretraining/benchmarking SAR flood-water segmentation weights | GeoTIFF/COG |
| **ICIMOD inventories** | Open data/reports | Glacial-lake baselines and regional GLOF context | GeoJSON/GeoPackage |

### Prepared demo dataset (on disk, `data/`)

- **Sentinel-1 GRD triplet:** 2026-07-23 (obs-001) + 2026-08-04 (obs-002) + 2026-08-12 (obs-003), IW dual-pol VV/VH, full AOI coverage.
- **Sentinel-2 L2A:** 2025-11-22, tile **T45RVL** (covers 100% of AOI — the clean post-monsoon optical baseline). Note: the AOI spans 4 S2 tiles; T45RVL is the correct one for this basin.
- **SRTM 30 m clip:** 1188×1260, EPSG:4326, elevation 1930–8429 m, no nodata gaps.
- **OSM extract:** 1100 features — 63 settlements, 92 bridges (incl. Hillary suspension bridges), 16 drinking-water points, 3 clinics, 1 hospital, Dudh Koshi/Imja rivers.
- **Weather context:** `data/assets/weather_series.json` (prepared demo context; refresh with `backend/siren/ingest/openmeteo.py`).

### Data hygiene rules (Hard Rule 7)

- Only `ingest/` scripts write to `data/raw/`.
- Only the pipeline writes `data/processed/`.
- Never commit rasters.
- Never hand-edit data files.

---

## 8. The Pipeline — Step by Step

The pipeline is orchestrated by `backend/siren/pipeline.py`. Here is the complete 17-step flow:

```
Step 1:  Select basin and time range
Step 2:  Acquire scenes and contextual datasets (or load from local cache)
Step 3:  Validate provenance and spatial metadata
Step 4:  Clip, reproject, resample, align to SRTM baseline grid
Step 5:  Apply cloud and invalid-pixel masks
Step 6:  Run weather-adaptive change detection (SAR and/or optical path)
Step 7:  Extract water, glacier, debris, and change statistics
Step 8:  Join rainfall, temperature, and hydrology features
Step 9:  Derive slope, drainage, river proximity, and exposure features
Step 10: Calculate temporal persistence and trend
Step 11: Intersect hazard polygon + corridor with settlements and infrastructure
Step 12: Fuse evidence into hazard, exposure, and disease-risk scores
Step 13: Apply data-quality and alert policies
Step 14: Create human-review alert
Step 15: Confirm, reject, or postpone
Step 16: Dispatch simulated geofenced alert if confirmed
Step 17: Store complete audit record
```

### How the pipeline is triggered

- **Single observation:** `POST /runs` with `{"observation_id": "obs-003"}`. Runs synchronously.
- **All observations:** `POST /runs/process-all`. Processes the prepared demo sequence (obs-001 → obs-002 → obs-003) sequentially.

### What happens in each step

**Steps 1–3 (Ingestion & Validation):** The pipeline selects the basin (Dudh Koshi / Imja), loads the observation's raster from `data/processed/`, validates the CRS (EPSG:4326), and checks provenance metadata.

**Steps 4–5 (Preprocessing):** Scenes are clipped to the basin boundary, reprojected onto the SRTM baseline grid, and aligned. Cloud masks are applied to optical scenes. The quality gate (Step 6) determines which detection path to use.

**Step 6 (Change Detection):** The weather-adaptive router decides: if optical cloud fraction ≥ 20%, use SAR backscatter differencing; otherwise, use optical NDWI differencing. Both paths produce a water/change mask.

**Steps 7–9 (Feature Extraction):** Water area, expansion percentage, rainfall (24h + 7d), mean slope, and terrain features are extracted and joined.

**Steps 10–11 (Corridor & Exposure):** D8 flow accumulation traces the downstream path from the change source. OSM river segments are selected and buffered. The corridor is intersected against critical assets using tolerance buffers.

**Steps 12–13 (Risk Fusion & Policy):** Hazard score H, exposure priority E, and disease risk D_risk are computed using fixed weights. Severity is classified (informational / watch / elevated / critical).

**Steps 14–17 (Review, Dispatch, Audit):** A review alert is created. After human confirmation, a compressed payload is dispatched and the full lineage is written to the audit log.

---

## 9. Quality Gate

The quality gate (`backend/siren/preprocess/quality.py`) is the first decision point in the pipeline. It outputs a `QualityVerdict` JSON object:

```json
{
  "quality_score": 0.88,
  "cloud_fraction": 0.11,
  "alignment_ok": true,
  "usable": true,
  "confidence_adjustment": 0.95
}
```

### How it works

1. **Cloud fraction** is computed from the optical scene (fraction of bright pixels in both bands). For SAR, cloud_fraction is treated as 0.0 (SAR is all-weather capable).
2. **Alignment check** verifies that the moving scene aligns with the reference (SRTM baseline grid). Returns an alignment error metric.
3. **Usability** is determined: if cloud_fraction ≥ 0.20, `usable` is set to `false` and the pipeline routes to the SAR path.
4. **Confidence multiplier** = `(1.0 - cloud_fraction) × sensor_freshness_weight`. For a cloud-blocked optical scene, the gate routes to SAR and sets `cloud_fraction: 0.0` for that path.

### Why it matters

The quality gate is the **weather-adaptive router's input**. It determines whether the system uses optical or SAR change detection. This is the core differentiator: when the Himalayas are cloud-covered (the expected case during monsoon), the system automatically switches to SAR, which penetrates clouds.

---

## 10. Change Detection — Two Paths

SIREN uses a **hybrid, weather-adaptive** change detection approach with two parallel paths:

### Path A: Optical NDWI (when skies are clear)

**File:** `backend/siren/detect/ndwi.py`

- Computes the **Normalized Difference Water Index (NDWI)** from Sentinel-2 bands:
  - NDWI = (Green - NIR) / (Green + NIR)
- Differences the current observation against the baseline (2025-11-22 clear-sky scene).
- Pixels with NDWI above a threshold are flagged as water expansion.
- Produces a binary water-change mask.

**When used:** Optical cloud fraction < 20%.

### Path B: SAR Backscatter Ratio (when cloudy / monsoon)

**File:** `backend/siren/detect/sar.py`

- Uses **Sentinel-1 IW GRD** dual-polarization (VV and VH) backscatter.
- Computes the **log-ratio** between current and baseline backscatter:
  - Ratio = log10(σ⁰VV_current / σ⁰VV_baseline)
- Multi-look speckle suppression reduces noise.
- DEM slope masking removes false positives on steep terrain.
- Thresholding flags open-water expansion and surface scouring.

**When used:** Optical cloud fraction ≥ 20% (the monsoon case).

### Scenario masks (fallback)

**File:** `backend/siren/detect/scenario.py`

When the available SAR swath doesn't cover the change source (Imja lake at 86.925°E is outside the ascending-orbit S1 swath), deterministic scenario masks are used. These are clearly labeled in the code and produce reproducible results.

---

## 11. Weather-Adaptive Router

**File:** `backend/siren/detect/router.py` (33 lines)

The weather-adaptive router is the decision point between the two change detection paths:

```
if optical_cloud_fraction >= 0.20:
    → Route to SAR path (primary)
    → Set cloud_fraction = 0.0 for SAR (all-weather)
else:
    → Route to optical NDWI path (primary)
    → Use optical cloud_fraction for confidence adjustment
```

### Why this matters

Monsoon cloud cover is the **expected case** in the Himalayas, not the edge case. A system that goes blind in the rain is not operationally useful. The router ensures the pipeline always has a working detection path.

### Demo demonstration

In the demo:
- **Baseline (2025-11-22):** S2 optical, 5% cloud → optical path.
- **obs-001 (2026-07-23):** S1 SAR, 0% effective cloud → SAR path.
- **obs-002 (2026-08-04):** S1 SAR, 95% optical cloud → **SWITCHED TO SAR PATH** (this is the demo's dramatic beat).
- **obs-003 (2026-08-12):** S1 SAR, 90% optical cloud → SAR path.

The TimelineView's router strip visually shows this switch.

---

## 12. Corridor & Exposure Mapping

**File:** `backend/siren/geo/corridor.py` (295 lines)

This is the most complex geospatial module. It combines two evidence sources to identify which assets are downstream of a detected change.

### The problem

A pure D8 steepest-descent path from the change polygon **missed all settlements** on the real SRTM DEM. The nearest settlement was 2.8 km away. At 30 m resolution in steep Himalayan terrain, DEMs suffer from:
- Drainage-trenching artifacts
- Narrow gorges
- Lateral moraine walls
- Interpolation errors that divert a single-pixel D8 path over a ridge or into a dry side-gully

### The solution: Combined D8 + OSM River Buffering

**Step 1 — D8 reachability (physical validation):**
- SRTM-derived D8 flow accumulation traces the downstream flow path from the change polygon centroid.
- Confirms the change source drains into the expected sub-basin (e.g., Imja lake → Imja Khola / Dudh Koshi), not an adjacent drainage divide.

**Step 2 — OSM river selection:**
- Selects `waterway=river/stream` segments reachable by the D8 path (within a reachability radius).
- OSM captures the real, surveyed riverbed through inhabited valleys.

**Step 3 — Floodplain buffer:**
- Buffers the reachable river segments by 100–150 m (nominal flood-plain width).

**Step 4 — Exposure intersection:**
- Intersects the buffered corridor against OSM-sourced assets using **resolution-aware tolerance buffers**:
  - Bridges: ±75 m
  - Roads: ±50 m
  - Settlements / wells: ±100 m

### Why this works

D8 provides **physical validation** (gravity gradient check). OSM rivers provide **operational accuracy** (surveyed ground truth of where water actually flows through inhabited valleys). Together, they reliably catch the demo assets: Benkar (75 m), Jorsale (69 m), the Hillary suspension bridges, and drinking wells along the Dudh Koshi.

### Tolerance buffers (PRD §6.4)

| Asset Type | Tolerance Buffer | Rationale |
|---|---|---|
| Bridges | ±75 m | Bridge approach roads, abutment scour |
| Roads | ±50 m | Road surface, shoulder, drainage |
| Settlements | ±100 m | Settlement boundary, perimeter structures |
| Wells | ±100 m | Well head, immediate contamination zone |

---

## 13. Risk Fusion — How Scores Are Calculated

**File:** `backend/siren/risk/fusion.py`

Every score carries a deterministic `reasons` array (≥3 entries on elevated+). Never return a bare number.

### Hazard Score (H)

```text
H = 0.30 × satellite-change trend (S_trend)
  + 0.25 × water-area expansion (A_expansion)
  + 0.20 × rainfall / snowmelt indicator (R_rain, 24h + 7d)
  + 0.15 × terrain and slope risk (T_slope)
  + 0.10 × downstream proximity (D_prox)
```

**Weights are fixed:** 0.30 / 0.25 / 0.20 / 0.15 / 0.10. These sum to 1.0.

### Exposure Priority (E)

```text
E = H × Population Vulnerability × Critical Infrastructure Weight
```

### Waterborne Disease Risk Index (D_risk)

```text
D_risk = Inundated Water Points × Population Density × Temperature Index
```

`D_risk` flags zones for immediate water-purification and medical-supply dispatch. It is explicitly a **triage priority signal**, not a medical diagnosis.

### Severity Classification

| Expansion % | Severity | Action |
|---|---|---|
| < 5% | Informational | Log only |
| ≥ 5% | Watch | Monitor |
| ≥ 20% | Elevated | Review |
| ≥ 40% | Critical | Human review required |

### Reasons generation

Every elevated+ alert carries at least 3 deterministic reasons explaining why the score is elevated. For example, obs-002 (critical) has 8 reasons; obs-003 (critical) has 8 reasons. Reasons are generated from deterministic templates based on the computed features.

---

## 14. Disease Prevention Layer (Track 7.iii)

### Disease Risk Index

```text
D_risk = Inundated Water Points × Population Density × Temperature Index
```

- **Inundated Water Points:** Count of municipal water points, wells, and storage tanks that are submerged or encircled by the detected inundation polygon.
- **Population Density:** Estimated population in the affected zone (from settlement-level data).
- **Temperature Index:** Ambient temperature factor (higher temperature → faster bacterial growth in contaminated water).

### Disease Prevention Action Sheet

Auto-generated per affected zone:
- **Boil-water advisory** for submerged wells.
- **Chlorine dispatch quota** per affected water point.
- **Medical supply manifest** for encircled clinics.
- **Per-zone priority ranking** based on D_risk score.

### Where it appears

The Disease Prevention Action Sheet is rendered in **ReviewView** (right dock, 320px panel) alongside the ranked exposed-infrastructure table.

---

## 15. Human-in-the-Loop Review

### The review workflow

```
Pipeline produces elevated/critical alert
    → ReviewView card appears with evidence panel
    → Escalation policy badge shows: "Advisory auto-routed to First Responders.
       Public broadcast held for Human Gate confirmation."
    → Early warning banner shows: "★ Early warning 12 days"
    → Coordinator inspects:
        - Before/after rasters (swipe compare)
        - Change overlay
        - Hazard score + reasons (≥3)
        - Exposure list (assets, population, status)
        - Disease action sheet
    → Coordinator decides:
        ✓ Confirm SOS  →  Dispatch proceeds + ntfy.sh push fires automatically
        ✗ Reject        →  Alert suppressed, no dispatch
        ⏸ Postpone      →  Request local verification, alert held
```

### Auto-SOS on CONFIRM

When the coordinator clicks **Confirm SOS**, the system:

1. Records the review decision in the database (`POST /runs/{id}/review` with `decision=confirm`)
2. **Automatically fires an ntfy.sh push notification** to the coordinator's phone (when online)
3. Shows a toast: "Decision confirmed — SOS sent to phone" (or "air-gap mode, SOS simulated" when offline)
4. Invalidates React Query caches to refresh the Audit trail

This does **not** violate Hard Rule #3 — the human made the decision. The ntfy.sh call is a side-effect of the confirm action, not an autonomous dispatch. The Audit tab's SEND TO PHONE button remains as a secondary manual send.

The ntfy.sh utility is shared between ReviewView and AuditView in `frontend/src/utils/ntfy.ts`.

### Decision states

| State | Effect |
|---|---|
| `null` (no review) | Dispatch endpoint returns 409 (human gate enforced) |
| `confirm` | Dispatch endpoint proceeds; audit log records decision |
| `reject` | Alert suppressed; no dispatch; audit log records rejection |
| `postpone` | Alert held; audit log records postponement |

### Human gate enforcement

The human gate is enforced at **two levels**:
1. **Application layer:** `POST /runs/{id}/dispatch` returns 409 if no `confirm` review exists.
2. **Database layer:** SQLite trigger prevents dispatch without a prior confirm review.

This is Hard Rule 3: **No code path may dispatch an alert without a recorded review decision (`confirm`).**

---

## 16. Resilient Dispatch — The ≤250-Byte Payload

**File:** `backend/siren/alerting/codec.py`

### The payload format

Confirmed alerts serialize to a compact JSON object under 250 bytes:

```json
{"aid":"siren-04","sec":"B","haz":"GLOF_FL","lvl":3,"exp_pop":1240,"crit":["BR-12","RD-4"],"med_act":"BOIL_WATER_NOW"}
```

### Field meanings

| Field | Meaning | Example |
|---|---|---|
| `aid` | Alert ID (prefix `siren-`) | `siren-04` |
| `sec` | Geofence sector | `B` |
| `haz` | Hazard type code | `GLOF_FL` (glacial lake outburst flood) |
| `lvl` | Severity level (1–4) | `3` (critical) |
| `exp_pop` | Exposed population | `1240` |
| `crit` | Critical asset IDs | `["BR-12","RD-4"]` |
| `med_act` | Medical/disease action | `BOIL_WATER_NOW` |

### Actual size

The demo payload is **118 bytes** (well under the 250-byte limit). This is verified by a unit test (`test_codec`).

### Why ≤250 bytes?

This is the maximum size for:
- **LoRa mesh** messages (typical payload limit)
- **Satellite messengers** (e.g., Garmin inReach, Zoleo)
- **Low-bandwidth SMS** (standard SMS is 160 characters; this is well under)

### Public-facing message

The compressed payload is accompanied by a plain-text message:

> **Potential flood or debris-flow risk detected in Sector B. Follow instructions from local authorities, avoid the downstream river corridor, and move toward the designated shelter if instructed.**

The message avoids false certainty — it says "potential" and defers to local authorities.

---

## 17. Audit Log & Hash Chain

**File:** `backend/siren/audit/writer.py` + `backend/siren/audit/hash_chain.py`

### Append-only enforcement

The audit log is **strictly append-only**:
- Repository exposes only INSERT + SELECT for `audit_log`.
- No UPDATE or DELETE paths exist.
- SQLite triggers enforce immutability at the database level.

### SHA-256 hash chain

Each audit entry carries two hashes:
- `prev_hash`: SHA-256 of the previous entry's event hash (genesis = `"0" * 64`).
- `event_hash`: `sha256(prev_hash + timestamp + payload)`.

This creates a **tamper-evident chain**: altering any entry invalidates all subsequent hashes.

### Audit entry structure

```json
{
  "entry_id": 1,
  "alert_id": "alert-0091",
  "run_id": "run-0004",
  "actor": "pipeline",
  "action": "run",
  "detail_json": "{}",
  "prev_hash": "0000000000000000000000000000000000000000000000000000000000000000",
  "event_hash": "8475547e1039023e...",
  "created_at": "2026-08-13T04:10:33Z"
}
```

### Query API

```
GET /audit?alert_id={alert_id}&run_id={run_id}
```

Either parameter filters the result; both can be combined. Returns the full lineage for an alert or run.

### What gets logged

Every run, model version, input snapshot, reviewer decision, and dispatch action is recorded:
1. Pipeline run (actor: `pipeline`, action: `run`)
2. Review decision (actor: `coordinator-01`, action: `review`)
3. Dispatch action (actor: `coordinator-01`, action: `dispatch`)

---

## 18. ML Evidence Layer (Optional)

**File:** `backend/siren/ml/`

### What it is

An **optional** change-detection evidence layer that provides additional visual evidence (heatmap, mask) alongside the deterministic pipeline. It is **never the sole source of a score** — the rule-based pipeline remains the critical path.

### Components

| File | Purpose |
|---|---|
| `ml/consensus.py` | Deterministic consensus mask (NDWI + SAR backscatter agreement) |
| `ml/engine.py` | Siamese U-Net / ChangeFormer inference (torch-gated) |
| `ml/model.py` | Model definitions |
| `ml/train.py` | Training entry point (not in demo critical path) |
| `ml/visualize.py` | Heatmap and preview generation |

### Deterministic fallback

When torch is **not** installed, the ML endpoint returns deterministic results:
- Heatmap from the consensus mask.
- Change mask from NDWI + SAR agreement.
- Same outputs every run (fully reproducible).

### When torch is available

When `pip install -e ".[ml]"` is run (torch + torchvision), the ML path produces:
- A heatmap from the Siamese U-Net / ChangeFormer inference.
- A change mask from the deep learning model.
- Preview images (baseline optical crop, after observation crop).

### Endpoint

```
GET /runs/{run_id}/ml-evidence
```

Returns:
```json
{
  "run_id": "run-0004",
  "source": "deterministic-fallback",  // or "siamese-unet" / "changeformer"
  "heatmap_uri": "/data/processed/obs-003-heatmap.png",
  "mask_uri": "/data/processed/obs-003-ml-mask.tif",
  "baseline_mask_uri": "/data/processed/baseline-water-mask.tif",
  "preview_baseline_uri": "/data/map-assets/obs-003/baseline-optical.png",
  "preview_after_uri": "/data/processed/obs-003-preview.png",
  "bounds": [86.65, 27.65, 86.95, 27.95]
}
```

### Test coverage

13 tests in `tests/test_ml.py` (3 torch-gated, 10 deterministic).

---

## 19. Search & Rescue Priority Layer

**File:** `backend/siren/risk/sar_priority.py`

### What it does

Ranks downstream sectors by `population × access-loss`. Assets with severed access (bridges/roads cut) get higher SAR priority because they are harder to reach.

### Priority score

```text
priority_score = population × access_risk_factor × inundation_factor
```

- **access_risk_factor:** `severed` = 1.0, `degraded` = 0.7, `intact` = 0.3.
- **inundation_factor:** `inundated` = 1.0, `buffered` = 0.5, `safe` = 0.1.

### Endpoint

```
GET /runs/{run_id}/sar-priority
```

Returns assets sorted by priority score (highest first).

### Test coverage

9 tests in `tests/test_sar_priority.py`.

### Build status

Built as a **stretch goal** in Phase 6 (after the core loop was complete). ~3–4 hours of work.

---

## 20. Frontend — The Four Views

### Overview

The frontend is a React + Vite + TypeScript application with MapLibre GL JS for mapping. It has four views, all sharing a dark ops-console design system.

### Design system

| Token | Hex | Usage |
|---|---|---|
| `bg` | `#0F172A` | App background (slate-900) |
| `panel` | `#1E293B` | Panels/cards (slate-800) |
| `panel-2` | `#334155` | Nested panels, borders (slate-700) |
| `text` | `#E2E8F0` | Primary text (slate-200) |
| `text-dim` | `#94A3B8` | Secondary text (slate-400) |
| `accent` | `#06B6D4` | SIREN brand, active states (cyan-500) |
| `safe` | `#22C55E` | Green — safe assets, success |
| `warn` | `#F59E0B` | Amber — buffered risk, advisory |
| `danger` | `#EF4444` | Red — inundated, critical, reject |
| `info` | `#3B82F6` | Blue — informational |

### Themes

Three themes are available via the theme toggle in the header bar:
1. **Ops Dark** (default) — Dark slate background, high-contrast status colors.
2. **Professional Light** — Light background for daytime operations.
3. **Satellite** — Satellite imagery-optimized theme.

### View 1: MapView

The core visual canvas. Full-bleed MapLibre map with:

- **Left dock (collapsible, 220px):** Layer toggles — Basin AOI, DEM hillshade, Optical baseline (S2 L2A), SAR backscatter (S1 VV), Water expansion mask, D8 + OSM corridor, OSM critical assets.
- **Right dock (280px):** Asset State legend (SAFE / BUFFERED / INUNDATED) + selected-asset detail card.
- **Bottom control:** Before/after swipe compare OR opacity slider.
- **Top overlay:** Observation badge (e.g., "OBS 03 | 2026-08-12"), coordinate readout, [SWIPE COMPARE] button.
- **Asset markers:** Green (safe) / Amber (buffered) / Red (inundated).

**Demo-critical beat:** The swipe reveal shows the +28% SAR water surge cutting through clouds.

### View 2: TimelineView

Observation sequence & run controller:

- **Simulation controller:** [▶ Run Simulation] button. Starts in the **before** state (baseline only, all assets green). Clicking advances through observations to the disaster day.
- **Weather-adaptive router strip:** Shows optical→SAR switch (e.g., "Optical cloud 95% → SWITCHED TO SAR PATH").
- **Observation cards:** Horizontal cards showing date, sensor badge, cloud %, 24h/7d rain, water area, % change, severity chip. Severity chip color-coded (safe=green, advisory=amber, elevated=orange, critical=red).
- **Prevention callout:** After the run, highlights "12 days of warning between Obs 1 (+8%) and Obs 2 (+28%)".

### View 3: ReviewView

Human-in-the-loop coordinator console with Simple/Advanced mode toggle:

- **Simple (Triage) mode:** Satellite-first triage card with early warning banner ("★ Early warning 12 days"), heatmap, contamination event details, chlorine logistics formula (1,240 × 2 tablets/day × 14 days = 34,720 tablets), and read-only SOS protocol checklist.
- **Advanced (Analyst) mode:** Full evidence panel (before/after rasters with object-cover, change overlay, swipe compare), risk gauges (H / E / D_risk / confidence), ≥3 deterministic reasons, disease prevention action sheet, and exposed-infrastructure table.
- **Escalation policy badge:** Shows when review is pending (elevated/critical, no decision). Communicates two-tier routing: first responders get advisory, public broadcast held for human gate.
- **Early warning banner:** Surfaces the 12-day lead time from the satellite timeline (obs-001 → obs-003).
- **Decision bar (sticky bottom):** [✓ Confirm SOS] (green, primary) / [✗ Reject] (red) / [⏸ Postpone] (amber). Requires a confirmation state before Confirm fires.
- **Auto-SOS on CONFIRM:** Clicking CONFIRM fires a real ntfy.sh push notification automatically (when online). Toast confirms "Decision confirmed — SOS sent to phone".

### View 4: AuditView

Lineage & resilient alerting:

- **AIR-GAP VERIFIED badge:** Green badge in header indicating offline-first operation.
- **Export dropdown:** Ledger JSON export + SitRep TXT export (field situation report).
- **Payload box:** Raw compressed JSON + byte counter (must show ≤250). Green badge when compliant.
- **Channel simulator:** SMS / LoRa Mesh / Satellite with live status badges (QUEUED → TRANSMITTING → DELIVERED). SMS fires a real ntfy.sh push when online. LoRa and Satellite are simulated state machines.
- **RF telemetry specs:** LoRa (868.1 MHz ISM, SF9, 125 kHz, 222 bytes max) and Iridium SBD (1621 MHz L-Band, 340 bytes/SBD).
- **First Responder Advisory row:** Pre-confirmation advisory row (amber border, "simulated" hash) showing "Hospitals, Firefighters, SAR teams notified (standby)". Disappears after confirmation.
- **Audit trail:** Append-only table — timestamp, actor, action, JSON detail, hash. Real SHA-256 hashes (not placeholders).
- **Verify Chain:** Web Crypto API verification modal. Recomputes SHA-256 for each entry and checks the chain. Shows "ALL 3 BLOCKS CRYPTOGRAPHICALLY LINKED" + "0 TAMPERING DETECTED".
- **Secondary SEND TO PHONE:** Manual ntfy.sh send button (in addition to auto-fire on CONFIRM in ReviewView).

### Keyboard shortcuts

| Key | Action |
|---|---|
| 1–4 | Switch tabs (Map, Timeline, Review, Audit) |
| R | Run simulation (from Timeline tab) |
| Esc | Close toast/modal |

---

## 21. Demo Scenario — The 36-Hour Script

The demo is framed as a **retrospective reconstruction**: "what would SIREN have caught, and how could it have prevented the disaster?"

### The observation sequence

| Observation | Date | Sensor | Cloud | Rain 24h | Expansion | Severity | Story |
|---|---|---|---|---|---|---|---|
| **Baseline** | 2025-11-22 | S2 Optical | 5% | 0.0 mm | — | — | Clear post-monsoon baseline |
| **obs-001** | 2026-07-23 | S1 SAR | 0% eff | 18.2 mm | +8% | Watch | Early warning sign |
| **obs-002** | 2026-08-04 | S1 SAR | 0% eff (95% optical) | 84.6 mm | +28% | Critical | Disaster day trigger |
| **obs-003** | 2026-08-12 | S1 SAR | 0% eff (90% optical) | 60.0 mm | +43% | Critical | Peak expansion |

### The prevention story

The +8% expansion on 07-23 was the **early warning**. Had SIREN been monitoring in real time, the watch would have escalated to a critical alert **20 days before the peak** (08-12), buying lead time to evacuate.

### Demo script (step by step)

1. **Before state:** Dudh Koshi basin loaded in its normal state — clear post-monsoon optical baseline (2025-11-22), normal glacial-lake boundary, intact access roads, all assets green (safe).
2. **Click "Simulation":** The console advances to the disaster window. The optical scene is 95% cloud-blocked (monsoon); the **Weather-Adaptive Router** switches to the Sentinel-1 SAR path.
3. **Observation 1 (2026-07-23):** SAR pass reveals small supraglacial pond expansion (+8% area); rainfall 18.2 mm → **Watch**. SIREN logs a watch.
4. **Observation 2 (2026-08-04):** SAR reveals moraine shift and rapid water expansion (+28% area); 24h rainfall 84.6 mm → **Critical**. *This is the disaster-day trigger.*
5. **Observation 3 (2026-08-12):** SAR reveals continued peak expansion (+43% area); 24h rainfall 60.0 mm → **Critical**. *This is the peak.*
6. **The prevention story:** The console shows that the +8% expansion on 07-23 was the early warning — had SIREN been monitoring in real time, the watch would have escalated to a critical alert 20 days before the peak (08-12), buying lead time to evacuate.
7. **Trigger & review:** System raises an **Elevated/Critical** review card, highlighting the combined D8 + OSM downstream corridor, 2 flagged villages (**Benkar**, **Jorsale**), 1 critical suspension bridge (**Hillary Bridge**), and 3 primary drinking wells along the Dudh Koshi corridor.
8. **Coordinator action:** Presenter inspects the evidence panel and the Disease Prevention Action Sheet, then clicks **Confirm SOS** — **the coordinator's phone receives an SOS push notification automatically via ntfy.sh**.
9. **Dispatch & response:** System shows the simulated geofenced compressed-payload dispatch (Track 7.ii) alongside the water/medical distribution manifest (Track 7.iii); the audit panel records reviewer, decision, and timestamp. The SHA-256 hash chain can be verified in-browser via the Verify Chain modal.

### Closing line for judges

> "SIREN doesn't replace emergency authorities — it buys them the lead time to identify who to rescue, how to reach them when networks are down, and how to stop the outbreak that follows the flood. This demo shows the 20 days of warning we could have had."

---

## 22. API Endpoints — Complete Reference

Base URL: `http://localhost:8010` · Frontend proxy: `/api` → `http://localhost:8010`

### Conventions

- All responses are JSON.
- Errors: `{"error": "<type>", "detail": "<message>"}` with appropriate HTTP status.
- Timestamps: ISO-8601 UTC strings.
- Geometry: GeoJSON (Polygon / Point / LineString / MultiLineString).
- IDs: `observation_id` (`obs-*`), `run_id` (`run-*`), `alert_id` (`alert-*`), `asset_id` (`BR-*`, `RD-*`, `village-*`, `well-*`).

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/basin` | Active basin configuration + basemap metadata |
| GET | `/observations` | List all observations (newest first) |
| GET | `/observations/{id}` | Single observation |
| POST | `/runs` | Trigger pipeline for one observation (synchronous) |
| POST | `/runs/process-all` | Run all demo observations in sequence |
| GET | `/runs` | List all runs with scores |
| GET | `/runs/{id}` | Single run with score, exposures, decision |
| GET | `/runs/{id}/exposures` | Exposed assets for a run |
| GET | `/runs/{id}/sar-priority` | Search & Rescue priority ranking |
| GET | `/runs/{id}/ml-evidence` | ML change detection evidence layer |
| POST | `/runs/{id}/review` | Human decision (confirm/reject/postpone) |
| POST | `/runs/{id}/dispatch` | Dispatch alert (requires prior confirm, 409 otherwise) |
| GET | `/audit?alert_id={id}&run_id={id}` | Audit lineage (filter by alert_id and/or run_id) |
| GET | `/data/processed/{file}` | Static raster/PNG file access |
| GET | `/data/map-assets/dem-hillshade.png` | DEM hillshade raster |
| GET | `/data/map-assets/sar-backscatter.png` | SAR backscatter raster |
| GET | `/data/map-assets/{obs_id}/baseline-optical.png` | Per-observation baseline optical crop |

### Invariants (enforced by code + tests)

- `POST /runs/{run_id}/dispatch` returns 409 if no `confirm` review exists (human gate).
- `payload_bytes` ≤ 250 always (enforced by unit test).
- `reasons` has ≥ 3 entries when `severity` is `elevated` or `critical`.
- Audit entries are append-only; no update/delete endpoints exist.
- Audit `event_hash` = `sha256(prev_hash + timestamp + payload)`; genesis `prev_hash` = `"0" * 64`.
- `optical_cloud_fraction` ≥ 0.20 routes to SAR-primary; `cloud_fraction` is set to 0.0 on the SAR path.
- Severity thresholds: expansion ≥40% → critical, ≥20% → elevated, ≥5% → watch, <5% → informational.
- ML evidence endpoint always returns a result — deterministic fallback when torch is unavailable.

---

## 23. Data Contracts — Exact Schemas

### Quality Verdict (PRD §9.1)

```json
{
  "quality_score": 0.88,
  "cloud_fraction": 0.11,
  "alignment_ok": true,
  "usable": true,
  "confidence_adjustment": 0.95
}
```

### Observation (PRD §10.2)

```json
{
  "observation_id": "obs-003",
  "basin_id": "dudh-koshi-demo-01",
  "acquired_at": "2026-08-12T12:00:00Z",
  "source": "sentinel-1-grd-nrt",
  "raster_uri": "data/processed/obs-003.tif",
  "crs": "EPSG:4326",
  "quality_score": 0.88,
  "cloud_fraction": 0.0,
  "optical_cloud_fraction": 0.90,
  "alignment_ok": true,
  "usable": true,
  "confidence_adjustment": 0.95,
  "water_area_km2": 4.3,
  "water_area_change_percent": 43.0,
  "rainfall_24h_mm": 60.0,
  "rainfall_7d_mm": 160.0,
  "mean_slope_degrees": 31.0,
  "processing_version": "0.1.0",
  "status": "processed"
}
```

### Alert (PRD §10.3)

```json
{
  "alert_id": "alert-0091",
  "geofence_id": "sector-b",
  "severity": "HIGH",
  "hazard_type": "possible_flood_or_debris_flow",
  "confidence": 0.76,
  "exposed_population": 1248,
  "critical_assets": ["bridge-12", "road-4", "village-2"],
  "disease_flags": ["well-3-submerged", "well-7-encircled"],
  "recommended_action": "Verify locally and prepare downstream warning",
  "human_review_required": true
}
```

### Compressed Payload (PRD §10.4)

```json
{"aid":"siren-04","sec":"B","haz":"GLOF_FL","lvl":3,"exp_pop":1240,"crit":["BR-12","RD-4"],"med_act":"BOIL_WATER_NOW"}
```

### Score (from API)

```json
{
  "hazard_score": 0.82,
  "exposure_priority": 0.68,
  "disease_risk": 0.45,
  "confidence": 0.88,
  "severity": "critical",
  "reasons": ["reason 1", "reason 2", "reason 3"]
}
```

### Audit Entry

```json
{
  "entry_id": 1,
  "alert_id": "alert-0091",
  "run_id": "run-0004",
  "actor": "pipeline",
  "action": "run",
  "detail_json": "{}",
  "prev_hash": "0000000000000000000000000000000000000000000000000000000000000000",
  "event_hash": "8475547e1039023e...",
  "created_at": "2026-08-13T04:10:33Z"
}
```

---

## 24. Architecture Decision Records (ADRs)

### ADR-001: SQLite over PostGIS

**Decision:** Use SQLite with JSON columns for persistence, and GeoJSON files on disk for geometry. All spatial joins run in-memory via geopandas.

**Why:** Zero setup, no server process, offline-safe, trivially reproducible (single file). At hackathon scale, the basin extract is small enough that in-memory geopandas joins are effectively instant.

**Tradeoff:** No native spatial indexing. Not suitable for multi-basin national scale. Migration path to PostGIS is mechanical (schema is deliberately PostGIS-shaped).

### ADR-002: Deterministic-First (No Trained ML in the Critical Path)

**Decision:** The MVP critical path is rule-based and deterministic. Trained models are a stretch goal gated on the core loop working.

**Why:** Every score is explainable (deterministic `reasons`), reproducible (same inputs + version → same output), and the demo never depends on a model that might not train in time.

**Addendum:** ML path implemented as optional evidence layer (`ml/`). Deterministic fallback runs without torch. When torch is available, ML produces additional evidence via `GET /runs/{id}/ml-evidence`. Never the sole source of a score.

### ADR-003: SAR-First, Weather-Adaptive Change Detection

**Decision:** The pipeline is weather-adaptive. If optical cloud fraction ≥ 0.20, SAR backscatter differencing is promoted to the primary change-detection path.

**Why:** Monsoon cloud cover is the expected case in the Himalayas, not the edge case. A system that goes blind in the rain is not operationally useful. SAR is all-weather capable.

### ADR-004: Offline-First Demo

**Decision:** Zero network calls at runtime. All data loads from `data/`. Live API ingestion exists only as bonus scripts, never as a runtime dependency.

**Why:** Reliability of the one-click demo chain is the #1 acceptance target. A live-API dependency converts a demo risk into a network risk. Prepared data is the standard, defensible choice for a 36-hour build.

### ADR-005: Combined D8 + OSM River Buffering Corridor

**Decision:** Use a combined D8 + OSM river buffering corridor instead of pure D8.

**Why:** A raw D8 path missed all settlements on the real SRTM DEM (nearest was 2.8 km away). At 30 m resolution in steep Himalayan terrain, DEMs suffer from drainage-trenching artifacts. OSM rivers capture the real, surveyed riverbed through inhabited valleys. Combining D8 (physical validation) + OSM (operational accuracy) reliably catches the demo assets.

---

## 25. Build Roadmap — Phases 0–7

### Phase 0 — Pre-Event Prep (before the clock starts)

- Lock the basin (Dudh Koshi/Imja, Nepal).
- Download all data to disk (Sentinel-1, Sentinel-2, SRTM, OSM, weather).
- Verify data opens in rasterio/geopandas.
- Environment setup (Python 3.11+, Node 18+).
- Repo scaffold.

**Exit criteria:** Every dataset loads locally; D8 produces a flow-accumulation raster in a 30-second smoke test.

### Phase 1 — Foundation (Hours 0–4)

- SQLite schema definition.
- Ingest script: load baseline + assets into `data/processed/`.
- Frontend shell: MapLibre + basin polygon + baseline raster overlay.
- API skeleton: `GET /basin`, `GET /observations`, `POST /runs`.

**Checkpoint (hour 4):** Map shows the basin.

### Phase 2 — Change Detection Core (Hours 4–10)

- Quality gate module.
- Optical NDWI differencing.
- SAR backscatter ratio thresholding.
- Weather-adaptive router.
- Change stats computation.
- Timeline UI.

**Checkpoint (hour 10):** +8% and +28% expansion numbers are computable from real masks.

### Phase 3 — Corridor, Exposure & Scores (Hours 10–16)

- Combined D8 + OSM corridor.
- Tolerance-buffer intersections.
- Hazard score H (5-factor weighted).
- Exposure priority E.
- Map overlays.

**Checkpoint (hour 16):** "2 villages, 1 bridge, 3 wells" emerges from real geometry.

### Phase 4 — Disease Layer + Review Console (Hours 16–22)

- D_risk index.
- Disease Prevention Action Sheet generator.
- Review panel UI.
- Policy engine (severity classification).

**Checkpoint (hour 22):** Review card tells the full story without a presenter narrating.

### Phase 5 — Decision Loop: Dispatch + Audit (Hours 22–28)

- Confirm/Reject/Postpone workflow.
- ≤250-byte payload codec.
- Simulated dispatch.
- Append-only audit log.
- Audit & dispatch panel UI.

**Checkpoint (hour 28):** Confirm → dispatch → audit works. **This is the spine's last node.**

### Phase 6 — Demo Wiring (Hours 28–32)

- Wire 3-observation sequence end-to-end.
- Evidence explanation panel (≥3 reasons on elevated+).
- Stretch goal: SAR Priority Layer ✅.
- ML evidence layer ✅.
- SHA-256 audit hash chain ✅.
- Map asset endpoints ✅.
- Tailwind operational console ✅.
- Docker deployment ✅.

### Phase 7 — Rehearsal & Hardening (Hours 32–36)

- Full offline rehearsal (airplane mode).
- Backup video.
- Known-limitations doc.
- Pitch pass.
- Freeze: tag the demo commit.

---

## 26. Testing — 104 Tests Explained

```bash
cd backend && pytest  # 104 tests, ~20s
```

| Test Suite | Tests | Coverage |
|---|---|---|
| `test_quality` | 11 | Quality gate (PRD §9.1): cloud fraction, alignment, confidence multiplier, SAR routing |
| `test_codec` | 12 | Payload codec: ≤250 bytes, round-trip, `siren-` prefix, oversize rejection |
| `test_audit` | 11 | Append-only enforcement, hash chain verification, SQLite trigger validation |
| `test_api` | 12 | All API endpoints, human gate (409 without confirm), error shapes |
| `test_preprocess` | 6 | Clip, reproject, co-register on synthetic 100×100 GeoTIFFs |
| `test_ingest` | 25 | CLI argument parsing, provenance sidecars, offline-safe error handling |
| `test_pipeline` | 5 | Full orchestrator: detect→geo→risk→DB→audit (DoD chain) |
| `test_ml` | 13 | ML evidence layer: deterministic fallback (10 tests) + torch-gated (3 tests) |
| `test_sar_priority` | 9 | SAR priority ranking: population × access-loss scoring |

**Total:** 104 tests (101 active + 3 torch-gated). All passing.

### Test fixtures

- `tests/fixtures/rasters/baseline.tif` — 100×100, 100 water pixels.
- `tests/fixtures/rasters/expanded_water.tif` — 128 water pixels = **+28%** expansion.
- `tests/fixtures/rasters/cloudy_optical.tif` — 2-band optical, cloud_fraction **0.25**.
- `tests/fixtures/osm/fake_assets.geojson` — 2 villages, 1 bridge, 3 wells, 1 road line.

---

## 27. Definition of Done

The MVP is done when, **offline**, in one click-chain:

1. ✅ Baseline loads (`GET /basin` returns Dudh Koshi/Imja with real AOI polygon).
2. ✅ 3 observations process (`POST /runs/process-all` — obs-001: watch, obs-002: critical, obs-003: critical).
3. ✅ Elevated/critical card with ≥3 evidence factors (obs-002: 8 reasons, obs-003: 8 reasons).
4. ✅ Human confirm (`POST /runs/{id}/review` with `decision=confirm`).
5. ✅ ≤250-byte dispatch (118 bytes actual — `POST /runs/{id}/dispatch`).
6. ✅ Audit lineage reconstructable with SHA-256 hash chain (`GET /audit?run_id=...`).
7. ✅ 104/104 tests passing (101 active + 3 torch-gated).

**Everything else is negotiable; that chain is not.**

---

## 28. Known Limitations

### What's deterministic (real code, real data)

- Change detection: NDWI + SAR backscatter ratio thresholding on actual downloaded scenes.
- D8 flow corridor: pysheds on real SRTM 1-arc-second DEM.
- Tolerance-buffer intersections: computed against real OSM extract (1100 features).
- Risk fusion: fixed PRD §9.5 weights. Same inputs → identical outputs. No unseeded randomness.
- Payload codec: real encoder/decoder with round-trip tests. 118 bytes actual.
- Audit log: append-only, enforced by SQLite triggers. SHA-256 hash chain.
- Human gate: no dispatch without `confirm` review. Enforced at DB layer.
- SAR priority ranking: real code on real OSM exposure data.
- ML evidence layer: deterministic fallback when torch unavailable.

### What's simulated (not real at runtime)

- Alert channels — LoRa and Satellite: Simulated state machines (QUEUED → TRANSMITTING → DELIVERED). No real radio or Iridium modem transmission.
- Alert channel — SMS (partially live): SMS is the only live integration via ntfy.sh. Clicking CONFIRM in ReviewView or SEND TO PHONE in AuditView fires a real push notification when online. When offline, the dispatch is simulated.
- First Responder Advisory: Simulated visual in AuditView. No real pre-confirmation notification is sent to hospitals or fire crews.
- Escalation policy badge: Informational only. No auto-escalation dispatch fires without human confirmation.
- Live satellite ingestion: All scenes pre-downloaded. Zero network calls for pipeline data at runtime.
- Weather data: Prepared JSON file, not live API call.
- Synchronous pipeline: Full chain runs in the request. No background task queue.
- Single reviewer: Hardcoded `coordinator-01`. No authentication or RBAC.

### Known data gaps

- **Sentinel-1 swath coverage:** The available ascending-orbit S1 pair covers only the western AOI; the Imja lake proper (86.925°E) is outside the swath. Demo uses prepared scenario masks near Imja (clearly labeled in `detect/scenario.py`).
- **Single basin:** Only Dudh Koshi / Imja is configured. Multi-basin is V2.
- **No ground-truth validation set:** Change masks not validated against a held-out labeled flood dataset.

### Latency realities

- Pipeline processing: ~2–4 seconds per observation (Docker, single core). Production target: <30 seconds.
- Dispatch latency: Simulated as instant. Real LoRa: 30–120s; satellite SBD: 1–5 min; SMS: depends on tower availability.
- Review latency: Depends on the human coordinator. The escalation policy badge communicates the two-tier concept, but no auto-escalation dispatch fires without human confirmation (Hard Rule #3). The ntfy.sh push on CONFIRM is a side-effect of the human decision.

### What SIREN does NOT do (PRD §14)

- Predict exact time of glacial-lake collapse.
- Guarantee exact flood depth or flow path.
- Diagnose specific diseases from satellite imagery.
- Issue autonomous evacuation orders.
- Guarantee delivery to every person in an area.
- Replace government early-warning systems.
- Identify individuals, locate trapped persons, or handle family reunification.

---

## 29. Future Roadmap (V1–V4)

### V1 — Hackathon MVP (completed)

Prepared SAR/optical image sequence, backscatter/NDWI change detection, rainfall context, terrain and exposure overlays, disease-risk index, explainable hazard score, human confirmation, simulated resilient dispatch, audit log. Stretch: SAR Priority Layer.

### V2 — Operational prototype

- Full Copernicus API ingestion.
- Automated quality checks.
- Live weather ingestion.
- Field-report validation.
- Role-based authentication.
- PostGIS migration.
- Approved alert-channel integration.
- Productionized SAR Priority Layer.
- Two-way alert acknowledgment (delivery confirmation + recipient check-in over constrained links).
- Evacuation-route optimization to shelters around severed road segments.

### V3 — Research system

- Fine-tuned Siamese U-Net / ChangeFormer + SegFormer.
- ConvLSTM / temporal Transformer trend modeling.
- Calibrated risk fusion with uncertainty estimation.
- Regional transfer testing.
- Hydrodynamic flow modeling.
- Post-event damage assessment.
- What-if scenario simulation (partial lake-release planning mode).
- UAV/drone tasking for high-resolution local verification.
- Landslide-susceptibility modeling.

### V4 — Resilience platform

- Integration with river gauges, local field reports, telecom status feeds.
- Food/water logistics, shelter capacity.
- Multilingual alert templates.
- Offline field applications.
- Cross-border basin coordination.
- Multi-basin national portfolio dashboard.
- **Area i expansion — personnel identification and family reunification** (missing-persons registry, survivor tracking, reunification workflow), built only in partnership with approved disaster-management authorities and subject to privacy constraints.

---

## 30. Quick Start — Running the Demo

### Option A: Docker (recommended)

```bash
./start.sh
```

That's it. The app is at `http://localhost:5175`. The script kills any stale processes on ports 8010/5175, then runs `docker compose up -d --build`.

- Backend: Python 3.12 + GDAL + rasterio + geopandas (port 8010)
- Frontend: nginx serving the Vite production build (port 5175)
- Data: `./data` is volume-mounted

```bash
# Stop
docker compose down

# Rebuild after code changes
docker compose up --build

# View logs
docker compose logs -f
```

### Option B: Local development

**Backend:**
```bash
cd backend
pip install -e ".[dev]"
uvicorn siren.api:app --port 8010 --reload
pytest  # 104 tests
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev  # serves on http://localhost:5175, proxies /api → :8010
```

### Running the demo

1. Start the backend on port 8010.
2. Start the frontend on port 5175.
3. Open `http://localhost:5175`.
4. Click the **Timeline** tab.
5. Click **Run Simulation** (or press `R`).
6. Watch three observations process through the pipeline.
7. An alert banner appears — click it to open the **Review** tab.
8. Inspect the evidence, scores, and disease-prevention actions.
9. Click **Confirm SOS** → **Yes, confirm** — **your phone receives an SOS push notification automatically** (install the ntfy app and subscribe to topic `siren-emergency-alert`).
10. Go to the **Audit** tab to see the dispatch payload, audit trail, and verify the SHA-256 hash chain.

---

## 31. Glossary of Terms

| Term | Definition |
|---|---|
| **SIREN** | Satellite-Informed Risk & Emergency Network |
| **GLOF** | Glacial Lake Outburst Flood |
| **SAR** | Synthetic Aperture Radar (Sentinel-1) |
| **NDWI** | Normalized Difference Water Index (optical water detection) |
| **D8** | D8 flow accumulation algorithm (hydrological flow routing) |
| **OSM** | OpenStreetMap (open-source geographic database) |
| **CDSE** | Copernicus Data Space Ecosystem (Sentinel data portal) |
| **STAC** | SpatioTemporal Asset Catalog (API for searching satellite data) |
| **GRD** | Ground Range Detected (Sentinel-1 product type) |
| **L2A** | Level-2A (Sentinel-2 surface reflectance product) |
| **IW** | Interferometric Wide swath (Sentinel-1 acquisition mode) |
| **VV/VH** | Dual polarization (vertical transmit/vertical receive, vertical transmit/horizontal receive) |
| **SRTM** | Shuttle Radar Topography Mission (30 m DEM) |
| **IMERG** | Integrated Multi-satellitE Retrievals for GPM (rainfall product) |
| **AOI** | Area of Interest (the basin boundary polygon) |
| **CRS** | Coordinate Reference System (e.g., EPSG:4326 = WGS 84) |
| **COG** | Cloud Optimized GeoTIFF |
| **LoRa** | Long Range wireless communication (low-bandwidth mesh network) |
| **SBD** | Satellite Broadband (satellite messaging) |
| **DoD** | Definition of Done |
| **ADR** | Architecture Decision Record |
| **PRD** | Product Requirements Document |
| **RBAC** | Role-Based Access Control |
| **NRT** | Near Real-Time (Sentinel-1 data delivery latency) |
| **IoU** | Intersection over Union (segmentation evaluation metric) |
| **ConvLSTM** | Convolutional Long Short-Term Memory (temporal deep learning) |
| **Siamese U-Net** | Shared-encoder neural network for change detection |
| **ChangeFormer** | Transformer-based change detection model |
| **SegFormer** | Segmentation Transformer (land-cover classification) |
| **ntfy.sh** | Free push notification service used for live SOS alerts (topic: `siren-emergency-alert`) |

---

*This document is the single source of truth for understanding SIREN. For implementation details, refer to the source code. For design decisions, refer to the ADRs. For the build plan, refer to BUILD_ROADMAP.md. For the product spec, refer to PRD.md.*
