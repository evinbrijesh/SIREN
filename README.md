# SIREN — Satellite-Informed Risk & Emergency Network

**Human-in-the-loop, satellite-assisted early-warning and disaster-response platform for Himalayan basins.**

Track 7: *Living with Uncertainties, Building with Resilience*
- Area ii: Communication Systems During Disasters for Effective Response
- Area iii: Curbing Diseases That Arise During Disasters

> What changed? How serious is it? Who and what are in the path? What should responders do right now?

---

## What SIREN Does

SIREN fuses Sentinel-1 SAR and Sentinel-2 optical imagery with rainfall, terrain, river, population, and infrastructure data to model hazard progression and downstream exposure. It surfaces evidence to an authorized emergency coordinator through an explainable review console and — only after human confirmation — dispatches a geofenced, bandwidth-light alert alongside a disease-prevention action sheet.

The offline demo runs a retrospective "what-if" prevention scenario for the **Dudh Koshi / Imja glacial basin, Nepal**:

1. Baseline loads (clear post-monsoon optical scene, 2025-11-22)
2. Three observations process through the pipeline (2026-07-23, 2026-08-04, 2026-08-12)
3. An elevated/critical review card appears with ≥3 evidence reasons
4. The coordinator confirms the alert — **an SOS push notification is sent to their phone automatically via ntfy.sh**
5. A compressed dispatch payload (≤250 bytes) is sent
6. The audit log reconstructs the full lineage with SHA-256 hash chain verification

---

## Architecture

```
Sentinel-1 SAR / Sentinel-2 Optical / SRTM / GPM IMERG / OSM
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

### Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, FastAPI, Pydantic, SQLite |
| Geospatial | rasterio, geopandas, shapely, numpy, pysheds |
| Frontend | React, Vite, TypeScript, Tailwind CSS, MapLibre GL JS, TanStack Query |
| Storage (demo) | SQLite + GeoJSON + GeoTIFF on disk (no PostGIS, no Redis) |
| Storage (hosted, planned) | PostgreSQL/RDS + PostGIS, S3 object storage (ADR-007) |
| Deployment (demo) | Docker Compose (one-command: backend + frontend) |
| Deployment (hosted, planned) | ECS Fargate + ALB + EventBridge + SQS (ADR-006/008) |

---

## Repository Structure

```
backend/
  siren/
    api/          # FastAPI routes + Pydantic models + map asset endpoints
    ingest/       # CDSE STAC, SRTM, IMERG, Open-Meteo, Overpass downloaders
    preprocess/   # clip, reproject, co-register, cloud mask, quality gate, SAR calibration
    detect/       # NDWI, SAR backscatter, weather-adaptive router, scenario masks
    geo/          # D8 corridor, tolerance buffers, exposure intersections
    risk/         # hazard H, exposure E, disease D_risk, SAR priority scoring + reasons
    ml/           # WaterUNet shadow evidence layer (trained on Sen1Floods11; shadow-only per ADR-010)
    alerting/     # ≤250-byte payload codec, validator
    audit/        # append-only log writer + SHA-256 hash chain
    db/           # SQLite schema + repositories
    pipeline.py   # orchestrator: detect→geo→risk→DB→audit
  tests/          # 124 tests (pytest: 121 active + 3 torch-gated)
frontend/
  src/
    views/        # MapView, TimelineView, ReviewView, AuditView
    api/          # typed API client + offline mock fallback
    simulation/   # SimulationContext (shared demo state)
    components/   # OfflineBadge (online/offline event listener)
    theme/        # ThemeProvider + ThemeToggle (Ops Dark / Professional Light / Satellite)
    utils/        # ntfy.ts — shared ntfy.sh live alert utility
    index.css     # Tailwind base + design tokens
data/
  raw/            # downloaded scenes (gitignored)
  processed/      # masks, aligned rasters (gitignored, pipeline-written)
  assets/         # basin GeoJSON, OSM extracts, weather series (committed)
docs/
  spec/
    PRD.md           # Product Requirements Document (v4.5)
    BUILD_ROADMAP.md # 36-hour build plan + live service transition roadmap
    API_CONTRACT.md  # HTTP API surface
    DEVIN_BRIEFS.md  # Devin task dispatch briefs (D1-D8, archived)
  design/
    UI_DESIGN.md     # Coordinator console design spec
  adr/
    ADR-001..005     # Architecture decision records (hackathon, accepted)
    ADR-006..009     # Architecture decision records (live service, proposed)
  reference/
    KNOWN_LIMITATIONS.md  # Demo limitations + production transition gaps
Dockerfile.backend  # Backend image (Python + GDAL + geospatial stack)
Dockerfile.frontend # Frontend image (Node build → nginx serve)
docker-compose.yml  # One-command orchestration
start.sh            # Port-conflict-aware launcher (kills stale processes, starts Docker)
```

---

## Quick Start

### Option A: Docker (recommended for demo/presentation)

**Prerequisites:** Docker + Docker Compose installed. The `data/` directory must exist locally with the demo datasets.

```bash
./start.sh
```

That's it. The app is at `http://localhost:5175`. (The script kills any stale processes on ports 8010/5175 first, then runs `docker compose up -d --build`.)

- Backend: Python 3.12 + GDAL + rasterio + geopandas (port 8010)
- Frontend: nginx serving the Vite production build (port 5175)
- Data: `./data` is volume-mounted (rasters, assets, SQLite DB persistence)
- nginx proxies `/api/*` and `/data/*` to the backend container

```bash
# Stop
docker compose down

# Rebuild after code changes
docker compose up --build

# View logs
docker compose logs -f
```

### Option B: Local development

**Prerequisites:**

- Python 3.11+ (3.14 works but pyproject.toml restricts to `<3.13` for editable install; pytest runs directly)
- Node.js 18+
- The demo data in `data/` (Sentinel-1 pair, Sentinel-2 baseline, SRTM DEM, OSM extract)

### Backend

```bash
cd backend
pip install -e ".[dev]"          # or use existing venv
uvicorn siren.api:app --port 8010 --reload

# run tests
pytest                           # 124 tests, ~10s
```

### Frontend

```bash
cd frontend
npm install
npm run dev                      # serves on http://localhost:5175, proxies /api → :8010

# production build
npm run build                    # tsc + vite build
```

### Running the Demo

1. Start the backend on port 8010
2. Start the frontend on port 5175
3. Open `http://localhost:5175` in your browser
4. Click the **Timeline** tab
5. Click **Run Simulation** (or press `R`)
6. Watch three observations process through the pipeline
7. An alert banner appears — click it to open the **Review** tab
8. Inspect the evidence, scores, and disease-prevention actions
9. Click **Confirm SOS** → **Yes, confirm** — **your phone receives an SOS push notification automatically** (install the [ntfy app](https://ntfy.sh) and subscribe to topic `siren-emergency-alert`)
10. Go to the **Audit** tab to see the dispatch payload, audit trail, and verify the SHA-256 hash chain

> **Live phone alerts:** When online, clicking CONFIRM fires a real ntfy.sh push notification. When offline (air-gap mode), the dispatch is simulated. The Audit tab also has a secondary **SEND TO PHONE** button for manual re-send.

### Keyboard Shortcuts

| Key | Action |
|---|---|
| 1–4 | Switch tabs (Map, Timeline, Review, Audit) |
| R | Run simulation (from Timeline tab) |
| Esc | Close toast/modal |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/basin` | Active basin configuration + basemap metadata |
| GET | `/observations` | List all observations |
| GET | `/observations/{id}` | Single observation |
| POST | `/runs` | Trigger pipeline for one observation (synchronous) |
| POST | `/runs/process-all` | Run all demo observations in sequence |
| GET | `/runs` | List all runs with scores |
| GET | `/runs/{id}` | Single run with score, exposures, decision |
| GET | `/runs/{id}/exposures` | Exposed assets for a run |
| GET | `/runs/{id}/sar-priority` | Search & Rescue priority ranking (PRD §15) |
| GET | `/runs/{id}/ml-evidence` | ML change detection evidence layer |
| POST | `/runs/{id}/review` | Human decision (confirm/reject/postpone) |
| POST | `/runs/{id}/dispatch` | Dispatch alert (requires prior confirm, 409 otherwise) |
| GET | `/audit?alert_id={id}&run_id={id}` | Audit lineage (filter by alert_id and/or run_id) |
| GET | `/data/processed/{file}` | Static raster/PNG file access |
| GET | `/data/map-assets/dem-hillshade.png` | DEM hillshade raster |
| GET | `/data/map-assets/sar-backscatter.png` | SAR backscatter raster |
| GET | `/data/map-assets/{obs_id}/baseline-optical.png` | Per-observation baseline optical crop |

See `docs/spec/API_CONTRACT.md` for full request/response schemas.

---

## Key Design Decisions

### Hackathon MVP (ADR-001 → ADR-005, Accepted)

- **Deterministic-first.** No trained ML in the critical path. Rule-based masks and weighted scores (ADR-002). *Audit note (2026-09-07): the current implementation diverges — a 0.20-weight ML term sits inside H and the trend class can be model-replaced; ADR-010 (Proposed) restores compliance.*
- **Offline demo.** Zero network calls at runtime for pipeline data. All data loads from `data/` (ADR-004). The only live network call is the ntfy.sh phone push on CONFIRM, gated by `navigator.onLine`.
- **SAR-first.** Weather-adaptive router switches to SAR when cloud ≥20% (ADR-003).
- **SQLite over PostGIS.** Zero-ops, offline-safe for hackathon scale (ADR-001).
- **Combined D8 + OSM corridor.** D8 validates gravity gradient; OSM rivers capture the real surveyed riverbed (ADR-005).
- **Human gate.** No dispatch without a recorded `confirm` review (enforced by SQLite trigger). The ntfy.sh push on CONFIRM is a side-effect of the human decision, not an autonomous dispatch.
- **≤250-byte payload.** Compact JSON for LoRa mesh / satellite messenger / low-bandwidth SMS.
- **≥3 reasons on elevated+.** Every elevated/critical score carries at least three evidence factors.
- **Two-tier alert routing.** First responders receive an advisory (simulated, pre-confirmation); public broadcast requires human confirmation. An escalation policy badge in ReviewView communicates this clearly.
- **Live phone alerts via ntfy.sh.** Clicking CONFIRM fires a real push notification to the coordinator's phone. A secondary manual SEND TO PHONE button exists in AuditView.

### Live Service Transition (ADR-006 → ADR-009, Proposed)

- **Connected acquisition, isolated execution.** A dedicated acquisition service makes all network calls; the frozen pipeline consumes only locally materialized, verified inputs (ADR-006, supersedes ADR-004 for hosted deployment).
- **PostgreSQL for hosted operation.** SQLite retained for demo; RDS PostgreSQL + PostGIS for multi-instance hosted service with concurrent writers, idempotency constraints, and managed backups (ADR-007, supersedes ADR-001 for hosted deployment).
- **Durable orchestration.** Acquisition job ledger with `UNIQUE(source, provider_product_id)` idempotency; atomic download → verify → publish; immutable input manifests binding each run to exact raster/weather/DEM/OSM versions (ADR-008).
- **Authenticated review and server-side delivery.** OIDC roles separate ingestion workers from coordinators; transactional delivery outbox with provider receipts replaces browser-side ntfy (ADR-009).
- **Frozen pipeline unchanged.** The deterministic quality gate → router → change detection → corridor → risk fusion → DB → audit chain is not modified. What changes is what feeds it and how often it is triggered.

---

## ML Shadow Architecture & Verified Metrics

### Deterministic-first design (ADR-010)

SIREN's load-bearing path is entirely deterministic: SAR log-ratio thresholding, hydrological corridor routing, and a five-factor risk score (weights 0.30/0.25/0.20/0.15/0.10). No trained model influences hazard score, exposure, disease risk, severity, corridor routing, or dispatch eligibility.

WaterUNet (trained on Sen1Floods11) runs as a **shadow evidence layer only** — its output is displayed alongside the deterministic mask for comparison but never enters the load-bearing path. The ML weight in the risk formula is `W_ML = 0.00`.

### WaterUNet runtime tensor contract

The model expects calibrated Sentinel-1 VV/VH sigma0 in decibels, normalized to [0, 1] via `ml/contract.py::normalize_sar()`:

| Channel | Polarization | Value range | Normalization |
|---|---|---|---|
| 0 | VV | [-30, 0] dB (clamped) | Linear map to [0, 1] |
| 1 | VH | [-30, 0] dB (clamped) | Linear map to [0, 1] |

The pipeline extracts real calibrated dB from Sentinel-1 SAFE archives via `preprocess/sar_calibrate.py`:
1. Read VV/VH measurement TIFFs from the SAFE ZIP
2. Parse the ESA calibration LUT (sigmaNought vectors)
3. Apply `sigma0 = DN² / sigmaNought²`
4. Convert to dB: `sigma0_dB = 10 * log10(sigma0)`
5. Cache as 2-band float32 GeoTIFF in `data/processed/`

Verified on real Dudh Koshi scenes: VV mean -12.3 dB, VH mean -18.6 dB (physically realistic for C-band GRD).

### Dual-split evaluation (Sen1Floods11)

| Split | IoU | Precision | Recall | F1 | Verdict |
|---|---|---|---|---|---|
| Official test (90 chips) | **0.671** | 0.773 | 0.836 | 0.803 | Generalizes within ecoregion |
| Event-holdout (flood events unseen during training) | **0.239** | — | — | — | Below 0.65 load-bearing gate → shadow-only |

The event-holdout IoU of 0.24 confirms WaterUNet does not generalize to unseen flood events. This is why it remains shadow-only and the deterministic pipeline is authoritative.

### Compact alert payload

The dispatch codec encodes 7 keys into a **128-byte** JSON payload (well within the 250-byte LoRa/SMS limit):

| Key | Meaning |
|---|---|
| `aid` | Alert ID (`siren-alert-XXXX`) |
| `sec` | Severity level |
| `haz` | Hazard score |
| `lvl` | Risk level |
| `exp_pop` | Exposed population |
| `crit` | Critical assets count |
| `med_act` | Medical action code |

### Real data inventory

| Data | Source | Status |
|---|---|---|
| Sentinel-1 GRD (obs-001) | S1D, 2026-07-23, track 12, ascending | Real SAFE, calibrated VV/VH dB |
| Sentinel-1 GRD (obs-002) | S1D, 2026-08-04, track 12, ascending | Real SAFE, calibrated VV/VH dB |
| Sentinel-1 GRD (obs-003) | S1D, 2026-08-11, track 12, ascending | Real SAFE, calibrated VV/VH dB (downloaded from CDSE 2026-09-08) |
| Sentinel-1 GRD (descending pair) | S1D, 2026-07-02 + 07-14, descending | Real SAFE, covers Imja lake (86.925°E) — outside ascending swath |
| SRTM 1 Arc-Second DEM | Earthdata | Real, 30m resolution |
| OSM infrastructure | Overpass API | Real, 5,691 features (51 bridges, 23 settlements, 3 wells, 1,536 roads, 1 health) |
| Rainfall (all obs) | Open-Meteo ERA5 reanalysis | Real, daily precipitation + temperature |
| WaterUNet weights | Trained on Sen1Floods11 (252 train / 89 valid / 90 test) | Official + event-holdout checkpoints |

---

## Demo Scenario

The demo is a retrospective "what-if" reconstruction of a GLOF (glacial lake outburst flood) event:

| Observation | Date | Sensor | Cloud | Rain 24h | Rain 7d | Expansion | Severity | Story |
|---|---|---|---|---|---|---|---|---|
| Baseline | 2025-11-22 | S2 Optical | 5% | 0.0 mm | 0.0 mm | — | — | Clear post-monsoon baseline |
| obs-001 | 2026-07-23 | S1 SAR (real SAFE) | 0% eff | 3.2 mm | 58.9 mm | +8% | Watch | Early warning sign |
| obs-002 | 2026-08-04 | S1 SAR (real SAFE) | 0% eff (95% optical) | 12.1 mm | 48.5 mm | +28% | Elevated | Disaster day |
| obs-003 | 2026-08-11 | S1 SAR (real SAFE) | 0% eff (90% optical) | 3.7 mm | 61.3 mm | +43% | Critical | Peak expansion |

> **All 3 observations run on real ESA Sentinel-1 GRD archives** downloaded from Copernicus CDSE. Calibrated VV/VH sigma0 dB is extracted from SAFE archives via `preprocess/sar_calibrate.py` (ESA XML calibration LUT → sigma0 → dB). Rainfall values are real ERA5 reanalysis from the Open-Meteo Archive API (no auth required). A descending-pass S1 pair (2026-07-02 + 07-14) provides Imja lake coverage outside the ascending swath.

The prevention story: the +8% expansion on 07-23 was the early warning. Had SIREN been monitoring in real time, the watch would have escalated 20 days before the peak (08-12), buying lead time to evacuate.

---

## Testing

```bash
cd backend
pytest                           # 124 tests, ~10s
```

| Test Suite | Tests | Coverage |
|---|---|---|
| test_quality | 11 | Quality gate (PRD §9.1) |
| test_codec | 12 | Payload codec ≤250 bytes, round-trip |
| test_audit | 14 | Append-only enforcement, hash chain, trigger validation, stored-hash integrity, tamper detection |
| test_api | 13 | All API endpoints, human gate, error shapes, confirm-then-reject suppression |
| test_preprocess | 6 | Clip, reproject, co-register on synthetic rasters |
| test_ingest | 34 | CLI parsing, provenance sidecars, streaming downloads, flat OSM properties, empty-response protection, acquisition jobs, live observation pipeline |
| test_pipeline | 5 | Full orchestrator: detect→geo→risk→DB→audit |
| test_ml | 40 | ML evidence layer (deterministic fallback, torch-gated) |
| test_sar_priority | 9 | SAR priority ranking (PRD §15) |
| test_sar_calibrate | 6 | SAR calibration: sigma0 dB formula, normalize_sar contract, NaN handling |
| test_open_meteo | 8 | Real ERA5 rainfall fetcher: antecedent computation, temp index, mocked API |

> **Pre-existing failure:** `test_ml.py::test_model_registry_metadata_loads` (1 test) fails due to missing Siamese U-Net weight metadata — unrelated to the pipeline. 123/124 tests pass.

---

## Documentation

### Specs

- [`docs/spec/PRD.md`](docs/spec/PRD.md) — Product Requirements Document (v4.6)
- [`docs/spec/BUILD_ROADMAP.md`](docs/spec/BUILD_ROADMAP.md) — 36-hour build plan + live service transition roadmap (Phases 0–6)
- [`docs/spec/API_CONTRACT.md`](docs/spec/API_CONTRACT.md) — HTTP API surface
- [`docs/design/UI_DESIGN.md`](docs/design/UI_DESIGN.md) — Coordinator console design spec
- [`docs/spec/DEVIN_BRIEFS.md`](docs/spec/DEVIN_BRIEFS.md) — Devin task dispatch briefs (archived)

### Architecture Decision Records

| ADR | Status | Topic |
|---|---|---|
| [ADR-001](docs/adr/ADR-001-sqlite-over-postgis.md) | Accepted (demo) | SQLite over PostGIS |
| [ADR-002](docs/adr/ADR-002-deterministic-first-ml.md) | Accepted | Deterministic-first, ML as optional evidence layer |
| [ADR-003](docs/adr/ADR-003-sar-first-weather-adaptive.md) | Accepted | SAR-first, weather-adaptive routing |
| [ADR-004](docs/adr/ADR-004-offline-first-demo.md) | Accepted (demo) | Offline-first demo |
| [ADR-005](docs/adr/ADR-005-combined-d8-osm-corridor.md) | Accepted | Combined D8 + OSM corridor |
| [ADR-006](docs/adr/ADR-006-connected-acquisition-isolated-execution.md) | **Proposed** | Connected acquisition, isolated execution (hosted) |
| [ADR-007](docs/adr/ADR-007-postgresql-operational-persistence.md) | **Proposed** | PostgreSQL operational persistence (hosted) |
| [ADR-008](docs/adr/ADR-008-durable-orchestration-immutable-manifests.md) | **Proposed** | Durable orchestration, job ledger, immutable manifests |
| [ADR-009](docs/adr/ADR-009-authenticated-review-server-side-delivery.md) | **Proposed** | Authenticated review, server-side delivery outbox |
| [ADR-010](docs/adr/ADR-010-ml-evidence-isolation-and-retraining-path.md) | **Proposed** | ML evidence isolation, model verdicts, retraining path |
| [ADR-011](docs/adr/ADR-011-production-multimodal-upgrade.md) | **Accepted** | Production multimodal upgrade: 4-ch tensor, production deps, gated ML load-bearing |

### Reference

- [`docs/reference/KNOWN_LIMITATIONS.md`](docs/reference/KNOWN_LIMITATIONS.md) — Demo limitations + production transition gaps (phase-tagged)
- [`docs/reference/DL_MODEL_AUDIT.md`](docs/reference/DL_MODEL_AUDIT.md) — 2026-09-07 audit of the four PRD-nominated ML models; verdict: no existing checkpoint is qualified for live hazard assessment
- [`docs/reference/PRODUCTION_ML_PLAN.md`](docs/reference/PRODUCTION_ML_PLAN.md) — Recommended production pipeline, models, datasets, and the dual-basin strategy (Imja monitoring + South Lhonak event validation)
- [`docs/spec/V3_RESEARCH_PROPOSAL.md`](docs/spec/V3_RESEARCH_PROPOSAL.md) — V3 research RFC: 4-channel DEM-conditioned segmenter, XGBoost/TreeSHAP breach susceptibility, FNO hydrodynamic surrogate, physics-informed loss (L_gravity), conformal prediction, RTC γ⁰ + DANN domain adaptation.
- [`docs/spec/PRODUCTION_ROADMAP.md`](docs/spec/PRODUCTION_ROADMAP.md) — **Production transition roadmap:** 4-phase plan (unfreeze → ingestion daemon → load-bearing AI → production infrastructure) with 3-sprint execution sequence. Active engineering spec for the post-hackathon production system.

---

## Production Transition

The hackathon MVP (v1.0.0-hackathon-final) is preserved as a frozen release. The project is now transitioning to a production-grade, autonomous disaster-response platform per [ADR-011](docs/adr/ADR-011-production-multimodal-upgrade.md) (Accepted) and the [Production Transition Roadmap](docs/spec/PRODUCTION_ROADMAP.md).

### Two deployment profiles

| Profile | Network | Database | Storage | Auth | Delivery |
|---|---|---|---|---|---|
| **Offline/demo** (v1.0.0, frozen) | Zero runtime calls | SQLite | Local disk | None | Browser-side ntfy |
| **Production** (target) | STAC polling daemon (Celery) | PostgreSQL/PostGIS | S3/MinIO | OIDC + RBAC | Dual-path: SMS + Iridium SBD/LoRa |

### Production transition (4 phases, 3 sprints)

| Phase | Goal | Sprint |
|---|---|---|
| 1 | Unfreeze ADR-010, retire mock artifacts, adopt production deps | Sprint 1 |
| 2 | Autonomous STAC ingestion daemon, COG /vsicurl/, RTC γ⁰ | Sprint 1 |
| 3 | 4-channel WaterResUNet (L_gravity + DANN), HAND exposure, XGBoost+TreeSHAP, FNO surrogate | Sprints 2–3 |
| 4 | PostgreSQL/PostGIS, S3, Celery, hardware dispatch, RFC 3161 audit | Sprint 1 + Sprint 3 |

> **ADR-011 accepted 2026-09-08:** authorizes 4-channel tensor contract, production dependency addendum (psycopg, geoalchemy2, xgboost, shap, celery, redis), and gated ML load-bearing role. Human gate (Hard Rule 3) preserved.

See [`docs/spec/PRODUCTION_ROADMAP.md`](docs/spec/PRODUCTION_ROADMAP.md) for the full 4-phase plan with acceptance criteria, sprint deliverables, and exit gates. The hackathon `BUILD_ROADMAP.md` is preserved as the historical record of v1.0.0-hackathon-final.

### Source-specific polling cadence (planned)

| Source | Recommended interval | Rationale |
|---|---|---|
| Sentinel-1 catalogue | Hourly | Orbit 12 (ascending) and orbit 121 (descending) cover Imja with ~12-day repeat each; hourly metadata checks minimize publication latency |
| Sentinel-2 catalogue | Every 2 hours | ~5-day tile repeat; monsoon cloud cover makes usable observations much less frequent |
| IMERG Early (half-hourly) | Hourly | ~4-hour publication latency; daily polling discards most of the operational value |
| OSM / Overpass | Daily (small area) | Editing is continuous but irregular; frequent polling cannot discover unmapped infrastructure |
| SRTM | Once, then weekly integrity check | Static historical DEM; not a daily observation |

### Rough AWS cost estimate (hosted profile)

~$300–450/month for a single-basin pilot (Multi-AZ RDS, two API tasks, batch workers, ALB, S3, NAT, monitoring). ~$100–200/month for a reduced-availability shadow pilot. See the architecture assessment for the full breakdown.

---

## Known Limitations

### Demo limitations (hackathon scope)

- The ascending-orbit Sentinel-1 pair (relative orbit 85) covers the western AOI; the Imja lake (86.925°E) is outside the ascending swath. A descending-pass pair (2026-07-02 + 07-14) was downloaded to cover Imja, but is not yet wired into the demo pipeline (the 3 demo observations use the ascending track). The SAR pipeline itself is real and validated on the covered region. All 3 demo observations use real calibrated Sentinel-1 VV/VH sigma0 dB from ESA SAFE archives.
- **SAR physics caveat.** C-band SAR penetrates clouds, but the high Himalaya introduces SAR-specific challenges: wet snow on glaciers causes backscatter drops that mimic open water in a simple log-ratio threshold; steep terrain causes layover and shadowing; debris-covered ice alters backscatter unpredictably. The V1 demo uses deterministic scenario masks as the rule-based detection layer and real calibrated SAR feeds the ML shadow layer only. No glacier/snow classification mask is applied. See `docs/reference/KNOWN_LIMITATIONS.md` → "Domain Physics Limitations".
- **Hydrology caveat.** The corridor uses D8 flow accumulation (steady-state steepest descent) plus a static 125 m planar buffer along OSM waterways. A GLOF is a dynamic dam-break wave that superelevates in canyon bends far beyond a static buffer. A 125 m horizontal buffer does not account for vertical clearance — an asset perched 80 m above the river is marked as exposed. A Height Above Nearest Drainage (HAND) model using Copernicus GLO-30 is the V2 roadmap solution. See `docs/reference/KNOWN_LIMITATIONS.md` → "Domain Physics Limitations".
- **Latency framing.** Sentinel-1 has a 12-day repeat orbit (6 days combining constellations). A moraine breach unfolds in minutes. SIREN is a medium-term situational awareness and rapid post-event triage tool, not a real-time breach warning system. The demo models progressive 19-day expansion (07-23 → 08-11), not seconds-or-minutes flash flood detection. SIREN complements — rather than replaces — in-situ acoustic ground sensors.
- Rainfall values are real ERA5 reanalysis data from the Open-Meteo Archive API (not GPM IMERG, which requires Earthdata auth). The values are raster-derived daily precipitation sums for the basin center.
- The pipeline runs synchronously in the API request (no background task queue). This is intentional for demo simplicity.
- The frontend uses mock fallback data when the backend is unreachable. This is by design for offline resilience.
- No authentication or role-based access control in the MVP.
- **SMS** is the only live alert channel (via ntfy.sh push when online). **LoRa** and **Satellite** remain simulated state machines (QUEUED → TRANSMITTING → DELIVERED). No real radio or Iridium modem transmission occurs.
- The **First Responder Advisory** row in AuditView is a simulated visual — no real pre-confirmation notification is sent to hospitals or fire crews. It communicates the two-tier routing concept for the demo.
- The **escalation policy badge** in ReviewView is informational only — no auto-escalation dispatch fires without human confirmation (Hard Rule #3).

### Production transition gaps (must fix before unattended operation)

15 specific gaps were identified during the live-service architecture audit. 8 have been resolved (2026-09-07):

- ~~**Phase 4 blocker:** `run_pipeline()` accepts only the three demo observation IDs.~~ **Resolved** — live observations now accepted via `repo.register_observation()`.
- ~~**Ingest scripts exit 0 on failure**~~ **Resolved** — `--strict` flag added; API errors return non-zero.
- ~~**CDSE uses deprecated endpoint**~~ **Resolved** — uses `stac.dataspace.copernicus.eu/v1/search` with pagination.
- ~~**Overpass query missing river geometry**~~ **Resolved** — `waterway=river/stream` queried; flat properties emitted.
- ~~**openmeteo.py date window bug**~~ **Resolved** — backward 7-day window; null for missing precip.
- ~~**No job ledger or idempotency**~~ **Partially resolved** — `acquisition_jobs` table added with idempotency constraint.
- ~~**SRTM uses outdated URL**~~ **Resolved** — Earthdata Cloud endpoint with Bearer token auth.
- ~~**CDSE downloads into memory**~~ **Resolved** — streaming downloads in 64 KiB chunks.
- **SQLite unsuitable for multi-instance** — WAL/network filesystem limitation.
- **Review suppression logic incorrect** — historical confirm still authorizes dispatch after later reject.
- **ntfy delivery is browser-side** — `"sent"` does not mean delivered.
- **No S3/object storage** — local disk exhausts within weeks at production volume.
- **ML not qualified for live use (audited 2026-09-07)** — WaterUNet event-holdout IoU is 0.24 (below the 0.65 load-bearing gate). The runtime tensor contract is now fixed (calibrated VV/VH dB via `normalize_sar()`), and ML paths cannot suppress rule evidence per ADR-010. WaterUNet remains shadow-only. See `docs/reference/DL_MODEL_AUDIT.md` + ADR-010.
- **Chorabari is pre-Sentinel-1** — the 2013 Kedarnath event cannot validate the SAR-primary pipeline (S1A launched April 2014); the Sentinel-era validation event is the South Lhonak GLOF (October 2023), hence the dual-basin strategy.

See [`docs/reference/KNOWN_LIMITATIONS.md`](docs/reference/KNOWN_LIMITATIONS.md) → "Production Transition Gaps" for the complete list with phase tags.

---

## License

Hackathon project. See competition rules for usage terms.

---

## Team

Built for `>.hack();'26`, 7th Edition — 36-hour hackathon.

**Closing line:**

> SIREN doesn't replace emergency authorities — it buys them the lead time to identify who to rescue, how to reach them when networks are down, and how to stop the outbreak that follows the flood. This demo shows the 20 days of warning we could have had.
