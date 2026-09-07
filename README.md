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
    ingest/       # CDSE STAC, SRTM, IMERG, Overpass downloaders
    preprocess/   # clip, reproject, co-register, cloud mask, quality gate
    detect/       # NDWI, SAR backscatter, weather-adaptive router, scenario masks
    geo/          # D8 corridor, tolerance buffers, exposure intersections
    risk/         # hazard H, exposure E, disease D_risk, SAR priority scoring + reasons
    ml/           # ML evidence layer (audited 2026-09-07 — not qualified for live use; see docs/reference/DL_MODEL_AUDIT.md)
    alerting/     # ≤250-byte payload codec, validator
    audit/        # append-only log writer + SHA-256 hash chain
    db/           # SQLite schema + repositories
    pipeline.py   # orchestrator: detect→geo→risk→DB→audit
  tests/          # 104 tests (pytest: 101 active + 3 torch-gated)
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
pytest                           # 104 tests
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

## Demo Scenario

The demo is a retrospective "what-if" reconstruction of a GLOF (glacial lake outburst flood) event:

| Observation | Date | Sensor | Cloud | Rain 24h | Expansion | Severity | Story |
|---|---|---|---|---|---|---|---|
| Baseline | 2025-11-22 | S2 Optical | 5% | 0.0 mm | — | — | Clear post-monsoon baseline |
| obs-001 | 2026-07-23 | S1 SAR | 0% eff | 18.2 mm | +8% | Watch | Early warning sign |
| obs-002 | 2026-08-04 | S1 SAR | 0% eff (95% optical) | 84.6 mm | +28% | Critical | Disaster day |
| obs-003 | 2026-08-12 | S1 SAR | 0% eff (90% optical) | 60.0 mm | +43% | Critical | Peak expansion |

The prevention story: the +8% expansion on 07-23 was the early warning. Had SIREN been monitoring in real time, the watch would have escalated 20 days before the peak (08-12), buying lead time to evacuate.

---

## Testing

```bash
cd backend
pytest                           # 104 tests, ~20s
```

| Test Suite | Tests | Coverage |
|---|---|---|
| test_quality | 11 | Quality gate (PRD §9.1) |
| test_codec | 12 | Payload codec ≤250 bytes, round-trip |
| test_audit | 11 | Append-only enforcement, hash chain, trigger validation |
| test_api | 12 | All API endpoints, human gate, error shapes |
| test_preprocess | 6 | Clip, reproject, co-register on synthetic rasters |
| test_ingest | 25 | CLI argument parsing, provenance sidecars |
| test_pipeline | 5 | Full orchestrator: detect→geo→risk→DB→audit |
| test_ml | 13 | ML evidence layer (deterministic fallback, torch-gated) |
| test_sar_priority | 9 | SAR priority ranking (PRD §15) |

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

### Reference

- [`docs/reference/KNOWN_LIMITATIONS.md`](docs/reference/KNOWN_LIMITATIONS.md) — Demo limitations + production transition gaps (phase-tagged)
- [`docs/reference/DL_MODEL_AUDIT.md`](docs/reference/DL_MODEL_AUDIT.md) — 2026-09-07 audit of the four PRD-nominated ML models; verdict: no existing checkpoint is qualified for live hazard assessment
- [`docs/reference/PRODUCTION_ML_PLAN.md`](docs/reference/PRODUCTION_ML_PLAN.md) — Recommended production pipeline, models, datasets, and the dual-basin strategy (Imja monitoring + South Lhonak event validation)

---

## Live Service Transition

The hackathon MVP is an offline demo with prepared data. The project is being evaluated for transition to a continuously running hosted service that polls live satellite, weather, and OSM sources. The frozen deterministic pipeline is not changed — what changes is what feeds it and how often it is triggered.

### Two deployment profiles

| Profile | Network | Database | Storage | Auth | Delivery |
|---|---|---|---|---|---|
| **Offline/demo** (current) | Zero runtime calls | SQLite | Local disk | None | Browser-side ntfy |
| **Hosted/live** (planned) | Acquisition service polls sources | PostgreSQL/RDS | S3 + local ephemeral | OIDC + RBAC | Server-side outbox |

### Transition roadmap (7 phases)

| Phase | Goal | Status |
|---|---|---|
| 0 | Frozen release and acceptance boundary | Pending |
| 1 | Reliable acquisition-only service | Pending |
| 2 | Basin and input qualification | Pending |
| 3 | Hosted persistence, security, live UI | Pending |
| 4 | Automatic scoring integration | **Blocked** |
| 5 | Extended shadow operation (30-day) | Pending |
| 6 | Authority-supervised operational pilot | Pending |

> **Phase 4 blocker:** `run_pipeline()` currently accepts only the three hardcoded demo observation IDs (`obs-001`, `obs-002`, `obs-003`). A live acquisition service can download real satellite products, but the frozen pipeline will reject them with `ValueError: Unknown observation`. An approved scope decision is required to extend the observation-acceptance interface before automatic live scoring is possible.

See [`docs/spec/BUILD_ROADMAP.md`](docs/spec/BUILD_ROADMAP.md) → "Live Service Transition Roadmap" for full task tables, GO/NO-GO criteria, and rollback plans per phase.

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

- The available ascending-orbit Sentinel-1 pair covers only the western AOI; the Imja lake (86.925°E) is outside the swath. The demo uses prepared scenario masks near Imja (clearly labeled in `detect/scenario.py`). The SAR pipeline itself is real and validated on the covered region.
- The pipeline runs synchronously in the API request (no background task queue). This is intentional for demo simplicity.
- The frontend uses mock fallback data when the backend is unreachable. This is by design for offline resilience.
- No authentication or role-based access control in the MVP.
- **SMS** is the only live alert channel (via ntfy.sh push when online). **LoRa** and **Satellite** remain simulated state machines (QUEUED → TRANSMITTING → DELIVERED). No real radio or Iridium modem transmission occurs.
- The **First Responder Advisory** row in AuditView is a simulated visual — no real pre-confirmation notification is sent to hospitals or fire crews. It communicates the two-tier routing concept for the demo.
- The **escalation policy badge** in ReviewView is informational only — no auto-escalation dispatch fires without human confirmation (Hard Rule #3).

### Production transition gaps (must fix before unattended operation)

15 specific gaps were identified during the live-service architecture audit, including:

- **Phase 4 blocker:** `run_pipeline()` accepts only the three demo observation IDs. Live scoring is impossible without an approved scope decision.
- **Ingest scripts exit 0 on failure** — silent data loss under a scheduler.
- **CDSE uses deprecated endpoint** — may stop returning results.
- **Overpass query missing river geometry** — automated OSM refresh would break the corridor.
- **openmeteo.py date window bug** — seven-day antecedent rainfall computed over wrong dates.
- **No job ledger or idempotency** — duplicate observations on scheduler retry.
- **SQLite unsuitable for multi-instance** — WAL/network filesystem limitation.
- **Review suppression logic incorrect** — historical confirm still authorizes dispatch after later reject.
- **ntfy delivery is browser-side** — `"sent"` does not mean delivered.
- **No S3/object storage** — local disk exhausts within weeks at production volume.
- **ML not qualified for live use (audited 2026-09-07)** — train/inference input mismatch, no held-out evaluation, and ML paths that can suppress rule evidence; see `docs/reference/DL_MODEL_AUDIT.md` + ADR-010.
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
