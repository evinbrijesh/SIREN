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
4. The coordinator confirms the alert
5. A compressed dispatch payload (≤250 bytes) is sent
6. The audit log reconstructs the full lineage

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
| Storage | SQLite + GeoJSON + GeoTIFF on disk (no PostGIS, no Redis) |
| Deployment | Docker Compose (one-command: backend + frontend) |

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
    ml/           # Optional ML evidence layer (deterministic fallback, torch-gated)
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
    index.css     # Tailwind base + design tokens
data/
  raw/            # downloaded scenes (gitignored)
  processed/      # masks, aligned rasters (gitignored, pipeline-written)
  assets/         # basin GeoJSON, OSM extracts, weather series (committed)
docs/
  PRD.md          # Product Requirements Document (v4.3)
  BUILD_ROADMAP.md # 36-hour build plan with phase checkpoints
  API_CONTRACT.md  # HTTP API surface
  UI_DESIGN.md     # Coordinator console design spec
  DEVIN_BRIEFS.md  # Devin task dispatch briefs (D1-D8, archived)
  ADR-001..005     # Architecture decision records
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
9. Click **Confirm SOS** → **Yes, confirm**
10. Go to the **Audit** tab to see the dispatch payload and audit trail

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

See `docs/API_CONTRACT.md` for full request/response schemas.

---

## Key Design Decisions

- **Deterministic-first.** No trained ML in the critical path. Rule-based masks and weighted scores (ADR-002).
- **Offline demo.** Zero network calls at runtime. All data loads from `data/` (ADR-004).
- **SAR-first.** Weather-adaptive router switches to SAR when cloud ≥20% (ADR-003).
- **SQLite over PostGIS.** Zero-ops, offline-safe for hackathon scale (ADR-001).
- **Combined D8 + OSM corridor.** D8 validates gravity gradient; OSM rivers capture the real surveyed riverbed (ADR-005).
- **Human gate.** No dispatch without a recorded `confirm` review (enforced by SQLite trigger).
- **≤250-byte payload.** Compact JSON for LoRa mesh / satellite messenger / low-bandwidth SMS.
- **≥3 reasons on elevated+.** Every elevated/critical score carries at least three evidence factors.

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

- [`docs/PRD.md`](docs/PRD.md) — Product Requirements Document (v4.3)
- [`docs/BUILD_ROADMAP.md`](docs/BUILD_ROADMAP.md) — 36-hour build plan
- [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) — HTTP API surface
- [`docs/UI_DESIGN.md`](docs/UI_DESIGN.md) — Coordinator console design spec
- [`docs/DEVIN_BRIEFS.md`](docs/DEVIN_BRIEFS.md) — Devin task briefs
- [`docs/ADR-*.md`](docs/) — Architecture decision records

---

## Known Limitations

- The available ascending-orbit Sentinel-1 pair covers only the western AOI; the Imja lake (86.925°E) is outside the swath. The demo uses prepared scenario masks near Imja (clearly labeled in `detect/scenario.py`). The SAR pipeline itself is real and validated on the covered region.
- The pipeline runs synchronously in the API request (no background task queue). This is intentional for demo simplicity.
- The frontend uses mock fallback data when the backend is unreachable. This is by design for offline resilience.
- No authentication or role-based access control in the MVP.
- No real alert channel integration (SMS/LoRa/satellite are simulated).

---

## License

Hackathon project. See competition rules for usage terms.

---

## Team

Built for `>.hack();'26`, 7th Edition — 36-hour hackathon.

**Closing line:**

> SIREN doesn't replace emergency authorities — it buys them the lead time to identify who to rescue, how to reach them when networks are down, and how to stop the outbreak that follows the flood. This demo shows the 20 days of warning we could have had.
