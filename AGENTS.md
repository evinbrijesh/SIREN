# AGENTS.md — SIREN

**SIREN** — Satellite-Informed Risk & Emergency Network. Satellite-assisted early warning and disaster-response decision platform (Track 7: Area ii resilient alerting + Area iii disease prevention). 36-hour hackathon build.

**Read before writing code:** `docs/PRD.md` (v4.1 — product spec, data contracts, scoring formulas) and `docs/BUILD_ROADMAP.md` (phase order, checkpoints, fallbacks). This file tells you *how to work*; those tell you *what to build*.

---

## Stack

- **Backend:** Python 3.11+, FastAPI, rasterio, geopandas, shapely, numpy, xarray, pysheds (fallback: whitebox), SQLite (JSON columns)
- **Frontend:** React + Vite + TypeScript, MapLibre GL JS, TanStack Query
- **Storage:** SQLite + GeoJSON files + GeoTIFF/COG on disk. No PostGIS. No Redis. No Docker.

## Repo Structure

```text
backend/
  siren/
    api/          # FastAPI routes
    ingest/       # CDSE STAC, Earthdata SRTM, IMERG, Overpass downloaders
    preprocess/   # clip, reproject, co-register, quality gate
    detect/       # NDWI diff, SAR backscatter ratio, weather-adaptive router, change stats
    geo/          # D8 corridor, tolerance buffers, exposure intersections
    risk/         # hazard H, exposure E, disease D_risk scoring + reasons
    alerting/     # <250-byte payload codec, simulated dispatch
    audit/        # append-only log writer
    db/           # SQLite schema + repositories
  tests/
    fixtures/     # synthetic rasters + fake OSM GeoJSON (tiny, committed)
frontend/
  src/
    views/        # MapView, TimelineView, ReviewView, AuditView
    api/          # typed client
    components/
data/
  raw/            # downloaded scenes — gitignored, never hand-edited
  processed/      # aligned rasters, masks — gitignored, written only by pipeline
  assets/         # basin GeoJSON, OSM extracts — small, committed
docs/
```

## Commands

```bash
# backend
cd backend && pip install -e ".[dev]"
uvicorn siren.api:app --reload --port 8000
pytest

# frontend
cd frontend && npm install && npm run dev   # port 5173, proxies /api → 8000
```

---

## Hard Rules (all agents, no exceptions)

1. **Deterministic-first.** No trained ML models in the critical path. Rule-based masks and weighted scores are the deliverable. ML is a stretch goal, gated on the core loop working (Roadmap Phase 6).
2. **Offline demo.** Zero network calls at runtime. All data loads from `data/`. Live API ingestion is a bonus script, never a runtime dependency.
3. **Human gate.** No code path may dispatch an alert without a recorded review decision (`confirm`). Reject/postpone must suppress dispatch.
4. **Payload ≤ 250 bytes.** Enforced by a unit test, not by hope.
5. **Explainability.** Every score object carries a `reasons` array (≥3 entries on elevated+). Never return a bare number.
6. **Reproducibility.** Same inputs + processing version → identical outputs. No unseeded randomness anywhere.
7. **Data hygiene.** Only `ingest/` scripts write to `data/raw`; only the pipeline writes `data/processed`. Never commit rasters. Never hand-edit data files.
8. **Dependency whitelist.** rasterio, geopandas, shapely, numpy, xarray, pysheds, fastapi, pydantic, pytest. Anything else: stop and ask.
9. **Scope discipline.** If a feature isn't in the PRD or Roadmap, don't build it. Out of scope list: PRD §14.

## Data Contracts

Authoritative schemas live in the PRD — implement exactly these, field names and types:
- Quality gate verdict: PRD §9.1
- Observation: PRD §10.2
- Alert: PRD §10.3
- Compressed payload: PRD §10.4 (`aid` prefix `siren-`)
- Scoring formulas: PRD §9.5 (weights are fixed: 0.30/0.25/0.20/0.15/0.10)
- Tolerance buffers: bridges ±75 m, roads ±50 m, settlements/wells ±100 m (PRD §6.4)

---

## Task Routing: Devin vs OpenCode

**Routing principle:** Devin runs async in the cloud with a slow feedback loop and **no access to the local demo data**. OpenCode runs locally where the rasters live, iterates in real time, and can see rendered output.

> **If the task is spec-complete, testable against synthetic fixtures, and independent of the real basin data → Devin.**
> **If the task needs the actual rasters, visual tuning, or live debugging → OpenCode.**

### Devin Tasks (dispatch all at hour 0, in this priority order)

Devin works from PRD sections alone — every task below has a complete spec in the docs. Each must land as a PR with passing tests against `tests/fixtures/`, never against real basin data.

| # | Task | Spec | Acceptance criteria |
|---|---|---|---|
| D1 | Ingest toolkit: CDSE STAC search/download, SRTM from Earthdata, IMERG pull, Overpass extract → `ingest/` | PRD §11 | CLI with `--bbox`; downloads to `data/raw/`; works on any bbox; retries + provenance metadata sidecar per file |
| D2 | Preprocessing: clip-to-basin, reproject, co-register to DEM grid, cloud mask → `preprocess/` | PRD §6.2 | Pure functions; unit tests on synthetic 100×100 GeoTIFFs; alignment error metric returned |
| D3 | Quality gate module → `preprocess/quality.py` | PRD §9.1 | Emits exact §9.1 JSON contract; cloud ≥ 0.20 flips `usable` routing flag; confidence multiplier formula implemented |
| D4 | FastAPI scaffold + Pydantic models for all data contracts + route stubs → `api/`, `db/` | PRD §10.2–10.3 | App boots; `/basin`, `/observations`, `/runs` return fixture data; SQLite schema creates |
| D5 | Payload codec: alert JSON → ≤250-byte packet, validator, round-trip → `alerting/` | PRD §10.4 | Round-trip property test; oversize alert raises; `siren-` ID prefix |
| D6 | Audit log: append-only writer + query API → `audit/` | PRD §7.8 | Append-only enforced (no update/delete paths); full lineage queryable by alert_id |
| D7 | Frontend scaffold: Vite React TS + MapLibre init + 4-view layout with mocked data → `frontend/` | PRD §12 | All four views render with mock JSON; typed API client generated from D4 models |
| D8 | Test fixtures: synthetic rasters (baseline + "expanded water" pair) + fake OSM GeoJSON with 2 villages, 1 bridge, 3 wells | Roadmap Phase 0 | Fixtures committed; used by all above tests |

**Dispatch order = D8, D4, D3, D5, D2, D1, D6, D7** (fixtures and API scaffold unblock everything else). Review and merge PRs as they land — do not batch-merge at hour 12.

### OpenCode Tasks (start immediately, owns the spine)

Everything that requires the real basin rasters on disk, threshold tuning against what you can see, or end-to-end integration:

| Phase | Task | Why local |
|---|---|---|
| 0 | Verify Devin-downloaded data opens; lock basin; smoke-test D8 on real DEM | Real files, visual check |
| 2 | NDWI differencing + SAR backscatter ratio on real scenes; **tune thresholds by eye**; weather-adaptive router wiring | Threshold tuning is visual iteration on actual rasters |
| 3 | Combined D8 + OSM river buffering corridor (D8 reachability validation + OSM river floodplain buffer); tolerance-buffer intersections against real OSM extract; verify "2 villages, 1 bridge, 3 wells" emerges from geometry | Geospatial iteration on local data |
| 3 | Risk fusion H + E + disease D_risk with real inputs; wire `reasons` arrays | Needs real feature values |
| 4–5 | Integrate Devin modules (gate, codec, audit) into the live pipeline; Confirm/Reject/Postpone flow; simulated dispatch | Integration + debugging |
| 4–5 | Wire real data into the 4 frontend views; swipe-compare; evidence panel | Visual, demo-critical |
| 6 | **Run Monitoring** end-to-end sequence; polish; stretch goal gate | Demo wiring |
| 7 | Offline rehearsal support; live fixes during rehearsal | Must be local |

### Handoff Protocol

1. Hour 0: dispatch all Devin tasks; OpenCode starts Phase 0 with whatever data exists locally.
2. As Devin PRs land: OpenCode reviews, merges, and **re-points the module from fixtures to real data** — that re-pointing is always OpenCode's job.
3. Devin never blocks the spine: if a PR is late, OpenCode writes a minimal stub behind the same interface and moves on. Replace later.
4. After hour 28: **no new Devin tasks.** Late PRs at that point are risk, not help.

### What Neither Agent Does

- Lock the basin (human decision — judge-resonance call)
- Record the backup video or rehearse (human)
- Approve scope changes (human — PRD §14 is the boundary)

---

## Definition of Done

Offline, in one click-chain: baseline loads → 4 observations process through the pipeline → elevated review card appears with ≥3 evidence reasons → Confirm produces a ≤250-byte simulated dispatch → audit log reconstructs the full lineage. If your change breaks this chain, fix it before anything else.
