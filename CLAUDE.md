# CLAUDE.md — SIREN

Companion to `AGENTS.md` (agent routing + hard rules) and `docs/PRD.md` (spec). Read all three before writing code.

**SIREN** — Satellite-Informed Risk & Emergency Network. Satellite-assisted early warning and disaster-response platform for Himalayan basins. Track 7 (resilient alerting + disease prevention). 36-hour hackathon build.

---

## Stack

- **Backend:** Python 3.11+, FastAPI, rasterio, geopandas, shapely, numpy, xarray, pysheds (fallback: whitebox), SQLite (JSON columns)
- **Frontend:** React + Vite + TypeScript, MapLibre GL JS, TanStack Query
- **Storage:** SQLite + GeoJSON files + GeoTIFF/COG on disk. No PostGIS, no Redis, no Docker.

## Structure

```text
backend/
  siren/
    api/          # FastAPI routes (thin; delegates to modules)
    ingest/       # CDSE STAC, Earthdata SRTM, IMERG, Overpass downloaders
    preprocess/   # clip, reproject, co-register, quality gate
    detect/       # NDWI diff, SAR backscatter ratio, weather-adaptive router, change stats
    geo/          # D8 corridor, tolerance buffers, exposure intersections
    risk/         # hazard H, exposure E, disease D_risk scoring + reasons
    alerting/     # <250-byte payload codec, simulated dispatch
    audit/        # append-only log writer
    db/           # schema.sql + repositories
  tests/
    fixtures/     # synthetic rasters + fake OSM GeoJSON
frontend/
  src/
    views/        # MapView, TimelineView, ReviewView, AuditView
    api/          # typed client
    components/
data/
  raw/            # downloaded scenes (gitignored)
  processed/      # aligned rasters, masks (gitignored)
  assets/         # basin GeoJSON, OSM extracts (committed)
docs/
```

## Conventions

- **Module boundaries:** `api/` routes must not contain logic — delegate to `detect/`, `geo/`, `risk/`, etc. Keeps the pipeline testable without HTTP.
- **Data contracts:** implement PRD §10 field names/types exactly. Never rename a field to "make it nicer."
- **Scoring:** every score object carries a `reasons` array (≥3 entries on elevated+). Never return a bare number.
- **Errors:** raise typed exceptions; FastAPI handlers map them to structured JSON `{error, detail}`. No silent fallbacks that hide data gaps.
- **Logging:** use Python `logging`; log run_id/observation_id on every pipeline step for lineage.
- **Reproducibility:** no unseeded randomness. Seed any RNG explicitly.
- **Payload size:** the ≤250-byte alert constraint is enforced by a unit test, not by hope.

## Known Gotchas

- **WhiteboxTools vs pysheds:** prefer pysheds (pure Python, no binary). If D8 flow accumulation misbehaves on steep/karst terrain, fall back to OSM river buffering (Roadmap Phase 3 fallback).
- **Cloud routing:** optical cloud_fraction ≥ 0.20 flips the pipeline to SAR-primary. SAR is treated as all-weather (cloud_fraction = 0.0 on that path).
- **Tolerance buffers:** bridges ±75 m, roads ±50 m, settlements/wells ±100 m. These exist to prevent false intersections at 10–30 m satellite resolution — do not "tighten" them.
- **Offline demo:** zero network calls at runtime. All data loads from `data/`. Live ingestion is a bonus script, never a runtime dependency.
- **SQLite spatial joins:** run in-memory via geopandas on the small basin extract. Do not reach for PostGIS.

## Module Map

| Concern | Owned by |
|---|---|
| Downloading scenes | `ingest/` |
| Clip/reproject/align/quality | `preprocess/` |
| Change masks + stats | `detect/` |
| Corridor + exposure | `geo/` |
| H / E / D_risk scores | `risk/` |
| Payload codec + dispatch | `alerting/` |
| Append-only lineage | `audit/` |
| Persistence | `db/` |
| HTTP surface | `api/` |
| Map/timeline/review/audit UI | `frontend/src/views/` |

## Definition of Done

Offline, in one click-chain: baseline loads → 4 observations process → elevated review card with ≥3 evidence reasons → Confirm produces a ≤250-byte simulated dispatch → audit log reconstructs the full lineage. If a change breaks this chain, fix it before anything else.