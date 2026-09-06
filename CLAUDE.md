# CLAUDE.md — SIREN

Companion to `AGENTS.md` (agent routing + hard rules) and `docs/spec/PRD.md` (spec). Read all three before writing code.

**SIREN** — Satellite-Informed Risk & Emergency Network. Satellite-assisted early warning and disaster-response platform for Himalayan basins. Track 7 (resilient alerting + disease prevention). 36-hour hackathon build.

---

## Stack

- **Backend:** Python 3.11+, FastAPI, rasterio, geopandas, shapely, numpy, xarray, pysheds (fallback: whitebox), SQLite (JSON columns). Optional ML extra: torch/torchvision (evidence layer only, deterministic fallback).
- **Frontend:** React + Vite + TypeScript, MapLibre GL JS, TanStack Query, Tailwind CSS
- **Storage:** SQLite + GeoJSON files + GeoTIFF/COG on disk. No PostGIS, no Redis.
- **Deployment:** Docker Compose (backend + frontend, one-command via `./start.sh`)

## Structure

```text
backend/
  siren/
    api/          # FastAPI routes (thin; delegates to modules) + map_assets.py
    ingest/       # CDSE STAC, Earthdata SRTM, IMERG, Overpass downloaders
    preprocess/   # clip, reproject, co-register, quality gate
    detect/       # NDWI diff, SAR backscatter ratio, weather-adaptive router, change stats
    geo/          # combined D8 + OSM river corridor, tolerance buffers, exposure intersections
    risk/         # hazard H, exposure E, disease D_risk, SAR priority + reasons
    ml/           # optional ML evidence layer (deterministic fallback, torch-gated)
    alerting/     # <250-byte payload codec, simulated dispatch
    audit/        # append-only log writer + SHA-256 hash chain (hash_chain.py)
    db/           # schema.sql + repositories
  tests/
    fixtures/     # synthetic rasters + fake OSM GeoJSON
frontend/
  src/
    views/        # MapView, TimelineView, ReviewView, AuditView
    api/          # typed client + offline mock fallback
    simulation/   # SimulationContext (shared demo state)
    components/   # OfflineBadge (online/offline event listener)
    theme/        # ThemeProvider + ThemeToggle (Ops Dark / Light / Satellite)
    utils/        # ntfy.ts — shared ntfy.sh live alert utility
data/
  raw/            # downloaded scenes (gitignored)
  processed/      # aligned rasters, masks (gitignored)
  assets/         # basin GeoJSON, OSM extracts, weather series (committed)
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
- **pysheds 0.5 + numpy 2.x incompatibility (CRITICAL):** pysheds 0.5 calls `np.in1d`, which was removed in numpy 2.x. **Must monkeypatch `np.in1d = np.isin` before any pysheds call.** The `geo/` module does this at import. Do not "fix" by downgrading numpy — rasterio/geopandas need numpy 2.
- **pysheds 0.5 API:** use `grid.read_raster(path)` which returns a `Raster` object (NOT `grid.dem` / `grid.view('dem')` — those don't exist in 0.5). Pass the returned Raster to `fill_depressions(dem=...)`.
- **AOI is a rectangle, not a watershed:** D8 accumulation max is limited (~7k cells) because the river exits the box edges. Expected — the corridor is computed within the AOI. If you need the full river, extend the DEM downstream.
- **Cloud routing:** optical cloud_fraction ≥ 0.20 flips the pipeline to SAR-primary. SAR is treated as all-weather (cloud_fraction = 0.0 on that path). The original optical cloud is preserved as `optical_cloud_fraction` for display.
- **Severity thresholds:** expansion ≥40% → critical; ≥20% → elevated; ≥5% → watch; <5% → informational. These are policy thresholds in `risk/fusion.py::classify_severity()`.
- **SAR always routed:** all three demo observations use S1 SAR. obs-002/obs-003 have 95%/90% optical cloud → SAR-primary. obs-001 is SAR by source.
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
| SAR priority ranking | `risk/sar_priority.py` |
| ML evidence layer (optional) | `ml/` |
| Payload codec + dispatch | `alerting/` |
| Append-only lineage + hash chain | `audit/` |
| Persistence | `db/` |
| HTTP surface + map assets | `api/` |
| Map/timeline/review/audit UI | `frontend/src/views/` |
| Theme system | `frontend/src/theme/` |
| Offline status | `frontend/src/components/` |

## Definition of Done

Offline, in one click-chain: baseline loads → 3 observations process → elevated/critical review card with ≥3 evidence reasons → Confirm produces a ≤250-byte simulated dispatch → audit log reconstructs the full lineage with SHA-256 hash chain. If a change breaks this chain, fix it before anything else.