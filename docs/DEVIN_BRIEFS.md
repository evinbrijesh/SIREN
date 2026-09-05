# Devin Dispatch Briefs — SIREN

> **Status:** All Devin tasks (D1–D8) are complete and committed. The pipeline orchestrator (`backend/siren/pipeline.py`) wires all modules together. 80/80 tests passing. DoD chain verified end-to-end.

Copy-paste each brief into Devin as a task. **Dispatch order: D4, D3, D5, D2, D1, D6, D7** (D8 is already satisfied — see below; API scaffold unblocks the rest).

**Rules for every task:**
- Work from `docs/PRD.md` (v4.3) and `backend/siren/db/schema.sql` — they are authoritative.
- Test against `backend/tests/fixtures/` only. **Never** against real basin data in `data/`.
- Land as a PR with passing tests. Do not batch-merge.
- Dependency whitelist: rasterio, geopandas, shapely, numpy, xarray, pysheds, fastapi, pydantic, pytest. Anything else: stop and ask.

---

## D8 — Test Fixtures ✅ ALREADY SATISFIED (do not dispatch)

**Status:** Completed locally by OpenCode. Fixtures exist and are committed. **Do not dispatch this task to Devin** — it would duplicate work.

**What exists:**
- `backend/tests/fixtures/make_fixtures.py` — generator script (run once)
- `backend/tests/fixtures/rasters/baseline.tif` — 100×100, 100 water px
- `backend/tests/fixtures/rasters/expanded_water.tif` — 128 water px = **+28%** (verified)
- `backend/tests/fixtures/rasters/cloudy_optical.tif` — **2-band** optical scene (GREEN, NIR), cloud_fraction **0.25** (computed from scene stats, not a pre-made mask)
- `backend/tests/fixtures/osm/fake_assets.geojson` — 2 villages, 1 bridge, 3 wells, 1 road line
- `backend/tests/fixtures/__init__.py` — `fixture_path()` helper

**Note for downstream tasks:** `cloudy_optical.tif` is a 2-band scene; the quality gate (D3) must compute cloud_fraction from scene statistics (bright pixels in both bands), not read a pre-made mask.

**Spec:** Roadmap Phase 0. Create synthetic fixtures used by all other tests.

**Deliverables:**
- `backend/tests/fixtures/rasters/baseline.tif` — synthetic 100×100 GeoTIFF, EPSG:4326, single-band, representing a "normal" glacial lake + terrain.
- `backend/tests/fixtures/rasters/expanded_water.tif` — same grid, water body expanded ~+28% (a clearly detectable change).
- `backend/tests/fixtures/rasters/cloudy_optical.tif` — same grid with a large cloud mask region (cloud_fraction ≥ 0.20).
- `backend/tests/fixtures/osm/fake_assets.geojson` — fake OSM extract with exactly: **2 villages, 1 bridge, 3 wells** (point features), plus 1 road line.
- A `backend/tests/fixtures/__init__.py` exposing paths via a helper (e.g. `fixture_path(name)`).

**Acceptance criteria:**
- Fixtures are tiny (< 1 MB total) and committed.
- `expanded_water.tif` vs `baseline.tif` produce a water-area change of roughly +28% when differenced (test asserts within ±5%).
- `cloudy_optical.tif` reports cloud_fraction ≥ 0.20.
- Fake OSM has exactly the 2/1/3 asset counts.

---

## D4 — FastAPI Scaffold + Pydantic Models + Route Stubs ✅ COMPLETE

**Spec:** PRD §10.2–10.3, §12. Schema: `backend/siren/db/schema.sql`.

**Deliverables:**
- `backend/siren/api/` — FastAPI app (`siren.api:app`), CORS enabled for `http://localhost:5173`.
- Pydantic models in `backend/siren/api/models.py` matching PRD §10.2 (Observation) and §10.3 (Alert) field names/types exactly.
- Route stubs returning fixture data:
  - `GET /basin` → basin config
  - `GET /observations` → list of observations
  - `GET /runs` → list of runs
  - `POST /runs` → trigger a pipeline run (stub: returns 501 or a placeholder run)
- SQLite bootstrap: `backend/siren/db/` repository that executes `schema.sql` and seeds basins + observations from fixtures.
- `backend/pyproject.toml` with `[project.optional-dependencies] dev = ["pytest", "httpx"]`.

**Acceptance criteria:**
- `uvicorn siren.api:app` boots.
- `/basin`, `/observations`, `/runs` return valid JSON matching the Pydantic models.
- `pytest` passes a smoke test hitting the routes via `TestClient`.

---

## D3 — Quality Gate Module ✅ COMPLETE

**Spec:** PRD §9.1. Output contract is exact JSON.

**Deliverables:**
- `backend/siren/preprocess/quality.py` — pure function `assess_quality(cloud_fraction, alignment_error, sensor) -> QualityVerdict`.
- `QualityVerdict` Pydantic model with fields: `quality_score`, `cloud_fraction`, `alignment_ok`, `usable`, `confidence_adjustment`.
- Confidence multiplier = `(1.0 - cloud_fraction) * sensor_freshness_weight`. For SAR, cloud_fraction is treated as 0.0 (all-weather).
- `usable` flips to `false` when cloud_fraction ≥ 0.20 (routes to SAR path).

**Acceptance criteria:**
- Emits the exact §9.1 JSON contract.
- Unit tests: cloud ≥ 0.20 → `usable=false`; SAR input → `cloud_fraction=0.0`; confidence multiplier formula verified.

---

## D5 — Payload Codec (<250 bytes) ✅ COMPLETE

**Spec:** PRD §10.4. `aid` prefix `siren-`.

**Deliverables:**
- `backend/siren/alerting/codec.py` — `encode(alert: Alert) -> bytes` and `decode(payload: bytes) -> Alert`.
- `backend/siren/alerting/validate.py` — `validate_size(payload)` raising on > 250 bytes.
- Round-trip: `decode(encode(a)) == a`.

**Acceptance criteria:**
- Property test: round-trip holds for a set of representative alerts.
- Oversize alert raises.
- `aid` always starts with `siren-`.

---

## D2 — Preprocessing (clip, reproject, co-register, cloud mask) ✅ COMPLETE

**Spec:** PRD §6.2.

**Deliverables:**
- `backend/siren/preprocess/` — pure functions:
  - `clip_to_basin(raster, basin_geojson)`
  - `reproject(raster, target_crs)`
  - `co_register(reference, moving) -> (aligned, alignment_error)`
  - `cloud_mask(optical_raster) -> (mask, cloud_fraction)`
- Alignment error metric returned (e.g. RMSE of tie points).

**Acceptance criteria:**
- Unit tests on synthetic 100×100 GeoTIFFs from D8 fixtures.
- `co_register` returns an alignment error value; aligned output has same grid as reference.
- `cloud_mask` returns cloud_fraction matching the fixture's known value.

---

## D1 — Ingest Toolkit ✅ COMPLETE

**Spec:** PRD §11.

**Deliverables:**
- `backend/siren/ingest/` — CLI scripts with `--bbox`:
  - `cdse.py` — Sentinel-1/2 STAC search + download from Copernicus Data Space Ecosystem
  - `srtm.py` — SRTM 1-arc-second from NASA Earthdata
  - `imerg.py` — GPM IMERG rainfall pull
  - `overpass.py` — OSM/Overpass extract for critical facilities
- Downloads to `data/raw/` with a provenance metadata sidecar (`.json`) per file: source, bbox, acquired_at, retries.
- Retry logic with backoff on transient failures.

**Acceptance criteria:**
- CLI runs with `--bbox` on any bbox.
- Writes to `data/raw/` + sidecar.
- Works offline-safe: if network unavailable, exits cleanly with a clear message (never crashes the demo).

---

## D6 — Audit Log (append-only) ✅ COMPLETE

**Spec:** PRD §7.8.

**Deliverables:**
- `backend/siren/audit/` — append-only writer + query API:
  - `append(actor, action, detail_json, alert_id=None)`
  - `query_by_alert(alert_id) -> list[entries]` (full lineage)
- Enforced append-only: repository exposes only INSERT + SELECT for `audit_log` (no update/delete paths).

**Acceptance criteria:**
- No update/delete methods exist on the audit repository.
- Full lineage queryable by `alert_id`.
- Unit test: attempting to mutate a past entry is impossible via the public API.

---

## D7 — Frontend Scaffold (4 views, mocked data) ✅ COMPLETE

**Spec:** PRD §12, **`docs/UI_DESIGN.md` (layout + design system — authoritative for visual structure)**.

**Deliverables:**
- `frontend/` — Vite + React + TypeScript app.
- MapLibre GL JS map initialized with a basin polygon + raster overlay (mock).
- Four views: `MapView`, `TimelineView`, `ReviewView`, `AuditView` — all render with mock JSON, laid out per `docs/UI_DESIGN.md` §3–§6.
- Dark ops-console design system per `docs/UI_DESIGN.md` §7 (slate-900 bg, status colors green/amber/red, monospace for telemetry).
- Typed API client in `frontend/src/api/` generated from the D4 Pydantic models (field names/types must match).
- Vite dev proxy: `/api` → `http://localhost:8000`.

**Acceptance criteria:**
- `npm run dev` boots; all four views render with mock data.
- Typed client compiles against the D4 contract.
- No runtime network calls to external services (offline-safe).
- The 5 demo-critical visual beats in `docs/UI_DESIGN.md` §9 render correctly (swipe, router badge, action sheet, payload box, decision bar).

---

## Handoff Note

All Devin tasks are complete. OpenCode has integrated all modules into the live pipeline (`backend/siren/pipeline.py`) and wired the frontend to the real API. The DoD chain is verified end-to-end:

- **D1 (Ingest):** `backend/siren/ingest/{cdse,srtm,imerg,overpass}.py` — 25 tests
- **D2 (Preprocess):** `backend/siren/preprocess/` — 6 tests
- **D3 (Quality gate):** `backend/siren/preprocess/quality.py` — 11 tests
- **D4 (API scaffold):** `backend/siren/api/{app,models}.py` — 10 tests
- **D5 (Payload codec):** `backend/siren/alerting/{codec,validate}.py` — 12 tests
- **D6 (Audit log):** `backend/siren/audit/writer.py` — 11 tests
- **D7 (Frontend):** `frontend/src/` — 4 views, TypeScript clean, Vite build passes
- **D8 (Fixtures):** `backend/tests/fixtures/` — committed, used by all tests

**Pipeline integration (OpenCode):** `backend/siren/pipeline.py` orchestrates detect→geo→risk→DB→audit. `backend/tests/test_pipeline.py` adds 5 tests covering the full DoD chain. Total: 80/80 tests passing.