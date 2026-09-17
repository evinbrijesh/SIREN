# AGENTS.md — SIREN

**SIREN** — Satellite-Informed Risk & Emergency Network. Satellite-assisted early warning and disaster-response decision platform for Himalayan glacial lake outburst flood (GLOF) events.

> **Origin:** Originally built at the >.hack();'26 36-hour hackathon (Track 7: resilient alerting + disease prevention). Now an ongoing personal research project pursuing an end-to-end differentiable neural pipeline from raw radar bytes to downstream flood dynamics.

**Read before writing code:** `docs/spec/PRD.md` (v5.0 — product spec, data contracts, scoring formulas, end-to-end neural pipeline §9.7) and `docs/spec/BUILD_ROADMAP.md` (phase history + active development roadmap). This file tells you *how to work*; those tell you *what to build*.

> **Build status:** Dual-track architecture. Deterministic/empirical baseline (Huggel + calibrated 2-channel FNO) is the load-bearing primary path. Real-data Kuro Siwo 6-channel SAR model trained (ADR-011.1 gate-passed, IoU 0.62) and **now wired into the runtime ML evidence layer** — `ChangeDetectionEngine` auto-detects checkpoint architecture and loads the gate-passed 6-channel `WaterResUNet`, building the `(VV_post, VH_post, VV_pre, VH_pre, dVV, dVH)` multi-temporal tensor from the calibrated pre/post SAR pair. The runtime prefers the verified **descending pair (2026-07-02 + 2026-07-14) that covers Imja Lake** (`find_imja_descending_pair` in `preprocess/sar_calibrate.py`), with per-observation fallback; SAR-pair + model provenance persists to `change_stats` and `/runs/{run_id}/ml-evidence`. E1 MC Dropout is wired end-to-end: runtime engine runs T=20 stochastic passes (dropout=0.1 is checkpoint-compatible), emits a per-pixel std map + stats, and the **conformal gate PASSED on the Kuro Siwo test split** — split-conformal, event-level cal/eval split, q*=0.375, empirical coverage 0.8885 vs nominal 0.90 (error 0.0115 ≤ 0.05); caveat: post-hoc dropout on a net trained with dropout=0.0, per-pixel coverage on spatially correlated pixels. Sidecar: `models/checkpoints/water_resunet_kuro_siwo_full/conformal_calibration.json` (auto-loaded by the engine). E0, E2, E3 remain research scaffolds — E2 gate evaluated and failing (see below). See ADR-013 for promotion criteria. Human gate (Hard Rule 3) and explainability (Hard Rule 5) preserved.

> **E2 neural bathymetry — LOO benchmark evaluated (2026-09-15):** 20 surveyed Himalayan glacial lakes (Zhang 2023 + Das 2025, 117k depth points) with Copernicus DEM GLO30 terrain now support real-data training. Leave-one-lake-out volume estimation benchmark: **Huggel formula MAPE 75.6%**, fitted power-law regression MAPE 82.1%, neural `BathymetryUNet` (100 epochs, 19 training lakes) MAPE 676.4%. All three methods fail the 15% ADR-013 gate. The neural model is severely overparameterised for 19 training samples — it overfits and predicts implausible bed depths. Huggel remains the best available method at this dataset size. **Transfer learning attempted (2026-09-17):** pretraining `BathymetryUNet` on 2,000 synthetic contract-matching basins (`generate_transfer_bathymetry_data`) then fine-tuning per LOO fold (60 epochs, lr 1e-4) improves sample MAPE to **328.8%** (gt MAPE 202.4%) vs 676.4% random-init — a ~2× gain but still far above the 15% gate and worse than Huggel. The E2 gate cannot be passed with 20 lakes; paths forward include more lakes (global compilation has 64 unique), or simpler terrain-feature regression. See `ml/bathymetry_benchmark.py`, `ml/bathymetry_training_data.py`, `train_bathymetry.py --benchmark`, `--train-real`, `--pretrain-synthetic` / `--pretrain-weights`.

> **Known domain-shift finding (2026-09-14, updated 2026-09-17):** the Kuro Siwo model reproduces its gate metrics exactly on the Kuro Siwo test set (pooled IoU 0.6147, P 0.8710, R 0.6762 at τ=0.5 over 3,081 chips) but produces implausible water masks on the full Imja scene (61k "water" pixels vs ~2,500 rule-based, mostly outside the AOI and on glacier surfaces). **Geographically resampled overlap with the rule mask is only ~13–22 px** — the earlier 7.4% "agreement" was an index-resize artifact (the masks share no extent). The model is out-of-distribution on high-Himalaya terrain — exactly why it remains **shadow-only evidence** (ADR-010). The displayed shadow mask is now terrain-gated: AOI polygon + slope >15° (within DEM coverage) + RGI glacier outlines except near mapped baseline water (RGI includes terminus lakes — a hard glacier gate erases Imja itself). 61,088 → 124 px on the descending pair. Per-chip pooling also matters: mean per-chip IoU 0.29, median 0.06.
>
> **Also fixed 2026-09-17:** `detect/sar.py::dem_slope` divided per-pixel gradients by metres-per-degree (~1e5× too small) — every slope gate was silently inert. Calibrated SAR caches now persist S1 GCP geolocation (the measurement TIFFs carry no geotransform; a fitted affine has 1–4 km residuals over mountain terrain, so GCPs + `GCPTransformer` are the honest georeference). New helpers: `sar_grid_lonlat`, `sar_grid_dem_slope`, `sar_grid_polygon_mask`, `sar_grid_sample` in `detect/sar.py`.

---

## Stack

- **Backend:** Python 3.11+, FastAPI, rasterio, geopandas, shapely, numpy, xarray, pysheds (fallback: whitebox), SQLite (JSON columns). ML: torch/torchvision (primary analytical path, ADR-013).
- **Frontend:** React + Vite + TypeScript, MapLibre GL JS, TanStack Query, Tailwind CSS
- **Storage:** SQLite + GeoJSON files + GeoTIFF/COG on disk. No PostGIS. No Redis.
- **Deployment:** Docker Compose (backend + frontend, one-command via `./start.sh`)

## Repo Structure

```text
backend/
  siren/
    api/          # FastAPI routes + Pydantic models + map_assets.py
    ingest/       # CDSE STAC, Earthdata SRTM, IMERG, Overpass downloaders
    preprocess/   # clip, reproject, co-register, quality gate
    detect/       # NDWI diff, SAR backscatter ratio, weather-adaptive router, scenario masks
    geo/          # D8 corridor, tolerance buffers, exposure intersections
    risk/         # hazard H, exposure E, disease D_risk, SAR priority + reasons
    ml/           # optional ML evidence layer (deterministic fallback, torch-gated)
    alerting/     # <250-byte payload codec, simulated dispatch
    audit/        # append-only log writer + SHA-256 hash chain
    db/           # SQLite schema + repositories
    pipeline.py   # orchestrator: detect→geo→risk→DB→audit
  tests/
    fixtures/     # synthetic rasters + fake OSM GeoJSON (tiny, committed)
frontend/
  src/
    views/        # MapView, TimelineView, ReviewView, AuditView
    api/          # typed client + offline mock fallback
    simulation/   # SimulationContext (shared demo state)
    components/   # OfflineBadge (online/offline event listener)
    theme/        # ThemeProvider + ThemeToggle (Ops Dark / Light / Satellite)
    utils/        # ntfy.ts — shared ntfy.sh live alert utility
data/
  raw/            # downloaded scenes — gitignored, never hand-edited
  processed/      # aligned rasters, masks — gitignored, written only by pipeline
  assets/         # basin GeoJSON, OSM extracts, weather series — small, committed
docs/
```

## Commands

```bash
# backend
cd backend && pip install -e ".[dev]"
uvicorn siren.api:app --reload --port 8010
pytest                             # 782 tests

# frontend
cd frontend && npm install && npm run dev   # port 5175, proxies /api → 8010
```

---

## Hard Rules (no exceptions)

1. **Dual-track architecture (ADR-013).** The deterministic/empirical baseline (NDWI, SAR backscatter ratio, D8 corridor, Huggel formula, calibrated 2-channel FNO) is the active, load-bearing primary path. Neural modules (E0–E3: latent conditioning, MC Dropout uncertainty, neural bathymetry, multi-modal fusion) are experimental research scaffolds gated behind ADR-013 promotion criteria. No neural module is load-bearing until its gate is evaluated on real held-out data. The deterministic baseline must never be silently removed or bypassed.
2. **Offline demo.** Zero network calls at runtime. All data loads from `data/`. Live API ingestion is a bonus script, never a runtime dependency.
3. **Human gate.** No code path may dispatch an alert without a recorded review decision (`confirm`). Reject/postpone must suppress dispatch.
4. **Payload ≤ 250 bytes.** Enforced by a unit test, not by hope.
5. **Explainability.** Every score object carries a `reasons` array (≥3 entries on elevated+). Never return a bare number.
6. **Reproducibility.** Same inputs + processing version → identical outputs. No unseeded randomness anywhere.
7. **Data hygiene.** Only `ingest/` scripts write to `data/raw`; only the pipeline writes `data/processed`. Never commit rasters. Never hand-edit data files.
8. **Dependency whitelist.** rasterio, geopandas, shapely, numpy, xarray, pysheds, fastapi, pydantic, pytest. **Exception:** torch/torchvision are allowed under the `[ml]` extra for the evidence layer and E0–E3 neural research scaffolds (ADR-013); the deterministic fallback runs without them. Anything else: stop and ask.
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

## Definition of Done

Offline, in one click-chain: baseline loads → 3 observations process through the pipeline → elevated/critical review card appears with ≥3 evidence reasons → Confirm produces a ≤250-byte simulated dispatch → audit log reconstructs the full lineage with SHA-256 hash chain. If your change breaks this chain, fix it before anything else.
