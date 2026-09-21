# CLAUDE.md — SIREN

Companion to `AGENTS.md` (hard rules + data contracts) and `docs/spec/PRD.md` (v5.1 spec). Read all three before writing code.

**SIREN** — Satellite-Informed Risk & Emergency Network. Satellite-assisted early warning and disaster-response platform for Himalayan glacial lake outburst flood (GLOF) events. DL-primary target architecture: trained neural components (segmentation, bathymetry, latent-conditioned FNO, learned risk fusion) form the intended analytical spine; deterministic modules are the interim primary path and permanent fallback/cross-check, promoted component-wise through the §9.8/§17.2 gates (ADR-013, PRD v5.1).

---

## Stack

- **Backend:** Python 3.11+, FastAPI, rasterio, geopandas, shapely, numpy, xarray, pysheds (fallback: whitebox), SQLite (JSON columns). ML: torch/torchvision under `[ml]` extra for evidence layer + E0–E3 neural research scaffolds (ADR-013); deterministic fallback runs without them.
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
    detect/       # NDWI diff, SAR backscatter ratio, weather-adaptive router, change stats (deterministic fallback)
    geo/          # combined D8 + OSM river corridor, tolerance buffers, exposure intersections
                   # hydro_surrogate.py — FNO2D (latent-conditioned, ADR-013)
    risk/         # hazard H, exposure E, disease D_risk, SAR priority + reasons
                   # breach_volume.py — hypsometric / Huggel / neural bathymetry cascade
    ml/           # ML pipeline (primary analytical path, ADR-013)
                   #   model.py — WaterResUNet (6ch SAR, MC Dropout-ready)
                   #   fusion.py — multi-modal SAR+optical cross-attention (E3)
                   #   bathymetry.py — neural bed elevation estimator (E2)
                   #   bathymetry_dataset.py — unified loader for 20 surveyed lakes (Zhang 2023 + Das 2025)
                   #   bathymetry_benchmark.py — LOO volume estimation benchmark (Huggel vs regression vs neural)
                   #   bathymetry_training_data.py — (DEM, lake_mask, bed_elevation) sample builder from Copernicus GLO30
                   #   uncertainty.py — MC Dropout inference + conformal calibration (E1)
                   #   latent_coupling.py — segmentation bottleneck → FNO conditioning (E0)
                   #   registry.py — model registry with gate status + provenance
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

- **Module boundaries:** `api/` routes must not contain logic — delegate to `detect/`, `geo/`, `risk/`, `ml/`, etc. Keeps the pipeline testable without HTTP.
- **Data contracts:** implement PRD §10 field names/types exactly. Never rename a field to "make it nicer."
- **Scoring:** every score object carries a `reasons` array (≥3 entries on elevated+). Never return a bare number.
- **Errors:** raise typed exceptions; FastAPI handlers map them to structured JSON `{error, detail}`. No silent fallbacks that hide data gaps.
- **Logging:** use Python `logging`; log run_id/observation_id on every pipeline step for lineage.
- **Reproducibility:** no unseeded randomness. Seed any RNG explicitly.
- **Payload size:** the ≤250-byte alert constraint is enforced by a unit test, not by hope.
- **Commit granularity:** group related tasks into a few logical commits — never one commit per action. N tasks sharing a theme → 1 commit (e.g. 5 tasks, 3+2 related → 2 commits). No "Generated with …", co-author trailers, or tool/agent mentions in commit messages.
- **Neural promotion + fallback provenance (v5.1):** the target architecture is neural-primary (PRD §9.8 promotion pathway); deterministic modules are the interim primary path and permanent labeled fallback/cross-check. No neural module is load-bearing until its §9.8/§17.2 gate is evaluated on held-out real deployment-domain data; promotion is component-wise and reversible. Every component — neural or deterministic — records its method (primary vs fallback) in the result provenance. The deterministic baseline must never be silently removed or bypassed, before or after promotion.

## Known Gotchas

- **WhiteboxTools vs pysheds:** prefer pysheds (pure Python, no binary). If D8 flow accumulation misbehaves on steep/karst terrain, fall back to OSM river buffering (Roadmap Phase 3 fallback).
- **pysheds 0.5 + numpy 2.x incompatibility (CRITICAL):** pysheds 0.5 calls `np.in1d`, which was removed in numpy 2.x. **Must monkeypatch `np.in1d = np.isin` before any pysheds call.** The `geo/` module does this at import. Do not "fix" by downgrading numpy — rasterio/geopandas need numpy 2.
- **pysheds 0.5 API:** use `grid.read_raster(path)` which returns a `Raster` object (NOT `grid.dem` / `grid.view('dem')` — those don't exist in 0.5). Pass the returned Raster to `fill_depressions(dem=...)`.
- **AOI is a rectangle, not a watershed:** D8 accumulation max is limited (~7k cells) because the river exits the box edges. Expected — the corridor is computed within the AOI. If you need the full river, extend the DEM downstream.
- **Cloud routing:** optical cloud_fraction ≥ 0.20 flips the pipeline to SAR-primary. SAR is treated as all-weather (cloud_fraction = 0.0 on that path). The original optical cloud is preserved as `optical_cloud_fraction` for display.
- **Severity thresholds:** expansion ≥40% → critical; ≥20% → elevated; ≥5% → watch; <5% → informational. These are policy thresholds in `risk/fusion.py::classify_severity()`.
- **SAR always routed:** all three demo observations use S1 SAR. obs-002/obs-003 have 95%/90% optical cloud → SAR-primary. obs-001 is SAR by source. All 3 obs have real ESA SAFE archives downloaded from CDSE (obs-003 downloaded 2026-09-08). The ML evidence layer prefers the descending-pass pair (07-02 + 07-14), which covers Imja lake — resolved via `find_imja_descending_pair()` in `preprocess/sar_calibrate.py`, with per-observation ascending-scene fallback; provenance persists to `change_stats` and the `/ml-evidence` API.
- **OSM extract:** 5,691 features refreshed via Overpass (2026-09-08). Previous extract was ~1,100 features. Exposure intersections now yield ~1,614 per observation (51 bridges, 23 settlements, 3 wells, 1,536 roads, 1 health).
- **Tolerance buffers:** bridges ±75 m, roads ±50 m, settlements/wells ±100 m. These exist to prevent false intersections at 10–30 m satellite resolution — do not "tighten" them.
- **Offline demo:** zero network calls at runtime. All data loads from `data/`. Live ingestion is a bonus script, never a runtime dependency.
- **SQLite spatial joins:** run in-memory via geopandas on the small basin extract. Do not reach for PostGIS.
- **SAR shadow evidence is terrain-gated (2026-09-17):** the displayed `ml_shadow_mask` is gated by AOI + slope >15° + RGI glacier-minus-lake-vicinity via GCP geolocation (`sar_grid_lonlat`/`sar_grid_polygon_mask`/`sar_grid_sample`/`sar_grid_dem_slope` in `detect/sar.py`). Rule mask and scoring untouched. `dem_slope` had a per-pixel-vs-per-degree bug — it returned ~0 everywhere; fixed. True geographic ML∩rule overlap on Imja is ~13–22 px (index-resize "agreement" was spurious).
- **`baseline_water_mask` windowed-profile bug (fixed 2026-09-17):** `detect/ndwi.py::read_s2_band` returned the full-tile profile for an AOI-windowed read, stamping the baseline water mask ~65 km west in Rolwaling. Now returns `window_transform(win)`; the tif/png were regenerated over the AOI.
- **E3 optical separability evaluated (2026-09-17):** `ml/s2_spectral_eval.py` measures lake-vs-glacier NDWI/MNDWI/SCL separability per scene. Verdict: indices do not separate — glacier reads *higher* MNDWI than the frozen lake (AUC 0.18–0.29), and the S1-paired Jul-05 scene is 72.7% cloud. Fusion training is not justified on this data; reports in `models/checkpoints/s2_spectral_eval_*.json`.
- **SRTM is a DSM (v5.0):** SRTM over water returns the flat water surface, not the lake bed. The `breach_volume.py` `auto` mode detects this via mean-depth criterion (`MIN_BATHYMETRIC_MEAN_DEPTH_M = 1.0`) and falls back to Huggel. The neural bathymetry model (E2) will replace this fallback when trained.
- **E2 neural bathymetry — LOO benchmark evaluated (2026-09-15):** 20 surveyed Himalayan glacial lakes (117k depth points) with Copernicus DEM GLO30 terrain now support real-data training. LOO volume estimation: Huggel MAPE 75.6%, power-law regression MAPE 82.1%, neural `BathymetryUNet` (100 epochs, 19 training lakes) MAPE 676.4%. All three fail the 15% ADR-013 gate. The neural model is overparameterised for 19 samples. Transfer learning (2,000 contract-matching synthetic basins → per-fold fine-tune, `train_bathymetry.py --pretrain-synthetic` / `--pretrain-weights`) improves sample MAPE to 328.8% vs 676.4% — still far above gate; Huggel remains the best method at this dataset size. **Metadata benchmark (2026-09-17):** `load_global_compilation` reads all 5 sheets (323 entries / 267 lakes); `train_bathymetry.py --benchmark-metadata` runs grouped-LOO — Himalaya subset Huggel MAPE 33.5% (regression 37.5%), overall 76.6%/67.5%. Published lookups for dense lakes are proglacial-scoped (`_published_entries_for_dense_lakes`); other sheets have same-named lakes with incomparable values (Bencoguoco). See `ml/bathymetry_benchmark.py`, `ml/bathymetry_training_data.py`, `train_bathymetry.py --benchmark` / `--benchmark-metadata` / `--train-real`.
- **South Lhonak real-DEM diagnostic (2026-09-17, CORRECTED 2026-09-21):** `ml/south_lhonak_lake_eval.py` measures the 2023-10-03 GLOF from the on-disk 1 m Pleiades pre/post DEMs (CC BY 4.0). **The original `LAKE_BOX_UTM45N` (620500–622200 E) sat ~5 km east of the lake and measured a different flat feature — the box is now corrected to the actual lake (615640–619550 E, 3086680–3089488 N), verified against the published coordinates, the ICIMOD inventory polygon, DEM differencing, S2 masks, and SAR backscatter.** Corrected numbers: flat-surface area 0.385 km² (a lower bound — photogrammetric DEMs are not flat over water); S2 NDWI mask area 1.496 km² (2023-09-26, the trustworthy input); pre/post elevation 5209→5168 m (the 41 m reflects deepest-bed exposure, not the water-level drop); visible emptied volume 8.23 MCM (strict lower bound). Huggel with the S2 area → **61.9 MCM vs the surveyed total ~65.8 MCM (Sharma et al. 2018), within ~6%** — the dominant error was the area input, not the exponent (with the old wrong-location area of 0.866 km² the comparison was misleading). The FNO `eval_south_lhonak.py` uses a *synthetic* parametric corridor and clips wave speed into the documented range — its 12.3% MAPE is not real-terrain evidence.
- **SAR segmentation real-event OOD probe — South Lhonak 2023 FAILED (2026-09-21):** `ml/south_lhonak_sar_eval.py` runs the promoted checkpoint on the only on-disk SAR pair bracketing a real GLOF (2023-09-28 → 2023-10-10, burst 2023-10-03). Result: post-event IoU **0.028**, recall **3.6%** at τ=0.3; **38,481 of 38,489 predicted pixels (99.98%) are false positives** elsewhere in the scene. The lake's SAR signature (VV −15.4 → −12.3 dB, dVV +3.1 dB) is in the change regime where Imja succeeds, but its absolute backscatter sits above Imja's water range (−16 to −19 dB) and the post-event surface carried icebergs/debris (rough, not specular). The SRTM tile covers only the Dudh Koshi AOI, so the runtime slope gate is **inoperative** at South Lhonak (only the RGI glacier gate applies — it removes ~1k of the 38k FPs). **Conclusion: the certified scope is Imja-area only — a hard requirement.** Multi-lake monitoring needs multi-lake training data + region-wide DEM coverage. Report: `models/checkpoints/south_lhonak_sar_eval.json`.
- **Promoted advisory-primary components (2026-09-19):** `PROMOTED_COMPONENTS` in `ml/promotion.py` now contains `sar_segmentation_expansion` (checkpoint: `water_resunet_6ch_himalayan_adapter_multidate.pt` — multi-date fine-tune, 50% verified-change recall on the gold pair), `susceptibility` (spatial XGBoost + isotonic, conformal q=0.82 — wide interval → `requires_manual_inspection`), and `dynamic_escalation` (Tier-2 weather/morphology XGBoost on NASA POWER, offline asset `data/assets/imja_power_series.json`). `ChangeDetectionEngine._candidate_paths` resolves the promoted checkpoint through the registry — the promotion record describes what runs. Promoted outputs are advisory evidence on the review card; deterministic severity + human gate unchanged. Residual-segmentation corrector was evaluated and rejected (saturated negative logits OOD — see `residual_seg_gold_eval.json`).
- **Operational scope — monsoon window Jun–Sep (2026-09-21):** `backend/siren/scope.py` declares the vulnerable-season scope: **Jun–Sep** (82.6% of 230 dated GLOF events, ~80% of annual rainfall, liquid lake surface +3 to +6 °C). In scope = monsoon window + descending orbit + liquid surface + Imja-area lakes. Out-of-scope (explicitly deferred): frozen/shoulder seasons, ascending orbit, multi-lake. The pipeline annotates every observation with `change_stats["operational_scope"]` and appends the scope note as a review reason when out of window; the deterministic baseline stays authoritative either way. The operational gate verdict is scope-aware (`in_scope_pairs_*` gate; `out_of_scope_pairs` documented probes). When adding new components/gates: declare their scope in the same vocabulary (months + orbit + surface state + region) and route out-of-scope observations to the fallback path with a visible reason — never silently drop them.
- **FNO input contract (v5.0):** `FNO2D` accepts `in_channels=2` (scalar-only, the frozen ADR-012 baseline) or `in_channels=2+d_latent` (experimental latent-conditioned, ADR-013). The frozen scalar checkpoint (`input_proj.weight: [32, 2]`) loads only with `in_channels=2`. The latent-conditioned variant is experimental — no trained checkpoint exists yet. See ADR-013-addendum for the dual-contract policy.

## Module Map

| Concern | Owned by |
|---|---|
| Downloading scenes | `ingest/` |
| Clip/reproject/align/quality | `preprocess/` |
| Change masks + stats (deterministic fallback) | `detect/` |
| Corridor + exposure | `geo/` |
| FNO hydrodynamic surrogate (latent-conditioned) | `geo/hydro_surrogate.py` |
| H / E / D_risk scores | `risk/` |
| Breach volume (neural/Huggel/hypsometric cascade) | `risk/breach_volume.py` |
| SAR priority ranking | `risk/sar_priority.py` |
| Water segmentation (SAR 6ch + multi-modal fusion) | `ml/model.py`, `ml/engine.py`, `ml/fusion.py`, `ml/fusion_dataset.py`, `ml/train_fusion.py` |
| S2 optical feature extraction (NDWI/MNDWI/cloud) | `preprocess/s2_optical.py` |
| Neural bathymetry inversion | `ml/bathymetry.py` |
| Bathymetry dataset + LOO splits | `ml/bathymetry_dataset.py` |
| Bathymetry volume benchmark | `ml/bathymetry_benchmark.py` |
| Bathymetry training data pipeline | `ml/bathymetry_training_data.py` |
| Bayesian uncertainty (MC Dropout + conformal) | `ml/uncertainty.py` |
| Latent coupling (segmentation → FNO) | `ml/latent_coupling.py` |
| Model registry + gate status | `ml/registry.py` |
| Payload codec + dispatch | `alerting/` |
| Append-only lineage + hash chain | `audit/` |
| Persistence | `db/` |
| HTTP surface + map assets | `api/` |
| Map/timeline/review/audit UI | `frontend/src/views/` |
| Theme system | `frontend/src/theme/` |
| Offline status | `frontend/src/components/` |

## Definition of Done

Offline, in one click-chain: baseline loads → 3 observations process → elevated/critical review card with ≥3 evidence reasons → Confirm produces a ≤250-byte simulated dispatch → audit log reconstructs the full lineage with SHA-256 hash chain. If a change breaks this chain, fix it before anything else.

**v5.1 extension:** the click-chain should also display the method provenance (neural-primary vs deterministic-fallback) at each stage and the uncertainty map alongside the water mask. **Note:** neural modules promote to primary only through the §9.8/§17.2 gates on held-out real data — until then the deterministic baseline remains the load-bearing path and every neural output is labeled shadow evidence.