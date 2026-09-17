# SIREN — Post-Hackathon Improvement Plan

SIREN stays a decision-support monitoring system for the Dudh Koshi / Imja basin.
This file tracks the portfolio-grade improvement roadmap agreed 2026-03-15.

## Scope

- System scope (unchanged): Imja / Dudh Koshi AOI, human gate, ≤250B dispatch, SHA-256 audit, offline runtime. Architecture direction (updated 2026-09-17, PRD v5.1): DL-primary target — neural components promote to primary through §9.8/§17.2 held-out gates; deterministic modules are the interim primary path and permanent fallback/cross-check.
- Training-data scope (widens): the ML shadow model needs high-altitude training data. `data/datasets/glacial_lake_2022-2024/` already contains 31,698 verified glacial lake polygons (67–104°E), incl. 21,003 lakes at 4500–6000 m and Imja Tsho itself (GL_27.89829_86.92818, 1.74 km²).

## Ordered tasks

| # | Task | Why | Status |
|---|------|-----|--------|
| 1 | State/change split — `water_extent_t1` + `expansion` + `drainage` through engine → processed rasters → API bounds → ReviewView layer toggle + stats | Verified design flaw: a better model makes a persistent lake disappear under change-only semantics | DONE — engine `predict_state_and_change`, gated state layers, `/ml-evidence` fields, ReviewView toggles + delineation summary, real-pipeline verified (obs-002: extent 10.6 km² / expansion 1.30 / drainage 3.87 km²) |
| 2 | Himalayan lake chip dataset — rasterize lake polygons onto the SAR grid, build S1 dual-pol chip extractor (reuse `sar_grid_*` helpers), coverage stats report | Real fix for the 61k-px glacier false-positive problem; training data now on disk | DONE — `ml/himalayan_lake_dataset.py`, GCP vertex-mapped rasterization, 697 lake chips + 13 bg @ 96px (elev 4031–5844m, area 0.02–5.45 km²), weak-label semantics recorded |
| 3 | Decoder fine-tune on lake chips — freeze encoder, fine-tune bottleneck/decoder, eval vs terrain-gate metrics | Real domain adaptation vs the weak-label experiment | DONE — `ml/lake_adapter_finetune.py`, frozen enc1–4 (6.7M/7.9M trainable). pos_weight=5 variant: glacier water-extent FPs 115,100→8,738 px (−92%), expansion-on-glacier 30,151→3,396 (−89%), water extent 729k→125k (−83%). Imja recall 71% t1 / 87% t0 — model now detects persistent lake at both dates (rule_overlap→0 is correct behaviour, not failure). Shadow-only, not wired to runtime |
| 4 | Pitch/case-study docs — reframe as audited monitoring pipeline; write up dem_slope units, GCP geotransform, 65 km window-shift, Kuro Siwo domain shift | Portfolio framing | DONE — README pitch reframed ("audited satellite monitoring pipeline", "does not predict floods"), state/change + adapter sections added, `docs/reference/CASE_STUDY.md` written (3 silent bugs, domain-shift defenses, state/change discovery, South Lhonak validation, negative results) |
| 5 | GeoClaw on South Lhonak corridor (validate vs Pleiades post-event DEM) — replaces synthetic FNO eval | Real-terrain hydrodynamics; no deadline pressure now — viable when chosen | deferred |
| 6 | Phase 4 production items (Twilio/SNS, LoRa hardware, Celery) | Conflicts with offline demo; no hardware | skipped |
| 7 | **Held-out adapter eval** — independent S1A descending pairs (shoulder 2025-11-09/11-21, winter 2026-01-08/01-20), same track; calibrate via `extract_and_cache_vv_vh_db`, run `ml/heldout_eval.py` (base vs adapter: glacier FPs, inventory/Imja recall, state/change) | Prior metrics were in-scene (train+eval on July pair) — circular; winter pair stress-tests FP suppression, shoulder pair keeps recall valid | DONE — 4 scenes downloaded, calibrated, evaluated. FP suppression generalises: glacier extent FPs −77% shoulder / −70% winter; Imja recall 4.8%→41.3% shoulder. Caveats: inventory-wide recall 27.8%→16.3% (selective toward large lakes), winter recall 0% both models (frozen ≠ water, physics). `heldout_eval_report.json` committed; README + CASE_STUDY updated |
| 8 | **Deterministic thermal-state gate** — `LakeThermalState` enum + ERA5-Land lapse-rate estimator (`detect/thermal_state.py`, committed `data/assets/lake_thermal_series.json` via `ingest/lake_thermal_series.py`); pipeline flags frozen SAR layers unreliable, marks drainage non-hydrological, suppresses breach-volume/hydro trigger; FROZEN badge in ReviewView | Held-out eval proved 0% winter recall for ALL models — a sensor limitation to gate deterministically, not a training problem | DONE — estimator verified on real dates (Nov −8°C frozen, Jan −15°C frozen, Jul +3°C liquid); 9 tests incl. hydro-trigger suppression; full backend suite + frontend vitest clean; obs-002 pipeline run shows `liquid` +3.1°C with no flags (DoD intact) |
| 9 | **Stratified area rebalancing** — `--stratified` flag in `lake_adapter_finetune.py` (equal per-epoch sampling across micro/medium/large bins) + optional `--loss-weight invsqrt`; per-bin recall added to `heldout_eval.py` | Held-out inventory recall regressed 27.8%→16.3% under v1 adapter — test whether gradient imbalance vs resolution floor | DONE — v2 stratified: aggregate recall recovered (24.7%) + glacier suppression improved (−87%) BUT Imja recall collapsed 41.3%→4.2%; micro-tarns unrecoverable at ~90m pitch (2–6 px = resolution floor). v1 remains the deployment candidate; stratification trades target recall for breadth |

## Rules carried forward

- Neural promotion is gate-gated and component-wise (PRD §9.8): deterministic masks/scores stay authoritative until a component passes its held-out real-data gate, then it becomes primary with the deterministic module demoted to labeled cross-check — never silently, always reversible.
- Human confirmation stays mandatory; offline runtime stays network-free.
- DoD click-chain must keep working end-to-end.
