# SIREN — Post-Hackathon Improvement Plan

SIREN stays a decision-support monitoring system for the Dudh Koshi / Imja basin.
This file tracks the portfolio-grade improvement roadmap agreed 2026-03-15.

## Scope

- System scope (unchanged): Imja / Dudh Koshi AOI, deterministic pipeline, human gate, ≤250B dispatch, SHA-256 audit, offline runtime.
- Training-data scope (widens): the ML shadow model needs high-altitude training data. `data/datasets/glacial_lake_2022-2024/` already contains 31,698 verified glacial lake polygons (67–104°E), incl. 21,003 lakes at 4500–6000 m and Imja Tsho itself (GL_27.89829_86.92818, 1.74 km²).

## Ordered tasks

| # | Task | Why | Status |
|---|------|-----|--------|
| 1 | State/change split — `water_extent_t1` + `expansion` + `drainage` through engine → processed rasters → API bounds → ReviewView layer toggle + stats | Verified design flaw: a better model makes a persistent lake disappear under change-only semantics | DONE — engine `predict_state_and_change`, gated state layers, `/ml-evidence` fields, ReviewView toggles + delineation summary, real-pipeline verified (obs-002: extent 10.6 km² / expansion 1.30 / drainage 3.87 km²) |
| 2 | Himalayan lake chip dataset — rasterize lake polygons onto the SAR grid, build S1 dual-pol chip extractor (reuse `sar_grid_*` helpers), coverage stats report | Real fix for the 61k-px glacier false-positive problem; training data now on disk | DONE — `ml/himalayan_lake_dataset.py`, GCP vertex-mapped rasterization, 697 lake chips + 13 bg @ 96px (elev 4031–5844m, area 0.02–5.45 km²), weak-label semantics recorded |
| 3 | Decoder fine-tune on lake chips — freeze encoder, fine-tune bottleneck/decoder, eval vs terrain-gate metrics | Real domain adaptation vs the weak-label experiment | DONE — `ml/lake_adapter_finetune.py`, frozen enc1–4 (6.7M/7.9M trainable). pos_weight=5 variant: glacier water-extent FPs 115,100→8,738 px (−92%), expansion-on-glacier 30,151→3,396 (−89%), water extent 729k→125k (−83%). Imja recall 71% t1 / 87% t0 — model now detects persistent lake at both dates (rule_overlap→0 is correct behaviour, not failure). Shadow-only, not wired to runtime |
| 4 | Pitch/case-study docs — reframe as audited monitoring pipeline; write up dem_slope units, GCP geotransform, 65 km window-shift, Kuro Siwo domain shift | Portfolio framing | pending |
| 5 | GeoClaw on South Lhonak corridor (validate vs Pleiades post-event DEM) — replaces synthetic FNO eval | Real-terrain hydrodynamics; heavy, deferred | deferred |
| 6 | Phase 4 production items (Twilio/SNS, LoRa hardware, Celery) | Conflicts with offline demo; no hardware | skipped |

## Rules carried forward

- Deterministic masks/scores stay authoritative; neural stays shadow-only.
- Human confirmation stays mandatory; offline runtime stays network-free.
- DoD click-chain must keep working end-to-end.
