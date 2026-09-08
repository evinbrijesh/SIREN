# V3 Research Proposal — Predictive Upgrade (ADR-011 Draft RFC)

**Status:** Proposal (not accepted) · **Date:** 2026-09-08 · **Supersedes:** selected clauses of [ADR-010](../adr/ADR-010-ml-evidence-isolation-and-retraining-path.md) (only on acceptance — see §6)
**Companions:** [ADR-002](../adr/ADR-002-deterministic-first-ml.md), [ADR-010](../adr/ADR-010-ml-evidence-isolation-and-retraining-path.md), [`PRODUCTION_ML_PLAN.md`](../reference/PRODUCTION_ML_PLAN.md), [`DL_MODEL_AUDIT.md`](../reference/DL_MODEL_AUDIT.md)

> **This document is a research RFC, not a build order.** Nothing here authorizes changes to the frozen hackathon MVP. The active pipeline stays 2-channel, deterministic-first, and shadow-gated per ADR-010. Hard Rule 8 (dependency whitelist) is unchanged until §5 is accepted. All work below is post-hackathon.

---

## 1. Motivation

The hackathon MVP (ADR-010 Stage 1) ships a 2-channel WaterUNet shadow segmenter with event-holdout IoU **0.24** — below the 0.65 load-bearing gate. Three failure modes remain that the deterministic rules cannot resolve by threshold-tuning:

1. **Mountain radar-shadow false positives.** Steep Himalayan topography produces dark backscatter in SAR layover/shadow that is indistinguishable from open water by VV/VH alone. A slope/elevation prior is the known fix.
2. **Static breach heuristics.** The five-factor hazard score uses fixed linear weights. Calibrated breach probability from historical inventories would replace guesswork with a defensible P_breach.
3. **Static buffer corridors.** D8 + OSM buffers answer "what is in the path" but not "when does the wave arrive." A hydrodynamic surrogate turns buffers into time-of-arrival estimates.

This RFC specifies a three-phase upgrade that addresses all three, gated so that no ML enters a load-bearing path until it measurably beats the rules-only baseline on held-out data.

---

## 2. Phase 1 — Elevation-Conditioned Segmenter (DEM-Aware WaterResUNet)

### 2.1 Tensor contract expansion

Expand `backend/siren/ml/contract.py` from 2 → 4 channels. This **invalidates all existing 2-channel weights** — a new checkpoint must be trained from scratch against the new contract.

| Channel | Name | Source | Normalization |
|---|---|---|---|
| 0 | VV σ0 dB | Sentinel-1 GRD | clamp [-30, 0] → [0, 1] (unchanged) |
| 1 | VH σ0 dB | Sentinel-1 GRD | clamp [-30, 0] → [0, 1] (unchanged) |
| 2 | DEM | Copernicus GLO-30 (or SRTM 1″) | `elevation / 8848.0` → [0, 1] |
| 3 | Slope | derived from DEM | `slope_deg / 90.0` → [0, 1] |

Contract constants to add: `DEM_MAX_M = 8848.0`, `SLOPE_MAX_DEG = 90.0`, `SAR_CHANNELS = 4`, `CHANNEL_NAMES = ("VV", "VH", "DEM", "Slope")`.

### 2.2 Data preparation

- Co-register Copernicus GLO-30 (or SRTM 1″) to each Sen1Floods11 chip grid (extend `preprocess/clip.py`).
- Derive slope in degrees via numpy gradient on the DEM (the display-only hillshade calc in `api/map_assets.py` is a reference, not the training input — write a reusable `preprocess/dem.py::slope_degrees()`).
- Augment `ml/dataset.py` to stack (VV, VH, DEM, Slope) per chip.

### 2.3 Architecture

WaterResUNet — a ResUNet variant of the existing `WaterUNet` (≤ ~10M params), 4-channel input, residual encoder blocks. Built from scratch (no ImageNet init — wrong modality, per ADR-010 audit).

### 2.4 Training & evaluation gate

- Train on Sen1Floods11 hand-labeled chips with the **official event-level holdout split** (never random chip splits — ADR-010 §4.7).
- **Load-bearing gate: event-holdout water IoU > 0.65** (the current 2-channel model scores 0.24). The slope prior is expected to filter mountain-shadow false positives, the primary failure mode.
- Report permanent-vs-new-water error separation and a prospective evaluation on untouched basin dates before any display as evidence.
- Until the gate is met, the 4-channel model stays shadow-only and the 2-channel contract remains frozen.

### 2.5 Acceptance criteria

- [ ] 4-channel contract implemented and frozen
- [ ] DEM + slope co-registration pipeline on synthetic 100×100 fixtures (unit tests)
- [ ] WaterResUNet trained; event-holdout IoU reported
- [ ] IoU > 0.65 demonstrated on the strict event-holdout split
- [ ] Shadow-mode wiring into `pipeline._try_ml_evidence_layer` (display only, no hazard-score term)

---

## 3. Phase 2 — Spatio-Temporal Trend & Breach Susceptibility

### 3.1 Trailing persistence engine (deterministic, replaces ConvLSTM per ADR-010)

Measures non-linear ΔArea/Δt across 5+ SAR passes with explicit elapsed-day features and an "insufficient observations" outcome. This is the deterministic trend estimator ADR-010 mandates; it is **not** a learned model. A learned spatio-temporal patch transformer is deferred until ≥ 2 seasons of real observations exist (ADR-010 §2).

### 3.2 Susceptibility scorer — `backend/siren/risk/susceptibility.py` (new)

An XGBoost gradient-boosted tree classifier trained on ICIMOD historical glacial-lake attributes:

| Feature | Source |
|---|---|
| Lake expansion rate (ΔArea/Δt) | Phase 3.1 trailing engine |
| Moraine dam width / height | ICIMOD GLOF inventory |
| Rain anomaly (7d vs climatology) | IMERG / ERA5 |
| Mean slope upstream | DEM |
| Lake area (absolute) | HMA Glacial Lake Catalog |

Output: calibrated breach probability `P_breach ∈ [0, 1]`.

### 3.3 Explainability — TreeSHAP

Compute feature contributions via `shap.TreeExplainer`. Pass the top-k SHAP contributions into the `ReviewView` evidence `reasons` array (Hard Rule 5: ≥3 entries on elevated+). The susceptibility score is recorded as separate evidence; it does **not** enter the five-factor hazard score until §6 acceptance (shadow-first).

### 3.4 Required datasets

- **ICIMOD Glacial Lake & GLOF Inventories** — historical moraine-dammed lake failures, lake area evolution, dam geometry.
- **High Mountain Asia (HMA) Glacial Lake Catalog** — lake boundary tracking 1990–present for temporal expansion features.

### 3.5 Acceptance criteria

- [ ] ICIMOD + HMA ingest scripts under `ingest/` with provenance sidecars
- [ ] XGBoost classifier trained; calibrated P_breach (Brier score reported)
- [ ] TreeSHAP contributions wired into `reasons` arrays
- [ ] Shadow-mode evaluation: disagreement rate vs rules-only baseline measured before any load-bearing role

---

## 4. Phase 3 — FNO Hydrodynamic Surrogate

### 4.1 Model — `backend/siren/geo/hydro_surrogate.py` (new)

A 2D Fourier Neural Operator (FNO-2D) trained on synthetic shallow-water simulation runs:

- **Input:** estimated breach volume `V_breach` (from Phase 2 P_breach + lake geometry) + DEM valley profile.
- **Output:** dynamic water depth grid `h_water` and wave arrival time `T_arrival` (minutes) at named points (Hillary Bridge, Benkar, Jorsale for Dudh Koshi; Teesta III + 31 bridges for South Lhonak).

### 4.2 Training data

Generate 500–1,000 synthetic HEC-RAS 2D simulation runs across varying `V_breach` and hydrographs to produce ground-truth flood wave arrival times and dynamic flow depths. No external HEC-RAS license assumption is documented here — resolve before build.

### 4.3 Trigger gate

FNO inference is triggered only when `P_breach ≥ 0.70` from Phase 2. Below the gate, the deterministic D8 + OSM buffer corridor (ADR-005, frozen) remains authoritative.

### 4.4 Validation

Retrospective validation on the **South Lhonak Oct 2023 GLOF** (pre/post Sentinel-1/2 + documented Teesta valley inundation extents). The success criterion: predicted `T_arrival` at Teesta III within an agreed tolerance of the documented flood travel time.

### 4.5 Downstream wiring (Phase 4)

- Replace static tolerance-buffer intersection with `h_water > 0.3 m` grid intersect against OSM (currently `geo/corridor.py` uses ±75/50/100 m buffers — do not tighten the buffers per CLAUDE.md gotchas; the dynamic grid supersedes them only when FNO output is available).
- Feed `T_arrival` into `risk/personnel.py` to prioritize evacuation urgency.
- Extend the ≤250-byte payload codec to serialize peak wave height + `T_arrival` within the byte budget (the 250-byte unit test must stay green).

### 4.6 Acceptance criteria

- [ ] HEC-RAS synthetic run generation harness
- [ ] FNO-2D trained; arrival-time MAE reported on held-out runs
- [ ] South Lhonak retrospective validation within tolerance
- [ ] `h_water` grid intersect + `T_arrival` wired into corridor + personnel + payload (250-byte test green)

---

## 5. Dependency Addendum Request (Hard Rule 8 amendment)

Hard Rule 8 currently whitelists: rasterio, geopandas, shapely, numpy, xarray, pysheds, fastapi, pydantic, pytest (+ torch/torchvision for the evidence layer only). This RFC requests adding, **as optional `[v3]` extras only**, mirroring the torch exception pattern:

| Package | Phase | Justification | Risk |
|---|---|---|---|
| `xgboost` | 2 | Gradient-boosted trees for calibrated P_breach; TreeSHAP-compatible | Pure-python wheel; low binary risk |
| `shap` | 2 | Feature attribution for explainability (Hard Rule 5) | Pure-python; low risk |
| `neuraloperator` (or in-torch FNO impl) | 3 | FNO-2D surrogate | Review license + binary deps before adoption |

**Until this section is accepted by a new ADR, none of these packages may be installed or imported.** The active build remains strictly whitelist-compliant.

---

## 6. Relationship to ADR-010 (supersession scope)

This RFC does **not** replace ADR-010 wholesale. It supersedes only these clauses, and only on acceptance of the relevant phase's evaluation gate:

| ADR-010 clause | V3 change | Condition |
|---|---|---|
| 2-channel frozen contract (§4.1) | 4-channel contract | Phase 1 event-holdout IoU > 0.65 |
| "No ML term in hazard score" (§3) | P_breach may enter H as a sixth factor | Phase 2 shadow-mode shows calibrated improvement over rules-only baseline + new ADR |
| ConvLSTM replaced by deterministic trend (§1, §2) | Deterministic trailing engine stays; learned transformer still deferred | (no change — Phase 3.1 is the deterministic engine ADR-010 already mandates) |
| ML shadow-only, never load-bearing (§3) | FNO `h_water`/`T_arrival` may supersede static buffers | Phase 3 South Lhonak validation within tolerance + new ADR |

All other ADR-010 clauses (input contract discipline, event-level splits, license review, model registry honesty, shadow-mode-first) remain in force.

---

## 7. End-to-end target pipeline (post-acceptance)

```
[S1 SAR VV/VH] + [Copernicus GLO-30 DEM] + [IMERG/ERA5]
                      │
                      ▼
   Phase 1: 4-channel DEM-aware WaterResUNet  → water probability mask
                      │
                      ▼
   Phase 2: trailing persistence (ΔArea/Δt) + XGBoost P_breach (TreeSHAP reasons)
                      │
                      ▼
   Phase 3 (if P_breach ≥ 0.70): FNO-2D → h_water grid + T_arrival
                      │
                      ▼
   Phase 4: h_water>0.3m ∩ OSM exposure + D_risk + personnel urgency by T_arrival
                      │
                      ▼
   Human gate → ≤250-byte dispatch (now incl. peak height + T_arrival) → SHA-256 audit
```

The deterministic five-factor path remains the fallback at every stage when ML is unavailable or below gate.

---

## 8. Open items before any V3 build

- License review: Sen1Floods11 terms, WorldFloods (CC non-commercial), SSL4EO-S12, neuraloperator.
- HEC-RAS availability / licensing for synthetic run generation.
- ICIMOD + HMA inventory access terms.
- Copernicus DEM vs SRTM qualification for slope/D8 (ADR-005 requalification if DEM switched).
- South Lhonak Sentinel-1 covering orbit identification (PRODUCTION_ML_PLAN §5).

---

## 9. Consequences

- **Positive:** addresses the three failure modes the deterministic rules cannot; every upgrade is gated on beating the rules-only baseline on held-out data; the frozen MVP is untouched.
- **Negative:** three new optional dependencies; retraining cost; HEC-RAS data generation is non-trivial; a future load-bearing ML role requires a new ADR beyond this RFC.
- **Frozen-pipeline note:** no clause here authorizes changes to the active hackathon build. All V3 work is post-demo.
