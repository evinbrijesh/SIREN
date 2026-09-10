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

## 1.5 Level 2 Empirical Results — Locked Benchmark Matrix

**Status: LOCKED.** The following results were measured on the Sen1Floods11 event-holdout test split (Pakistan + Somalia, 54 chips, zero event overlap with training). The ML remains gated in shadow mode per ADR-011 (IoU gate = 0.65, not passed).

### Benchmark matrix

| Configuration | Test IoU | FP pixels | TP pixels | Notes |
|---|---|---|---|---|
| 2ch Baseline (VV, VH) | 0.2238 | 705,698 | 383,752 | Legacy 2-channel WaterResUNet, 45 epochs |
| 2ch + HAND (10m) | 0.2658 | 690,432 | 383,678 | +0.0027 IoU; 15,266 FP removed, 74 TP lost |
| 4ch (Fixed L_gravity) | 0.2777 | 445,750 | 332,917 | VV+VH+DEM+Slope, lambda=1.0, 45 epochs |
| **4ch + HAND (10m)** | **0.2779** | 444,887 | 332,913 | +0.0002 IoU; 863 FP removed, 4 TP lost |

### Key findings

1. **Gravity loss normalization fix (Phase 1):** The original L_gravity penalty divided elevation variance by DEM_MAX_M² (8848² ≈ 78M), collapsing the gradient to ~0.0005 — a no-op. Replacing with per-chip local elevation range (Δz_local = max(z) - min(z) + 1.0) and switching from weighted variance ratio to weighted sum made the penalty active (~0.04, ~1-4% of total loss). This fix flipped the 4ch vs 2ch comparison from -0.0355 (4ch worse) to +0.0540 (4ch better).

2. **HAND post-filter asymmetry:** The deterministic HAND filter removes physically impossible water predictions (HAND > stage threshold) without confusing convolutional filters. It provides strong improvement on the 2ch baseline (+0.0027, 15,266 FP removed) but negligible improvement on the 4ch model (+0.0002, 863 FP removed). This confirms the 4ch model's soft terrain priors already suppress most high-elevation false positives, leaving little work for the hard HAND cutoff.

3. **Indus Basin flat-terrain bottleneck:** Pakistan (Indus plain, DEM 104-118m, HAND 0-8m) benefits minimally from HAND because the flat floodplain has minimal vertical relief relative to the channel. Somalia (hillier, HAND up to 126m) benefits more. A static 10m threshold cannot filter false positives in valleys where everything sits below 5m HAND. Dynamic stage thresholds conditioned on upstream flow accumulation (h_stage ∝ A_accum^β) or temporal SAR differencing (σ0_event - σ0_pre-event) are needed for flat arid basins.

4. **OOD generalization gap:** All configurations show a large validation-to-test gap (val IoU ~0.70 → test IoU ~0.28), consistent with Bonafilia et al. (CVPRW 2020) who reported 0.28-0.33 IoU on unseen holdouts. The event-holdout split is deliberately strict — most published high scores use chip-level splits that leak event geography.

5. **Safety contract held:** Despite reaching 0.2779 (a solid step up from 0.2275), the score remains well below the 0.65 gate. The model is locked in shadow mode per ADR-011.

### Reproducibility

- Training: `python -m siren.ml.train_water_resunet --epochs 45 --lambda-gravity 1.0`
- HAND eval: `python -m siren.ml.eval_hand_filter --model 4ch --stage-threshold 10.0`
- Checkpoints: `models/checkpoints/water_resunet_4ch_v1.pt`, `water_resunet_2ch_baseline.pt`
- Results: `models/checkpoints/hand_filter_eval_4ch.json`, `hand_filter_eval_2ch.json`
- DEM: 48 Copernicus GLO-30 tiles cached in `data/raw/dem/copernicus_glo30/`

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

### 2.6 Physics-informed training penalty (L_gravity)

Standard pixel-level losses (BCE, Dice) optimize for visual similarity only — they have no representation of potential energy or gravity. A network trained this way can predict water flowing uphill, which is physically impossible.

**Fix:** introduce a gravity penalty term into the loss function using the 30 m DEM elevation:

$$\mathcal{L} = \mathcal{L}_{\text{Dice}} + \lambda \cdot \text{ReLU}\big(z(p_i) - z(p_j)\big) \quad \text{for connected flow components}$$

where $z(p)$ is the DEM elevation at pixel $p$, $p_i$ and $p_j$ are connected water-class pixels, and $\lambda$ is a penalty weight (start at $\lambda = 0.1$, tune on validation). If the network predicts water at a higher-elevation cell $p_i$ connected to a lower-elevation cell $p_j$ without sufficient upstream kinetic head, the penalty explodes.

**Implementation notes:**
- Compute connected components on the predicted water mask per training step (use `scipy.ndimage.label` on the binarized prediction).
- For each component, check elevation monotonicity along the drainage direction (use the D8 flow direction from the DEM, precomputed as a static input).
- The penalty is a soft constraint — it does not hard-clip predictions, it gradients the network toward physically plausible water connectivity.
- $\lambda$ is a hyperparameter: too high overconstrains and the network ignores valid superelevated ponds; too low and the penalty is decorative. Tune on the event-holdout split, not training loss.

**Expected effect:** eliminates the "water on a ridge" false positives that pure Dice loss produces on steep terrain. The DEM is already a 4-channel input (§2.1), so the elevation is available in-graph; the penalty reuses it in the loss without new data.

### 2.7 Radiometric Terrain Correction (γ⁰) and adversarial domain adaptation

Two additional techniques to close the OOD gap from 0.67 → 0.24 IoU:

**Radiometric Terrain Correction (RTC) — γ⁰ vs σ⁰:**

The current calibration produces σ⁰ (sigma-nought), which is the radar backscatter normalized to the incident angle but **not** corrected for local terrain. In steep terrain, a slope facing the satellite appears artificially bright (foreshortening) and a slope facing away appears artificially dark (shadowing) — independent of the actual surface material.

**Fix:** convert to radiometrically terrain-corrected gamma-nought ($\gamma^0$):

$$\gamma^0 = \frac{\sigma^0}{\cos\theta_{\text{local}}}$$

where $\theta_{\text{local}}$ is the local incidence angle derived from the DEM and the satellite look vector. This removes the terrain-induced brightness distortion so that water on a flat valley floor and water on a 30° slope produce comparable backscatter values.

**Implementation:** use the Copernicus DEM (already a 4-channel input) to compute $\theta_{\text{local}}$ per pixel. Apply the correction in `preprocess/sar_calibrate.py` as an optional post-calibration step. The corrected $\gamma^0$ replaces $\sigma^0$ as channel 0/1 in the 4-channel tensor.

**Adversarial Domain Adaptation (DANN):**

The OOD collapse occurs because Sen1Floods11 training chips are predominantly flat terrain, while the deployment domain is steep Himalayan topography. The feature extractor learns flat-terrain-specific representations that do not transfer.

**Fix:** add a Gradient Reversal Layer (GANN / DANN architecture):
1. **Feature extractor:** the ResUNet encoder (shared).
2. **Segmentation head:** predicts water mask (primary task).
3. **Domain discriminator:** binary classifier predicting whether a chip is "flat benchmark" or "Himalayan" (auxiliary task).
4. **Gradient reversal:** the domain discriminator's gradients are reversed before reaching the feature extractor, so the encoder learns features that are **invariant** to the terrain domain.

**Training data:** unlabeled SAR chips from the Himalayan deployment region (we have 5 real SAFE archives already downloaded). The domain discriminator uses only the terrain label (flat vs Himalayan), not water labels — so the Himalayan chips need no annotation.

**Expected effect:** the encoder stops relying on flat-terrain-specific backscatter patterns and learns terrain-invariant water features. Combined with the 4-channel DEM/slope input (§2.1) and the physics-informed loss (§2.6), this targets the 0.65 IoU gate from three complementary directions: input conditioning, loss constraint, and domain alignment.

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

Example SHAP output on an elevated alert:
```
Lake area expansion rate:     +0.32 to log-odds
48h antecedent precipitation:  +0.21 to log-odds
Moraine freeboard height:     -0.05 to log-odds
```

### 3.4 Conformal prediction for risk thresholds

A single point estimate ("P_breach = 82%") gives the coordinator no sense of epistemic uncertainty. A wide interval signals the model is unsure and the decision requires manual inspection.

**Fix:** use split conformal prediction to produce a mathematically guaranteed uncertainty interval at a user-defined significance level:

$$\text{Risk} \in [lo, hi] \quad \text{with } P(Y \in C(X)) \geq 1 - \alpha$$

where $\alpha = 0.05$ (95% coverage guarantee), $C(X)$ is the prediction set for input $X$, and the guarantee holds **regardless of the model or data distribution** (finite-sample, distribution-free).

**Implementation:**
1. **Split conformal:** reserve a calibration set (disjoint from training and event-holdout). For each calibration point $i$, compute the nonconformity score $s_i = |y_i - \hat{y}_i|$.
2. **Quantile:** take the $\lceil(1-\alpha)(n+1)\rceil$-th order statistic of the calibration scores as the interval half-width $\hat{q}$.
3. **Prediction:** for a new input, output $[\hat{y} - \hat{q}, \hat{y} + \hat{q}]$.

**Automatic fallback to human gate:** when the interval width $|C(X)| = 2\hat{q} > 0.35$, the system flags **high epistemic uncertainty** and refuses to issue an autonomous recommendation. The review card displays the interval as a range bar with the 0.35 threshold marked, and the coordinator sees "Model uncertainty too high for automated triage — manual inspection required."

**Expected effect:** the coordinator sees not just "P_breach = 0.82" but "P_breach ∈ [0.74, 0.89] (95% coverage)" — or, when the model is unsure, "P_breach ∈ [0.20, 0.85] — manual inspection required." This converts a black-box number into a calibrated, auditable uncertainty statement.

### 3.5 Required datasets

- **ICIMOD Glacial Lake & GLOF Inventories** — historical moraine-dammed lake failures, lake area evolution, dam geometry.
- **High Mountain Asia (HMA) Glacial Lake Catalog** — lake boundary tracking 1990–present for temporal expansion features.

### 3.6 Acceptance criteria

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
[S1 SAR VV/VH] ──► RTC γ⁰ correction ──► [γ⁰_VV, γ⁰_VH]
[Copernicus GLO-30 DEM] ──► [Elevation, Slope]
[ERA5 / IMERG Rainfall]
                      │
                      ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 1: 4-Channel Terrain-Aware WaterResUNet              │
│   Input:  [γ⁰_VV, γ⁰_VH, DEM, Slope]  ∈ ℝ^{B×4×H×W}       │
│   Loss:   L_Dice + λ·L_gravity (uphill water penalty)      │
│   Adapt:  DANN gradient reversal (flat ↔ Himalayan)       │
│   Output: Validated water extent polygon (IoU > 0.65 gate) │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 2: Calibrated Breach Susceptibility                  │
│   Trend:  Trailing ΔArea/Δt (deterministic, ≥5 passes)     │
│   Model:  XGBoost on (expansion, rain anomaly, moraine,     │
│           slope, lake area) → P_breach ∈ [0, 1]            │
│   Explain: TreeSHAP → top-k feature contributions → reasons │
│   Uncert: Conformal prediction [lo, hi] at 95% coverage    │
│           → if |C(X)| > 0.35, force manual inspection      │
└────────────────────────┬───────────────────────────────────┘
                         │ (if P_breach ≥ 0.70)
                         ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 3: Differentiable Hydrodynamic Forward Pass           │
│   Model:  FNO-2D (trained on synthetic HEC-RAS runs)        │
│   Physics: Saint-Venant PDE (mass + momentum conservation) │
│   Output: h_water grid + T_arrival at named points          │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 4: Dynamic Exposure + Dispatch                        │
│   Exposure: h_water > 0.3m ∩ OSM (supersedes static buffer)│
│   Triage:   T_arrival → evacuation urgency ranking          │
│   Disease:  D_risk from inundated wells + population        │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
           Human gate → ≤250-byte dispatch → SHA-256 audit
```

The deterministic five-factor path remains the fallback at every stage when ML is unavailable or below gate. The conformal interval (§3.4) ensures the coordinator always sees the model's confidence — never a bare number.

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
