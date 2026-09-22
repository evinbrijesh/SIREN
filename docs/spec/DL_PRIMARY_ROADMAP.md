# SIREN — DL-Primary Transition Roadmap

**Version:** 1.0 (2026-09-21)  
**Companion to:** `docs/spec/PRD.md` v5.1, `docs/spec/BUILD_ROADMAP.md`, `AGENTS.md`, `CLAUDE.md`  
**Status:** Level 1 operational-primary (in certified scope); Level 2 next

---

## 1. What "DL-primary" actually means for SIREN

A component is **DL-primary** only when:

1. It passes its held-out, deployment-domain evaluation gate on real data it never trained on.
2. It is wired as the default/authoritative inference path for that stage.
3. The deterministic fallback remains callable, recorded in provenance, and is automatically re-activated if the neural component is demoted or fails.
4. Uncertainty, data gaps, and failure modes are exposed in the review UI.

A **DL-primary system** means that every load-bearing inference stage (segmentation, bathymetry, dynamics, risk fusion) is neural and gate-passed. The deterministic modules stay as permanent, labeled fallbacks and cross-checks — they are not deleted, only demoted from the operational path.

Current reality (2026-09-19): the system is **deterministic/Huggel-load-bearing with neural advisory evidence**. Three neural components are promoted to *advisory*-primary (`sar_segmentation_expansion`, `susceptibility`, `dynamic_escalation`), but operational severity, exposure, and dispatch authority still flow through the deterministic pipeline.

---

## 2. Ranked levels

Each level must be completed before the next is meaningful. The ranking follows **dependency order** and **data availability**, not engineering effort.

| Level | Goal | Why it gates the rest |
|-------|------|-----------------------|
| 0 | Honest baseline + readiness dashboard | Prevents "DL-primary" from becoming marketing while tests fail and component status is invisible |
| 1 | SAR segmentation becomes operational-primary | Water mask / expansion % is the upstream input to everything downstream |
| 2 | Bayesian uncertainty becomes a calibrated safety bound | Needed before neural outputs can drive severity thresholds autonomously |
| 3 | Neural bathymetry replaces Huggel | Volume estimate feeds the FNO; without it dynamics rest on empirical formula |
| 4 | Latent-conditioned FNO becomes the dynamics engine | Connects lake geometry to downstream flood dynamics end-to-end |
| 5 | Multi-modal SAR+optical fusion becomes primary segmenter | Optional improvement; only justified after Level 1 and clean optical-SAR pairs exist |
| 6 | Learned risk fusion replaces deterministic severity | Last component because it consumes all upstream neural outputs |
| 7 | System integration + operational certification | Differentiable smoke test, audit provenance, hosted acquisition, domain sign-off |

---

## 3. Level 0 — Honest baseline

**Goal:** make the current state mechanically correct and transparent.

| # | Task | Deliverable / gate |
|---|------|--------------------|
| 0.1 | Fix the failing engine test | `pytest` passes 100% on the current suite |
| 0.2 | Add a `/system/ml-status` endpoint and UI panel | Returns per-component `{status, gate, metric, blocker}` |
| 0.3 | Lock deterministic fallback provenance | Every score/observation carries `method: neural_primary` or `deterministic_fallback` |
| 0.4 | Freeze the demo as a regression harness | Automated end-to-end test: demo → elevated review → ≤250-byte dispatch → audit hash chain |

**Level 0 exit criterion:** a maintainer can open the app and immediately see which components are shadow, advisory-primary, or operational-primary, and the test suite is green.

---

## 4. Level 1 — SAR segmentation operational-primary

**Status (2026-09-22): GATE PASSED → PROMOTED to operational-primary.** The model achieved `operational_primary_eligible: true` on 2 unfrozen descending pairs from 2025, and the pipeline now consumes the neural Δp measurement as the load-bearing `water_area_change_percent` when the run is in certified scope (promoted + descending orbit covering Imja + monsoon window + liquid surface + registered baseline). The deterministic registry value stays recorded as `expansion_pct_deterministic`, disagreement surfaces via the `cross_check` reason, and `SIREN_ML_DEMOTE` reverses the promotion at runtime. Score `method` reads `mixed` when the expansion factor is neural-sourced.

### 4.0 Operational scope — monsoon window (declared 2026-09-21)

The whole project is scoped to the **vulnerable season: June–September (JJAS)**. This is a deliberate product decision, grounded in two independent datasets (`siren/scope.py`):

| Evidence | Jun | Jul | Aug | Sep | Window share |
|---|---:|---:|---:|---:|---:|
| **Real GLOF events** (230 dated, HMAGLOFDB) | 40 | 77 | 62 | 11 | **82.6%** |
| **Rainfall** (NASA POWER, Imja 1981–2026) | 72 mm | 144 mm | 133 mm | 70 mm | **~80% of annual** |
| **Lake temp** (same series) | +4.1°C | +5.6°C | +5.2°C | +3.1°C | liquid surface |

**In-scope** = monsoon window (Jun–Sep) + descending orbit + liquid surface + Imja-area monitorable lakes.

**Out-of-scope (future work, explicitly deferred):**
- **Frozen/shoulder seasons (Oct–May)** — the lake is frozen or refreezing; C-band SAR cannot see liquid water through ice (physics, not model failure). The thermal-state gate annotates and suppresses; the deterministic baseline remains authoritative.
- **Ascending orbit** — different backscatter statistics; needs ascending training data or a pass-invariant model.
- **Other lakes (multi-lake)** — the South Lhonak 2023 real-event probe failed (IoU 0.028); needs multi-lake training data + region-wide DEM coverage.

Out-of-scope observations are **not unsupported** — the deterministic baseline still runs and stays authoritative, and the scope note surfaces as a review reason. A GLOF can occur outside the window (South Lhonak burst 2023-10-03), so out-of-window monitoring is lower priority, not off.

**Enforcement:** `siren/scope.py` (declaration + `scope_for_date`), pipeline annotation (`change_stats["operational_scope"]` + review reason), scope-aware gate verdict (`in_scope_pairs_*` / `out_of_scope_pairs`), promotion record scope, tests (`tests/test_scope.py`, pipeline scope tests).

### 4.1 Gate result

| Pair | Interval | IoU | Precision | Glacier-FP | Verdict |
|---|---|---:|---:|---:|---|
| `monsoon_2025_desc` | 2025-08-29 → 09-10 | **0.68** | **0.97** | 0% | **Pass** |
| `monsoon_2025_desc2` | 2025-08-29 → 09-22 | **0.62** | **0.95** | 0% | **Pass** |
| `unfrozen_desc` (2026) | 08-19 → 09-12 | 0.50 | 0.74 | 0% | Fail |
| `monsoon_asc` (2026) | 08-11 → 09-16 | 0.03 | 1.00 | 0% | Fail (ascending) |
| `winter` (2026) | 01-08 → 01-20 | — | — | 0% | Pass (frozen gate) |

**Scope:** the model is certified for **descending unfrozen-season** operation (monsoon, June–September). It is a change-detection model — performance correlates with the magnitude of the backscatter change (dVV). When the lake's backscatter increases between pre and post scenes, the model detects water well; when the change is small, the model is less confident.

**What this means for GLOF monitoring:** the model detects lake expansion — the key precursor to GLOF. During monsoon when lakes are fullest and expanding, the model works. In winter when lakes are frozen and stable, the model correctly predicts ~0 liquid water.

### 4.2 What was done

1. **Label acquisition:** 8 gold labels now exist (2025 + 2026, unfrozen + frozen). AI-assisted candidates were auto-cleaned (largest-component + hole-fill + opening) and promoted with QA caveats.
2. **Season-aware gate:** frozen/shoulder pairs are evaluated for glacier-FP control + low liquid-water prediction, not extent metrics. Unfrozen pairs require IoU ≥ 0.60 + P ≥ 0.84 + glacier-FP < 5%.
3. **Terrain gate:** the gate eval now measures the *gated* model output (AOI + slope>15° + glacier-minus-lake-vicinity) — same as the runtime shadow-mask gating. This drops glacier-FP to 0% on all pairs.
4. **2025 data acquisition:** downloaded 3 additional SAR scenes + 3 S2 label scenes for 2025 monsoon (independent year from training data).
5. **Ice/water label semantics:** winter labels corrected to all-zero (frozen lake = no liquid water). NDWI/SCL incorrectly marked ice as water.

### 4.3 Real-event validation — South Lhonak 2023 GLOF (FAILED)

`ml/south_lhonak_sar_eval.py` runs the promoted checkpoint on the only on-disk SAR pair that brackets a real GLOF: South Lhonak 2023-09-28 → 2023-10-10 (burst 2023-10-03, ~40-50 MCM released, 10-20 m drawdown). Labels are S2-derived water masks (T45RXL, 2023-09-26 / 2023-10-09).

| Metric @ τ=0.3 | Value |
|---|---:|
| Post-event IoU | **0.028** |
| Post-event recall | **3.6%** |
| Scene false positives | **38,481 / 38,489 (99.98%)** |
| Lake SAR signature | VV −15.4 → −12.3 dB (dVV **+3.1 dB**) |

The model misses the lake entirely and floods the scene with false positives — even though the change signal (dVV +3.1 dB) is in the regime where Imja succeeds. The lake's absolute backscatter (−15.4/−12.3 dB) sits above Imja's water range (−16 to −19 dB), and the post-event surface was rough (documented icebergs/debris from the calving), not specular.

**Conclusion: the certified scope is Imja-area only — a hard requirement, not a conservative choice.** Multi-lake monitoring with this checkpoint is not viable; it would miss real lakes and flood the review queue. Additional findings:
- The SRTM tile covers only the Dudh Koshi AOI — the runtime slope gate is **inoperative** at South Lhonak (only the RGI glacier gate applies). Multi-lake expansion needs region-wide DEM coverage.
- The runtime terrain gates remove only ~1k of the 38k false positives (they are not on glaciers).

### 4.4 Remaining gaps

- **2026 eval pair under-performs** (IoU 0.50 vs 0.68 on 2025). The model was trained on 2026 data — the 2026 eval scene may have unusual conditions (small dVV change, georegistration issues). Needs investigation.
- **Ascending pass fails** (IoU 0.03). The adapter was trained on descending only. Requires ascending training data or a separate model.
- **Multi-lake generalization fails** (South Lhonak IoU 0.03). Requires multi-lake training data.
- **Model is change-sensitive** — it detects water better when the lake's backscatter changes. For monitoring, this is actually useful (detects expansion), but for static extent mapping it's a limitation.

### 4.5 Next steps

- ~~Wire the pipeline to use the neural mask for `water_area_change_percent` (task 1.3).~~ **DONE 2026-09-22** — `pipeline.py` resolves `expansion_pct_source` per run (`neural_primary` | `deterministic_fallback`); the gated Δp expansion inside monitorable-lake vicinity (`ml_expansion_lake_vicinity_km2`) divided by the registered baseline area drives the hazard score when in scope.
- ~~Add a runtime demotion flag to force deterministic fallback (task 1.5).~~ **DONE 2026-09-22** — `SIREN_ML_DEMOTE=<component|all>` in `ml/promotion.py`; demotion is read at call time, surfaces a review reason, and is reported on `/system/ml-status`.
- For multi-lake coverage: acquire multi-lake training labels (the inventory has 12,667 lakes mapped; ~10-20 gold labels across 5-10 lakes would be a start) + region-wide DEM coverage.
- For year-round coverage: acquire ascending training data or add a season-aware routing layer.
- Known consequence: on the verified Jul-02→14 runtime pair the model measures ~0 km² lake-vicinity expansion (persistent lake already at extent — correct change-detector behaviour), so demo-observation severities now reflect the neural measurement (obs-002 watch, obs-003 elevated) with scripted values kept as labeled cross-checks.

---

## 5. Level 2 — Calibrated Bayesian uncertainty

**Goal:** uncertainty maps become statistically trustworthy enough to drive severity thresholds and review confidence.

| # | Task | Success criterion |
|---|------|-------------------|
| 2.1 | ~~Re-train WaterResUNet with dropout active during training (`dropout=0.1` in encoder + bottleneck)~~ **DONE 2026-09-22** — `lake_adapter_finetune.py --dropout 0.1` → `water_resunet_6ch_himalayan_adapter_multidate_mc.pt`; deterministic gate metrics improved (IoU 0.82/0.80/0.64 vs 0.68/0.62/0.50), glacier FP still 0% | New checkpoint that supports MC Dropout natively |
| 2.2 | ~~Run split-conformal calibration on **whole Imja-area scenes**~~ **EVALUATED 2026-09-22 — GATE FAILED.** `calibrate_uncertainty_scenes.py` runs leave-one-scene-out conformal on the 3 in-scope gold scenes: coverage 0.881 / 0.816 / 0.931 (nominal 0.90, tol ±0.05); pooled q*=0.639. One scene (09-22, n=223 labelled px) dips below. Interval stays advisory | Empirical coverage 0.85–0.95 of nominal 90% on ≥3 held-out scenes |
| 2.3 | ~~Add uncertainty overlay to the review UI~~ **DONE** — `uncertainty_map_uri` flows to `/runs/{id}/ml-evidence`; ReviewView gains an "Uncertainty σ" mask layer + "Expansion: uncertain" badge | Per-pixel std map + scene-level "uncertain expansion" flag |
| 2.4 | ~~Gate auto-escalation on uncertainty~~ **DONE (reason-path)** — when the promoted checkpoint ships a conformal sidecar, the pipeline records `expansion_pct_ci90` + `expansion_trend_uncertain` and appends a "trend uncertain" review reason when the interval includes zero; severity is not silently downgraded (human gate authoritative) | If the 90% conformal interval for expansion % includes zero, downgrade severity or add "trend uncertain" reason |

### 5.1 Gate

Uncertainty becomes a calibrated safety bound when:

- Coverage is within **±5% of the nominal 90% level** on Imja-area held-out scenes.
- The model was trained with dropout (post-hoc dropout on a `dropout=0.0` model is explicitly disqualified).

**Status (2026-09-22): GATE NOT MET.** The dropout-native checkpoint exists and is promoted for segmentation (it improved the operational-gate metrics), and its whole-scene conformal sidecar is wired per-checkpoint (`<stem>.conformal.json` — directory-level sidecars no longer leak across checkpoints). But leave-one-scene-out coverage on the 3 in-scope gold scenes is 0.816–0.931 — the 09-22 scene misses the band. The conformal interval therefore stays *advisory*: the runtime computes `expansion_pct_ci90` and the "trend uncertain" flag with `uncertainty_conformal_gate_passed: false` recorded alongside. To pass, either more calibration scenes (more gold-labelled monsoon pairs) or a stronger uncertainty source (deep ensembles, heteroscedastic head) is needed — not a wider dropout rate tuned to pass.

---

## 6. Level 3 — Neural bathymetry

**Goal:** replace the Huggel area-volume formula with a neural volume estimator.

### 6.1 Hard truth

Training a U-Net on 20 lakes produced **676% MAPE** (random init) and **329% MAPE** with synthetic pre-training. Huggel is 75.6%. More model capacity is not the answer; more curated data and a simpler model are.

### 6.2 Tasks

| # | Task | Success criterion |
|---|------|-------------------|
| 3.1 | Expand to the full global bathymetry/metadata compilation | ≥60 unique lakes with measured bathymetry/volume in grouped-LOO — **DONE earlier: 323 entries / 267 unique lakes** |
| 3.2 | ~~Build a strong **terrain-feature regression baseline**~~ **DONE 2026-09-22 — NEGATIVE RESULT.** `bathymetry_terrain.py` extracts 11 deployable terrain/morphology features per dense lake (log area, perimeter, compactness, elongation, rim slope mean/p90, window relief + mean slope, z_surface, glacier distance + window fraction from RGI v7) and runs grouped-LOO ridge+GPR residual over Huggel on the 20 surveyed lakes. Ridge MAPE 74.5% vs Huggel 75.6% (median 55.6% vs 50.2% — a wash at n=20); GPR 86.8% with calibrated 0.90 interval coverage. Terrain features do not beat the area-only formula → 3.3 (neural head) is not justified. Report: `models/checkpoints/bathymetry_terrain_loo.json`, feature table `bathymetry_terrain_features.json` | Inputs: lake area, elevation, glacier distance, glacier area, slope, aspect, dam type; beats Huggel MAPE on grouped-LOO |
| 3.3 | Add a small neural head only if regression is close | 2–3 layer MLP or tiny CNN with heavy regularization; grouped-LOO MAPE <15% — **NOT PURSUED: regression is ~75% MAPE, 5× the gate** |
| 3.4 | Re-qualify with **Copernicus DEM GLO30** where available | Document coverage, nodata, and co-registration vs SRTM |
| 3.5 | Wire into `breach_volume.py` | Neural → Huggel → hypsometric cascade; fallback is loud |

### 6.3 Gate

Neural bathymetry becomes primary when it achieves **<15% MAPE on grouped leave-one-lake-out volume estimation** on held-out real lakes, and Huggel becomes the labeled fallback.

---

## 7. Level 4 — Latent-conditioned FNO

**Goal:** the dynamics surrogate runs end-to-end from segmentation bottleneck + DEM to flood depth/arrival, replacing the scalar `V_breach` injection.

### 7.1 Data blocker

This requires **real-terrain inundation simulations** for training. Synthetic parametric corridors are explicitly disqualified for operational claims (the previous 12.3% MAPE result was invalidated).

### 7.2 Tasks

| # | Task | Success criterion |
|---|------|-------------------|
| 4.1 | Build a GeoClaw (or equivalent) scenario library on South Lhonak / Imja DEMs | 500+ parametric breach scenarios with per-pixel depth/arrival targets |
| 4.2 | Modify `FNO2D` to accept `(B, 2+d_latent, H, W)` and spatially upsample the segmentation bottleneck | Unit tests pass on synthetic inputs; gradient flows back to segmentation encoder |
| 4.3 | Hold out ≥2 real documented events for evaluation | South Lhonak 2023 + one other Himalayan GLOF with independent timing/depth data |
| 4.4 | Replace scalar `V_breach` in the operational path once the gate passes | `hydro_surrogate.py` defaults to latent FNO when promoted |
| 4.5 | Add an FNO OOD flag | If input lake geometry is far from training distribution, mark arrival/depth estimates unavailable |

### 7.3 Gate

Latent FNO becomes primary when:

- **MAPE ≤ 20% on ≥2/3 held-out real events**, no event >30%.
- Gradient-flow test from FNO loss back to segmentation encoder passes.
- Scalar `V_breach` path remains as labeled fallback.

---

## 8. Level 5 — Multi-modal SAR+optical fusion

**Goal:** improve segmentation by fusing cloud-free Sentinel-2 with SAR.

### 8.1 Precondition

Do **not** start this until Level 1 is operational-primary **and** optical features separate lake from glacier. The current evaluation (`ml/s2_spectral_eval.py`) shows NDWI/MNDWI are worse than random for Imja (AUC 0.18–0.29) because glacier surfaces read higher MNDWI than the frozen lake.

### 8.2 Tasks

| # | Task | Success criterion |
|---|------|-------------------|
| 5.1 | Acquire ≥3 cloud-free S1+S2 pairs over Imja across seasons | Pixel registration <1 residual |
| 5.2 | Find an optical discriminator that beats raw NDWI | SCL class, NDSI, SWIR texture, or composite achieves lake-vs-glacier AUC ≥ 0.75 |
| 5.3 | Build cloud-gated cross-attention fusion | SAR 6ch + optical features + attention mask; train with event-held-out splits |
| 5.4 | Fallback to SAR-only under cloud | Cloud fraction ≥ 0.20 routes to Level 1 SAR model automatically |
| 5.5 | Promote only if it beats SAR-only | Improvement in IoU and/or glacier-FP on held-out pairs |

### 8.3 Gate

Fusion becomes primary when it reaches **event-held-out IoU > 0.75 and Precision ≥ 0.85 on real paired SAR+optical data**, and is superior to the Level 1 SAR-only model.

---

## 9. Level 6 — Learned risk fusion

**Goal:** replace the deterministic five-factor severity classifier with a learned, calibrated model that consumes upstream neural outputs.

**Status (2026-09-22): GATE PASSED → advisory-primary.** `ml/train_risk_fusion.py` evaluated the fused scorer on the 887-window corpus (230 dated events + stable/within-lake negatives) under the same spatio-temporal holdout as Tier-2: **mean ROC-AUC 0.835, mean Brier 0.142** vs the deterministic five-factor baseline's **0.504 / 0.187** on identical folds — both gate legs pass. Caveat: the fused model is statistically identical to Tier-2 alone (AUC 0.838) — the stacked fold-honest susceptibility prior and monsoon flag add no measurable accuracy. Wired advisory-first (`risk/learned_fusion.py` → `shadow_evidence["learned_risk_fusion"]` + review reason at p_fused ≥ 0.65); `classify_severity` stays deterministic-authoritative. Operational promotion (learned score driving severity) requires a separate severity-mapping evaluation — the §9.2 gate as measured covers event probability, not the four-class severity policy.

### 9.1 Tasks

| # | Task | Success criterion |
|---|------|-------------------|
| 6.1 | Build an honest event/non-event dataset | ≥500 windows from HMAGLOFDB + stable-lake + within-lake controls; spatial/temporal split; no label-contaminated features — **DONE: 887 windows reused from the dynamic-escalation corpus** |
| 6.2 | Train a calibrated classifier | Small MLP or XGBoost with isotonic/Platt calibration — **DONE: XGBoost + Platt OOF sidecar (`xgboost_risk_fusion.json`)** |
| 6.3 | Beat the deterministic baseline | **Brier < 0.15** on held-out real events; ROC-AUC > five-factor baseline — **PASS: 0.142 < 0.15; AUC 0.835 > 0.504** |
| 6.4 | Wire into `risk/fusion.py` | `classify_severity` uses learned score when promoted; deterministic stays fallback — **PARTIAL: advisory-first wiring (`risk/learned_fusion.py` + shadow evidence + review reason); severity stays deterministic pending a severity-mapping eval** |
| 6.5 | Expose feature attributions | Review card shows top 3 drivers via SHAP or permutation importance — **DONE: TreeSHAP top-3 log-odds contributions in `reasons`** |

### 9.2 Gate

Learned risk fusion becomes primary when it achieves **Brier < 0.15 on spatio-temporally held-out real events** and beats the deterministic five-factor baseline on the same split — **met for the advisory event-probability contract**. The baseline leg was evaluated on the formula's honestly-measurable inputs only (measured rain; neutral trend/expansion/drainage proxies — no SAR history exists for historical windows), documented in `risk_fusion_eval_report.json`.

---

## 10. Level 7 — System integration and operational certification

| # | Task | Deliverable |
|---|------|-------------|
| 7.1 | End-to-end differentiable smoke test | One script: S1 → segmentation → latent FNO → exposure → review → dispatch with gradient sanity check |
| 7.2 | Audit provenance for every neural output | Every score records model version, checkpoint hash, input hashes, gate evidence |
| 7.3 | Host acquisition service (Live Phase 1) | Scheduler + `acquisition_jobs` ledger + secrets management + ingestion health alerts |
| 7.4 | Operational acceptance with a domain reviewer | Signed-off thresholds, false-alarm budget, human-review policy, local-language templates |
| 7.5 | Release PRD v6.0 "DL-primary certified" | Document which gates passed, on what data, when, and the remaining caveats |

---

## 11. What to do right now

1. ~~**Finish Level 0**~~ — `/system/ml-status` live, fallback provenance locked (`method` field), suite green.
2. ~~**Level 1**~~ — **DONE 2026-09-22**: segmentation is operational-primary in certified scope; neural Δp expansion drives `water_area_change_percent`, deterministic stays labeled cross-check, `SIREN_ML_DEMOTE` reverses. Promoted checkpoint is now the **dropout-native** `multidate_mc` (better gate metrics + native MC Dropout).
3. ~~**Level 2**~~ — **EVALUATED 2026-09-22, GATE NOT MET**: dropout-native retrain done, whole-scene LOSO conformal ran on the 3 in-scope gold scenes (coverage 0.816–0.931), conformal interval + "trend uncertain" flag + σ overlay are wired but advisory. Next: more gold calibration scenes or a stronger uncertainty source.
4. ~~**Level 3.1/3.2**~~ — **DONE 2026-09-22, NEGATIVE**: terrain-feature regression does not beat Huggel (74.5% vs 75.6% MAPE at n=20); the small neural head is not justified at this sample size. Huggel stays operational.
5. **Do not start:** FNO retraining (Level 4) until bathymetry is credible; fusion (Level 5) until clean optical-SAR pairs exist; learned risk fusion (Level 6) until upstream neural outputs are trustworthy.

---

## 12. Anti-patterns to avoid

- **Do not** train more architectures on bad labels. The residual-segmentation corrector and the 100-epoch bathymetry U-Net already showed this wastes GPU time.
- **Do not** declare a component primary because it passes a generic benchmark (Kuro Siwo, Sen1Floods11). The gate is deployment-domain held-out data.
- **Do not** remove deterministic fallbacks. They are an invariant of the architecture.
- **Do not** present synthetic-corridor or post-hoc-dropout results as calibrated operational evidence.
- **Do not** manufacture negative labels, event times, or dam geometry. The label-contaminated susceptibility model was already disqualified for this.

---

## 13. Open questions

- Which second real event will validate the FNO besides South Lhonak 2023? Dig Tsho 1985, Chamoli 2021, and other HMAGLOFDB entries need data feasibility review.
- Is 60 lakes enough for bathymetry, or should the project pursue ICESat-2 bathymetric LiDAR or stereophotogrammetry-derived depths?
- What is the acceptable false-alarm budget for an operational deployment? This must be answered by a domain reviewer before Level 7.
