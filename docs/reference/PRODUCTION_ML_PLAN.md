# SIREN — Production ML Plan (Pipeline, Models, Datasets, Basins)

**Status:** Proposal (companion to ADR-010) · **Date:** 2026-09-07
**Read with:** [`DL_MODEL_AUDIT.md`](DL_MODEL_AUDIT.md) (what is wrong now), ADR-002/003/005/006/008/009

Answers three questions for taking SIREN to a full-fledged service: (1) the recommended pipeline and models, (2) the datasets required, (3) whether to keep, switch, or add monitoring locations.

---

## 1. Recommended production pipeline

Preserves ADR-002 deterministic-first semantics; ML enters only as qualified, isolated, shadow-mode evidence (ADR-010).

```text
Acquisition (ADR-006/008)                Deterministic core                          Decision (ADR-009)
──────────────────────────              ──────────────────────────────              ──────────────────
S1 covering orbits        ─┐        1. SAR preprocess: calibrate → σ0 VV/VH,
S2 L2A                    │            terrain-flatten, speckle filter, project
IMERG Early/Late          ├──►        onto a fixed per-basin grid (DEM-referenced)
OSM (incl. waterways)     │        2. Water segmentation, per date:
Copernicus DEM / SRTM    ─┘            ├─ rule-based σ0 threshold + slope mask   (frozen semantics)
                                      ├─ ML segmenter (shadow evidence)         (ADR-010 Stage 1)
                                      └─ JRC GSW / DynamicWorld priors          (cross-check)
                                 3. Change detection: deterministic bi-temporal
                                    differencing on the fixed grid
                                    (new-water | persistent | recession | invalid)
                                 4. Time-aware trend: area change per elapsed day,
                                    min-observation count, data-age gate,
                                    explicit "insufficient observations" outcome
                                 5. Corridor: D8 + OSM                         (frozen, ADR-005)
                                 6. Risk fusion: PRD §9.5 five-factor weights   (restored)
                                    ML recorded as separate evidence, not in H
                                 7. Human gate → dispatch → audit               (ADR-009)
```

| # | Stage | Method | Frozen? |
|---|---|---|---|
| 1 | Acquisition | scheduler + job ledger + atomic verify (ADR-006/008) | New — outside the pipeline |
| 2 | SAR preprocess | σ0 calibration, terrain flattening, speckle, DEM projection to a fixed basin grid | Current `preprocess/` requires qualification for this contract (audit: co-registration metric is grid-corner-based, not feature-based) |
| 3 | Water segmentation | rule σ0-threshold + slope mask (frozen); ML segmenter in shadow; JRC GSW + DynamicWorld as priors | Rule frozen; ML new (ADR-010) |
| 4 | Change detection | deterministic differencing of per-date water masks on the fixed grid | New, deterministic |
| 5 | Trend | time-aware deterministic (km²/day, min obs, age gate) | Replaces ConvLSTM per ADR-010 |
| 6 | Corridor | D8 + OSM rivers (ADR-005 + fixed Overpass query) | Frozen |
| 7 | Risk fusion | PRD §9.5 five-factor weights restored; ML evidence separate | Restore documented weights |
| 8 | Review / dispatch / audit | OIDC, delivery outbox, hash chain (ADR-009) | New |

**Why this shape:** the audit showed the risk is not model capacity but unverifiable supervision and ML leaking into load-bearing paths. This pipeline makes every ML claim measurable against a rules-only baseline, and keeps the frozen modules' semantics intact.

## 2. Model recommendations

| Stage | Model | When | Notes |
|---|---|---|---|
| **1 — build now** | Compact single-date **SAR water-segmentation U-Net/ResUNet** (≤ ~10M params, 2-ch VV/VH σ0) | After ADR-010 acceptance | Train on Sen1Floods11 hand-labeled with **official event-level splits**; optional SSL4EO-S12 SAR encoder init; evaluation gate before any display |
| **1b — optical, opportunistic** | Keep NDWI (frozen) + **DynamicWorld water/flooded-vegetation/snow probabilities** as a cross-check; WorldFloods-v2 cloud-aware segmentation model later | 1b cheap now; WorldFloods later | WorldFloods pretrained models are **CC non-commercial** — license review before hosted use |
| **2 — deferred** | Paired SAR change detection (FC-Siam-diff / ChangeFormer adapted to SAR) | Only when real labeled bi-temporal pairs exist | Until then, per-date segmentation + deterministic differencing **is** the change detector |
| **3 — deferred** | Semantic classes (debris vs snow vs water) | Only with defensible labels | DynamicWorld-derived weak supervision is an option to evaluate; the current 5-class crop classifier is dropped (ADR-010) |
| **Temporal** | Deterministic time-aware trend now; learned temporal model **with elapsed-time features** later | Learned: after ≥ 2 seasons of real observations | ConvLSTM is replaced (ADR-010) |
| **Foundation models** | SSL4EO-S12 (SAR encoders), Prithvi-EO-2.0 (HLS optical, flood-scar fine-tunes exist) | Optional encoder initialization | Initialization, not a substitute for labels or evaluation |

**What not to build:** ChangeFormer near-term; a real SegFormer fine-tune before labels exist; any learned model with a load-bearing role before shadow-mode evaluation.

## 3. Dataset plan

### A. Training / evaluation (basin-independent)

| Dataset | What | Access | Role | Flags |
|---|---|---|---|---|
| **Sen1Floods11** | 446 hand-labeled S1 water chips + official event-level splits; 4,385 weakly labeled; 815 permanent-water | `gs://sen1floods11` (gsutil, ~14 GB) | Stage-1 training/eval | Binary water labels; verify license terms before commercial use |
| **WorldFloods v2** | 509 S2 L1C flood events 2016–23 with curated flood + cloud masks | HuggingFace `isp-uv-es/WorldFloodsv2` (~76 GB); masks on Zenodo (record 8153514) | Stage-1b optical cloud-aware segmentation | **CC non-commercial** — review before commercial deployment |
| **DynamicWorld V1** | NRT 10 m S2 LULC class probabilities (water, flooded_vegetation, snow_and_ice, …) | Google Earth Engine `GOOGLE/DYNAMICWORLD/V1` | Optical cross-check; weak supervision candidate | GEE-native (cloud ≤ 35 % scenes) |
| **SSL4EO-S12** | Self-supervised S1/S2 encoders (ResNet/ViT; MoCo/DINO/MAE) | `github.com/zhu-xlab/SSL4EO-S12` + HF mirrors | Optional encoder init | Pretraining, not labels |
| LEVIR-CD / OSCD / S2Looking (optional) | Optical change-detection benchmarks | Various | Only if Stage-2 optical CD is ever built | Not core |

### B. Per-basin operational data

| Data | Source | Notes |
|---|---|---|
| Sentinel-1 GRD IW VV/VH | Copernicus CDSE | Covering orbits per basin — **Imja: orbit 12 (asc) + 121 (desc), verified**; South Lhonak: to identify in Live Phase 2 |
| Sentinel-2 L2A + SCL | CDSE | Opportunistic optical; monsoon-limited (all 17 T45RVL products Jul–Sep 2026 were > 20 % tile cloud) |
| Copernicus DEM GLO-30 (or SRTM 1″) | CDSE / Earthdata | Slope + D8; switching DEM requires corridor requalification (ADR-005) |
| JRC Global Surface Water | GEE `JRC/GSW1_4` | Permanent-water prior for baseline construction and persistent-vs-new separation |
| IMERG Early / Late / Final | NASA GES DISC | Early ≈ 4 h latency for ops; Late/Final for reanalysis; 0.1° — contextual, not ground truth |
| OSM (incl. `waterway=river/stream`) | Overpass — fixed query per ADR-005 addendum | Exposure + corridor; currently broken in `overpass.py` (missing rivers, nested tags) |
| Open-Meteo / ERA5 | API / CDS | Temperature context (fix the `_days_before` bug first) |
| Rainfall/river gauges (Nepal DHM; India CWC) | Agencies | IMERG bias correction — access varies; pursue via partners |

### C. Event-validation data (retrospective)

| Event | What exists | Role |
|---|---|---|
| **South Lhonak GLOF (Oct 3–4 2023, Sikkim)** | Pre/post S1+S2 (lake 167.4 → 60.3 ha, −64 %; Sentinel-1A captured Oct 4 ~06:00 per ISRO); damage record: Teesta III dam destroyed 63 km downstream, 31 major bridges, ~25,900 buildings, 55 deaths + 74 missing (Sattar et al., *Science* 2025); flood traveled 385 km to Bangladesh; multi-paper reconstructions (Science, EGU 2025, Sci. Rep.) | **Primary validation: "would SIREN have flagged it?"** — a real, Sentinel-era GLOF with quantified lake change and a rich downstream exposure record |
| Imja controlled drainage (2016–17) | Documented lake-lowering timeline; ICIMOD reports | **Benign-change control:** does SIREN avoid false alarms on a real, intentional, gradual change? |
| Chamoli ice/rock avalanche (Feb 2021) | Sentinel-era imagery; Rishiganga & Tapovan Vishnugad damage | Complementary Sentinel-era hazard — mechanism differs from lake outburst (slope failure, not water expansion) |

## 4. Basin / location assessment

**Should SIREN switch basins? No — add, don't switch.** Training data is global and basin-independent (§3A). Basin choice determines (a) validation events, (b) OSM/asset quality, (c) partner and data access.

| Basin | Sentinel-era real event | Data strengths | Constraints | Recommended role |
|---|---|---|---|---|
| **Imja / Dudh Koshi (Nepal)** | None (2016–17 controlled drainage is a benign change) | Covering orbits verified (12/121); best-in-region OSM (Everest trekking corridor); ICIMOD baselines; easiest field access | No positive GLOF event to validate detection against | **Keep — monitoring pilot** (does it monitor cleanly without false alarms?) |
| **South Lhonak / Teesta (Sikkim, India)** | **Yes — Oct 2023 GLOF** | Pre/post S1+S2; quantified lake change; extensive damage record (dam, 31 bridges, ~25,900 buildings); multi-paper science; Indian agencies (ISRO/NRSC, Sikkim SDMA) | Covering orbits TBD; north-Sikkim field access is restricted (Protected Area Permit) — irrelevant for satellite ops, matters for ground campaigns | **Add — event-validation basin + prospective monitoring** |
| **Chorabari / Kedarnath (Uttarakhand)** | 2013 — **pre-Sentinel-1** (S1A launched Apr 2014; S2A Jun 2015) | Rich literature; iconic story | **No SAR exists for the event** — the SAR-primary pipeline cannot be validated on it | **Drop for validation; narrative only** |
| **Cordillera Blanca (Peru)** | 1941 Huaraz, 2010 Lake 513 — both pre-S1 | Most-studied GLOF region globally; UGRH active monitoring | Outside the Himalayan scope | **Optional V3 generalization basin** |

**Dual-basin rationale:** Imja answers *"does it monitor cleanly without false alarms?"*; South Lhonak answers *"would it have caught a real disaster?"* — the exact retrospective "what-if" story the PRD tells, but with a real event and real casualties instead of scenario masks.

**If restricted to a single basin for a full-fledged deployment: South Lhonak/Teesta** — retrospective validation on a real Sentinel-era GLOF + prospective monitoring + a real exposure story (Teesta III, 31 bridges, transboundary Teesta) is strictly stronger evidence than a basin with no event.

## 5. Open verification items (Live Phase 2)

- Identify Sentinel-1 covering orbits and repeat pattern over South Lhonak (CDSE catalogue query).
- Gauge data access (Nepal DHM / India CWC) for IMERG bias correction.
- License review: Sen1Floods11 terms, WorldFloods (CC non-commercial), SSL4EO-S12, SegFormer/ChangeFormer code licenses.
- Copernicus DEM vs. SRTM qualification for slope/D8 (ADR-005 requalification if switched).
- OSM completeness audit for the Teesta corridor (settlements, bridges, dams) — population metadata was nearly absent even in the Imja extract.
