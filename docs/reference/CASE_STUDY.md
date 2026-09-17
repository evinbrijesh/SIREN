# SIREN Case Study — Audited Satellite Monitoring for Glacial Lake Hazards

A technical write-up of what SIREN actually is, what it measured, and the
engineering findings that came out of building it. Every claim below is
backed by a number produced by code in this repository — including the
negative results, which are the more interesting half.

**The one-line pitch:** SIREN is an audited satellite monitoring pipeline
for glacial lake hazards — deterministic hydrological baselines plus
shadow-track deep learning, strict spatial gating, and split-conformal
uncertainty bounds, with a mandatory human gate before any dispatch.

It does not predict floods. It watches a basin, measures change, scores
hazard deterministically, and puts an auditable evidence card in front
of a person.

---

## 1. Architecture that survived contact with real data

```
S1 SAR / S2 optical / DEM / rainfall / OSM
            ↓
calibration (ESA LUT → sigma0 dB) + GCP geolocation + quality gate
            ↓
deterministic change detection ─────────┐
   (NDWI / backscatter ratio,           │   shadow ML segmenter
    scenario masks for the demo)        │   (6-ch WaterResUNet,
            ↓                            │    terrain-gated,
   D8 + OSM corridor & exposure          │    state/change split,
            ↓                            │    MC Dropout σ² + conformal)
   risk fusion H + E + D_risk           │
            ↓
   human review (mandatory) → ≤250-byte dispatch → SHA-256 audit chain
```

The load-bearing path — detection, corridor, scoring, dispatch — is
entirely deterministic and empirically grounded. Neural components run
alongside as labelled *shadow evidence* and cannot influence any
decision until they clear an explicit promotion gate (ADR-013) on
held-out real data. That separation is what made the bugs below
survivable: every neural failure was caught *because* the deterministic
path was there to compare against.

## 2. Three silent geospatial bugs

These were all "the pipeline runs fine, the numbers look plausible" bugs
— invisible without geolocation-aware verification.

### 2.1 The `dem_slope` units bug — a gate that did nothing

The terrain gate for the ML shadow mask compares per-pixel slope against
15° (water surfaces are flat). `dem_slope()` returned slope in
**radians**; the gate compared against 15 — i.e., it gated pixels steeper
than ~859°. Nothing on Earth is steeper than 859°, so the gate passed
everything and reported "gate applied" in the stats. Found when the
shadow mask's pixel counts looked physically impossible (tens of
thousands of "water" px at 40°+ slopes). Fix: normalize slope to degrees
at the computation site. **Lesson: a threshold check with mismatched
units fails open, not closed.**

### 2.2 GCP geotransform distortion — 1–4 km errors from a "good" affine

Sentinel-1 GRD scenes ship ground-control points, not an affine
transform. Fitting a global affine to the GCPs produced residuals of
1–4 km over mountain terrain — plausible-looking, wrong answers. The
fix was to keep the GCPs and evaluate geolocation through GDAL's GCP
polynomial transformer (`GCPTransformer`), which is the geolocation the
runtime shadow layer, the terrain gate, and the training-chip
rasterizer all share. **Lesson: a fitted affine hides its error; the
sensor's own ground-control metadata is the honest geolocation.**

### 2.3 The 65 km window-transform shift — correct pixels, wrong place

`read_s2_band` read a windowed AOI crop from a Sentinel-2 tile but
returned the **full-tile profile**. The mask content was the correct
AOI; its transform placed it at the tile origin — stamping the baseline
water mask ~65 km west into the Rolwaling valley. The bug was invisible
in every numeric statistic (pixel count, area, % expansion) because
those don't depend on geolocation; it was caught by overlaying the mask
on the map. Fix: return the window-specific transform; regression test
with a synthetic SAFE archive. **Lesson: content-correct does not mean
position-correct — verify bounds, not just values.**

## 3. The Kuro Siwo domain shift — and how it was bounded

The gate-passed model (WaterResUNet-6ch, pooled IoU 0.62 on the Kuro
Siwo held-out test set — lowland flood events) reproduces those metrics
exactly on its own test distribution. On the full Imja scene it
produces ~729k "water" pixels, ~115k of them on glacier — orders of
magnitude over truth. Glacier ice and calm water are dielectrically
similar at C-band; a lowland-trained boundary cannot separate them.

Three layered responses, all measured:

| Defense | Mechanism | Measured effect |
|---|---|---|
| Terrain gate | AOI clip + DEM slope >15° + RGI glacier exclusion (exempting known-lake vicinity) | 61k raw → 132 gated expansion px |
| MC Dropout + conformal | T=20 stochastic passes → per-pixel σ²; split-conformal quantile calibrated on held-out chips | Uncertainty concentrated exactly on glacier/snow confusion regions |
| High-altitude adapter | Decoder fine-tune on 697 verified lake-inventory chips, encoder frozen | Glacier water-extent FPs **−92%** in-scene (115,100 → 8,738); **−77%/−70%** on held-out Nov/Jan S1A pairs; Imja recall 71–87% (in-scene), 41% held-out shoulder |

The adapter is the real fix — it teaches the model that "high-altitude
water" is a thing, using the 31,698-polygon glacial lake inventory
already on disk (a single descending swath covers 1,395 of those lakes).
Labels are weak (median-outlined polygons, monsoon season only), so the
checkpoint stays an experiment, not an operational weight.

**Held-out test (`ml/heldout_eval.py`).** The in-scene −92% figure was
circular — same acquisition for train and eval. On two independent S1A
descending pairs (Nov 2025 shoulder, Jan 2026 winter — different
season, different satellite, zero training chips), suppression holds:
glacier extent FPs fall 77% (40,788 → 9,207) in shoulder and 70%
(27,484 → 8,221) in deep winter, and Imja recall jumps 4.8% → 41.3%
in shoulder season. Two honest caveats: inventory-wide recall over all
720 in-swath lakes *degrades* (27.8% → 16.3% — the adapter favours
large Imja-like lakes over tiny tarns), and winter recall collapses to
0% for **both** models — a frozen lake is not liquid water at C-band,
which is physics, not failure.

## 4. The state/change discovery — a metric that fights itself

The weak-label adaptation run surfaced a design flaw: the shadow mask
was *expansion-only* (`water_t1 & ~water_t0`). Imja is a persistent
lake — a perfectly-adapted model detects water at both dates and
produces **zero** expansion, so *better* detection *reduced* the shadow
metric. The fix decomposes evidence into persistent extent (t1),
expansion, and drainage — so the lake stays visible while the delta
tracks shoreline movement. The adapted model's overlap with the rule
mask reads 0 not because it fails, but because it now sees Imja at both
dates — visible in the extent layer at 71–87% recall.

## 5. What the deterministic path validated against a real GLOF

South Lhonak (Oct 2023, ~40–50 MCM released) is a documented real event
with 1 m Pléiades pre/post DEMs on disk. Measured from the DEMs:
pre-event lake 0.87 km² (published 0.9–1.2), surface drawdown ~14 m
(documented 10–20). Huggel's area→volume on measured area gives 28.4
MCM; across the published area range, 28–45 MCM — the formula brackets
the documented release. The dominant error is the *area input*, not the
exponent. A DSM can only see the dewatered rim — the submerged bowl is
invisible, so DEM-difference volume is an honest lower bound (0.13 MCM
measured).

## 6. Honest negative results

| Experiment | Result | Decision |
|---|---|---|
| Neural bathymetry (E2) vs Huggel | 676% MAPE vs 75.6% on 20 surveyed lakes (LOO) | Huggel stays load-bearing; neural needs more sonar surveys |
| S2 optical separability (E3) | NDWI/MNDWI cannot separate frozen lake from glacier; July pair 73% cloud | Fusion not justified on current data |
| Weak-label SAR adaptation (single-scene) | Glacier FPs −62% but expansion overlap degraded | Single-scene labels insufficient → motivated the lake-inventory approach |
| Adapter inventory-wide recall (held-out) | 27.8% → 16.3% over 720 swath lakes; winter recall 0% for all models | Adapter is selective (large lakes > tarns); frozen-lake invisibility is a sensor limitation — needs optical confirmation in winter |
| FNO surrogate on real terrain | Eval uses synthetic corridor + wave-speed clipped to documented range | Semi-circular; real hydrodynamics (GeoClaw) deferred |

## 7. What holds up

- **The decision chain is real and tested.** Baseline → 3 observations
  → elevated/critical card with ≥3 evidence reasons → human confirm →
  118-byte dispatch → SHA-256 lineage. ~920 backend tests + 11 frontend
  tests covering the human-gate flow.
- **Volume estimation brackets reality** on a real GLOF (South Lhonak).
- **The shadow/gate architecture works as designed** — it caught,
  bounded, and quantified every neural failure above instead of letting
  it reach a decision.
- **Domain adaptation is feasible from on-disk data** — the lake
  inventory converts the missing-label problem into a solved one
  (697 chips → −92% glacier FPs in-scene, −77%/−70% held-out).

## 8. What it would take to go further

- ~~Held-out evaluation for the adapter~~ — **done** (`heldout_eval.py`,
  Nov + Jan S1A pairs): FP suppression generalises; remaining gate
  question is the tarn-vs-large-lake recall trade-off.
- **Frozen-lake handling** — winter recall is 0% for every model
  (physics); a production system needs an optical or rule-based
  "frozen" state, or explicit seasonal suppression of the water layer.
- **Real hydrodynamics** — GeoClaw/Clawpack on the South Lhonak, Dig
  Tsho, and Imja corridors to replace the synthetic-corridor FNO eval;
  the post-event Pléiades DEM is a ready-made validation target.
- **Triggering models** — moraine/seismic/hydromet precursor estimation
  is a separate research problem; SIREN's scope is detection and
  response, not prediction.

---

*Artifact trail: `models/checkpoints/lake_adapter_report.json`,
`sar_domain_adapt_report.json`, `south_lhonak_lake_eval.json`,
`docs/DATA_LICENSES.md`, `docs/spec/BUILD_ROADMAP.md` (E-track data
requirements), `PROGRESS.md` (ordered improvement plan).*
