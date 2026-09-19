# Session Notes 2026-09-18 — Segmentation Promotion Gate

**Status:** gate evaluated, **NOT passed** — next action is a label-refined
fine-tune round. Companion to [ADR-014](ADR-014-segmentation-deployment-domain.md)
(gate evidence table lives there).

## Where things stand

The stratified adapter (`water_resunet_6ch_himalayan_adapter_stratified.pt`)
was evaluated on two independent unfrozen ro-121 descending held-out pairs
(`unfrozen_desc` = 08-19/09-12, `unfrozen_desc2` = 07-26/08-07; all dates
post-training, thermal gate LIQUID on desc2, t1 UNKNOWN on desc):

| Gate leg | Result |
|---|---|
| Imja IoU ≥ 0.60 vs inventory | ✅ 0.63–0.67 both pairs |
| Imja IoU ≥ 0.60 vs SCL | ✅ 0.63–0.73 |
| Glacier-FP bound | ✅ −91% / −94% vs base |
| Precision ≥ 0.84 | ❌ 0.63–0.80 all labels |
| **vs GOLD truth** | ❌ IoU 0.37–0.55, P 0.38–0.62 |

**Key finding:** the precision gap is real model error, not label bias.
Hand-verified gold labels (NDWI>0.15 in a hand-drawn lake region,
RGB-verified, cloud-masked) on S2 09-08 and 08-24 show the model
over-segments ~1–2 SAR px (~90–180 m) beyond the true waterline — it
learned the *median* 2022–24 inventory outline (~19% wider than any
given date's edge: gold ⊂ inventory, P 0.99 / R 0.81). SCL also
under-labels turbid water (marks open water cloud/unclassified near the
terminus). Inventory polygons *flattered* the model — its ring lands
inside their over-wide boundary.

## Label-refined round — EXECUTED 2026-09-18 (gate still NOT passed)

Ran the recommended path below: per-date NDWI targets (a) + 1-px-eroded
inventory fallback (b) + boundary-aware loss (c). New artifacts:

- `ml/lake_label_refine.py` — rebuilds chip labels: >50%-of-~90m-footprint
  NDWI>0.15 on SCL-valid px from S2 07-05 where usable, 1-px-eroded
  inventory elsewhere. Output `data/datasets/himalayan_lake_chips_refined/`
  (169 chips partial S2 coverage, 0 full, 528 none — 7.7% grid coverage;
  pos px 84,893→35,329, median 79→13; 98 micro-tarn chips eroded to zero).
- `lake_adapter_finetune.py` — loads `v`/`src` from npz; new
  `--boundary-alpha/--boundary-band` (per-pixel edge emphasis).
- `ml/imja_gold_eval.py` — gold-label scorer reusing held-out machinery;
  report `models/checkpoints/imja_gold_eval_report.json`.
- Checkpoint `water_resunet_6ch_himalayan_adapter_labelrefined.pt`
  (stratified, α=1.0, band=2px, 15 ep, lr 5e-5) + report
  `lake_adapter_labelrefined_report.json`.

Held-out gate legs vs stratified adapter (τ=0.30, Imja-scoped):

| Metric | unfrozen_desc strat→refined | unfrozen_desc2 strat→refined |
|---|---|---|
| Imja IoU inv (t1/t0) | 0.654/0.660 → 0.610/0.546 | 0.627/0.674 → 0.613/0.620 |
| Imja IoU SCL | 0.734/0.638 → 0.692/0.604 | 0.634 → 0.721 |
| Imja P inv | 0.720/0.698 → 0.751/0.685 | 0.652/0.716 → 0.712/0.717 |
| Imja P SCL | 0.797/0.669 → 0.829/0.674 | 0.634 → 0.733 |
| Glacier water px | 11,516 → 10,654 | 7,771 → 8,688 |
| Imja recall t1 | 0.877 → 0.765 | 0.942 → 0.815 |

Gold re-score (unfrozen_desc only; `imja_gold_eval_report.json`):

| Date | strat IoU/P/R | refined IoU/P/R |
|---|---|---|
| t1 (τ=0.30) | 0.647 / 0.708 / 0.882 | 0.565 / 0.704 / 0.741 |
| t0 (τ=0.30) | 0.472 / 0.484 / 0.951 | 0.486 / 0.526 / 0.864 |

**Verdict:** SCL-precision legs improved (+0.03–0.10) and glacier FPs
edged down, but against gold truth the refinement is a wash — t1 gold
IoU dropped (0.65→0.57, recall traded away), t0 barely moved
(P 0.48→0.53). Gate still fails: no label source reaches P ≥ 0.84.

**Interpretation:** the residual precision deficit is not recoverable
by label tightening. The remaining gap is consistent with (i) ~1-px
systematic boundary offset — GCP geolocation jitter vs the S2 grid and
the model's true boundary uncertainty at ~90 m pitch — on a lake only
~13–19 px wide, where a 1-px offset caps gold IoU ~0.6 and P ~0.7 by
geometry alone; and (ii) 4–5-day label offsets. P ≥ 0.84 at Imja scope
may be unreachable at this resolution/label granularity — consider
either relaxing the gate leg to a displacement-bounded criterion
(e.g. boundary F1 within 1 px), scoring a *change*-detection target
where the constant boundary offset cancels between t0/t1, or finer
input resolution (S1 GRD is already ~10 m; the decimated ~90 m cache
pitch is the binding constraint — a full-resolution pipeline is a
larger change).

## Next session — recommended path

Label-refined fine-tune round, same `lake_adapter_finetune.py` machinery:

1. **(a) Per-date NDWI targets on training-scene lakes** — gold-style
   water masks for lakes in the Jul-02/14 training swath from the 07-05
   S2 (72% tile cloud, ~27% AOI clear — partial coverage only)
2. **(b) Label contraction** — erode inventory targets ~1 px, justified
   by the measured 19% median-vs-date bias; works on all 697 chips
3. **(c) Boundary-aware loss** — weight loss toward polygon edge

Suggested mix: (a) where S2 coverage exists, (b) elsewhere. Goal:
teach ">50% water" rather than "any water" at the 90 m shoreline band.
Gold labels are **eval-only** — do not train on them (they sit on the
held-out eval dates).

After retraining: re-run `python -m siren.ml.heldout_eval --pair both`,
then re-score vs gold labels (script in session history / adapt
`sar_grid_sample` over `imja_gold_label_*.tif`).

## Artifacts produced this session

- `ingest/swath_coverage.py` — GCP footprint audit; report
  `models/checkpoints/swath_coverage.json`. Orbit map: ro-121 desc
  covers Imja; ro-85 asc misses; ro-12 asc covers (docs had it wrong)
- `ml/cross_check.py` + `tests/test_cross_check.py` + pipeline wiring —
  §9.8.3 neural-vs-deterministic overlap verdict → `change_stats` →
  review reason on material disagreement. 9 tests pass
- `ml/calibrate_uncertainty_deployment.py` — E1 deployment-domain
  conformal **PASSED** (coverage 0.8868 vs 0.90, lake_id-split, stratified
  adapter): `water_resunet_6ch_himalayan_adapter_stratified_conformal_deployment.json`
- `ml/imja_label_roi.py` — S2 ROI extractor + RGB/NDWI/SCL/label panel
  renderer + label GeoTIFF writer
- `heldout_eval.py` — 5 pairs, IoU + precision vs inventory + SCL +
  Imja-scoped optical metrics; report `models/checkpoints/heldout_eval_report.json`
- ADR-014 — deployment domain declaration (ro-121 desc, PROPOSED) +
  full gate evidence + gold adjudication

## Data on disk (all with provenance sidecars)

- S1 ro-121 desc held-out pairs: 08-19/09-12, 07-26/08-07 (+Nov/Jan S1A, Jul S1D runtime)
- S1 ro-12 asc pair: 08-11/09-16 (out-of-domain stress test)
- S2 T45RVL: 05-26, 07-05, 08-11, 08-24, 09-08 (+Nov-2025)
- Gold labels: `data/processed/imja_gold_label_{20260824,20260908}.tif` + sidecars

## Known caveats to carry forward

- Monsoon optical labels carry 4–5 day offsets; gold labels same
- Centre-sampling 10 m labels onto ~90 m SAR grid is stricter than the
  polygon-burn used for inventory masks (bounds gold-vs-model by ~1 px)
- 09-12/09-16 thermal state UNKNOWN (ERA5 series ends) — treat as
  probably-liquid, honestly labelled
- `unfrozen_desc2` is the cleanest gate pair (both dates LIQUID, full
  coverage); prefer it for go/no-go reads

---

# Session 2026-09-19 — multi-scene label coverage + change-product evidence

Second label-refinement round, motivated by the finding that the
single-scene (a)-labels covered only 7.7% of the grid (169/697 chips)
and that the residual precision deficit is mostly a sub-footprint
boundary offset, not scattered FPs.

## What changed

- **CDSE downloader bugfix** (`ingest/cdse.py`): the current STAC API
  exposes the product archive as the `Product` asset (HTTPS OData
  `Products(<id>)/$value`); `_pick_asset` looked for lowercase
  `product`, then fell back to the first asset href — an `s3://` band
  URL, so every download failed. Now prefers `Product` and skips
  non-HTTP hrefs. Regression tests added in `test_ingest.py`.
- **16 new S2 L2A scenes** downloaded (~15 GB, each <5 GB) across the
  six MGRS tiles covering the lake swath (T45RUL/RUM/RVL/RVM/RWL/RWM),
  best-cloud picks near the Jul-02/14 training pair plus the eval dates
  (07-25 for desc2 t0, 08-11/08-24/09-08 RWL for AOI east coverage).
- **Multi-scene label merge** (`lake_label_refine.py`): every in-window
  (±21 d of t1) scene contributes per-date NDWI labels where its
  footprint is ≥50% usable SCL; per pixel the scene nearest to t1 wins,
  eroded-inventory fallback elsewhere. Output:
  `data/datasets/himalayan_lake_chips_refined_ms/`.
  - grid per-date coverage: **7.7% → 65.5%**
  - lake chips with any S2 coverage: **169 → 692 of 697** (2 fully, 5 none)
  - pos px 84,893 → 40,508; chips eroded to zero 98 → 77
- **Merged eval label rasters** (`heldout_eval.py`): each SAR date maps
  to a list of scenes; per-pixel first-valid merge fills one tile's
  cloud with another's. `unfrozen_desc2` t0 (07-26) now has optical
  labels (S2 07-25, 1-day offset — previously none).
- **Auto-candidate label tier** (`imja_label_roi.auto_candidate_label`):
  NDWI>0.15 inside the inventory polygon +200 m, SCL-clear only —
  explicitly **unverified** (tier below gold). Covers the desc2 dates.
- **Tolerant + change metrics** (`imja_gold_eval.py`): per tier per τ —
  strict IoU/P/R, P@1px/P@2px (pred px within tolerance counts),
  and the change block (expansion = water_t1 & ~water_t0 vs label
  inter-date change, FP vs t1-label land).
- **v2 adapter** trained on refined_ms (15 ep, lr 5e-5, stratified,
  boundary α=1.0 band=2):
  `water_resunet_6ch_himalayan_adapter_labelrefined_v2.pt`

## gold_change verification

The 32-px inter-date change in the gold labels is real signal, not
noise: two contiguous new-water bodies (1,607 + 1,202 px at 10 m) on
the western/terminus end, only 50 gone-px. 2,262 px are hidden behind
t0 cloud, so gold change is a *lower bound*.

## Results — gate still not passed on strict extent precision

unfrozen_desc, Imja-scoped vs gold (τ=0.30):

| ckpt | t1 IoU | t1 P | t1 P@2px | t0 P | t0 P@2px | chg FP | chg TP |
|---|---|---|---|---|---|---|---|
| base | 0.418 | 0.469 | 0.548 | 0.516 | 0.763 | 94 | 6/32 |
| adapter | 0.563 | 0.715 | 0.825 | 0.515 | 0.783 | 1 | 6/32 |
| labelrefined | 0.565 | 0.704 | 0.831 | 0.526 | 0.790 | 0 | 3/32 |
| **v2** | **0.577** | 0.655 | **0.842** | 0.429 | 0.724 | 3 | **8/32** |
| stratified | **0.647** | 0.708 | 0.821 | 0.484 | 0.761 | 1 | 1/32 |

unfrozen_desc2, Imja-scoped vs auto tier (unverified; scope_valid_frac
0.096 for change — thin):

| ckpt | t1 IoU | t1 P | t1 P@2px | chg FP |
|---|---|---|---|---|
| labelrefined | 0.734 | 0.810 | 0.966 | 1 |
| stratified | 0.743 | 0.754 | 0.971 | 1 |
| v2 | 0.635 | 0.800 | 0.980 | 0 |

Held-out SCL legs: v2 opt-IoU t1 0.6495/0.5441 on desc/desc2 — desc2
now *fails* the ≥0.60 IoU leg too (contracted boundary trades IoU for
precision). Glacier suppression intact (~91% vs base on all pairs).

## Reading

- Strict extent P still caps ~0.7 vs gold — unchanged conclusion: at
  ~90 m pitch a 1–2 px systematic boundary offset is geometry-bound;
  no label mix at this resolution reaches P ≥ 0.84.
- The **change product is the right contract** for the gate: expansion
  masks carry 0–3 FP px in scope across both pairs/tiers; v2 has the
  best change recall (8/32 gold-change px) though recall is still the
  weak leg.
- Auto tier corroborates: adapters' P@2px ~0.96–0.98 on desc2.
- v2 vs stratified is a trade — better change recall + best P@2px,
  worse strict P and desc2 IoU. Neither dominates.

## Gate recommendation for owner decision

Evidence now measured for a §9.8/§17.2 amendment: gate on the
operational output — expansion FP-rate + change recall — with a
displacement-bounded extent leg (P@2px or boundary F1) as extent
evidence, instead of strict per-pixel P ≥ 0.84 which is
geometry-unreachable at 90 m. Alternative: full-resolution pipeline
(recalibrate without decimation, retrain at ~10–20 m — the only route
that plausibly passes strict P). Awaiting owner decision; no spec
change made here.

## Artifacts

- `ingest/cdse.py` fix + 2 tests; `tests/test_ingest.py` 36 pass
- `ml/lake_label_refine.py` multi-scene merge (`s2_scenes`, ±21 d window)
- `ml/heldout_eval.py` list-valued `S2_LABEL_SCENES`, merged
  `s2_label_raster(s2_paths, sar_date)`
- `ml/imja_label_roi.auto_candidate_label` (tier-2, unverified)
- `ml/imja_gold_eval.py` gold+auto tiers, P@1/2px, change block
- `data/datasets/himalayan_lake_chips_refined_ms/` (gitignored)
- `data/processed/imja_autolabel_{20260725,20260811}.tif` + sidecars
- `data/processed/s2_water_label_for_*.tif` merged eval rasters
- 16 S2 zips in `data/raw/` + provenance sidecars
- `models/checkpoints/lake_adapter_labelrefined_v2_report.json`
- `models/checkpoints/imja_gold_eval_report.json` (both pairs, both tiers)
- `models/checkpoints/heldout_eval_report.json` (5 ckpts × 5 pairs)

---

# Session 2026-09-19 (cont.) — gate amendment + South Lhonak recall probe

## Amended gate adopted (ADR-014-am1, owner-directed)

Per owner direction: gate on the operational change product. Legs:
L1 expansion FP ≤5px per lake ROI vs best label tier; L2 ≥20% verified
change recall (≥1 contiguous component, ≥1 of 2 pairs); L3 Imja-scoped
P@2px ≥0.70 each labelled date; L4 ≥50% glacier-FP suppression; L5 ≥2
in-domain unfrozen pairs. Strict extent metrics remain required
reporting, de-gated. Promotion scope = expansion evidence only; the
per-date extent mask stays unqualified shadow.

**Leg results:** `labelrefined_v2` satisfies all five (L1 3/0px, L2
25% w/ 4px contiguous component, L3 min 0.72, L4 −90%, L5 ✓). Others
fail L2 (adapter 19%, labelrefined 9%, stratified 3%). Promotion is
still blocked by the honesty conditions below.

## South Lhonak real-event probe — lake is SAR-invisible

Downloaded S1 ro-48 descending pair straddling the 2023-10-03 GLOF
(09-28 pre, 10-10 post; only descending track over the site). Lake
footprint from the 1 m Pleiades pre-DEM (0.866 km² → 80 px on the
~90 m grid).

- Footprint backscatter at t0: **VV −6.9 dB mean — brighter than its
  own ring (−10.7 dB)**. Water-dark signature absent everywhere in the
  footprint; the steep moraine basin return is layover/rough-surface
  dominated (possibly ice-covered surface).
- The drain registers ~nothing: ΔVV −0.13 dB, ΔVH +0.28 dB across the
  footprint.
- **All 5 checkpoints detect 0/80 footprint px at t0** — not a model
  recall failure, a sensor/geometry blind spot: C-band SAR water
  detection cannot monitor this lake in this geometry.

Implication: SAR-invisible lakes are a documented domain limit of the
detection contract (any detector, not just the model). An inventory
coverage audit — which monitored lakes are actually SAR-visible —
is the honest follow-up for the alert product's coverage claims.
Probe data: `data/processed/lhonak_{20230928,20231010}_sar_vv_vh_db.tif`,
`lhonak_lake_footprint.tif`.

## AOI-wide change metrics (corrected)

Fixed a scoping bug in `heldout_eval`'s change block (label-change
denominator was unscoped). Correct numbers, both-valid & lake-vicinity
scope:

- unfrozen_desc: 41 label-change px; recall — v2 3px (7%), base 4px
  (10%), others 0–1px. FP — v2 97px, base 174px, adapter 27px.
- unfrozen_desc2: 45 label-change px; recall — base 7px (16%, P 0.64),
  labelrefined/adapter 1px, v2/stratified 0. FP — v2 9px, base 4px.

Caveats: label-change truth is SCL-flip noisy + label-window-offset;
v2's FP mass is component-coherent and shoreline-adjacent (43/97 px
within 2 px of t1-label water) — plausibly real detection the offset
labels missed, which cuts both ways. Reported-not-gated per am1.

## Gate status

Amended gate: v2 leads on paper (all 5 legs) but the change-recall
evidence remains thin — 32 px Imja gold + a failed (invisible) probe.
Promotion requires either ≥2 more verified-change lakes or an
in-domain real event. Deterministic baseline stays load-bearing.
