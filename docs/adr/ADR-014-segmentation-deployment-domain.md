# ADR-014 — Segmentation Deployment Domain (ro-121 Descending)

**Status:** PROPOSED — awaiting owner sign-off · **Date:** 2026-09-18
**Companions:** [ADR-013](ADR-013-end-to-end-neural-pipeline.md), [ADR-011.1](ADR-011.1-real-data-sar-gate-calibration.md), [PRD v5.1 §9.8](../spec/PRD.md)

---

## Context

PRD §9.8 requires a promoted component's evaluation set to cover **the deployment domain's failure modes** — but until now "deployment domain" was never pinned to a specific imaging geometry. The 2026-09-18 footprint audit (`ingest/swath_coverage.py`, report `models/checkpoints/swath_coverage.json`) established the actual track inventory:

| Track | Pass | Scenes on disk | Imja coverage |
|---|---|---|---|
| ro-121 | Descending | 6 (S1A Nov+Jan, S1D Jul) | ✅ all — swath 85.30–87.88°E at Imja lat |
| ro-85 | Ascending | 2 (S1D Jul-23, Aug-04) | ❌ eastern edge 86.685°E — misses Imja |
| ro-12 | Ascending | 2 (S1D Aug-11, Sep-16) | ✅ swath 86.65–88.91°E |

The first different-geometry held-out eval (`monsoon_asc` pair, `ml/heldout_eval.py`) then showed the high-altitude adapter is **geometry-bound**: Imja recall collapses from 0.41 (Nov, ro-121 descending) to **0.089** on ro-12 ascending, while the unadapted Kuro Siwo base reaches 0.68 there at the cost of 12× the glacier false positives. The adapter learned ro-121 descending signatures, not orbit-invariant water features.

## Decision

Declare the segmentation deployment domain for §9.8 promotion purposes:

> **In-domain:** Sentinel-1 **relative orbit 121, descending** GRD over the Dudh Koshi AOI, in liquid-water and shoulder seasons. Frozen-surface observations are handled by the deterministic thermal-state gate (`detect/thermal_state.py`) and are excluded from water-detection scoring, not counted as model failures.
>
> **Out-of-domain:** all other orbits/pass directions (ro-12, ro-85, any future track) and other basins. On out-of-domain scenes the deterministic detector remains primary and neural output — if produced at all — is shadow evidence labeled `out_of_deployment_domain`.

**Rationale:** an operational GLOF monitor pins its acquisition orbit — the runtime already prefers the verified ro-121 descending pair (`find_imja_descending_pair`). A geometry-bound model promoted inside a declared domain is honest; the same model silently evaluated only on its training geometry is not.

## Consequences

- The §9.8 gate set becomes concrete: held-out pairs on ro-121 descending — `shoulder` (Nov 2025, evaluated), `unfrozen_desc` (Aug-19/Sep-12 2026), `unfrozen_desc2` (Jul-26/Aug-07 2026), plus `winter` as a frozen-state stress pair (recall expected 0; glacier-FP bound still applies).
- `monsoon_asc` results are retained in the eval report as a documented out-of-domain stress test — informative, not gate-blocking.
- If operations ever require ascending-track monitoring (e.g., descending coverage loss), the adapter needs a new domain-qualified fine-tune; this ADR does not preclude that.
- Domain-scoped promotion means the runtime must *verify* a scene's relative orbit before applying the primary model — orbit metadata is parseable (`ingest/swath_coverage.py::_parse_orbit_meta`) and must become part of the promotion-time input contract.

## Gate evidence (measured 2026-09-18, `heldout_eval_report.json`)

Stratified adapter (`water_resunet_6ch_himalayan_adapter_stratified.pt`) at τ=0.30, Imja-scoped (10-px ROI dilation), on the two unfrozen ro-121 pairs — both dates thermally LIQUID on desc2; desc t1=Sep-12 is UNKNOWN (ERA5 series ends):

| Metric | Gate | unfrozen_desc | unfrozen_desc2 |
|---|---|---|---|
| Imja IoU vs inventory | ≥ 0.60 | **0.654 / 0.660** ✅ | **0.627 / 0.674** ✅ |
| Imja IoU vs SCL labels | ≥ 0.60 | **0.734 / 0.638** ✅ | **0.634** (t1 only) ✅ |
| Imja precision vs inventory | ≥ 0.84 | 0.720 / 0.698 ❌ | 0.652 / 0.716 ❌ |
| Imja precision vs SCL | ≥ 0.84 | 0.797 / 0.669 ❌ | 0.634 ❌ |
| Water-px on glacier (FP bound) | suppression | 11,516 vs base 128,336 (−91%) ✅ | 7,771 vs 132,583 (−94%) ✅ |

Threshold sweep (τ 0.30→0.65) and morphology tests: no operating point reaches P ≥ 0.84 while holding IoU ≥ 0.60 — precision caps at ~0.75 vs inventory / ~0.86 vs SCL; 1-px erosion lifts SCL precision to 0.90+ but drops inventory IoU below gate (the lake is ~13 px wide at 90 m pitch). The residual false positives form a shoreline ring ~1–2 px wide, not scattered speckle — consistent with the model reproducing the *median* inventory outline while labels measure a specific date's edge.

**Verdict: gate NOT passed — IoU and glacier-FP legs hold on both independent pairs; the precision leg fails against every label source.**

### Gold-label adjudication (2026-09-18)

Hand-verified labels (`ml/imja_label_roi.py`, `imja_gold_label_*.tif`) were produced on S2 09-08 (t1, 4-day offset) and S2 08-24 (t0, 5-day offset): NDWI > 0.15 inside a hand-drawn lake region, boundary visually verified against true-colour RGB, opaque cloud masked. Findings:

- The gold boundary sits ~19% **inside** the inventory median outline (gold ⊂ inventory: P 0.99, R 0.81 — the median polygon overstates a given date's waterline, mostly south shore + east end).
- SCL also under-labels: large parts of open water classify as cloud/unclassified near the terminus and on the north shore.
- Against gold truth the stratified adapter is **worse** than weak labels suggested — the inventory polygon was flattering it:

| Date | gold IoU | gold P | inv IoU | inv P |
|---|---|---|---|---|
| t1 (09-12 SAR vs 09-08 label) | 0.53–0.55 | 0.57–0.62 | 0.65 | 0.72–0.77 |
| t0 (08-19 SAR vs 08-24 label) | 0.37–0.39 | 0.38–0.41 | 0.66 | 0.70–0.74 |

(τ swept 0.30–0.55; ranges shown. Caveats: 4–5 day label offsets; centre-sampling the 10 m label onto the ~90 m SAR grid is stricter than the polygon-burn used for the inventory mask.)

**Conclusion:** the model genuinely over-segments ~1–2 SAR px (~90–180 m) beyond the true waterline — consistent with having learned the *median* inventory outline as its target. The precision deficit is real model error, not label bias. Promotion path: fine-tune round against per-date labels (gold-style NDWI masks / SCL water on clear scenes) instead of median polygons; gold labels retained as the eval's truth tier above SCL and inventory.

### Follow-up: label-refined round (2026-09-18, gate still NOT passed)

A fine-tune round on label-refined targets (`ml/lake_label_refine.py`: >50%-of-footprint NDWI from S2 07-05 where clear, 1-px-eroded inventory elsewhere; + boundary-aware loss) was evaluated on both unfrozen pairs and re-scored against gold (`imja_gold_eval_report.json`). SCL-precision legs improved (unfrozen_desc t1: 0.80→0.83; unfrozen_desc2 t1: 0.63→0.73) and glacier FPs held ~−92% vs base, but **gold truth was a wash** — t1 gold IoU 0.65→0.57 at equal P (~0.70), t0 P 0.48→0.53. No label source reaches P ≥ 0.84. Full numbers in the session notes. Working hypothesis: a ~1-px systematic boundary offset (GCP-vs-S2 geolocation jitter + true boundary uncertainty at ~90 m pitch) caps Imja-scoped P near ~0.7–0.8 on a ~13–19-px-wide lake regardless of label tightening — the precision leg may need a displacement-tolerant criterion or finer input resolution rather than more label work.

### Follow-up: multi-scene label coverage + change-product evidence (2026-09-19, gate still NOT passed)

A second refinement round expanded per-date supervision from one cloudy tile to **16 S2 L2A scenes across the six MGRS tiles covering the swath** (`lake_label_refine.py` now merges all in-window scenes, nearest-to-t1 wins per pixel): grid coverage 7.7% → 65.5%, lake chips with true labels 169 → 692/697. The v2 adapter trained on this set (`..._labelrefined_v2.pt`).

Measured against gold + a new unverified auto tier (desc2):

- Strict extent P still caps ~0.7 vs gold — the 90 m geometry bound stands; no label mix at this resolution reaches P ≥ 0.84.
- Tolerant precision confirms the displacement story: P@2px 0.72–0.84 vs gold, 0.96–0.98 vs desc2 auto labels.
- **Change-product evidence** (expansion = water_t1 & ~water_t0): 0–3 FP px in the Imja ROI across both pairs/tiers; v2 has the best change recall (8/32 gold-change px). The per-date boundary offset cancels in the difference — the model is far better at the operational task than at absolute extent.
- The inter-date gold change itself verified as real signal (contiguous new-water bodies on the terminus end, not label noise).
- Trade-off: v2 gains change recall + best P@2px but loses strict P and desc2 SCL-IoU (0.544 < 0.60); stratified remains the better extent model.

**Gate recommendation (owner decision pending):** amend §9.8/§17.2 to gate on the operational output — expansion FP-rate + change recall — with a displacement-bounded extent leg (P@2px or boundary F1), rather than strict per-pixel P ≥ 0.84 which is geometry-unreachable at ~90 m. The alternative is a full-resolution pipeline (non-decimated ~10–20 m input), the only route that plausibly passes strict P. Full evidence: `ADR-014-session-notes-2026-09-18.md` (2026-09-19 section) + `imja_gold_eval_report.json`.

---

## Amendment ADR-014-am1 — change-product gate legs (2026-09-19, owner-directed)

**Rationale.** The deployment product of this component is the per-pair
expansion mask `water_t1 & ~water_t0` inside monitored-lake vicinity —
the evidence that feeds the elevated/critical severity tiers. The
inherited extent gate (IoU ≥ 0.60 AND P ≥ 0.84 per date, from
ADR-011.1's lowland-flood calibration) is geometry-unreachable at the
~90 m decimated cache pitch: on a ~13–19 px-wide lake, a systematic
1–2 px boundary offset (GCP-vs-S2 geolocation jitter + label-date
offset + real boundary uncertainty) caps strict precision near ~0.7
regardless of model quality. That offset cancels between dates, so the
*change* contract is both measurable and what the system actually
uses. The strict extent metrics remain **required reporting** (below)
for regression tracking — they are not deleted, only de-gated.

**Promotion scope:** this amendment can promote the *expansion
evidence* of the SAR segmentation component (ro-121 descending,
unfrozen season) — the per-date extent mask remains unqualified
shadow evidence regardless.

### Amended legs

| Leg | Criterion | Basis |
|---|---|---|
| L1 FP bound | predicted-expansion FP ≤ 5 px inside each evaluated lake ROI vs the highest-quality label tier | 5 px ≈ 0.04 km² — below the watch-tier area (≥5% of a ~1 km² lake ≈ 0.05–0.07 km²); false alarms cannot reach the smallest alert tier |
| L2 change recall | ≥ 20% of verified inter-date change px detected, with ≥1 contiguous component, on ≥1 of the gated pairs | provisional bar — verified-change truth is thin (32 px gold at Imja; 9.5% valid scope on desc2); a real-event probe (South Lhonak S1, ro-48 desc 2023-09-28→10-10) is the stronger recall test and must corroborate |
| L3 extent floor | Imja-scoped tolerant precision P@2px ≥ 0.70 on each labelled date | sanity floor — confirms the extent mask is lake-shaped, not noise; ~90 m displacement bound acknowledged |
| L4 glacier FP | ≥ 50% reduction vs `kuro_siwo_base` in water px on glacier, each pair | unchanged intent from the original domain gate |
| L5 pairs | ≥ 2 independent held-out in-domain unfrozen pairs | unchanged |

**Not gated (required reporting):** strict per-date IoU/P vs every
label tier, AOI-wide change metrics vs merged SCL labels (whose
change-truth is itself SCL-flip noisy and window-offset), winter/
shoulder context, ascending-pair stress results.

### Measured status under the amended legs (2026-09-19)

| Leg | adapter | stratified | labelrefined | labelrefined_v2 |
|---|---|---|---|---|
| L1 (FP ≤5px: desc/desc2) | 1/1 ✅ | 1/1 ✅ | 0/1 ✅ | 3/0 ✅ |
| L2 (recall ≥20%) | 19% ❌ | 3% ❌ | 9% ❌ | **25%** ✅ |
| L3 (P@2px ≥0.70, all dates) | min 0.78 ✅ | min 0.76 ✅ | min 0.79 ✅ | min 0.72 ✅ |
| L4 (glacier FP) | −93% ✅ | −94% ✅ | −93% ✅ | −90% ✅ |
| L5 | ✅ | ✅ | ✅ | ✅ |

**Verdict: `labelrefined_v2` is the first checkpoint to satisfy all
five amended legs** — the others fail L2 (change recall < 20% vs the
gold change). Promotion remains blocked, however, pending the two
honesty conditions below — the gate legs are satisfied on thin
evidence and must be corroborated before the runtime flag flips.

*Honesty notes / conditions on the L2 leg:* (a) the 20% bar is
provisional pending the South Lhonak real-event probe — if the model
cannot detect catastrophic change, the leg is meaningless regardless
of the Imja numbers; (b) L1/L2 are Imja-scoped because Imja is the
only lake with verified-change truth — verified labels on ≥2 more
in-domain lakes are required before promotion is broad rather than
Imja-specific; (c) AOI-wide change metrics are reported but not gated
because the merged SCL label-change is dominated by classification
noise (SCL flips) and label-window offsets — measured FP mass is
largely shoreline-adjacent and plausibly real detection the offset
labels missed, which cuts both ways.

### South Lhonak real-event probe (2026-09-19) — inconclusive, domain-limit finding

S1 ro-48 descending pair straddling the 2023-10-03 GLOF (09-28 pre,
10-10 post; the only descending track over the site) was calibrated
and all five checkpoints scored against the Pleiades-measured lake
footprint (0.866 km² → 80 px on the ~90 m grid).

**The lake is SAR-invisible in this geometry:** footprint backscatter
is VV −6.9 dB at t0 — *brighter* than its own ring (−10.7 dB) — i.e.
layover/rough-surface dominated with no water-dark signature anywhere;
the drain registers ΔVV −0.13 dB, ΔVH +0.28 dB. Every checkpoint
detects 0/80 px — including the lowland-trained base model, so this is
a sensor/geometry blind spot, not model-specific recall failure. The
probe therefore provides no recall evidence either way, and documents
that **C-band SAR detection has a visibility floor**: steep-walled
(possibly ice-covered) high-altitude lakes can be unmonitorable by any
SAR water detector — relevant to coverage claims for the alert
product. An inventory audit of which monitored lakes are SAR-visible
is the follow-up.
