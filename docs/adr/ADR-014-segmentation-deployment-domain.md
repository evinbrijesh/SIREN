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
