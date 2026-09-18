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
