# ADR-010 — ML Evidence Isolation and Retraining Path

**Status:** Proposed · **Date:** 2026-09-07 · **Applies to:** ML layer (`backend/siren/ml/`), ML inputs to risk fusion, ML evaluation
**Companions:** ADR-002 (addendum 2), [`docs/reference/DL_MODEL_AUDIT.md`](../reference/DL_MODEL_AUDIT.md), [`docs/reference/PRODUCTION_ML_PLAN.md`](../reference/PRODUCTION_ML_PLAN.md)

## Context

The 2026-09-07 DL audit found that the implemented ML layer fails ADR-002 in both directions at once:

- **Not scientifically qualified:** synthetic supervision (label-derived "before" images, threshold-generated weak labels, generated temporal sequences), a train/inference input mismatch (model trained on SAR chips, fed binary masks at runtime), and no held-out evaluation anywhere in the codebase (all checkpoint metrics are training-set numbers).
- **Not safely subordinate to the rules:** a 0.20-weight ML-confidence term inside the hazard score (diverging from PRD §9.5's documented five-factor weights), a ConvLSTM hybrid that can replace the configured trend class feeding H, and a crop classifier that can delete rule-detected pixels from the evidence mask.

No existing checkpoint is qualified for live hazard assessment.

## Decision

### 1. Verdicts on the four PRD-nominated models

| Candidate | Decision | Rationale |
|---|---|---|
| Siamese U-Net | **Keep as a later candidate; replace the immediate experiment** | Right task family, but current training synthesizes "before" images from labels and runtime feeds binary masks instead of SAR σ0 |
| ChangeFormer | **Drop from the near-term roadmap** | Not implemented; released checkpoints are optical building-change (LEVIR-CD/DSIFN-CD), not SAR water; no advantage over a smaller baseline under current constraints |
| "SegFormer" crop classifier | **Drop from operational filtering; rename or remove** | Not the SegFormer architecture; threshold-generated 5-class supervision with an unreachable "shadow" class; component-level suppression can remove rule-detected evidence |
| ConvLSTM | **Replace with deterministic time-aware trend estimation** | Trained on synthetic mask progressions; no elapsed-time input; inference fabricates missing timesteps by dilating the last mask |

### 2. Replacement ML path — Stage 1

Build a **compact single-date SAR water-segmentation model** first:

- **Architecture:** small U-Net / ResUNet (≤ ~10M parameters), 2-channel input (VV/VH σ0 in dB, fixed normalization contract).
- **Training data:** Sen1Floods11 hand-labeled chips using the **official event-level train/val/test splits** (never random chip splits); the permanent-water split is a separate evaluation. Optional encoder initialization from SSL4EO-S12 self-supervised SAR weights.
- **Change detection stays deterministic:** per-date water masks are compared on a fixed basin grid (new-water / persistent / recession / invalid) with per-elapsed-day rates. The segmenter is never asked to detect change directly.
- **Evaluation gate (before any ML output is displayed as evidence):** held-out-event water IoU/precision/recall, permanent-vs-new-water error separation, and a prospective evaluation on untouched basin dates.

**Deferred until their data exists:** paired SAR change detection (FC-Siam-diff / ChangeFormer adapted to SAR) — only when real labeled bi-temporal pairs exist; semantic classification of changed areas — only with defensible labels (DynamicWorld-derived weak supervision is an option to evaluate); learned temporal models — only after ≥ 2 seasons of real observations, and they must take elapsed time as an input.

### 3. ML evidence isolation contract (all stages)

- The **rule-based assessment is computed and stored immutably** for every run: mask, measurements, trend, score, reasons.
- ML outputs are recorded as **separate evidence records**: model ID + version, exact inputs (input-manifest reference, ADR-008), prediction, uncertainty, availability status. They never overwrite or relabel the rule result.
- **No ML term in the hazard score.** Restore the documented PRD §9.5 five-factor weights (0.30/0.25/0.20/0.15/0.10). Adding a weighted ML factor later requires a new ADR and a calibrated shadow-mode evaluation demonstrating value over the rules-only baseline.
- Failed inference records "ML unavailable" — no synthetic confidence maps, no fabricated timesteps, no default confidence values.
- **Shadow mode first:** compute, store, and display ML output as clearly-labeled auxiliary evidence; measure disagreement rates against the rules; only then consider any load-bearing role.

### 4. Required engineering fixes (before any ML is live)

1. **Input contract:** normalized σ0 VV/VH rasters at inference — the same arrays the model was trained on. Never binary masks.
2. **Remove timestep fabrication** in `trend_engine.py` (padding by dilation).
3. **Remove or gate the crop-classifier's `filtered_mask` replacement** of the consensus mask; rule positives are never removed by ML.
4. **Tiled inference** (the "tiled" docstring is unimplemented) and bounded memory.
5. **Fix the Docker weights-path divergence** (`/data/processed` vs `/app/data/processed`).
6. **Model registry reports actual loadability**, not file existence; descriptions use accurate names.
7. **Event-level dataset splits** with a frozen final test set; model selection on validation events; results reported on test events.
8. **License review** (Sen1Floods11 terms; WorldFloods is CC non-commercial; SSL4EO-S12; SegFormer/ChangeFormer code licenses) against the deployment model before hosted use.

### 5. Basin dependency

Stage-1 training is basin-independent (global Sen1Floods11). Prospective evaluation uses the dual-basin strategy: **Imja/Dudh Koshi as the monitoring pilot** and **South Lhonak/Teesta as the event-validation basin** (real October 2023 GLOF with full Sentinel-era coverage). See `docs/reference/PRODUCTION_ML_PLAN.md` §4.

## Consequences

- **Positive:** the deterministic critical path ADR-002 specified is restored and enforceable; ML progress is decoupled from the pipeline freeze (training/evaluation is offline work); every ML claim becomes measurable against a rules-only baseline; the "four-model" narrative is retired in favor of one defensible model.
- **Negative:** the three existing checkpoints are demoted to experiments; PRD model claims are corrected (v4.6); a future weighted-ML fusion requires a new decision record.
- **Frozen-pipeline note:** fixes 3 and any fusion-weight restoration touch frozen modules and require the Live Phase 0/4 scope decision. Training, evaluation, and shadow evidence can proceed independently of the freeze.

## Related decisions

- ADR-002: deterministic-first — this ADR enforces its addendum's "never the sole source" as "never load-bearing until proven".
- ADR-006 / ADR-008: acquisition and input manifests that supply the model's input contract.
- ADR-009: authenticated review — ML evidence is display/evidence only and never authorizes dispatch.
