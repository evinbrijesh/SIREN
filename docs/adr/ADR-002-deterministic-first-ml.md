# ADR-002 — Deterministic-First (No Trained ML in the Critical Path)

**Status:** Accepted + Implemented-as-optional-layer · **Date:** 2026-09-04 · **Applies to:** change detection, risk fusion

## Context

SIREN is a safety-adjacent decision-support system. Judges and reviewers will probe whether scores are explainable and reproducible. Trained models risk black-box behavior, non-determinism, and a half-finished model that breaks the demo.

## Decision

The MVP critical path is **rule-based and deterministic**: NDWI differencing + SAR backscatter ratio thresholding for change detection, weighted scoring for risk fusion, persistence rules for trend. Trained models (Siamese U-Net / ChangeFormer, SegFormer, ConvLSTM) are a **stretch goal gated on the core loop working** (Roadmap Phase 6).

## Consequences

- **Positive:** every score is explainable (deterministic `reasons`), reproducible (same inputs + version → same output), and the demo never depends on a model that might not train in time.
- **Negative:** lower pixel-level accuracy than a fine-tuned model on complex scenes; thresholds must be tuned by eye on real data (OpenCode's job).
- **ML path:** if time permits, fine-tune a focused change-detection model as an *additional* evidence layer — never as the sole source of a score.

## Rationale

A rule-based change mask that completes the full evidence→review→dispatch loop beats a partially-trained neural net that doesn't. This is explicitly endorsed in PRD §15 ("prioritize the complete workflow over a sophisticated trained model").

---

## Addendum — Implemented as Optional Evidence Layer (post-build)

**Date:** 2026-08-12 · **Status change:** Accepted → Accepted + Implemented-as-optional-layer

The ML path described in §Consequences ("if time permits, fine-tune a focused change-detection model as an *additional* evidence layer") has been implemented as `backend/siren/ml/`:

- **`ml/consensus.py`** — deterministic consensus mask (NDWI + SAR backscatter agreement)
- **`ml/engine.py`** — Siamese U-Net / ChangeFormer inference path (torch-gated)
- **`ml/model.py`** — model definitions
- **`ml/train.py`** — training entry point (not in the demo critical path)
- **`ml/visualize.py`** — heatmap and preview generation

The deterministic fallback runs without torch installed. When torch is available (`pip install -e ".[ml]"`), the ML path produces a heatmap and change mask as an **additional evidence layer** exposed via `GET /runs/{run_id}/ml-evidence`. It is never the sole source of a score — the rule-based pipeline remains the critical path.

**Test coverage:** 13 tests in `tests/test_ml.py` (3 torch-gated, 10 deterministic).

**Dependency note:** torch/torchvision are an optional `[ml]` extra in `pyproject.toml`, outside the original AGENTS.md dependency whitelist. This is an approved exception (AGENTS.md rule 8 addendum).

---

## Addendum 2 — 2026-09-07 audit findings (implementation divergence)

**Status:** Accepted + Implemented-as-optional-layer (unchanged) · **Companions:** [`docs/reference/DL_MODEL_AUDIT.md`](../reference/DL_MODEL_AUDIT.md), [ADR-010](ADR-010-ml-evidence-isolation-and-retraining-path.md) (Proposed)

A full audit of the implemented ML layer against this ADR found:

1. **The rule-based critical path is not preserved as specified.** `risk/fusion.py` implements a six-factor H with a 0.20-weight ML-confidence term (0.25/0.20/0.15/0.10/0.10/0.20), diverging from the PRD §9.5 five-factor weights (0.30/0.25/0.20/0.15/0.10). The default 0.5 neutral value does not restore the original formula.
2. **Trend classification can be replaced by a trained model.** `run_pipeline()` starts from the configured deterministic trend but overwrites it with the ConvLSTM hybrid result, which feeds H directly. The ConvLSTM was trained on synthetic water-mask progressions and its inference wrapper fabricates missing timesteps by dilating the last mask.
3. **"ChangeFormer" is not implemented.** It is named in this ADR and the PRD, but no code or checkpoint exists.
4. **"SegFormer" is not the SegFormer architecture.** `ml/model.py::SegFormerHead` is a patch-embedding crop classifier (one attention block, global pooling, one class per crop), trained on threshold-generated weak labels with an unreachable "shadow" class, whose false-alarm filtering can remove rule-detected pixels from the consensus evidence mask.
5. **The Siamese U-Net exists but its training and runtime inputs do not match.** Training synthesizes "before" images by replacing labeled water with median land backscatter (the label leaks into the input); runtime feeds binary water masks, not SAR σ0.
6. **No held-out evaluation exists.** All three checkpoints report training-set metrics; model selection is by training loss; there is no event-level split, no held-out test set, and no prospective evaluation.

ADR-010 (Proposed) records the response: ML evidence isolation (shadow mode, immutable rule assessment, separately recorded ML evidence), restoration of the documented five-factor fusion weights, and a retraining path starting with a compact single-date SAR water-segmentation model on real labels with event-level splits.