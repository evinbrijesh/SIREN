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