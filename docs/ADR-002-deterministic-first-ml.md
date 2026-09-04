# ADR-002 — Deterministic-First (No Trained ML in the Critical Path)

**Status:** Accepted · **Date:** 2026-09-04 · **Applies to:** change detection, risk fusion

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