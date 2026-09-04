# ADR-003 — SAR-First, Weather-Adaptive Change Detection

**Status:** Accepted · **Date:** 2026-09-04 · **Applies to:** change detection routing

## Context

The demo geography (Himalayan monsoon) is cloud-covered for much of the season. Optical-only change detection goes blind exactly when it matters. Sentinel-1 SAR backscatter is unaffected by cloud or darkness.

## Decision

The pipeline is **weather-adaptive**: the quality gate computes optical cloud fraction; if it is ≥ 0.20, the pipeline promotes **Sentinel-1 SAR backscatter differencing** to the primary change-detection path. Optical NDWI is used when skies are clear. SAR is treated as all-weather capable (cloud_fraction = 0.0 on that path).

## Consequences

- **Positive:** the demo's "monsoon observation" (95% cloud) still produces a change mask via SAR — this is the core differentiator and the demo's dramatic beat.
- **Negative:** SAR preprocessing (calibration, orbit correction) is more complex than optical; risk of stalling. Mitigated by a precomputed-mask fallback (Roadmap Phase 2).
- **Threshold tuning:** SAR ratio thresholds must be tuned by eye on real scenes — an OpenCode task, not Devin.

## Rationale

Monsoon cloud cover is the *expected* case in the Himalayas, not the edge case (PRD §5, principle 4). A system that goes blind in the rain is not operationally useful. SAR-first is the single most defensible technical decision for Track 7's "uncertainty" framing.