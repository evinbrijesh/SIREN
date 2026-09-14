# ADR-013 — End-to-End Neural Pipeline

**Status:** ACCEPTED · **Date:** 2026-09-14 · **Supersedes:** ADR-002 (deterministic-first ML) in the research path · **Amends:** ADR-012 (FNO input contract) · **Companions:** [PRD v5.0 §9.7](../spec/PRD.md), [ADR-011.1](ADR-011.1-real-data-sar-gate-calibration.md), [ADR-010](ADR-010-ml-evidence-isolation-and-retraining-path.md)

---

## Context

The v4.7 shadow architecture (ADR-002 deterministic-first + ADR-010 ML isolation + ADR-011/011.1 gate calibration + ADR-012 FNO surrogate) achieved a working hybrid pipeline: real-data 6-channel SAR segmentation (IoU 0.62, ADR-011.1 gate-passed), Huggel empirical breach-volume fallback, and a scalar-conditioned FNO hydrodynamic surrogate. However, four non-neural bottlenecks constrain the system to a hybrid posture and prevent gradient flow from raw radar bytes to downstream flood dynamics:

1. **Empirical volume formula.** The Huggel et al. (2002) power law V = 0.104 · A^1.421 is a 24-year-old scalar approximation that discards all spatial structure of the lake basin. It is the only fallback when the DEM is a surface model (DSM), which is the common case for SRTM over water.

2. **Single-modality saturation.** The 6-channel SAR-only WaterResUNet saturates at ~0.62 IoU due to C-band layover, shadow, and speckle in steep Himalayan terrain. Three approaches (threshold sweep, TTA, pos-weight fine-tune) all failed to close the 0.03 gap to the original 0.65 gate.

3. **Broken gradient flow.** The pipeline transitions segmentation → discrete pixel counting → scalar V_breach → FNO. The FNO receives only a 1D scalar (log(1+V_breach)/20), discarding all spatial lake-basin geometry. The segmentation and hydrodynamic stages are detached.

4. **No epistemic uncertainty.** Hardcoded severity thresholds (expansion ≥ 40% → critical) provide no measure of confidence in the prediction itself. Shadow systems are retained because teams cannot trust neural networks to detect their own OOD errors.

---

## Decision

### Transition from hybrid shadow to end-to-end differentiable neural pipeline

The v5.0 pipeline eliminates each bottleneck with a corresponding neural component, creating a differentiable path from raw radar bytes to downstream flood dynamics. The deterministic baseline is retained as a labeled fallback and regression target at each stage — the neural component must pass its gate before promotion, and the fallback is always available with provenance recording.

#### 1. Neural Bathymetry Inversion (replaces Huggel formula)

Train a physics-informed neural network (PINN) or implicit neural representation (INR) to reconstruct submerged lake bed topography z_bed(x, y) from surrounding moraine DEM contours, lake boundary shape features, and glacier terminus velocity fields.

- **Training data:** Millan et al. (2022) consensus ice-thickness estimates and Farinotti et al. (2019) ITMIX 2 bed-inversion datasets.
- **Architecture:** small U-Net (3–5M params) or INR with coordinate encoding, conditioned on moraine DEM contours and lake boundary polygon.
- **Integration:** `breach_volume.py` `auto` mode cascade: neural bathymetry → Huggel fallback → strict hypsometric (if DEM resolves bathymetry).
- **Gate:** < 15% MAPE on held-out lakes with known bathymetry. Until this gate passes, Huggel remains primary.

#### 2. Multi-Modal Sensor Fusion (replaces single-modality SAR)

Implement cross-attention fusion of dual-pol Sentinel-1 SAR (all-weather) with cloud-masked Sentinel-2 multispectral imagery (NDWI, MNDWI).

- **Architecture:** lightweight cross-attention transformer (SegFormer-style) where SAR features query optical features when cloud cover is low; cloud-gated attention mask handles variable optical availability.
- **Input contract:** SAR (6ch) + optical (NDWI, MNDWI, cloud mask) when available. Falls back to SAR-only when clouds block optical.
- **Gate:** event-held-out IoU > 0.75 AND precision ≥ 0.85 on real paired SAR+optical data. Until this gate passes, the SAR-only 6-channel model (ADR-011.1 gate-passed) remains primary.

#### 3. Latent Spatial Conditioning (replaces scalar V_breach injection)

Feed the segmentation bottleneck embedding z_lake ∈ R^(C×H/16×W/16) directly into the FNO's lifting layer alongside the normalized DEM, instead of converting the segmented mask to a scalar V_breach.

- **Formula:** h_0(x, y) = P(DEM(x, y), W(z_lake)) where P is the FNO lifting projection and W is a learned projection from the segmentation bottleneck to the FNO input width.
- **Implementation:** `FNO2D` input contract changes from (B, 2, H, W) [DEM + V_breach] to (B, 2 + d_latent, H, W) [DEM + projected z_lake]. The scalar V_breach path is retained as a labeled fallback channel.
- **Gate:** ADR-012's MAPE ≤ 20% on at least two of three validation events. The scalar-injection FNO remains as a labeled fallback until the latent-conditioned variant passes.

#### 4. Bayesian Neural Uncertainty (augments hardcoded thresholds)

Add Monte Carlo Dropout or Deep Evidential Regression to WaterResUNet and output a spatially resolved uncertainty map σ²(x, y) alongside the water mask.

- **Mechanism:** MC Dropout — run T forward passes with dropout active at inference; compute per-pixel mean + variance. Deep Evidential Regression — interpret the evidential output distribution directly.
- **Conformal calibration:** split conformal prediction on a held-out calibration set to guarantee distribution-free coverage of the 90% confidence interval.
- **Gate:** empirical coverage within ±5% of the nominal 90% level on a held-out calibration set. Until this gate passes, uncertainty maps are informational only.

---

## Consequences

- **Positive:** the pipeline becomes a single differentiable system from raw radar bytes to downstream flood dynamics, enabling end-to-end gradient flow and joint optimization. The neural components replace lossy scalar approximations with spatially-resolved predictions. Uncertainty maps provide statistically rigorous safety bounds without hardcoded threshold fallbacks.
- **Negative:** each neural component requires its own training data, gate evaluation, and provenance trail. The system complexity increases — four new model architectures, four new gates, four new fallback paths. Training the neural bathymetry model requires external datasets (Millan/Farinotti) that may not be immediately available.
- **Risk:** a neural component that fails its gate leaves the system on the fallback path, which is the v4.7 hybrid architecture. This is safe but means the "end-to-end neural" claim is conditional on all four gates passing. The PRD and audit trail must clearly record which components are neural-primary vs fallback-primary at any given release.
- **Preserved:** the human gate (Hard Rule 3), explainability (Hard Rule 5), ≤250-byte payload (Hard Rule 4), reproducibility (Hard Rule 6), and the offline demo regression chain. ADR-002's deterministic-first principle is superseded in the research path but retained for the operational fallback.

---

## Relationship to existing ADRs

| ADR | Relationship |
|---|---|
| ADR-002 (deterministic-first) | Superseded in the research path; deterministic baseline retained as labeled fallback |
| ADR-010 (ML evidence isolation) | Preserved — each neural component runs in isolation with its own gate and provenance |
| ADR-011 (production multimodal upgrade) | Preserved — IoU > 0.65 gate remains for synthetic baselines; ADR-011.1 calibrated gate for real-data SAR |
| ADR-011.1 (real-data SAR gate) | Preserved — the Kuro Siwo 6-channel model is the SAR-only fallback for the multi-modal fusion model |
| ADR-012 (FNO hydrodynamic surrogate) | Amended — FNO input contract expanded from 2 channels (DEM + V_breach) to 2 + d_latent (DEM + projected z_lake); scalar path retained as fallback |
