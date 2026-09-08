# ADR-011 — Production Multimodal Upgrade

**Status:** ACCEPTED · **Date:** 2026-09-08 · **Supersedes:** ADR-010 §2 (2-channel frozen contract) and §3 (ML shadow-only, never load-bearing) — only on acceptance of the relevant phase's evaluation gate
**Applies to:** ML layer (`backend/siren/ml/`), tensor ingestion contract, dependency whitelist (Hard Rule 8), production infrastructure
**Companions:** [ADR-002](ADR-002-deterministic-first-ml.md), [ADR-010](ADR-010-ml-evidence-isolation-and-retraining-path.md), [`V3_RESEARCH_PROPOSAL.md`](../spec/V3_RESEARCH_PROPOSAL.md), [`PRODUCTION_ROADMAP.md`](../spec/PRODUCTION_ROADMAP.md)

---

## Context

The hackathon MVP (v1.0.0-hackathon-final) shipped under strict artificial constraints to prevent last-minute breaks:

- Hard Rule 8 froze the dependency whitelist to a minimal geospatial stack.
- ADR-010 §2 froze the ML tensor contract to 2 channels (VV/VH σ⁰ dB).
- ADR-010 §3 mandated ML shadow-only status (never load-bearing) due to 0.24 IoU on event-holdout.
- All observations were hardcoded demo IDs (`obs-001/002/003`) with static weather JSON.

The hackathon is over. The demo is done. The system is transitioning to a production-grade, autonomous disaster-response platform. The artificial constraints that protected the hackathon build now actively block the production transition.

Three concrete blockers require governance clearance:

1. **Tensor contract freeze.** The 2-channel contract cannot produce terrain-aware water masks. The 4-channel expansion (VV, VH, DEM, Slope) is the known fix for the 0.24 IoU OOD collapse (V3 §2.1).
2. **Dependency whitelist.** Production requires `psycopg`, `geoalchemy2`, `xgboost`, `shap`, `celery`, and `redis` — all currently banned under Hard Rule 8.
3. **ML load-bearing prohibition.** ADR-010 §3 prohibits ML from entering the hazard score. The XGBoost susceptibility scorer (V3 §3.2) and FNO hydrodynamic surrogate (V3 §4) require a load-bearing role once their evaluation gates are met.

---

## Decision

### 1. Unfreeze the tensor ingestion contract

Supersede ADR-010 §2. Authorize expanding `backend/siren/ml/contract.py` from 2 → 4 channels:

$$\mathbf{X} \in \mathbb{R}^{B \times 4 \times H \times W} \quad \big[\sigma^0_{VV}, \; \sigma^0_{VH}, \; \text{DEM Elevation}, \; \text{Slope Angle}\big]$$

**Condition:** this expansion invalidates all existing 2-channel weights. A new WaterResUNet checkpoint must be trained from scratch. The 4-channel model enters a load-bearing role **only** when event-holdout IoU > 0.65 (V3 §2.4 gate). Until then, the 2-channel shadow model and the deterministic scenario masks remain the fallback.

### 2. Amend Hard Rule 8 — production dependency addendum

Add the following packages under a `[production]` extra in `pyproject.toml`, mirroring the existing `[ml]` extra pattern:

| Package | Phase | Justification | Risk |
|---|---|---|---|
| `psycopg[binary]` | Sprint 1 | PostgreSQL + PostGIS driver | Pre-built wheel; low binary risk |
| `geoalchemy2` | Sprint 1 | PostGIS spatial ORM for SQLAlchemy | Pure-python; low risk |
| `celery` | Sprint 1 | Asynchronous job queue for ingestion daemon | Pure-python + redis dependency |
| `redis` | Sprint 1 | Celery broker + result backend | Low risk; standard infrastructure |
| `rtree` | Sprint 1 | Spatial indexing (GEOS-backed) | Pre-built wheel; low risk |
| `xgboost` | Sprint 3 | Gradient-boosted trees for calibrated P_breach; TreeSHAP-compatible | Pre-built wheel; low binary risk |
| `shap` | Sprint 3 | Feature attribution for explainability (Hard Rule 5) | Pure-python; low risk |
| `neuraloperator` | Sprint 3 | FNO-2D hydrodynamic surrogate | Review license + binary deps before adoption |

**The hackathon `[ml]` extra (torch/torchvision) remains unchanged.** The `[production]` extra is additive — `pip install -e ".[ml,production]"` installs everything.

### 3. Authorize ML load-bearing role (gated)

Supersede ADR-010 §3 **only on acceptance of the relevant phase's evaluation gate**:

| ML component | Load-bearing role | Gate condition |
|---|---|---|
| 4-channel WaterResUNet | Replaces deterministic scenario masks as primary water detection | Event-holdout IoU > 0.65 (V3 §2.4) |
| XGBoost P_breach | Enters hazard score as a sixth factor | Shadow-mode shows calibrated improvement over rules-only baseline + Brier score < 0.15 |
| FNO-2D h_water / T_arrival | Supersedes static tolerance buffers | South Lhonak retrospective validation within tolerance (V3 §4.4) |

**Until each gate is met, the corresponding ML component remains shadow-only.** The deterministic five-factor path remains the fallback at every stage.

### 4. Preserve human-in-the-loop confirmation gates

The human gate (Hard Rule 3) is **not** superseded. No code path may dispatch an alert without a recorded `confirm` review. The HTTP 409 safeguard on dispatch-without-confirm remains active. Conformal prediction intervals (V3 §3.4) add an automatic fallback to human inspection when epistemic uncertainty is too high — this strengthens the human gate, it does not weaken it.

### 5. Retain ADR-010 principles

The following ADR-010 clauses remain in force and are **not** superseded:

- ML evidence isolation contract (§3): rule-based assessment is computed and stored immutably; ML outputs are separate evidence records.
- Input contract discipline: exact tensor contract, normalization, and channel order are frozen per version.
- Event-level splits: model evaluation uses official event-holdout splits, never random chip splits.
- License review: all training data and model weights must have documented licenses before adoption.
- Model registry honesty: model IDs, versions, and checkpoint paths are recorded in audit log.
- Shadow-mode-first: every new ML component runs in shadow mode before any load-bearing role.

---

## Consequences

- **Positive:** clears the governance path for the production transition described in `PRODUCTION_ROADMAP.md`. The 4-channel expansion, production dependencies, and gated ML load-bearing role are now authorized. Sprint 1 can begin.
- **Negative:** three new optional dependency groups; retraining cost for 4-channel model; HEC-RAS data generation for FNO is non-trivial; PostgreSQL migration requires schema + data migration from SQLite.
- **Frozen-pipeline note:** the hackathon v1.0.0-hackathon-final tag is preserved. This ADR authorizes post-hackathon changes on a new branch; the frozen release is not modified.

---

## Relationship to V3 Research Proposal

This ADR accepts the **governance framework** for the V3 upgrade. The **technical implementation details** (L_gravity loss, DANN, conformal prediction, FNO architecture, HAND computation) remain specified in `V3_RESEARCH_PROPOSAL.md` as a research RFC. This ADR does not mandate specific architectures — it authorizes the dependency and contract changes needed to implement them.

The evaluation gates in V3 (IoU > 0.65, Brier < 0.15, South Lhonak validation) are the conditions under which ML components transition from shadow to load-bearing. These gates are not negotiable.
