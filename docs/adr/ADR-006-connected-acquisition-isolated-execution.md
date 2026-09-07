# ADR-006 — Connected Acquisition, Isolated Deterministic Execution

**Status:** Proposed · **Date:** 2026-09-07 · **Applies to:** runtime architecture (hosted operation)
**Supersedes:** ADR-004 for hosted deployment; ADR-004 remains Accepted for the offline/demo profile.

## Context

ADR-004 made "zero network calls at runtime" the non-negotiable constraint for the hackathon demo. That decision was correct for its purpose: reproducibility, demo reliability, and offline resilience. SIREN is now being evaluated for transition to a continuously running hosted service that polls live satellite, weather, and OSM sources. A hosted service that makes zero network calls can never acquire new observations.

The tension is:

- ADR-004's offline guarantee is still a valid *deployment mode* (demo, field disconnected, disaster network outage).
- A continuously hosted service must acquire data without manual preparation.
- The frozen deterministic pipeline must not be changed (ADR-002 principle).
- Human confirmation before dispatch must be preserved (hard rule 3).

## Decision

Separate the runtime into two independently deployed profiles:

**Offline / demo profile** (ADR-004, unchanged):
- Zero network calls during pipeline execution.
- All inputs loaded from `data/`.
- Suitable for presentations, field deployment with disconnected networks, and regression testing.

**Hosted / live profile** (this ADR):
- A separate acquisition service runs network calls: satellite catalogue discovery, product downloads, weather fetch, OSM refresh.
- Scientific execution (the frozen pipeline) consumes only **locally materialized, verified, immutable inputs**. It never calls an external API.
- Provider outages preserve the last verified result, labeled with observed age and degraded status.
- Live mode never silently substitutes scenario masks, demo metadata, or hardcoded expansion values. If a required input is missing, the assessment is blocked and the gap is visible.
- Demo and operational data use separate namespaces, basins, and observation IDs. The demo sequence cannot be triggered by operational schedulers.

The acquisition service is **not** part of the frozen pipeline. It produces verified, registered inputs that the pipeline then consumes. The interface between acquisition and execution is the `observations` table plus the local storage path of each verified product.

## Consequences

- **Positive:** the frozen pipeline's reproducibility and determinism are preserved. Each execution has the same input→output contract regardless of whether inputs arrived from a live download or were manually prepared.
- **Positive:** provider outages degrade acquisition; they do not degrade already-scored assessments.
- **Positive:** offline and live modes share the same pipeline code, test fixtures, and deterministic behavior.
- **Negative:** two deployment profiles must be maintained and tested separately.
- **Negative:** a live observation cannot be scored until all required inputs are locally materialized, which adds latency relative to streaming from object storage.
- **Blocker:** the current `run_pipeline()` only accepts the three hardcoded demo observation IDs. The live profile cannot produce honest live assessments until either (a) the observation-acceptance interface is extended under an approved scope change, or (b) an explicit adapter is built outside the frozen boundary. This must be resolved before Phase 4 of the transition roadmap.

## Rationale

An acquisition service that downloads then validates inputs before handing them to execution is the standard split for geospatial pipelines. It provides natural retry, checksum verification, and quarantine points without touching the scientific code. The frozen pipeline becomes a black box that receives local files and returns results — which is exactly the contract it already implements for the demo.

## Related decisions

- ADR-004: retained Accepted for offline/demo profile.
- ADR-007: database and storage for hosted operation.
- ADR-008: durable orchestration connecting acquisition to execution.
