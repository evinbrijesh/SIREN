# ADR-004 — Offline-First Demo

**Status:** Accepted · **Date:** 2026-09-04 · **Applies to:** runtime architecture

## Context

The hackathon demo must be reliable under live conditions — network flakiness, API rate limits, or a dead internet connection must not break the presentation. The demo narrative is a prepared 4-observation sequence.

## Decision

**Zero network calls at runtime.** All data loads from `data/` (prepared GeoTIFFs, OSM extracts, DEM clip, rainfall series). Live API ingestion (Copernicus, Earthdata, IMERG, Overpass) exists only as bonus scripts in `ingest/`, never as a runtime dependency.

## Consequences

- **Positive:** the demo runs in airplane mode; no external service can fail mid-presentation; reproducible across machines.
- **Negative:** the demo is a prepared sequence, not a live pull — must be framed honestly as "prepared scenes" (PRD §5, principle 6: realistic about latency).
- **Data hygiene:** only `ingest/` scripts write to `data/raw/`; only the pipeline writes `data/processed/`. Never commit rasters.

## Rationale

Reliability of the one-click demo chain is the #1 acceptance target (PRD §17.2). A live-API dependency converts a demo risk into a network risk. Prepared data is the standard, defensible choice for a 36-hour build.

---

## Addendum — Hosted service deployment profile (2026-09-07)

**Status change proposed:** Accepted (offline/demo profile) · See ADR-006 for hosted deployment.

ADR-004's "zero network calls at runtime" constraint is correct and retained for the offline/demo deployment profile. A continuously hosted service cannot acquire new observations without network access.

ADR-006 defines a separate **hosted/live profile** in which:
- A dedicated acquisition service makes all network calls (satellite catalogue, download, weather, OSM).
- The frozen scientific pipeline is never changed and never calls external APIs itself. It consumes only locally materialized, verified inputs.
- Provider outages degrade acquisition; they do not corrupt already-scored assessments.
- Live mode never silently substitutes demo scenario masks or hardcoded metadata for missing real inputs.

The two profiles share the same pipeline code and test fixtures. "Offline-first" remains the correct framing for demo and field-disconnected deployment. The hosted profile adds acquisition machinery around an unchanged execution core.

**Known blocker:** the current `run_pipeline()` implementation accepts only the three demo observation IDs. The hosted profile cannot produce live assessments until the observation-acceptance interface is extended under an approved scope decision. See the live service transition roadmap in `docs/spec/BUILD_ROADMAP.md`.