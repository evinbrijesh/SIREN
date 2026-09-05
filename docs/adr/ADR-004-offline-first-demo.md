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