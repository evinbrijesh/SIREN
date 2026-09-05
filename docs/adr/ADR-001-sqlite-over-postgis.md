# ADR-001 — SQLite over PostGIS

**Status:** Accepted · **Date:** 2026-09-04 · **Applies to:** storage layer

## Context

SIREN needs to persist observations, runs, scores, reviews, dispatches, and an append-only audit log, plus run spatial joins (corridor × asset intersections). PostGIS is the "proper" geospatial database; SQLite is zero-ops.

## Decision

Use **SQLite with JSON columns** for persistence, and **GeoJSON files on disk** for geometry. All spatial joins run in-memory via geopandas on the small basin extract (< 100 MB).

## Consequences

- **Positive:** zero setup, no server process, offline-safe, trivially reproducible (single file), ideal for a 36-hour hackathon and an offline demo.
- **Negative:** no native spatial indexing; spatial queries are in-memory only. Not suitable for multi-basin national scale.
- **Migration path:** the schema (`backend/siren/db/schema.sql`) is deliberately PostGIS-shaped (geometry as GeoJSON text, explicit FK relationships), so a V2 migration to PostGIS is mechanical.

## Rationale

At hackathon scale the basin extract is small enough that in-memory geopandas joins are effectively instant. The cost of running PostGIS (setup, ops, demo fragility) outweighs the indexing benefit for a single-basin demo. This is a scope decision, not a permanent one — see Roadmap V2 (PostGIS migration).