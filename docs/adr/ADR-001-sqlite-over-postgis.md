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

---

## Addendum — Hosted service considerations (2026-09-07)

**Status change proposed:** Accepted (demo/single-host) · See ADR-007 for hosted deployment.

The following limitations have been identified that make SQLite unsuitable as the sole persistence layer for a continuously hosted multi-instance service:

- **SQLite WAL does not work over network filesystems.** The SQLite documentation explicitly warns against using WAL mode on NFS, EFS, or any network-mounted path. Putting a shared `.db` file on EFS for multi-instance API access is unsupported and risks corruption.
- **Concurrent write contention.** A single persistent connection with `check_same_thread=False` and a five-second busy timeout does not serialize concurrent ingestion workers, processing workers, and audit writes correctly under sustained load.
- **Count-based ID generation.** The current `SELECT COUNT(*) + 1` pattern for run and observation IDs produces collisions under concurrent inserts.
- **No native idempotency constraints.** There is no provider-product-ID uniqueness constraint. Scheduler retries can create duplicate observations.
- **No backup/PITR.** File-copy backup can produce a corrupt snapshot if a writer is active at copy time.

ADR-007 proposes RDS PostgreSQL as the persistence layer for hosted operation. SQLite is retained and continues to be the correct choice for demo and single-host operation (ADR-001 remains Accepted for those profiles). The frozen pipeline's in-memory spatial joins via geopandas are not migrated to PostGIS.