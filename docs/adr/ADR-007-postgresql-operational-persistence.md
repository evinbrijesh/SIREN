# ADR-007 — PostgreSQL Operational Persistence; SQLite Retained for Demo

**Status:** Proposed · **Date:** 2026-09-07 · **Applies to:** storage layer (hosted operation)
**Supersedes:** ADR-001 for hosted multi-instance deployment; ADR-001 remains Accepted for demo/single-host operation.

## Context

ADR-001 chose SQLite for zero setup time at hackathon scale. In a hosted continuous service, the storage layer faces constraints SQLite was not designed for:

- Multiple concurrent writers: ingestion workers, the API server, processing workers, and audit writes all need to write simultaneously.
- Multiple API instances: SQLite WAL requires all cooperating processes to share the same filesystem host. The [SQLite documentation](https://www.sqlite.org/useovernet.html) explicitly warns that WAL does not work correctly over network filesystems (NFS, EFS). Putting a shared SQLite file on EFS for multi-instance access is not supported.
- Backups and recovery: SQLite's file-copy backup can produce corrupt output if a writer is active. Point-in-time recovery is not natively supported.
- Unique constraints for idempotency: the current schema has no provider-product-ID uniqueness constraint, meaning duplicate observations can be created on scheduler retry.
- Count-based ID generation: `SELECT COUNT(*) + 1` for run/observation IDs produces collisions under concurrent inserts.

Spatial indexing (PostGIS) is a secondary concern. The frozen pipeline performs all spatial joins in-memory via geopandas and must not be changed. PostGIS is useful for the operational asset catalogue, footprint queries, and future multi-basin support — not for replacing the frozen spatial logic.

## Decision

**For hosted multi-instance operation:** use **RDS PostgreSQL** with the **PostGIS extension** enabled.

- Concurrent writers are handled by PostgreSQL's MVCC isolation.
- Multi-AZ deployment provides managed failover.
- Automated backups and point-in-time recovery are available by default.
- Provider-product-ID uniqueness constraints are native.
- Sequence-based IDs replace the current count-based approach.
- The `acquisition_jobs` table (ADR-008) requires durability across process restarts, which SQLite in `:memory:` cannot provide.

**For demo / single-host operation:** SQLite remains the storage layer per ADR-001. The demo runs offline with zero additional setup.

**For spatial operations:** enable PostGIS on the operational database for the asset catalogue, footprint storage, and basin boundary queries. The frozen pipeline's spatial joins are not migrated — they continue to run in-memory via geopandas on the extracted GeoJSON. PostGIS is an operational catalogue index, not a pipeline dependency.

## Migration boundary

The migration must preserve the existing repository interface. `backend/siren/db/repo.py` must continue to expose the same public methods with the same field names and types. The storage adapter underneath can be swapped — the frozen pipeline must not be aware of the change.

Migration tasks that are NOT mechanical:

- SQLite-specific SQL, boolean representations, JSON columns, and timestamp handling differ from PostgreSQL.
- Current `per-method commits` need a defined transaction boundary appropriate for PostgreSQL's MVCC.
- Hash chain serialization must preserve exact historical bytes across the migration.
- Legacy assets with `geometry = [0, 0]` must not become authoritative spatial records in PostGIS.
- Migration requires row count verification, FK constraint checks, record reconciliation, and restore testing before switch-over.

## Consequences

- **Positive:** concurrent ingestion, processing, API reads, and audit writes coexist correctly.
- **Positive:** managed backups, PITR, and Multi-AZ failover for operational continuity.
- **Positive:** idempotency constraints prevent duplicate observations and duplicate runs on scheduler retry.
- **Negative:** adds operational complexity and cost (approximately $100–140/month for a modest Multi-AZ instance). Not appropriate for a single-machine demo.
- **Negative:** migration from the current SQLite schema requires careful field-by-field translation and restore testing.
- **Negative:** if the frozen pipeline's repository interface is treated as immutable, a compatibility adapter layer must be built and maintained.
- **Do not:** share a SQLite WAL file on EFS or any network filesystem.
- **Do not:** migrate the frozen spatial joins from geopandas into PostGIS queries. That would change frozen behavior.

## Rationale

PostgreSQL is the standard production persistence choice for Python geospatial services. RDS eliminates server management. The PostGIS extension is available on all RDS-supported PostgreSQL versions. The decision to pull this migration forward (relative to the PRD §18 V2 roadmap) is justified by the requirement for concurrent writers and multi-instance API deployment in the hosted service, not by spatial indexing alone.

## Related decisions

- ADR-001: retained Accepted for demo/single-host profile.
- ADR-006: acquisition service that produces the additional writes this decision accommodates.
- ADR-008: durable orchestration that requires the idempotency constraints this decision provides.
