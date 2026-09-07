# ADR-008 — Durable Orchestration and Immutable Input Manifests

**Status:** Proposed · **Date:** 2026-09-07 · **Applies to:** job lifecycle and input lineage (hosted operation)

## Context

The hackathon implementation has no durable job lifecycle. Ingestion scripts are prep-time CLIs that exit `0` on many failure conditions. There is no record of which download attempts were made, which succeeded, and which failed. There is no idempotency key preventing a scheduler retry from creating a duplicate observation or a duplicate run.

For a continuously running hosted service, the absence of job state causes:

- **Silent data loss:** a crashed download leaves a partial file at the destination path, which a subsequent reader may treat as valid.
- **Duplicate assessments:** a scheduler that fires twice before the first job completes creates two run records for the same satellite pass.
- **Undetectable gaps:** there is no record of a poll cycle that found nothing new, making it impossible to distinguish "no new products published" from "the poll failed."
- **Untraceable evidence:** a run record in the database does not link to the exact verified raster file, weather window, OSM version, DEM pin, and processing container that produced it. Reproducing or auditing a past assessment requires manual reconstruction.
- **Stale inputs masquerading as current:** a successfully downloaded 12-day-old SAR product with no acquisition timestamp check can enter the pipeline and score as if it were new.

## Decision

### Acquisition job ledger

Add an `acquisition_jobs` table to track every product discovery, download attempt, and verification result:

```sql
CREATE TABLE acquisition_jobs (
    job_id          TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    provider_product_id TEXT,
    acquisition_time TEXT,
    publication_time TEXT,
    status          TEXT NOT NULL,
    -- discovered | selected | downloading | verified | ready
    -- | processing | published | retry_wait | quarantined | failed
    storage_path    TEXT,
    checksum        TEXT,
    footprint_geojson TEXT,
    attempts        INTEGER DEFAULT 0,
    last_error      TEXT,
    next_retry_at   TEXT,
    lease_expires_at TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE (source, provider_product_id)
);
```

The `UNIQUE (source, provider_product_id)` constraint is the idempotency guarantee. A scheduler that fires twice discovers the product once.

### Immutable input manifest

Each run record must be extended with an `input_manifest_id` foreign key to a new `input_manifests` table:

```sql
CREATE TABLE input_manifests (
    manifest_id     TEXT PRIMARY KEY,
    basin_id        TEXT NOT NULL REFERENCES basins(basin_id),
    sar_job_id      TEXT REFERENCES acquisition_jobs(job_id),
    baseline_job_id TEXT REFERENCES acquisition_jobs(job_id),
    weather_job_id  TEXT REFERENCES acquisition_jobs(job_id),
    osm_job_id      TEXT REFERENCES acquisition_jobs(job_id),
    dem_version     TEXT NOT NULL,
    processing_image TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
```

A manifest is created once, before execution begins, and never mutated. Reprocessing creates a new manifest and a new run — it does not overwrite the historical record.

### Atomic download and verification

All downloads must:
1. Stream to a temporary path (e.g. `{destination}.part`).
2. Verify file size against the provider-reported content-length.
3. Verify checksum against the provider-supplied hash (SHA-256 or MD5, depending on source).
4. Validate that the file opens correctly as a raster/GeoJSON/NetCDF.
5. Validate that the file covers the required monitoring area (lake footprint + buffer).
6. Rename to the final destination path atomically.
7. Mark the job `verified`.

A job is never marked `ready` until all six checks pass. A failed check produces a `quarantined` status with the error detail, never a silent overwrite of an existing valid file.

### Retry policy

| Failure class | Retry behavior |
|---|---|
| Transient network (5xx, timeout) | Exponential backoff: 30 s → 2 min → 10 min → 30 min; jitter derived from job ID |
| HTTP 429 | Honor `Retry-After` header; if absent, treat as 10 min |
| HTTP 401 | Attempt one token refresh; if still 401, mark `failed` and trigger credential incident |
| HTTP 403 | Mark `failed`; no retry; alert operator |
| Not-yet-published granule | Delayed retry; distinguished from invalid catalogue metadata |
| Checksum / coverage failure | Quarantine; never publish as `ready` |

After a configured maximum attempt count or age limit, move to dead-letter status and alert the ingestion-health channel. Dead-letter jobs are not retried automatically; they require operator review.

### Worker lease

Processing jobs must acquire a lease before execution. The lease prevents two workers from processing the same job simultaneously (fence on concurrent scheduler triggers). The lease expiry is set to a safe multiple of the expected execution time. On worker restart, expired leases are reclaimed by a reconciliation scan.

## Consequences

- **Positive:** every acquisition attempt, outcome, and retry is traceable.
- **Positive:** idempotency prevents duplicate observations on scheduler retry.
- **Positive:** immutable manifests allow exact historical reproduction and independent audit.
- **Positive:** quarantine prevents corrupt or incomplete files from reaching the scientific pipeline.
- **Positive:** lease-based execution prevents duplicate runs on concurrent scheduler triggers.
- **Negative:** adds schema tables and code that do not exist in the demo implementation. Migration from the demo schema is required before the hosted service goes live.
- **Negative:** partial download cleanup (`.part` files, abandoned multipart uploads in S3) must be handled by lifecycle rules, not just application logic.
- **Constraint:** the `input_manifest_id` FK on `runs` requires a schema migration. If the frozen pipeline's DB writes cannot be extended to record the manifest ID, the link must be maintained in a separate operational table outside the frozen schema.

## Rationale

The job ledger is the minimum durable state required to distinguish "download succeeded" from "process exited 0." Without it, operational monitoring is guesswork. The immutable manifest is the minimum lineage record required to reproduce or dispute a historical assessment. Both are standard in any production data pipeline; their absence from the demo was an explicit scope decision, not an architectural endorsement.

## Related decisions

- ADR-006: the acquisition service whose download/verification outcomes this table records.
- ADR-007: the PostgreSQL database that provides the unique constraints and transaction isolation this table requires.
- ADR-009: the delivery outbox that extends the same durable-job pattern to confirmed alert dispatch.
