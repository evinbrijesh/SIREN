-- SIREN — PostgreSQL / PostGIS production schema
-- Authoritative source: docs/PRD.md §10 data contracts. Field names/types must match exactly.
-- Activated when DATABASE_URL=postgresql://... is set at runtime (ADR-011).
-- The SQLite schema (schema.sql) remains the offline/demo fallback.
--
-- This schema is idempotent: safe to re-run on an existing database.
-- Requires: PostgreSQL 14+ with the PostGIS extension installed.

CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------------------
-- Basins (monitoring configuration)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS basins (
    basin_id        TEXT PRIMARY KEY,          -- e.g. 'dudh-koshi-demo-01'
    name            TEXT NOT NULL,
    boundary        geometry(MultiPolygon, 4326) NOT NULL,
    boundary_geojson JSONB NOT NULL,           -- original GeoJSON for API responses
    crs             TEXT NOT NULL DEFAULT 'EPSG:4326',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Observations (PRD §10.2)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS observations (
    observation_id          TEXT PRIMARY KEY,  -- e.g. 'obs-003'
    basin_id                TEXT NOT NULL REFERENCES basins(basin_id),
    acquired_at             TIMESTAMPTZ NOT NULL,
    source                  TEXT NOT NULL,     -- 'sentinel-1-grd-nrt' | 'sentinel-2-l2a' | 'prepared-demo'
    raster_uri              TEXT NOT NULL,     -- path under data/processed/ or s3:// URI
    footprint              geometry(Polygon, 4326),  -- scene footprint for spatial queries
    crs                     TEXT NOT NULL DEFAULT 'EPSG:4326',
    quality_score           DOUBLE PRECISION,  -- 0..1
    cloud_fraction          DOUBLE PRECISION,  -- 0..1 effective sensor cloud
    optical_cloud_fraction  DOUBLE PRECISION,  -- 0..1 source optical scene before SAR routing
    alignment_ok            BOOLEAN,           -- was the scene aligned to the basin grid
    usable                  BOOLEAN,           -- routing flag
    confidence_adjustment   DOUBLE PRECISION,  -- 0..1 multiplier
    water_area_km2          DOUBLE PRECISION,
    water_area_change_percent DOUBLE PRECISION,
    rainfall_24h_mm         DOUBLE PRECISION,
    rainfall_7d_mm          DOUBLE PRECISION,
    temp_mean_c             DOUBLE PRECISION,  -- mean air temp at acquisition (disease driver)
    temp_index              DOUBLE PRECISION,  -- 0..1 normalized temp (disease_risk input)
    mean_slope_degrees      DOUBLE PRECISION,
    processing_version      TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'ingested',  -- ingested|processed|failed
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Pipeline runs (one per observation processing pass)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,      -- e.g. 'run-0007'
    observation_id      TEXT NOT NULL REFERENCES observations(observation_id),
    processing_version  TEXT NOT NULL,
    change_mask_uri     TEXT,                  -- GeoTIFF path or s3:// URI
    corridor            geometry(LineString, 4326),  -- D8 downstream corridor
    corridor_geojson    JSONB,                  -- corridor as GeoJSON for API
    change_stats        JSONB,                  -- area, % expansion, class counts
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- Scores (hazard H, exposure E, disease D_risk) — PRD §9.5
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scores (
    score_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    hazard_score    DOUBLE PRECISION NOT NULL, -- H
    exposure_priority DOUBLE PRECISION NOT NULL, -- E
    disease_risk    DOUBLE PRECISION,          -- D_risk (nullable if no water points)
    confidence      DOUBLE PRECISION NOT NULL,  -- 0..1
    severity        TEXT NOT NULL,             -- informational|watch|elevated|critical
    reasons         JSONB NOT NULL,            -- array, >=3 entries on elevated+
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Assets (OSM-sourced critical infrastructure)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assets (
    asset_id        TEXT PRIMARY KEY,          -- e.g. 'BR-12', 'RD-4', 'village-2', 'well-3'
    basin_id        TEXT NOT NULL REFERENCES basins(basin_id),
    asset_type      TEXT NOT NULL,             -- village|bridge|road|well|clinic|shelter|school|food
    name            TEXT,
    geometry        geometry(Geometry, 4326) NOT NULL,  -- point/line/polygon
    geometry_geojson JSONB NOT NULL,           -- original GeoJSON for API responses
    population      INTEGER,                   -- for settlements
    weight          DOUBLE PRECISION NOT NULL DEFAULT 1.0  -- critical-infrastructure weight
);

-- ---------------------------------------------------------------------------
-- Exposures (asset-to-run intersection results)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exposures (
    exposure_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    asset_id        TEXT NOT NULL REFERENCES assets(asset_id),
    distance_m      DOUBLE PRECISION,          -- distance to corridor
    buffer_m        DOUBLE PRECISION,          -- tolerance buffer applied
    inundated       BOOLEAN NOT NULL DEFAULT FALSE,  -- TRUE if water point submerged/encircled
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Reviews (human-in-the-loop gate) — PRD §7.6
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reviews (
    review_id       TEXT PRIMARY KEY,
    score_id        TEXT NOT NULL REFERENCES scores(score_id),
    reviewer        TEXT NOT NULL,             -- authenticated actor
    decision        TEXT NOT NULL,             -- confirm|reject|postpone|escalate
    note            TEXT,
    decided_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Dispatches (simulated geofenced alert) — PRD §7.7
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dispatches (
    dispatch_id     TEXT PRIMARY KEY,
    review_id       TEXT NOT NULL REFERENCES reviews(review_id),
    alert_id        TEXT NOT NULL,             -- e.g. 'alert-0091'
    geofence_id     TEXT NOT NULL,
    payload         TEXT NOT NULL,             -- compressed <250-byte packet
    payload_bytes   INTEGER NOT NULL CHECK (payload_bytes <= 250),  -- Hard Rule 4, enforced here
    channel         TEXT NOT NULL,             -- sms|push|lora|satellite
    recipient_group TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'sent',  -- sent|delivered|failed
    sent_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Audit log (append-only lineage) — PRD §7.8
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    entry_id        BIGSERIAL PRIMARY KEY,
    alert_id        TEXT,                      -- nullable; lineage key
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,             -- run|score|review|dispatch|reject
    detail          JSONB NOT NULL,            -- full snapshot of the event
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    prev_hash       TEXT NOT NULL,
    event_hash      TEXT NOT NULL
);

-- Append-only enforcement (Hard Rule 3 / PRD §7.8): PL/pgSQL trigger that
-- raises an exception on any UPDATE or DELETE against audit_log. This is the
-- PostgreSQL equivalent of the SQLite BEFORE UPDATE/DELETE triggers.
CREATE OR REPLACE FUNCTION audit_log_append_only()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only: % forbidden', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_log_no_update ON audit_log;
CREATE TRIGGER audit_log_no_update
    BEFORE UPDATE ON audit_log
    FOR EACH ROW
    EXECUTE FUNCTION audit_log_append_only();

DROP TRIGGER IF EXISTS audit_log_no_delete ON audit_log;
CREATE TRIGGER audit_log_no_delete
    BEFORE DELETE ON audit_log
    FOR EACH ROW
    EXECUTE FUNCTION audit_log_append_only();

-- Human gate (Hard Rule 3 / PRD §7.6): a dispatch may only reference a
-- review whose decision is 'confirm'. Reject/postpone cannot dispatch.
CREATE OR REPLACE FUNCTION dispatches_require_confirm()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM reviews
        WHERE reviews.review_id = NEW.review_id
          AND reviews.decision = 'confirm'
    ) THEN
        RAISE EXCEPTION 'human gate: dispatch requires a review decision = confirm';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS dispatches_require_confirm_trigger ON dispatches;
CREATE TRIGGER dispatches_require_confirm_trigger
    BEFORE INSERT ON dispatches
    FOR EACH ROW
    EXECUTE FUNCTION dispatches_require_confirm();

-- ---------------------------------------------------------------------------
-- Acquisition jobs (ADR-008: durable orchestration and immutable manifests)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS acquisition_jobs (
    job_id              TEXT PRIMARY KEY,          -- e.g. 'acq-0001'
    source              TEXT NOT NULL,             -- 'cdse-s1' | 'cdse-s2' | 'srtm' | 'imerg' | 'overpass'
    provider_product_id TEXT NOT NULL,             -- scene ID, tile name, or date key
    status              TEXT NOT NULL DEFAULT 'pending',  -- pending|downloading|verified|ready|failed|dead_letter
    download_url        TEXT,
    local_path          TEXT,                      -- path under data/raw/ or s3:// URI
    acquired_at         TIMESTAMPTZ,                -- ISO-8601 UTC of the source product
    footprint           geometry(Polygon, 4326),   -- scene footprint for spatial dedup
    attempts            INTEGER NOT NULL DEFAULT 0,
    last_error          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(source, provider_product_id)
);

-- ---------------------------------------------------------------------------
-- Spatial indexes (GiST) — replaces manual Python bounding-box tolerance checks
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_basins_boundary ON basins USING GIST (boundary);
CREATE INDEX IF NOT EXISTS idx_obs_footprint ON observations USING GIST (footprint);
CREATE INDEX IF NOT EXISTS idx_runs_corridor ON runs USING GIST (corridor);
CREATE INDEX IF NOT EXISTS idx_assets_geometry ON assets USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_acq_footprint ON acquisition_jobs USING GIST (footprint);

-- ---------------------------------------------------------------------------
-- B-tree indexes (same as SQLite schema)
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_obs_basin ON observations(basin_id);
CREATE INDEX IF NOT EXISTS idx_obs_acquired ON observations(acquired_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_obs ON runs(observation_id);
CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id);
CREATE INDEX IF NOT EXISTS idx_exposures_run ON exposures(run_id);
CREATE INDEX IF NOT EXISTS idx_audit_alert ON audit_log(alert_id);
CREATE INDEX IF NOT EXISTS idx_acq_status ON acquisition_jobs(status);
CREATE INDEX IF NOT EXISTS idx_acq_source ON acquisition_jobs(source);

-- ---------------------------------------------------------------------------
-- updated_at trigger for acquisition_jobs
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS acq_jobs_updated_at ON acquisition_jobs;
CREATE TRIGGER acq_jobs_updated_at
    BEFORE UPDATE ON acquisition_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
