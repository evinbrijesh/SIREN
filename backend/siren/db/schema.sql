-- SIREN — SQLite schema
-- Authoritative source: docs/PRD.md §10 data contracts. Field names/types must match exactly.
-- Storage: SQLite with JSON columns for nested structures. No PostGIS (see ADR-001).
--
-- IMPORTANT (W7): `PRAGMA foreign_keys = ON` is PER-CONNECTION in SQLite.
-- Every connection (API, pipeline, tests) MUST execute this pragma itself
-- after connecting — executing this schema file only affects that connection.
-- The db repository factory must do:  conn.execute("PRAGMA foreign_keys = ON")

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Basins (monitoring configuration)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS basins (
    basin_id        TEXT PRIMARY KEY,          -- e.g. 'dudh-koshi-demo-01'
    name            TEXT NOT NULL,
    boundary_geojson TEXT NOT NULL,            -- GeoJSON polygon
    crs             TEXT NOT NULL DEFAULT 'EPSG:4326',
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ---------------------------------------------------------------------------
-- Observations (PRD §10.2)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS observations (
    observation_id          TEXT PRIMARY KEY,  -- e.g. 'obs-003'
    basin_id                TEXT NOT NULL REFERENCES basins(basin_id),
    acquired_at             TEXT NOT NULL,     -- ISO-8601 UTC
    source                  TEXT NOT NULL,     -- 'sentinel-1-grd-nrt' | 'sentinel-2-l2a' | 'prepared-demo'
    raster_uri              TEXT NOT NULL,     -- path under data/processed/
    crs                     TEXT NOT NULL DEFAULT 'EPSG:4326',
    quality_score           REAL,              -- 0..1
    cloud_fraction          REAL,              -- 0..1
    alignment_ok            INTEGER,           -- 0/1
    usable                  INTEGER,           -- 0/1 (routing flag)
    confidence_adjustment   REAL,              -- 0..1 multiplier
    water_area_km2          REAL,
    water_area_change_percent REAL,
    rainfall_24h_mm         REAL,
    rainfall_7d_mm          REAL,
    mean_slope_degrees      REAL,
    processing_version      TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'ingested',  -- ingested|processed|failed
    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ---------------------------------------------------------------------------
-- Pipeline runs (one per observation processing pass)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,      -- e.g. 'run-0007'
    observation_id      TEXT NOT NULL REFERENCES observations(observation_id),
    processing_version  TEXT NOT NULL,
    change_mask_uri     TEXT,                  -- GeoTIFF path
    corridor_geojson    TEXT,                  -- D8 downstream corridor
    change_stats_json   TEXT,                  -- area, % expansion, class counts
    started_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    finished_at         TEXT
);

-- ---------------------------------------------------------------------------
-- Scores (hazard H, exposure E, disease D_risk) — PRD §9.5
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scores (
    score_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    hazard_score    REAL NOT NULL,             -- H
    exposure_priority REAL NOT NULL,           -- E
    disease_risk    REAL,                      -- D_risk (nullable if no water points)
    confidence      REAL NOT NULL,             -- 0..1
    severity        TEXT NOT NULL,             -- informational|watch|elevated|critical
    reasons_json    TEXT NOT NULL,             -- array, >=3 entries on elevated+
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ---------------------------------------------------------------------------
-- Assets (OSM-sourced critical infrastructure)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assets (
    asset_id        TEXT PRIMARY KEY,          -- e.g. 'BR-12', 'RD-4', 'village-2', 'well-3'
    basin_id        TEXT NOT NULL REFERENCES basins(basin_id),
    asset_type      TEXT NOT NULL,             -- village|bridge|road|well|clinic|shelter|school|food
    name            TEXT,
    geometry_geojson TEXT NOT NULL,            -- point/line/polygon
    population      INTEGER,                   -- for settlements
    weight          REAL NOT NULL DEFAULT 1.0  -- critical-infrastructure weight
);

-- ---------------------------------------------------------------------------
-- Exposures (asset-to-run intersection results)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exposures (
    exposure_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    asset_id        TEXT NOT NULL REFERENCES assets(asset_id),
    distance_m      REAL,                      -- distance to corridor
    buffer_m        REAL,                      -- tolerance buffer applied
    inundated       INTEGER NOT NULL DEFAULT 0, -- 1 if water point submerged/encircled
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ---------------------------------------------------------------------------
-- Reviews (human-in-the-loop gate) — PRD §7.6
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reviews (
    review_id       TEXT PRIMARY KEY,
    score_id        TEXT NOT NULL REFERENCES scores(score_id),
    reviewer        TEXT NOT NULL,             -- authenticated actor
    decision        TEXT NOT NULL,             -- confirm|reject|postpone
    note            TEXT,
    decided_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
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
    sent_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ---------------------------------------------------------------------------
-- Audit log (append-only lineage) — PRD §7.8
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    entry_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id        TEXT,                      -- nullable; lineage key
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,             -- run|score|review|dispatch|reject
    detail_json     TEXT NOT NULL,             -- full snapshot of the event
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- Append-only enforcement (Hard Rule 3 / PRD §7.8 / D6): enforced at the
-- schema level — any UPDATE or DELETE on audit_log aborts the transaction.
CREATE TRIGGER IF NOT EXISTS audit_log_no_update
    BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
    BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only: DELETE forbidden');
END;

-- Human gate (Hard Rule 3 / PRD §7.6): a dispatch may only reference a
-- review whose decision is 'confirm'. Reject/postpone cannot dispatch.
CREATE TRIGGER IF NOT EXISTS dispatches_require_confirm
    BEFORE INSERT ON dispatches
    WHEN EXISTS (
        SELECT 1 FROM reviews
        WHERE reviews.review_id = NEW.review_id
          AND reviews.decision != 'confirm'
    )
BEGIN
    SELECT RAISE(ABORT, 'human gate: dispatch requires a review decision = confirm');
END;

CREATE TRIGGER IF NOT EXISTS dispatches_require_existing_review
    BEFORE INSERT ON dispatches
    WHEN NOT EXISTS (
        SELECT 1 FROM reviews WHERE reviews.review_id = NEW.review_id
    )
BEGIN
    SELECT RAISE(ABORT, 'human gate: dispatch requires an existing review row');
END;

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_obs_basin ON observations(basin_id);
CREATE INDEX IF NOT EXISTS idx_runs_obs ON runs(observation_id);
CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id);
CREATE INDEX IF NOT EXISTS idx_exposures_run ON exposures(run_id);
CREATE INDEX IF NOT EXISTS idx_audit_alert ON audit_log(alert_id);