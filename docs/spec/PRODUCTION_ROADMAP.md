# SIREN — Production Transition Roadmap

**Status:** Active · **Date:** 2026-09-08 · **Supersedes:** Live Service Transition Roadmap in `BUILD_ROADMAP.md` (phases 0–6 there are preserved as hackathon history)
**Companions:** [ADR-011](../adr/ADR-011-production-multimodal-upgrade.md) (Accepted), [V3_RESEARCH_PROPOSAL.md](V3_RESEARCH_PROPOSAL.md), [KNOWN_LIMITATIONS.md](../reference/KNOWN_LIMITATIONS.md)

> **The rules have changed.** Hackathon code survives a 3-minute pitch. Production code runs unattended at 3 AM during a monsoon cloudburst without hallucinating an evacuation or dropping an alert. This document is the engineering spec for that transition.

---

## The Four Pillars

| Pillar | Hackathon State | Production Target |
|---|---|---|
| Data ingestion | Manual `curl -X POST /runs/process-all` | Autonomous STAC polling daemon (Celery/Temporal) |
| Vision & hydrology | 2-channel SAR + D8 + 125 m planar buffer | 4-channel terrain-aware WaterResUNet + HAND flood zoning |
| Risk scoring | 5-factor linear weights, static demo observations | XGBoost susceptibility + conformal prediction intervals |
| Infrastructure | SQLite + local disk + ntfy.sh | PostgreSQL/PostGIS + S3/MinIO + dual-path hardware dispatch + RFC 3161 audit |

---

## Phase 1 — Unfreezing ADR-010 & Retiring Mock Artifacts

**Goal:** Clear the technical debt from hackathon-era artificial constraints so the codebase can accept production changes without governance conflicts.

### 1.1 Unfreeze the tensor ingestion contract

Deprecate the strict 2-channel constraint in `backend/siren/ml/contract.py`. ADR-011 (Accepted) authorizes expanding the input tensor from $(B, 2, H, W)$ to $(B, 4, H, W)$:

$$\mathbf{X} \in \mathbb{R}^{B \times 4 \times H \times W} \quad \big[\sigma^0_{VV}, \; \sigma^0_{VH}, \; \text{DEM Elevation}, \; \text{Slope Angle}\big]$$

This invalidates all existing 2-channel weights — a new checkpoint must be trained from scratch against the 4-channel contract (V3 §2.1).

### 1.2 Purge hardcoded demo state

Remove static demo artifacts that have no place in a production system:

- `data/assets/weather_series.json` — replaced by live ERA5/IMERG API calls
- Static observation IDs (`obs-001`, `obs-002`, `obs-003`) — replaced by dynamic records indexed by acquisition timestamp and satellite orbit metadata
- Hardcoded `DEMO_OBSERVATIONS` in `pipeline.py` — replaced by database-registered observations from the ingestion daemon

The pipeline must treat every observation as a dynamic record: `{acquisition_time, orbit, sensor, cloud_fraction, raster_uri, quality_verdict}`.

### 1.3 Adopt scientific dependencies

Hard Rule 8's whitelist was a hackathon constraint to prevent last-minute breaks. ADR-011 amends it for production:

| Package | Purpose | Phase |
|---|---|---|
| `psycopg[binary]` | PostgreSQL + PostGIS driver | Sprint 1 |
| `geoalchemy2` | PostGIS spatial ORM for SQLAlchemy | Sprint 1 |
| `xgboost` | Gradient-boosted trees for calibrated P_breach | Sprint 3 |
| `shap` | TreeSHAP feature attribution for explainability | Sprint 3 |
| `celery` + `redis` | Asynchronous job queue for ingestion daemon | Sprint 1 |
| `rtree` / `pygeos` | Spatial indexing for high-throughput exposure joins | Sprint 1 |

All added under a `[production]` extra in `pyproject.toml`, mirroring the existing `[ml]` extra pattern.

### 1.4 Acceptance criteria

- [ ] ADR-011 accepted (done — see `docs/adr/ADR-011-production-multimodal-upgrade.md`)
- [ ] `pyproject.toml` updated with `[production]` extra
- [ ] `ml/contract.py` expanded to 4-channel (SAR_CHANNELS = 4)
- [ ] `DEMO_OBSERVATIONS` refactored to dynamic DB-backed observation registry
- [ ] `weather_series.json` removed; weather fetched live at runtime

---

## Phase 2 — Autonomous Ingestion Engine (Daemon Layer)

**Goal:** The system detects new satellite scenes and processes them without human intervention.

### 2.1 STAC poller & ingestion daemon

A background worker polls the Copernicus Data Space STAC API on a cron schedule for new Sentinel-1 (IW_GRDH_1S) passes intersecting the target AOI bounding box.

```
[Copernicus STAC / CDSE API] ──► [Celery / Temporal Polling Worker]
                                           │ (New Scene Detected)
                                           ▼
                                 [Cloud-Optimized GeoTIFF Pipeline]
                                           │
                                 [Radiometric Calibration (RTC γ⁰)]
                                           │
                                 [Quality Gate & Cloud Routing]
                                           │
                                 [Pipeline Execution (detect→geo→risk→DB→audit)]
```

**Polling cadence:** every 6 hours for Sentinel-1 (12-day repeat per orbit, 6 days combined constellation). Every 2 hours for Sentinel-2 during monsoon.

**Idempotency:** the `acquisition_jobs` table (ADR-008 schema, already implemented) with `UNIQUE(source, provider_product_id)` prevents duplicate processing.

### 2.2 Cloud-Optimized GeoTIFFs (COGs) & /vsicurl/

Do not download entire 1.7 GB SAFE ZIP archives to disk. Configure rasterio and GDAL with `/vsicurl/` to read only the spatial bounding box intersecting the target glacial basin directly from cloud storage.

**Benefits:**
- Eliminates 1.7 GB downloads for a basin that covers ~5% of the scene footprint
- Reduces storage costs (S3 lifecycle: raw SAFE → 30-day TTL, processed COG → permanent)
- Enables parallel tile processing for sub-minute pipeline latency

### 2.3 Radiometric Terrain Correction (RTC γ⁰)

Replace raw σ⁰ digital number conversion with true γ⁰ (gamma-nought) RTC using the Copernicus 30 m DEM:

$$\gamma^0 = \frac{\sigma^0}{\cos\theta_{\text{local}}}$$

where θ_local is the local incidence angle derived from the DEM and the satellite look vector. This flattens radar brightness variations caused by local slope aspect — the biggest source of false-positive water classifications in steep gorges.

**Implementation:** add `preprocess/rtc.py` with `gamma_nought(sigma0_db, dem, look_vector)`. The corrected γ⁰ replaces σ⁰ as channels 0/1 in the 4-channel tensor.

### 2.4 Acceptance criteria

- [ ] Celery worker polls CDSE STAC every 6 hours; detects new S1 scenes in AOI
- [ ] `/vsicurl/` reads only the AOI bounding box from cloud COGs
- [ ] RTC γ⁰ correction implemented and verified against a known flat-water scene
- [ ] New scene → full pipeline execution → review card, unattended
- [ ] Duplicate scene detection → idempotent skip (no duplicate runs)

---

## Phase 3 — Load-Bearing AI & Hydrology Upgrade

**Goal:** Execute the V3 research proposal. Move ML from shadow to load-bearing, and replace planar buffers with physics-based hydrology.

### 3.1 4-channel terrain-aware segmentation (WaterResUNet)

Per V3 §2.1–2.7:

- Concatenate normalized VV dB, VH dB, normalized Copernicus DEM elevation, and local slope (degrees) into the model input.
- Train with the L_gravity penalty (V3 §2.6) so pixels with high upward hydraulic head gradients are penalized during backpropagation.
- Apply DANN adversarial domain adaptation (V3 §2.7) to align flat-benchmark ↔ Himalayan feature distributions.
- **Target:** drive OOD event-holdout IoU from 0.239 → > 0.65.

### 3.2 Height Above Nearest Drainage (HAND) for exposure

Discard the planar 125 m Euclidean river buffer. Compute a true HAND raster over the Dudh Koshi basin using pysheds and the 30 m DEM:

$$\text{HAND}(x, y) \leq h_{\text{water\_stage}}$$

An asset is exposed only if its relative vertical height above the drainage channel is below the flood surge depth. This immediately stops hilltop monasteries from being flagged as inundated simply because they sit horizontally near a canyon bend.

**Implementation:** add `geo/hand.py` with `compute_hand(dem, flow_dir, drainage_raster)`. Replace `corridor.py`'s `FLOODPLAIN_BUFFER_M = 125.0` with a dynamic `h_water_stage` from the FNO surrogate (Phase 3.3) or a policy default (0.5 m for watch, 2.0 m for elevated, 5.0 m for critical).

### 3.3 Calibrated susceptibility scorer (XGBoost + TreeSHAP)

Per V3 §3.1–3.4:

Ingest the ICIMOD High Mountain Asia Glacial Lake Database. Train an XGBoost tabular model using:

- Lake area expansion rate (Δkm²/day)
- Moraine dam width-to-height ratio
- Glacier tongue contact length
- 7-day cumulative rainfall anomaly from ERA5 / GPM IMERG

Wrap predictions in conformal prediction intervals (95% coverage). If epistemic uncertainty is too high (interval width > 0.35), the system refuses autonomous dispatch and mandates human field verification.

TreeSHAP feature attributions feed the `reasons` array (Hard Rule 5 preserved).

### 3.4 FNO hydrodynamic surrogate

Per V3 §4: a 2D Fourier Neural Operator trained on synthetic HEC-RAS shallow-water runs. Triggered only when P_breach ≥ 0.70. Outputs dynamic water depth grid `h_water` and wave arrival time `T_arrival` at named points.

### 3.5 Acceptance criteria

- [ ] 4-channel WaterResUNet trained; event-holdout IoU > 0.65
- [ ] HAND raster computed for Dudh Koshi; exposure uses HAND ≤ h_water_stage
- [ ] XGBoost P_breach calibrated; Brier score reported; conformal intervals implemented
- [ ] TreeSHAP reasons wired into review card
- [ ] FNO-2D trained on synthetic HEC-RAS; South Lhonak retrospective validation within tolerance
- [ ] h_water grid intersect replaces static tolerance buffers when FNO output available

---

## Phase 4 — Production Infrastructure & Enterprise Reliability

**Goal:** The system runs unattended, survives component failures, and delivers alerts through real hardware.

### 4.1 Database: PostgreSQL + PostGIS

| Hackathon | Production |
|---|---|
| Local SQLite with file locks | PostgreSQL + PostGIS for high-throughput spatial indexing |
| In-memory geopandas joins | `ST_DWithin`, `ST_Intersects` for indexed spatial queries |
| Count-based IDs | UUIDs with database-generated sequences |

Migrate `db/schema.sql` to SQLAlchemy + Alembic migrations. Spatial columns use `GEOMETRY(Point, 4326)` / `GEOMETRY(Polygon, 4326)`. Exposure intersections use `ST_DWithin(geom, corridor, tolerance)` with GiST indexing.

### 4.2 Storage: S3-compatible object storage

| Hackathon | Production |
|---|---|
| `data/raw/` and `data/processed/` local files | S3-compatible (MinIO / AWS S3) with tiered lifecycle |

- Raw SAFE scenes: 30-day TTL (enough for reprocessing, not permanent)
- Processed COGs and masks: permanent with versioning
- Database backups: 30-day retention, point-in-time recovery

### 4.3 Orchestration: asynchronous job queue

| Hackathon | Production |
|---|---|
| Synchronous FastAPI request handlers | Celery / ARQ with retries, exponential backoff, dead-letter queues |

Failed scenes retry 3× with exponential backoff (1 min, 5 min, 30 min). After 3 failures, the scene is marked `failed` and an alert is sent to the operations channel (separate from hazard alerts).

### 4.4 Dispatch: dual-path hardware engine

| Hackathon | Production |
|---|---|
| ntfy.sh browser-side push | Dual-path: AWS SNS / Twilio for terrestrial SMS + serial bridge to RockBLOCK 9603 Iridium SBD modem or SX1262 LoRa gateway |

The ≤250-byte payload codec is preserved. The dispatch engine sends via both paths simultaneously; delivery receipts from either path mark the alert as delivered.

### 4.5 Audit: cryptographic timestamp authority

| Hackathon | Production |
|---|---|
| SQLite triggers with SHA-256 string concat | Cryptographic audit service with RFC 3161 timestamp authority |

Anchor hash chain roots to an external public witness (Rekor / OpenTimestamps) so the audit log is provably tamper-evident even if the database is compromised.

### 4.6 Acceptance criteria

- [ ] PostgreSQL + PostGIS running; all spatial queries use GiST indexes
- [ ] S3/MinIO storage configured; lifecycle policies active
- [ ] Celery worker processes scenes asynchronously with retry + dead-letter
- [ ] Hardware dispatch tested: SMS via Twilio + Iridium SBD via serial
- [ ] RFC 3161 timestamp anchoring verified on audit chain

---

## Sprint Execution Sequence

Do not attempt to build all four phases at once. Start with the data foundation.

### Sprint 1 — PostGIS & Ingestion (Weeks 1–3)

**Scope:** Phase 1 (unfreeze) + Phase 2 (ingestion daemon) + Phase 4.1 (PostgreSQL)

| Task | Deliverable |
|---|---|
| Accept ADR-011 | Governance cleared for 4-channel + new dependencies |
| Add `[production]` extra to `pyproject.toml` | psycopg, geoalchemy2, celery, redis, rtree |
| Migrate SQLite schema to PostgreSQL + PostGIS | Alembic migrations; GiST indexes on spatial columns |
| Write STAC polling worker (Celery) | Detects new S1 scenes in AOI every 6 hours |
| Implement `/vsicurl/` COG reading | Read only AOI bbox from cloud; no full SAFE download |
| Implement RTC γ⁰ correction | `preprocess/rtc.py`; verified on flat-water scene |
| Refactor `DEMO_OBSERVATIONS` to dynamic DB registry | Observations indexed by acquisition time + orbit |
| Remove `weather_series.json` | Weather fetched live via Open-Meteo / ERA5 API |

**Exit gate:** a new Sentinel-1 scene appears on CDSE → the system detects it, ingests it, processes it, and produces a review card — all unattended.

### Sprint 2 — 4-Channel Vision & Terrain Priors (Weeks 4–7)

**Scope:** Phase 3.1 (WaterResUNet) + Phase 3.3 (XGBoost susceptibility)

| Task | Deliverable |
|---|---|
| Expand `ml/contract.py` to 4 channels | SAR_CHANNELS = 4; DEM + Slope normalization |
| Generate slope rasters from Copernicus GLO-30 | `preprocess/dem.py::slope_degrees()` |
| Co-register DEM + slope to Sen1Floods11 chip grid | Training data pipeline ready |
| Train WaterResUNet with L_gravity + DANN | Event-holdout IoU reported |
| IoU > 0.65 gate | Load-bearing path authorized (V3 §2.4) |
| Ingest ICIMOD + HMA glacial lake inventories | `ingest/icimod.py`, `ingest/hma.py` |
| Train XGBoost P_breach | Calibrated; Brier score reported |
| Implement conformal prediction intervals | 95% coverage; auto-fallback at width > 0.35 |
| Wire TreeSHAP reasons into review card | Hard Rule 5 preserved |

**Exit gate:** the 4-channel model beats 0.65 IoU on event-holdout and the XGBoost susceptibility scorer produces calibrated P_breach with conformal intervals.

### Sprint 3 — HAND Hydrology & Tabular Risk (Weeks 8–10)

**Scope:** Phase 3.2 (HAND) + Phase 3.4 (FNO surrogate) + Phase 4.4–4.5 (hardware dispatch + audit)

| Task | Deliverable |
|---|---|
| Compute HAND raster for Dudh Koshi | `geo/hand.py`; verified against known hilltop assets |
| Replace planar buffer with HAND ≤ h_water_stage | Exposure uses vertical clearance, not Euclidean distance |
| Generate synthetic HEC-RAS 2D runs | 500–1000 runs across varying V_breach + hydrographs |
| Train FNO-2D surrogate | Arrival-time MAE reported on held-out runs |
| South Lhonak Oct 2023 retrospective validation | T_arrival within tolerance of documented flood travel time |
| Wire h_water grid intersect + T_arrival into corridor | Supersedes static buffers when FNO output available |
| Dual-path hardware dispatch | Twilio SMS + Iridium SBD serial bridge tested |
| RFC 3161 timestamp anchoring | Audit chain roots anchored to Rekor / OpenTimestamps |

**Exit gate:** HAND exposure eliminates false-positive hilltop assets; FNO T_arrival validated on South Lhonak; hardware dispatch delivers a real alert.

---

## Relationship to Existing Documents

| Document | Status | Role |
|---|---|---|
| `BUILD_ROADMAP.md` | **Preserved as history** | Hackathon phases 0–6 + Live Service Transition (phases 0–6) — frozen record of v1.0.0-hackathon-final |
| `V3_RESEARCH_PROPOSAL.md` | **Active research RFC** | Technical patterns (L_gravity, DANN, conformal, FNO) — referenced by this roadmap |
| `ADR-010` | **Superseded (partial)** | 2-channel frozen contract clause superseded by ADR-011; ML evidence isolation principles retained |
| `ADR-011` | **Accepted** | Authorizes 4-channel tensor, production dependencies, and the transition described here |
| `KNOWN_LIMITATIONS.md` | **Active** | Domain physics limitations (SAR, hydrology, latency, scoring) — addressed by this roadmap |

---

## Guardrails Preserved from Hackathon

These principles survive the transition — they are not hackathon constraints, they are engineering discipline:

1. **Human gate.** No dispatch without a recorded `confirm` review. HTTP 409 safeguard remains active. (Hard Rule 3)
2. **Explainability.** Every score carries a `reasons` array (≥3 on elevated+). TreeSHAP attributions feed this. (Hard Rule 5)
3. **Reproducibility.** Same inputs + processing version → identical outputs. No unseeded randomness. (Hard Rule 6)
4. **ML evaluation gate.** No ML enters a load-bearing path until it beats the rules-only baseline on held-out event data. IoU > 0.65 or P_breach Brier score < 0.15. (ADR-010 §3, preserved by ADR-011)
5. **Payload ≤ 250 bytes.** The codec constraint is preserved even as the payload content evolves (T_arrival, peak height added within the byte budget). (Hard Rule 4)
6. **Audit integrity.** Append-only, hash-chained, now anchored to external timestamp authority. (ADR-009 + RFC 3161)
