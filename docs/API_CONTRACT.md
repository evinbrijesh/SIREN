# SIREN — API Contract

Authoritative HTTP surface. Implemented in `backend/siren/api/app.py`. Field names/types match `docs/PRD.md` §10 and `backend/siren/db/schema.sql` exactly.

Base URL: `http://localhost:8010` · Frontend proxy: `/api` → `http://localhost:8010`

> **Note:** The default dev port is 8010 (port 8000 was occupied on the build machine). Update `frontend/vite.config.ts` if you need a different port.

---

## Conventions

- All responses are JSON.
- Errors: `{"error": "<type>", "detail": "<message>"}` with appropriate HTTP status.
- Timestamps: ISO-8601 UTC strings.
- Geometry: GeoJSON (Polygon / Point / LineString / MultiLineString).
- IDs: `observation_id` (`obs-*`), `run_id` (`run-*`), `alert_id` (`alert-*`), `asset_id` (`BR-*`, `RD-*`, `village-*`, `well-*`).

---

## Endpoints

### `GET /basin`
Returns the active basin configuration.

```json
{
  "basin_id": "dudh-koshi-demo-01",
  "name": "Dudh Koshi / Imja",
  "boundary_geojson": { "type": "Polygon", "coordinates": [] },
  "crs": "EPSG:4326"
}
```

### `GET /observations`
List all observations, newest first.

```json
{
  "observations": [
    {
      "observation_id": "obs-003",
      "basin_id": "dudh-koshi-demo-01",
      "acquired_at": "2026-08-12T12:00:00Z",
      "source": "sentinel-1-grd-nrt",
      "raster_uri": "data/processed/obs-003.tif",
      "crs": "EPSG:4326",
      "quality_score": 0.88,
      "cloud_fraction": 0.0,
      "optical_cloud_fraction": 0.90,
      "alignment_ok": true,
      "usable": true,
      "confidence_adjustment": 0.95,
      "water_area_km2": 4.3,
      "water_area_change_percent": 43.0,
      "rainfall_24h_mm": 60.0,
      "rainfall_7d_mm": 160.0,
      "mean_slope_degrees": 31.0,
      "processing_version": "0.1.0",
      "status": "processed"
    }
  ]
}
```

### `GET /observations/{observation_id}`
Single observation (same shape as above).

### `POST /runs`
Trigger a pipeline run for an observation. The pipeline runs **synchronously** — quality gate → weather-adaptive router → change detection → corridor + exposures → risk fusion → DB write → audit log. Body:

```json
{ "observation_id": "obs-003" }
```

Response (202 Accepted):

```json
{
  "run_id": "run-0004",
  "observation_id": "obs-003",
  "status": "processed",
  "started_at": "2026-08-13T04:10:33Z"
}
```

> **Note:** `status` is `"processed"` (not `"queued"`) because the pipeline completes synchronously within the request. The full run with scores and exposures is available via `GET /runs/{run_id}` immediately after.

### `GET /runs/{run_id}`
Single run with score, exposures, and decision state.

```json
{
  "run_id": "run-0004",
  "observation_id": "obs-003",
  "processing_version": "0.1.0",
  "change_mask_uri": "data/processed/obs-003-mask.tif",
  "corridor_geojson": { "type": "LineString", "coordinates": [] },
  "change_stats_json": { "water_area_km2": 4.3, "expansion_percent": 43.0, "routing": { "sar_primary": true, "optical_cloud_fraction": 0.90 } },
  "score": {
    "hazard_score": 0.82,
    "exposure_priority": 0.68,
    "disease_risk": 0.45,
    "confidence": 0.88,
    "severity": "critical",
    "reasons": ["reason 1", "reason 2", "reason 3"]
  },
  "decision": null
}
```

> `decision` is `null` until a review is recorded, then `"confirm"`, `"reject"`, or `"postpone"`.

### `POST /runs/process-all`
Run the pipeline for all demo observations in sequence. No request body required.

```json
{
  "runs": [ { "run_id": "run-0002", ... }, { "run_id": "run-0003", ... }, { "run_id": "run-0004", ... } ],
  "count": 3
}
```

### `GET /runs`
List runs with their scores.

```json
{
  "runs": [
    {
      "run_id": "run-0007",
      "observation_id": "obs-003",
      "processing_version": "0.1.0",
      "change_mask_uri": "data/processed/obs-003-mask.tif",
      "corridor_geojson": { "type": "LineString", "coordinates": [] },
      "change_stats_json": { "water_area_km2": 4.3, "expansion_percent": 43.0, "routing": { "sar_primary": true, "optical_cloud_fraction": 0.90 } },
      "score": {
        "hazard_score": 0.82,
        "exposure_priority": 0.68,
        "disease_risk": 0.45,
        "confidence": 0.88,
        "severity": "critical",
        "reasons": ["reason 1", "reason 2", "reason 3"]
      }
    }
  ]
}
```

### `GET /runs/{run_id}/exposures`
Affected assets for a run.

```json
{
  "exposures": [
    {
      "asset_id": "village-2",
      "asset_type": "village",
      "name": "Chhukung",
      "distance_m": 210.0,
      "buffer_m": 100.0,
      "inundated": false,
      "population": 1240
    }
  ]
}
```

### `GET /runs/{run_id}/sar-priority`
Search & Rescue priority ranking for exposed assets (PRD §15). Returns assets sorted by priority score.

```json
{
  "run_id": "run-0004",
  "priorities": [
    {
      "asset_id": "village-2",
      "asset_type": "village",
      "name": "Chhukung",
      "priority_score": 0.92,
      "population": 1240,
      "inundated": true,
      "access_risk": "severed",
      "recommended_action": "immediate_evacuation"
    }
  ]
}
```

### `GET /runs/{run_id}/ml-evidence`
ML change-detection evidence layer (ADR-002 addendum). Returns heatmap, mask, and preview URIs. Falls back to deterministic masks when torch is unavailable.

```json
{
  "run_id": "run-0004",
  "source": "deterministic-fallback",
  "heatmap_uri": "/data/processed/obs-003-heatmap.png",
  "mask_uri": "/data/processed/obs-003-ml-mask.tif",
  "baseline_mask_uri": "/data/processed/baseline-water-mask.tif",
  "preview_baseline_uri": "/data/map-assets/obs-003/baseline-optical.png",
  "preview_after_uri": "/data/processed/obs-003-preview.png",
  "bounds": [86.65, 27.65, 86.95, 27.95]
}
```

> `source` is `"deterministic-fallback"` when torch is not installed, or `"siamese-unet"` / `"changeformer"` when the ML extra is available. The deterministic path produces identical outputs every run.

### `POST /runs/{run_id}/review`
Human-in-the-loop decision. Body:

```json
{
  "reviewer": "coordinator-01",
  "decision": "confirm",
  "note": "Verified against field report"
}
```

`decision` ∈ `confirm | reject | postpone`. Response:

```json
{
  "review_id": "rev-0003",
  "score_id": "score-0007",
  "reviewer": "coordinator-01",
  "decision": "confirm",
  "decided_at": "2026-09-04T12:10:00Z"
}
```

### `POST /runs/{run_id}/dispatch`
Simulate dispatch of a confirmed alert. Only valid after a `confirm` review. Body:

```json
{
  "channel": "sms",
  "recipient_group": "sector-b"
}
```

Response:

```json
{
  "dispatch_id": "disp-0001",
  "alert_id": "alert-0091",
  "geofence_id": "sector-b",
  "payload": "{\"aid\":\"siren-04\",\"sec\":\"B\",\"haz\":\"GLOF_FL\",\"lvl\":3,\"exp_pop\":1240,\"crit\":[\"BR-12\",\"RD-4\"],\"med_act\":\"BOIL_WATER_NOW\"}",
  "payload_bytes": 118,
  "channel": "sms",
  "status": "sent",
  "sent_at": "2026-09-04T12:11:00Z"
}
```

### `GET /audit?alert_id={alert_id}&run_id={run_id}`
Full lineage for an alert or run (append-only). Either query parameter filters the result; both can be combined.

```json
{
  "entries": [
    { "entry_id": 1, "alert_id": "alert-0091", "run_id": "run-0004", "actor": "pipeline", "action": "run", "detail_json": "{}", "prev_hash": "0000000000000000000000000000000000000000000000000000000000000000", "event_hash": "8475547e1039023e...", "created_at": "..." },
    { "entry_id": 2, "alert_id": "alert-0091", "run_id": "run-0004", "actor": "coordinator-01", "action": "review", "detail_json": "{}", "prev_hash": "8475547e1039023e...", "event_hash": "a1b2c3d4e5f6...", "created_at": "..." },
    { "entry_id": 3, "alert_id": "alert-0091", "run_id": "run-0004", "actor": "coordinator-01", "action": "dispatch", "detail_json": "{}", "prev_hash": "a1b2c3d4e5f6...", "event_hash": "d3258283347643e2...", "created_at": "..." }
  ]
}
```

> Each entry's `event_hash` is `sha256(prev_hash + timestamp + payload)`. The genesis entry uses `prev_hash = "0" * 64`. The chain is tamper-evident: altering any entry invalidates all subsequent hashes.

### `GET /data/processed/{filename}`
Static file access for processed rasters and PNG sidecars (masks, overlays). Mounted via FastAPI `StaticFiles`.

```
GET /data/processed/obs-001_expansion_mask.tif  → GeoTIFF
GET /data/processed/obs-001_expansion_mask.png  → PNG sidecar
GET /data/processed/baseline_water_mask.tif     → baseline mask
```

### `GET /data/map-assets/{filename}`
Pre-rendered map tile overlays for the MapLibre frontend. Implemented in `api/map_assets.py`.

```
GET /data/map-assets/dem-hillshade.png              → DEM hillshade raster
GET /data/map-assets/sar-backscatter.png             → SAR backscatter raster
GET /data/map-assets/{obs_id}/baseline-optical.png   → Per-observation baseline optical crop
```

---

## Pydantic Models (implemented in `backend/siren/api/models.py`)

```python
class QualityVerdict(BaseModel):
    quality_score: float
    cloud_fraction: float
    alignment_ok: bool
    usable: bool
    confidence_adjustment: float

class Observation(BaseModel):
    observation_id: str
    basin_id: str
    acquired_at: datetime
    source: str
    raster_uri: str
    crs: str
    quality_score: float | None
    cloud_fraction: float | None
    optical_cloud_fraction: float | None  # original optical cloud before SAR routing
    alignment_ok: bool | None
    usable: bool | None
    confidence_adjustment: float | None
    water_area_km2: float | None
    water_area_change_percent: float | None
    rainfall_24h_mm: float | None
    rainfall_7d_mm: float | None
    mean_slope_degrees: float | None
    processing_version: str
    status: str

class Score(BaseModel):
    hazard_score: float
    exposure_priority: float
    disease_risk: float | None
    confidence: float
    severity: str  # informational | watch | elevated | critical
    reasons: list[str]  # >=3 on elevated+

class Run(BaseModel):
    run_id: str
    observation_id: str
    processing_version: str
    change_mask_uri: str | None
    corridor_geojson: dict | None
    change_stats_json: dict | None
    score: Score | None
    decision: str | None  # null | "confirm" | "reject" | "postpone"

class Alert(BaseModel):
    alert_id: str
    geofence_id: str
    severity: str
    hazard_type: str
    confidence: float
    exposed_population: int
    critical_assets: list[str]
    disease_flags: list[str]
    recommended_action: str
    human_review_required: bool

class SarPriorityItem(BaseModel):
    asset_id: str
    asset_type: str
    name: str
    priority_score: float
    population: int
    inundated: bool
    access_risk: str
    recommended_action: str

class SarPriorityList(BaseModel):
    run_id: str
    priorities: list[SarPriorityItem]

class MlEvidence(BaseModel):
    run_id: str
    source: str  # "deterministic-fallback" | "siamese-unet" | "changeformer"
    heatmap_uri: str
    mask_uri: str
    baseline_mask_uri: str
    preview_baseline_uri: str
    preview_after_uri: str
    bounds: list[float]  # [min_lon, min_lat, max_lon, max_lat]

class AuditEntry(BaseModel):
    entry_id: int
    alert_id: str | None
    run_id: str | None
    actor: str
    action: str
    detail_json: str
    prev_hash: str  # sha256 of previous entry (genesis = "0"*64)
    event_hash: str  # sha256(prev_hash + timestamp + payload)
    created_at: str

class AuditList(BaseModel):
    entries: list[AuditEntry]
```

---

## Invariants

- `POST /runs/{run_id}/dispatch` returns 409 if no `confirm` review exists for the run (human gate, PRD §7.6).
- `payload_bytes` ≤ 250 always (enforced by unit test, PRD §10.4).
- `reasons` has ≥ 3 entries when `severity` is `elevated` or `critical` (PRD §9.5).
- Audit entries are append-only; no update/delete endpoints exist.
- Audit `event_hash` = `sha256(prev_hash + timestamp + payload)`; genesis `prev_hash` = `"0" * 64`.
- `optical_cloud_fraction` ≥ 0.20 routes to SAR-primary; `cloud_fraction` is set to 0.0 on the SAR path.
- Severity thresholds: expansion ≥40% → critical, ≥20% → elevated, ≥5% → watch, <5% → informational.
- ML evidence endpoint always returns a result — deterministic fallback when torch is unavailable.