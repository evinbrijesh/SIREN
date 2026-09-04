# SIREN — API Contract

Authoritative HTTP surface. Devin D4 implements this; the frontend typed client consumes it. Field names/types must match `docs/PRD.md` §10 and `backend/siren/db/schema.sql` exactly.

Base URL: `http://localhost:8000` · Frontend proxy: `/api` → `http://localhost:8000`

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
      "acquired_at": "2026-09-04T12:00:00Z",
      "source": "sentinel-1-grd-nrt",
      "raster_uri": "data/processed/obs-003.tif",
      "crs": "EPSG:4326",
      "quality_score": 0.88,
      "cloud_fraction": 0.11,
      "alignment_ok": true,
      "usable": true,
      "confidence_adjustment": 0.95,
      "water_area_km2": 3.2,
      "water_area_change_percent": 14.3,
      "rainfall_24h_mm": 72.4,
      "rainfall_7d_mm": 188.0,
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
Trigger a pipeline run for an observation. Body:

```json
{ "observation_id": "obs-003" }
```

Response (202 Accepted):

```json
{
  "run_id": "run-0007",
  "observation_id": "obs-003",
  "status": "queued",
  "started_at": "2026-09-04T12:05:00Z"
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
      "change_stats_json": { "water_area_km2": 3.2, "expansion_percent": 14.3 },
      "score": {
        "hazard_score": 0.62,
        "exposure_priority": 0.48,
        "disease_risk": 0.31,
        "confidence": 0.76,
        "severity": "elevated",
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

### `GET /audit?alert_id={alert_id}`
Full lineage for an alert (append-only).

```json
{
  "entries": [
    { "entry_id": 1, "alert_id": "alert-0091", "actor": "pipeline", "action": "run", "detail_json": "{}", "created_at": "..." },
    { "entry_id": 2, "alert_id": "alert-0091", "actor": "coordinator-01", "action": "review", "detail_json": "{}", "created_at": "..." },
    { "entry_id": 3, "alert_id": "alert-0091", "actor": "coordinator-01", "action": "dispatch", "detail_json": "{}", "created_at": "..." }
  ]
}
```

---

## Pydantic Models (D4 must implement)

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
```

---

## Invariants

- `POST /runs/{run_id}/dispatch` returns 409 if no `confirm` review exists for the run (human gate, PRD §7.6).
- `payload_bytes` ≤ 250 always (enforced by unit test, PRD §10.4).
- `reasons` has ≥ 3 entries when `severity` is `elevated` or `critical` (PRD §9.5).
- Audit entries are append-only; no update/delete endpoints exist.