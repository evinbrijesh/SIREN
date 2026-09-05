"""Pydantic models for the SIREN API surface.

Field names and types match docs/spec/API_CONTRACT.md and docs/spec/PRD.md §10 exactly.
Never rename a field to "make it nicer" (CLAUDE.md / Hard Rules).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


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
    quality_score: float | None = None
    cloud_fraction: float | None = None
    optical_cloud_fraction: float | None = None
    alignment_ok: bool | None = None
    usable: bool | None = None
    confidence_adjustment: float | None = None
    water_area_km2: float | None = None
    water_area_change_percent: float | None = None
    rainfall_24h_mm: float | None = None
    rainfall_7d_mm: float | None = None
    mean_slope_degrees: float | None = None
    processing_version: str
    status: str


class Score(BaseModel):
    hazard_score: float
    exposure_priority: float
    disease_risk: float | None = None
    confidence: float
    severity: str  # informational | watch | elevated | critical
    reasons: list[str]  # >= 3 entries on elevated+ (PRD §9.5)


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


# --- Response wrappers / endpoint bodies (shapes from API_CONTRACT.md) ---


class BasinConfig(BaseModel):
    basin_id: str
    name: str
    boundary_geojson: dict[str, Any]
    crs: str
    basemap_uri: str | None = None
    basemap_bounds: list[list[float]] | None = None


class ObservationList(BaseModel):
    observations: list[Observation]


class RunCreateRequest(BaseModel):
    observation_id: str


class RunCreateResponse(BaseModel):
    run_id: str
    observation_id: str
    status: str
    started_at: datetime


class Run(BaseModel):
    run_id: str
    observation_id: str
    processing_version: str
    change_mask_uri: str | None = None
    corridor_geojson: dict[str, Any] | None = None
    change_stats_json: dict[str, Any] | None = None
    score: Score | None = None
    status: str
    started_at: str
    finished_at: str | None = None
    decision: str | None = None
    reviewer: str | None = None
    decided_at: str | None = None


class RunList(BaseModel):
    runs: list[Run]


class Exposure(BaseModel):
    asset_id: str
    asset_type: str
    name: str | None = None
    distance_m: float | None = None
    buffer_m: float | None = None
    inundated: bool
    population: int | None = None
    geometry_geojson: dict[str, Any] | None = None


class ExposureList(BaseModel):
    exposures: list[Exposure]


class ReviewRequest(BaseModel):
    reviewer: str
    decision: str  # confirm | reject | postpone | escalate
    note: str | None = None


class ReviewResponse(BaseModel):
    review_id: str
    score_id: str
    reviewer: str
    decision: str
    decided_at: datetime


class DispatchRequest(BaseModel):
    channel: str
    recipient_group: str


class DispatchResponse(BaseModel):
    dispatch_id: str
    alert_id: str
    geofence_id: str
    payload: str
    payload_bytes: int
    channel: str
    status: str
    sent_at: datetime


class AuditEntry(BaseModel):
    entry_id: int
    alert_id: str | None = None
    actor: str
    action: str
    detail_json: str
    created_at: str
    prev_hash: str
    event_hash: str


class AuditList(BaseModel):
    entries: list[AuditEntry]


class SarSector(BaseModel):
    sector_id: str
    name: str
    asset_type: str
    population: int
    access_loss: float
    access_label: str  # CUT | AT_RISK | ACCESSIBLE
    sar_priority: float
    reason: str
    assets: list[str]


class SarPriorityList(BaseModel):
    sectors: list[SarSector]
    top_priority: SarSector | None = None
    summary: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
