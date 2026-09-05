// SIREN API types — match backend/siren/api/models.py and docs/API_CONTRACT.md exactly.

export type GeoJSONFeature = Record<string, any>;

export interface QualityVerdict {
  quality_score: number;
  cloud_fraction: number;
  alignment_ok: boolean;
  usable: boolean;
  confidence_adjustment: number;
}

export interface Observation {
  observation_id: string;
  basin_id: string;
  acquired_at: string;
  source: string;
  raster_uri: string;
  crs: string;
  quality_score: number | null;
  cloud_fraction: number | null;
  alignment_ok: boolean | null;
  usable: boolean | null;
  confidence_adjustment: number | null;
  water_area_km2: number | null;
  water_area_change_percent: number | null;
  rainfall_24h_mm: number | null;
  rainfall_7d_mm: number | null;
  mean_slope_degrees: number | null;
  processing_version: string;
  status: string;
}

export interface Score {
  hazard_score: number;
  exposure_priority: number;
  disease_risk: number | null;
  confidence: number;
  severity: "informational" | "watch" | "elevated" | "critical";
  reasons: string[];
}

export interface Alert {
  alert_id: string;
  geofence_id: string;
  severity: string;
  hazard_type: string;
  confidence: number;
  exposed_population: number;
  critical_assets: string[];
  disease_flags: string[];
  recommended_action: string;
  human_review_required: boolean;
}

export interface BasinConfig {
  basin_id: string;
  name: string;
  boundary_geojson: GeoJSONFeature;
  crs: string;
  basemap_uri: string | null;
  basemap_bounds: number[][] | null;
}

export interface Run {
  run_id: string;
  observation_id: string;
  processing_version: string;
  change_mask_uri: string | null;
  corridor_geojson: GeoJSONFeature | null;
  change_stats_json: GeoJSONFeature | null;
  score: Score | null;
}

export interface Exposure {
  asset_id: string;
  asset_type: string;
  name: string | null;
  distance_m: number | null;
  buffer_m: number | null;
  inundated: boolean;
  population: number | null;
}

export interface ReviewResponse {
  review_id: string;
  score_id: string;
  reviewer: string;
  decision: "confirm" | "reject" | "postpone";
  decided_at: string;
}

export interface DispatchResponse {
  dispatch_id: string;
  alert_id: string;
  geofence_id: string;
  payload: string;
  payload_bytes: number;
  channel: string;
  status: string;
  sent_at: string;
}

export interface AuditEntry {
  entry_id: number;
  alert_id: string | null;
  actor: string;
  action: string;
  detail_json: string;
  created_at: string;
}

// Response wrappers
export interface ObservationList { observations: Observation[] }
export interface RunList { runs: Run[] }
export interface ExposureList { exposures: Exposure[] }
export interface AuditList { entries: AuditEntry[] }

export interface SarSector {
  sector_id: string;
  name: string;
  asset_type: string;
  population: number;
  access_loss: number;
  access_label: "CUT" | "AT_RISK" | "ACCESSIBLE";
  sar_priority: number;
  reason: string;
  assets: string[];
}

export interface SarPriorityList {
  sectors: SarSector[];
  top_priority: SarSector | null;
  summary: string;
}

export interface MlEvidence {
  run_id: string;
  observation_id: string;
  ml_source: string;
  ml_confidence_mean: number;
  ml_consensus_pixels: number;
  heatmap_uri: string;
  heatmap_bounds: number[][] | null;
  mask_uri: string;
  mask_bounds: number[][] | null;
  baseline_mask_uri: string;
  baseline_mask_bounds: number[][] | null;
  preview_baseline_uri: string;
  preview_after_uri: string;
  model_available: boolean;
  change_polygon: GeoJSONFeature | null;
}

export interface ApiError {
  error: string;
  detail: string;
}
