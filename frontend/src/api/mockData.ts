// Mock data for offline demo — matches the D4 seed data and API_CONTRACT.md shapes.
// Used when the backend is not reachable so the frontend always renders.

import type {
  BasinConfig,
  ObservationList,
  RunList,
  ExposureList,
  AuditList,
  DispatchResponse,
  SarPriorityList,
  MlEvidence,
} from "./types";

const basin: BasinConfig = {
  basin_id: "dudh-koshi-demo-01",
  name: "Dudh Koshi / Imja",
  boundary_geojson: {
    type: "Polygon",
    coordinates: [[[86.65, 27.65], [87.0, 27.65], [87.0, 27.98], [86.65, 27.98], [86.65, 27.65]]],
  },
  crs: "EPSG:4326",
};

const observations: ObservationList = {
  observations: [
    {
      observation_id: "obs-003",
      basin_id: "dudh-koshi-demo-01",
      acquired_at: "2026-09-04T12:00:00Z",
      source: "sentinel-1-grd-nrt",
      raster_uri: "data/processed/obs-003.tif",
      crs: "EPSG:4326",
      quality_score: 0.88, cloud_fraction: 0.11, alignment_ok: true, usable: true,
      confidence_adjustment: 0.95, water_area_km2: 3.2, water_area_change_percent: 14.3,
      rainfall_24h_mm: 72.4, rainfall_7d_mm: 188.0, mean_slope_degrees: 31.0,
      processing_version: "0.1.0", status: "processed",
    },
    {
      observation_id: "obs-002",
      basin_id: "dudh-koshi-demo-01",
      acquired_at: "2026-08-29T12:00:00Z",
      source: "sentinel-1-grd-nrt",
      raster_uri: "data/processed/obs-002.tif",
      crs: "EPSG:4326",
      quality_score: 0.90, cloud_fraction: 0.0, alignment_ok: true, usable: true,
      confidence_adjustment: 1.0, water_area_km2: 2.9, water_area_change_percent: 3.6,
      rainfall_24h_mm: 28.0, rainfall_7d_mm: 96.0, mean_slope_degrees: 31.0,
      processing_version: "0.1.0", status: "processed",
    },
    {
      observation_id: "obs-001",
      basin_id: "dudh-koshi-demo-01",
      acquired_at: "2026-08-23T12:00:00Z",
      source: "sentinel-2-l2a",
      raster_uri: "data/processed/obs-001.tif",
      crs: "EPSG:4326",
      quality_score: 0.92, cloud_fraction: 0.05, alignment_ok: true, usable: true,
      confidence_adjustment: 1.0, water_area_km2: 2.8, water_area_change_percent: 0.0,
      rainfall_24h_mm: 12.0, rainfall_7d_mm: 34.0, mean_slope_degrees: 31.0,
      processing_version: "0.1.0", status: "processed",
    },
  ],
};

const runs: RunList = {
  runs: [
    {
      run_id: "run-0001",
      observation_id: "obs-003",
      processing_version: "0.1.0",
      change_mask_uri: "data/processed/obs-003_expansion_mask.tif",
      corridor_geojson: { type: "LineString", coordinates: [[86.82, 27.88], [86.85, 27.91]] },
      change_stats_json: { water_area_km2: 3.2, expansion_percent: 14.3 },
      score: {
        hazard_score: 0.62,
        exposure_priority: 0.48,
        disease_risk: 0.31,
        confidence: 0.76,
        severity: "elevated",
        reasons: [
          "Water area expanded 14.3% vs baseline",
          "72.4 mm rainfall in 24h exceeds watch threshold",
          "188 mm 7-day rainfall saturates slopes",
        ],
      },
    },
  ],
};

const exposures: ExposureList = {
  exposures: [
    { asset_id: "village-2", asset_type: "village", name: "Chhukung", distance_m: 210.0, buffer_m: 100.0, inundated: false, population: 1240 },
    { asset_id: "BR-12", asset_type: "bridge", name: "Bridge 12", distance_m: 60.0, buffer_m: 75.0, inundated: false, population: null },
    { asset_id: "RD-4", asset_type: "road", name: "Road 4", distance_m: 40.0, buffer_m: 50.0, inundated: false, population: null },
    { asset_id: "well-3", asset_type: "well", name: "Well 3", distance_m: 90.0, buffer_m: 100.0, inundated: true, population: null },
  ],
};

const audit: AuditList = {
  entries: [
    { entry_id: 1, alert_id: "alert-0091", actor: "pipeline", action: "run", detail_json: '{"run_id":"run-0001","observation_id":"obs-003"}', created_at: "2026-09-04T12:05:00Z" },
    { entry_id: 2, alert_id: "alert-0091", actor: "coordinator-01", action: "review", detail_json: '{"decision":"confirm","note":"Verified against field report"}', created_at: "2026-09-04T12:10:00Z" },
    { entry_id: 3, alert_id: "alert-0091", actor: "coordinator-01", action: "dispatch", detail_json: '{"channel":"sms","recipient_group":"sector-b"}', created_at: "2026-09-04T12:11:00Z" },
  ],
};

const dispatch: DispatchResponse = {
  dispatch_id: "disp-0001",
  alert_id: "alert-0091",
  geofence_id: "sector-b",
  payload: '{"aid":"siren-04","sec":"B","haz":"GLOF_FL","lvl":3,"exp_pop":1240,"crit":["BR-12","RD-4"],"med_act":"BOIL_WATER_NOW"}',
  payload_bytes: 118,
  channel: "sms",
  status: "sent",
  sent_at: "2026-09-04T12:11:00Z",
};

const sarPriority: SarPriorityList = {
  sectors: [
    {
      sector_id: "village-2",
      name: "Chhukung",
      asset_type: "village",
      population: 1240,
      access_loss: 0.6,
      access_label: "AT_RISK",
      sar_priority: 0.372,
      reason: "Chhukung (1240 people, buffered) — access at risk, SAR priority 0.37",
      assets: ["village-2"],
    },
    {
      sector_id: "water-points",
      name: "Water Points",
      asset_type: "well",
      population: 1240,
      access_loss: 0.6,
      access_label: "AT_RISK",
      sar_priority: 0.261,
      reason: "1 water points (1 inundated) serve ~1240 people; access at risk",
      assets: ["well-3"],
    },
    {
      sector_id: "access-routes",
      name: "Access Routes",
      asset_type: "bridge",
      population: 1240,
      access_loss: 0.6,
      access_label: "AT_RISK",
      sar_priority: 0.186,
      reason: "2 access routes (0 cut) affect 1240 people downstream",
      assets: ["BR-12", "RD-4"],
    },
  ],
  top_priority: {
    sector_id: "village-2",
    name: "Chhukung",
    asset_type: "village",
    population: 1240,
    access_loss: 0.6,
    access_label: "AT_RISK",
    sar_priority: 0.372,
    reason: "Chhukung (1240 people, buffered) — access at risk, SAR priority 0.37",
    assets: ["village-2"],
  },
  summary: "3 sector(s) at risk. Top SAR priority: Chhukung (0.37).",
};

const mlEvidence: MlEvidence = {
  run_id: "run-0001",
  observation_id: "obs-003",
  ml_source: "deterministic_fallback",
  ml_confidence_mean: 0.78,
  ml_consensus_pixels: 2150,
  heatmap_uri: "/data/processed/obs-003_change_heatmap.png",
  mask_uri: "/data/processed/obs-003_expansion_mask.png",
  baseline_mask_uri: "/data/processed/baseline_water_mask.png",
  model_available: false,
};

export const mockData = {
  basin,
  observations,
  runs,
  exposures,
  audit,
  dispatch,
  sarPriority,
  mlEvidence,
};
