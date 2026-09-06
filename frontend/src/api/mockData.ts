import type {
  BasinConfig,
  ObservationList,
  RunList,
  ExposureList,
  AuditList,
  DispatchResponse,
  SarPriorityList,
  MlEvidence,
  ModelStatusResponse,
  TrendClassification,
} from "./types";

const basin: BasinConfig = {
  basin_id: "dudh-koshi-demo-01",
  name: "Dudh Koshi / Imja",
  boundary_geojson: {
    type: "Polygon",
    coordinates: [[[86.65, 27.65], [87.0, 27.65], [87.0, 27.98], [86.65, 27.98], [86.65, 27.65]]],
  },
  crs: "EPSG:4326",
  basemap_uri: "/data/processed/basemap.png",
  basemap_bounds: [[86.6489, 27.9805], [87.0003, 27.9805], [87.0003, 27.6494], [86.6489, 27.6494]],
};

const observations: ObservationList = {
  observations: [
    {
      observation_id: "obs-003", basin_id: basin.basin_id, acquired_at: "2026-08-12T12:00:00Z",
      source: "sentinel-1-grd-nrt", raster_uri: "data/processed/obs-003.tif", crs: "EPSG:4326",
      quality_score: 0.88, cloud_fraction: 0.0, optical_cloud_fraction: 0.90, alignment_ok: true,
      usable: true, confidence_adjustment: 0.95, water_area_km2: 4.3,
      water_area_change_percent: 43.0, rainfall_24h_mm: 60.0, rainfall_7d_mm: 160.0,
      mean_slope_degrees: 31.0, processing_version: "0.1.0", status: "processed",
    },
    {
      observation_id: "obs-002", basin_id: basin.basin_id, acquired_at: "2026-08-04T12:00:00Z",
      source: "sentinel-1-grd-nrt", raster_uri: "data/processed/obs-002.tif", crs: "EPSG:4326",
      quality_score: 0.90, cloud_fraction: 0.0, optical_cloud_fraction: 0.95, alignment_ok: true,
      usable: true, confidence_adjustment: 1.0, water_area_km2: 4.1,
      water_area_change_percent: 28.0, rainfall_24h_mm: 84.6, rainfall_7d_mm: 192.4,
      mean_slope_degrees: 31.0, processing_version: "0.1.0", status: "processed",
    },
    {
      observation_id: "obs-001", basin_id: basin.basin_id, acquired_at: "2026-07-23T12:00:00Z",
      source: "sentinel-1-grd-nrt", raster_uri: "data/processed/obs-001.tif", crs: "EPSG:4326",
      quality_score: 0.94, cloud_fraction: 0.0, optical_cloud_fraction: 0.0, alignment_ok: true,
      usable: true, confidence_adjustment: 1.0, water_area_km2: 3.2,
      water_area_change_percent: 8.0, rainfall_24h_mm: 18.2, rainfall_7d_mm: 64.0,
      mean_slope_degrees: 31.0, processing_version: "0.1.0", status: "processed",
    },
  ],
};

const runs: RunList = {
  runs: [{
    run_id: "run-mock-obs-002", observation_id: "obs-002", processing_version: "0.1.0",
    change_mask_uri: "data/processed/obs-002_expansion_mask.tif",
    corridor_geojson: { type: "LineString", coordinates: [[86.93, 27.90], [86.90, 27.895], [86.88, 27.89], [86.86, 27.90]] },
    change_stats_json: {
      water_area_km2: 4.1, expansion_percent: 28.0, rainfall_24h_mm: 84.6,
      source: "sentinel-1-grd-nrt",
      routing: { path: "sar", sar_primary: true, cloud_fraction_reported: 0.95 },
    },
    score: {
      hazard_score: 0.88, exposure_priority: 0.54, disease_risk: 0.08, confidence: 0.90,
      severity: "elevated",
      reasons: [
        "water-area expansion +28.0% contributes 0.25*0.93 to H",
        "rainfall 24h 84.6mm / 7d 192.4mm contributes 0.2*0.85 to H",
        "optical cloud 95% triggered the Sentinel-1 SAR path",
      ],
    },
    status: "processed", started_at: "2026-08-04T12:05:00Z", finished_at: "2026-08-04T12:05:09Z",
    decision: null, reviewer: null, decided_at: null,
  }],
};

const exposures: ExposureList = {
  exposures: [
    { asset_id: "village-2", asset_type: "village", name: "Chhukung", distance_m: 210, buffer_m: 100, inundated: false, population: 1240, geometry_geojson: { type: "Point", coordinates: [86.86, 27.90] } },
    { asset_id: "BR-12", asset_type: "bridge", name: "Hillary Bridge", distance_m: 60, buffer_m: 75, inundated: false, population: null, geometry_geojson: { type: "Point", coordinates: [86.85, 27.91] } },
    { asset_id: "RD-4", asset_type: "road", name: "Road 4", distance_m: 40, buffer_m: 50, inundated: false, population: null, geometry_geojson: { type: "LineString", coordinates: [[86.84, 27.90], [86.87, 27.92]] } },
    { asset_id: "well-3", asset_type: "well", name: "Well W-1", distance_m: 90, buffer_m: 100, inundated: true, population: 1240, geometry_geojson: { type: "Point", coordinates: [86.86, 27.89] } },
  ],
};

// Real SHA-256 hashes computed from the hash chain formula:
// SHA256(prev_hash + timestamp + detail_json)
const HASH_0 = "0000000000000000000000000000000000000000000000000000000000000000";
const HASH_1 = "de0190c18133b2700adcedb6b65350c4a500e6cd01bc48fbc191fe70133709e7";
const HASH_2 = "6deffb9b9d497309f039c07d9fce1bfeb24ab55ec360cdc7d7a5ca38074a5bc1";
const HASH_3 = "382e377c291fe3aa594a2e793deadb6ac3900f6538db5dfc834f88c75ed40245";
const audit: AuditList = {
  entries: [
    { entry_id: 1, alert_id: "alert-0091", actor: "pipeline", action: "run", detail_json: '{"run_id":"run-mock-obs-003","observation_id":"obs-003"}', created_at: "2026-08-12T12:05:00Z", prev_hash: HASH_0, event_hash: HASH_1 },
    { entry_id: 2, alert_id: "alert-0091", actor: "coordinator-01", action: "review", detail_json: '{"decision":"confirm","run_id":"run-mock-obs-003"}', created_at: "2026-08-12T12:10:00Z", prev_hash: HASH_1, event_hash: HASH_2 },
    { entry_id: 3, alert_id: "alert-0091", actor: "coordinator-01", action: "dispatch", detail_json: '{"channel":"sms","run_id":"run-mock-obs-003"}', created_at: "2026-08-12T12:11:00Z", prev_hash: HASH_2, event_hash: HASH_3 },
  ],
};

const dispatch: DispatchResponse = {
  dispatch_id: "disp-0001", alert_id: "alert-0091", geofence_id: "sector-b",
  payload: '{"aid":"siren-04","sec":"B","haz":"GLOF_FL","lvl":3,"exp_pop":1240,"crit":["BR-12","RD-4"],"med_act":"BOIL_WATER_NOW"}',
  payload_bytes: 118, channel: "sms", status: "sent", sent_at: "2026-08-04T12:11:00Z",
};

const sarPriority: SarPriorityList = {
  sectors: [
    { sector_id: "village-2", name: "Chhukung", asset_type: "village", population: 1240, access_loss: 0.6, access_label: "AT_RISK", sar_priority: 0.372, reason: "Chhukung (1240 people, buffered) — access at risk, SAR priority 0.37", assets: ["village-2"] },
    { sector_id: "water-points", name: "Water Points", asset_type: "well", population: 1240, access_loss: 0.6, access_label: "AT_RISK", sar_priority: 0.261, reason: "1 water point inundated; 1240 people require alternate supply", assets: ["well-3"] },
    { sector_id: "access-routes", name: "Access Routes", asset_type: "bridge", population: 1240, access_loss: 0.6, access_label: "AT_RISK", sar_priority: 0.186, reason: "2 access routes affect 1240 people downstream", assets: ["BR-12", "RD-4"] },
  ],
  top_priority: null,
  summary: "3 sectors at risk. Top SAR priority: Chhukung (0.37).",
};
sarPriority.top_priority = sarPriority.sectors[0];

const mlEvidence: MlEvidence = {
  run_id: "run-mock-obs-002", observation_id: "obs-002", ml_source: "deterministic_fallback",
  ml_confidence_mean: 0.78, ml_consensus_pixels: 2548,
  heatmap_uri: "/data/processed/obs-002_change_heatmap.png",
  heatmap_bounds: [[86.8945, 27.9219], [86.9555, 27.9219], [86.9555, 27.8681], [86.8945, 27.8681]],
  mask_uri: "/data/processed/obs-002_expansion_mask.png",
  mask_bounds: [[86.8945, 27.9219], [86.9555, 27.9219], [86.9555, 27.8681], [86.8945, 27.8681]],
  baseline_mask_uri: "/data/processed/baseline_water_mask.png", baseline_mask_bounds: null,
  preview_baseline_uri: "/map-assets/obs-002/baseline-optical.png",
  preview_after_uri: "/data/processed/obs-002_change_heatmap.png",
  model_available: false,
  change_polygon: { type: "Polygon", coordinates: [[[86.8945, 27.8681], [86.9555, 27.8681], [86.9555, 27.9219], [86.8945, 27.9219], [86.8945, 27.8681]]] },
};

const modelStatus: ModelStatusResponse = {
  models: {
    siamese_unet: {
      stage: 1,
      name: "Siamese U-Net Change Detector",
      loaded: true,
      weights_path: "data/processed/siamese_unet_weights.pt",
      weights_exists: true,
      weights_size_mb: 85.3,
      metadata: {
        epoch: 40,
        loss: 0.0842,
        accuracy: 0.941,
        in_channels: 3,
        backbone: "ResNet-34",
        input_resolution: "512x512 @ 10m GSD",
        dataset: "Sen1Floods11 (252 hand-labeled SAR/Optical chips)",
      },
      description:
        "Bi-temporal change detection using shared ResNet-34 encoder. Produces pixel-level change probability maps from satellite image pairs.",
      architecture: "SiameseUNet(ResNet-34, U-Net decoder)",
      training_data: "Sen1Floods11 (252 hand-labeled SAR chips)",
    },
    segformer: {
      stage: 2,
      name: "SegFormer Land-Cover Classifier",
      loaded: true,
      weights_path: "data/processed/segformer_classifier_weights.pt",
      weights_exists: true,
      weights_size_mb: 14.8,
      metadata: {
        epoch: 25,
        loss: 0.128,
        accuracy: 0.912,
        num_classes: 5,
        class_names: ["Water (Flood)", "Debris Flow", "Snowmelt (Benign)", "Cloud/Shadow", "Bare Rock"],
        dataset: "Sen1Floods11 weak-labeled (2580 crops, 5 classes)",
      },
      description:
        "Classifies changed pixels into functional categories: water, debris, snowmelt, shadow, bare rock. Filters false alarms from cloud shadows and snowmelt.",
      architecture: "SegFormerHead(MiT-B0 patch attention, 5-class)",
      training_data: "Sen1Floods11 weak-labeled (2580 crops, 5 classes)",
    },
    consensus_gating: {
      stage: 3,
      name: "Multi-Sensor Consensus Gating",
      loaded: true,
      weights_path: null,
      weights_exists: true,
      weights_size_mb: 0,
      metadata: {
        slope_threshold_deg: 35.0,
        ml_weight: 0.6,
        rule_weight: 0.4,
        engine: "Vectorized NumPy / C++ array operator",
      },
      description:
        "Fuses ML mask with rule-based mask and DEM slope gating. Eliminates ML false positives on steep terrain (>35°). Weighted fusion: 0.6×ML + 0.4×rule-based.",
      architecture: "Deterministic (consensus.py)",
      training_data: null,
    },
    convlstm_trend: {
      stage: 4,
      name: "ConvLSTM Temporal Trend Classifier",
      loaded: true,
      weights_path: "data/processed/convlstm_trend_weights.pt",
      weights_exists: true,
      weights_size_mb: 5.6,
      metadata: {
        epoch: 50,
        loss: 0.194,
        accuracy: 0.884,
        seq_len: 3,
        num_classes: 4,
        hybrid_cutoff: 0.75,
        dataset: "Synthetic sequences from Sen1Floods11 (840 sequences, 4 trend classes)",
      },
      description:
        "Classifies temporal trend from satellite sequences: stable, slowly expanding, rapidly expanding, or uncertain. Uses ConvLSTM cells over a CNN encoder. Hybrid inference — defers to deterministic thresholds when ML confidence < 0.75.",
      architecture: "ConvLSTMTrendClassifier(CNN encoder, ConvLSTM cells, 4-class)",
      training_data: "Synthetic sequences from Sen1Floods11 (840 sequences, 4 trend classes)",
    },
  },
};

const trend: TrendClassification = {
  trend_class: "rapidly",
  confidence: 0.82,
  source: "hybrid_inference",
  ml_model_available: true,
  sequence_length: 3,
  water_areas: [3.0, 3.32, 4.10, 4.30],
  expansion_pcts: [0.0, 10.5, 28.0, 43.3],
};

export const canonicalBaseline = {
  acquired_at: "2025-11-22T12:00:00Z",
  source: "sentinel-2-l2a",
  cloud_fraction: 0.05,
  rainfall_24h_mm: 0.0,
  water_area_km2: 3.0,
  water_area_change_percent: 0.0,
};

export const mockData = { basin, observations, runs, exposures, audit, dispatch, sarPriority, mlEvidence, modelStatus, trend };
