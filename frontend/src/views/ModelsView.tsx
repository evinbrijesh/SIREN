import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import type { ModelStatusResponse, MlEvaluation, TrendClassification, Run } from "../api/types";

interface ModelsViewProps {
  activeRun?: Run | null;
}

type TabMode = "architecture" | "registry" | "evaluation";

// Deterministic pipeline stages (the authoritative path)
const DETERMINISTIC_STAGES = [
  {
    stage: 1,
    name: "Sentinel-1 Preprocessing",
    role: "Calibration & Co-registration",
    desc: "VV/VH sigma0 calibration, speckle suppression, co-registration to SRTM grid, quality gate (cloud ≥ 0.20 → SAR route)",
    params: "Deterministic (0 params)",
    latencyMs: 12.0,
    input: "Sentinel-1 GRD IW (VV, VH)",
    output: "Aligned 2-band raster [2, H, W]",
    source: "Copernicus / ESA",
  },
  {
    stage: 2,
    name: "Deterministic Change Detection",
    role: "Dual-Pol Threshold + Set Difference",
    desc: "Per-date water masks via VV/VH backscatter ratio thresholding. Change = deterministic set difference between dates. This is the authoritative change mask.",
    params: "Deterministic (0 params)",
    latencyMs: 8.5,
    input: "Aligned VV/VH raster pair",
    output: "Binary change mask [H, W] (uint8)",
    source: "Rule-based (PRD §9.2)",
  },
  {
    stage: 3,
    name: "Hydrological Corridor & Exposure",
    role: "D8 Flow + OSM Buffering",
    desc: "D8 flow accumulation confirms gravity gradient. OSM waterway vectors buffered 100–150m. Tolerance-buffer intersections: bridges ±75m, roads ±50m, settlements/wells ±100m.",
    params: "Deterministic (0 params)",
    latencyMs: 15.3,
    input: "Change mask + SRTM DEM + OSM vectors",
    output: "Corridor GeoJSON + exposure list",
    source: "pysheds / OpenStreetMap",
  },
  {
    stage: 4,
    name: "Risk Fusion & Human Gate",
    role: "5-Factor Hazard + Coordinator Confirm",
    desc: "H = 0.30·S_trend + 0.25·A_exp + 0.20·R_rain + 0.15·T_slope + 0.10·D_prox. No ML term. Dispatch blocked until human confirm. Payload ≤ 250 bytes.",
    params: "Deterministic (0 params)",
    latencyMs: 3.2,
    input: "Exposure + weather + DEM slope",
    output: "H, E, D_risk scores + alert payload",
    source: "PRD §9.5 (5-factor formula)",
  },
];

// Shadow ML branch (supplementary only)
const SHADOW_MODEL = {
  name: "WaterUNet",
  role: "Single-Date Water Segmentation (Shadow)",
  params: "7.76M",
  latencyMs: 42.0,
  input: "[B, 2, H, W] VV/VH in dB",
  output: "[B, 1, H, W] P(water) ∈ [0, 1]",
  contract: "[-30, 0] dB → [0, 1] (normalize_sar)",
  weightsFile: "water_unet_weights.pt",
  weightsSize: "31.1 MB",
  trainingData: "Sen1Floods11 (431 hand-labeled SAR chips)",
  officialIou: 0.6708,
  eventHoldoutIou: 0.2394,
  deploymentGate: 0.65,
  influence: "0% — shadow only, no hazard weight",
};

// Archived / disqualified models (audit history only)
const ARCHIVED_MODELS = [
  {
    name: "Siamese U-Net",
    reason: "Label leakage: training synthesized 'before' images from the answer key. Test accuracy was memorization, not generalization.",
    disqualified: "2026-09-07 DL Audit",
  },
  {
    name: "SegFormer Classifier",
    reason: "Not the SegFormer architecture (was a 3-layer CNN). Could remove real flood pixels from the mask — load-bearing ML violation.",
    disqualified: "2026-09-07 DL Audit",
  },
  {
    name: "ConvLSTM Trend",
    reason: "Trained on synthetic mask progressions, not real satellite sequences. Temporal trend claims were unfounded.",
    disqualified: "2026-09-07 DL Audit",
  },
];

export default function ModelsView({ activeRun }: ModelsViewProps) {
  const [activeTab, setActiveTab] = useState<TabMode>("architecture");

  const { data: modelStatusData } = useQuery({
    queryKey: ["model-status"],
    queryFn: () => apiOrMock(() => api.getModelStatus(), "modelStatus") as Promise<ModelStatusResponse>,
    staleTime: 30_000,
  });

  const { data: mlEvalData } = useQuery({
    queryKey: ["ml-evaluation"],
    queryFn: () => apiOrMock(() => api.getMlEvaluation(), "mlEvaluation") as Promise<MlEvaluation>,
    staleTime: 60_000,
  });

  const { data: trendData } = useQuery({
    queryKey: ["trend"],
    queryFn: () => apiOrMock(() => api.getTrend(), "trend") as Promise<TrendClassification>,
    staleTime: 10_000,
  });

  const modelStatus = modelStatusData ?? mockData.modelStatus;
  const mlEvaluation = mlEvalData ?? mockData.mlEvaluation;
  const trend = trendData ?? mockData.trend;

  const models = Object.values(modelStatus.models);
  const activeCount = models.filter((m) => m.loaded && m.status !== "archived_disqualified").length;
  const archivedCount = models.filter((m) => m.status === "archived_disqualified").length;

  // Spatiotemporal series data (deterministic area history)
  const seriesPoints = [
    { label: "T0 (Baseline)", date: "2025-11-22", areaKm2: 3.0, deltaPct: 0.0, sensor: "Sentinel-2 MSI" },
    { label: "T1 (obs-001)", date: "2026-07-23", areaKm2: 3.32, deltaPct: 10.5, sensor: "Sentinel-1 SAR" },
    { label: "T2 (obs-002)", date: "2026-08-04", areaKm2: 4.1, deltaPct: 28.0, sensor: "Sentinel-1 SAR" },
    { label: "T3 (obs-003)", date: "2026-08-12", areaKm2: 4.3, deltaPct: 43.3, sensor: "Sentinel-1 SAR" },
  ];

  const chartW = 620;
  const chartH = 140;
  const pad = { top: 20, right: 30, bottom: 28, left: 45 };
  const plotW = chartW - pad.left - pad.right;
  const plotH = chartH - pad.top - pad.bottom;
  const minArea = 2.6;
  const maxArea = 4.8;

  const getX = (idx: number) => pad.left + (idx / (seriesPoints.length - 1)) * plotW;
  const getY = (val: number) => pad.top + plotH - ((val - minArea) / (maxArea - minArea)) * plotH;
  const trendLineD = seriesPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(i)} ${getY(p.areaKm2)}`).join(" ");

  return (
    <div className="flex flex-col h-full bg-surface-canvas text-text-primary overflow-y-auto">
      {/* Header */}
      <header className="flex-none flex items-center justify-between px-space-16 py-space-10 bg-surface-panel border-b border-border-subtle">
        <div className="flex items-center gap-space-16">
          <div className="flex items-center gap-space-8">
            <span className="w-2.5 h-2.5 rounded-full bg-status-safe animate-pulse" />
            <span className="label-caps font-mono tracking-wider text-headline-sm">
              ARCHITECTURE & MODEL AUDIT
            </span>
          </div>
          <span className="text-border-subtle hidden md:inline">|</span>
          <span className="text-body-sm text-text-dim hidden md:inline">
            Dudh Koshi / Imja Basin · Deterministic-First (ADR-010)
          </span>
        </div>

        <div className="flex items-center gap-space-8">
          <div className="flex items-center border border-border-subtle bg-surface-recessed p-space-2 text-body-sm">
            <button
              onClick={() => setActiveTab("architecture")}
              className={`px-space-10 py-space-4 font-mono text-caption transition-colors ${
                activeTab === "architecture"
                  ? "bg-surface-container text-primary font-semibold border-b border-primary"
                  : "text-text-dim hover:text-text-primary"
              }`}
            >
              PIPELINE ARCHITECTURE
            </button>
            <button
              onClick={() => setActiveTab("registry")}
              className={`px-space-10 py-space-4 font-mono text-caption transition-colors ${
                activeTab === "registry"
                  ? "bg-surface-container text-primary font-semibold border-b border-primary"
                  : "text-text-dim hover:text-text-primary"
              }`}
            >
              REGISTRY & ARCHIVE
            </button>
            <button
              onClick={() => setActiveTab("evaluation")}
              className={`px-space-10 py-space-4 font-mono text-caption transition-colors ${
                activeTab === "evaluation"
                  ? "bg-surface-container text-primary font-semibold border-b border-primary"
                  : "text-text-dim hover:text-text-primary"
              }`}
            >
              ML EVALUATION
            </button>
          </div>

          <div className="flex items-center gap-space-6 px-space-8 py-space-4 border border-border-subtle bg-surface-panel text-caption font-mono text-text-dim">
            <span className="text-status-safe">●</span>
            <span>{activeCount} ACTIVE · {archivedCount} ARCHIVED</span>
          </div>
        </div>
      </header>

      <div className="flex-1 p-space-16 flex flex-col gap-space-16 max-w-[1600px] w-full mx-auto">
        {/* ADR-010 Safety Banner */}
        <div className="flex items-center gap-space-12 px-space-16 py-space-8 bg-status-warn/10 border border-status-warn/30">
          <span className="text-status-warn text-headline-md">⚠</span>
          <div className="flex flex-col gap-space-2">
            <span className="text-body-md text-status-warn font-medium">ADR-010: ML Shadow Isolation</span>
            <span className="text-body-sm text-text-dim">
              WaterUNet (event-holdout IoU = 0.24) is below the 0.65 load-bearing gate. ML output is supplementary evidence only —
              it cannot modify the authoritative change mask, hazard score, or dispatch payload. The 5-factor hazard formula (PRD §9.5) has no ML term.
            </span>
          </div>
        </div>

        {/* Temporal Trend Chart — deterministic area history */}
        <section className="bg-surface-panel border border-border-subtle flex flex-col xl:flex-row divide-y xl:divide-y-0 xl:divide-x divide-border-subtle">
          <div className="flex-1 p-space-16 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-space-8">
              <div className="flex items-center gap-space-8">
                <span className="label-caps text-caption text-text-dim">DETERMINISTIC AREA HISTORY</span>
                <span className="text-caption font-mono text-text-muted">
                  ({seriesPoints.length} SATELLITE PASSES)
                </span>
              </div>
              <div className="flex items-center gap-space-12 text-caption font-mono">
                <span className="flex items-center gap-space-4">
                  <span className="w-3 h-0.5 bg-primary inline-block" />
                  <span className="text-text-dim">Water Surface (km²)</span>
                </span>
              </div>
            </div>

            <div className="w-full relative">
              <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-auto" preserveAspectRatio="none">
                {[2.8, 3.2, 3.6, 4.0, 4.4].map((v) => (
                  <g key={v}>
                    <line x1={pad.left} x2={chartW - pad.right} y1={getY(v)} y2={getY(v)}
                      stroke="var(--color-border-subtle)" strokeWidth="0.75" strokeDasharray="3 3" />
                    <text x={pad.left - 6} y={getY(v) + 3} textAnchor="end" fontSize="9"
                      fill="var(--color-text-muted)" fontFamily="monospace">{v.toFixed(1)}</text>
                  </g>
                ))}
                <path d={trendLineD} fill="none" stroke="var(--color-primary)" strokeWidth="2.5" />
                {seriesPoints.map((pt, i) => {
                  const cx = getX(i);
                  const cy = getY(pt.areaKm2);
                  const isCurrent = i === seriesPoints.length - 1;
                  return (
                    <g key={pt.label}>
                      <line x1={cx} x2={cx} y1={cy} y2={chartH - pad.bottom}
                        stroke="var(--color-border-subtle)" strokeWidth="1" strokeDasharray="2 2" />
                      {isCurrent && <circle cx={cx} cy={cy} r="7" fill="var(--color-primary)" fillOpacity="0.25" className="animate-ping" />}
                      <circle cx={cx} cy={cy} r={isCurrent ? "4.5" : "3.5"}
                        fill={isCurrent ? "var(--color-status-danger)" : "var(--color-primary)"}
                        stroke="var(--color-surface-panel)" strokeWidth="1.5" />
                      <text x={cx} y={cy - 9} textAnchor="middle" fontSize="10" fontWeight="600"
                        fill="var(--color-text-primary)" fontFamily="monospace">{pt.areaKm2.toFixed(2)} km²</text>
                      <text x={cx} y={chartH - pad.bottom + 14} textAnchor="middle" fontSize="9"
                        fill="var(--color-text-dim)" fontFamily="monospace">{pt.date}</text>
                    </g>
                  );
                })}
              </svg>
            </div>

            <div className="grid grid-cols-4 gap-space-8 mt-space-12 pt-space-8 border-t border-border-subtle">
              {seriesPoints.map((p) => (
                <div key={p.label} className="bg-surface-recessed p-space-6 border border-border-subtle text-caption">
                  <div className="flex items-center justify-between text-text-muted font-mono text-[10px]">
                    <span>{p.label.split(" ")[0]}</span>
                    <span>{p.date.slice(5)}</span>
                  </div>
                  <div className="flex items-baseline justify-between mt-space-2">
                    <span className="font-mono font-bold text-body-sm text-text-primary">{p.areaKm2.toFixed(2)} km²</span>
                    <span className={`font-mono text-caption font-bold ${
                      p.deltaPct > 20 ? "text-status-danger" : p.deltaPct > 0 ? "text-status-elevated" : "text-text-muted"
                    }`}>
                      {p.deltaPct > 0 ? `+${p.deltaPct.toFixed(1)}%` : "0.0%"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Trend classification — deterministic */}
          <div className="w-full xl:w-[460px] p-space-16 flex flex-col justify-between bg-surface-panel">
            <div>
              <div className="flex items-center justify-between">
                <span className="label-caps text-caption text-text-dim">TREND CLASSIFICATION</span>
                <span className="text-[10px] font-mono px-space-6 py-space-2 border border-status-safe text-status-safe bg-status-safe/10">
                  DETERMINISTIC
                </span>
              </div>
              <div className="mt-space-12 p-space-12 bg-surface-recessed border border-border-subtle">
                <div className="flex items-baseline justify-between">
                  <span className="text-caption font-mono text-text-muted uppercase">Active Trend State:</span>
                  <span className="text-caption font-mono text-text-dim">Source: {trend.source}</span>
                </div>
                <div className="flex items-center gap-space-12 mt-space-4">
                  <span className="text-metric-display font-headline font-bold text-status-danger tracking-wider">
                    {trend.trend_class.toUpperCase()}
                  </span>
                  <div className="flex flex-col">
                    <span className="font-mono text-body-md font-semibold text-text-primary">
                      {(trend.confidence * 100).toFixed(1)}% Deterministic
                    </span>
                    <span className="text-[10px] font-mono text-status-danger">HIGH-VELOCITY EXPANSION</span>
                  </div>
                </div>
                <div className="mt-space-8">
                  <div className="flex items-center justify-between text-[10px] font-mono text-text-muted mb-space-2">
                    <span>Area persistence score: {trend.confidence.toFixed(2)}</span>
                    <span>Computed from {trend.sequence_length} passes</span>
                  </div>
                  <div className="h-2 w-full bg-surface-container relative overflow-hidden border border-border-subtle">
                    <div className="h-full bg-status-danger transition-all duration-300"
                      style={{ width: `${trend.confidence * 100}%` }} />
                  </div>
                </div>
              </div>

              <div className="mt-space-12 space-y-space-6 text-caption font-mono">
                <div className="flex items-start gap-space-8 text-text-dim">
                  <span className="text-status-safe font-bold">✓</span>
                  <span>
                    <strong className="text-text-primary">Deterministic Area Persistence:</strong> Expansion confirmed across
                    3 consecutive Sentinel-1 passes. S_trend derived from area history, not ML.
                  </span>
                </div>
                <div className="flex items-start gap-space-8 text-text-dim">
                  <span className="text-status-safe font-bold">✓</span>
                  <span>
                    <strong className="text-text-primary">No ML Trend Model:</strong> ConvLSTM archived (2026-09-07 audit).
                    Trend class is computed from deterministic area deltas.
                  </span>
                </div>
                <div className="flex items-start gap-space-8 text-text-dim">
                  <span className="text-status-safe font-bold">✓</span>
                  <span>
                    <strong className="text-text-primary">Environmental Concordance:</strong> 24h GPM rainfall concords with
                    physical expansion signal.
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-space-12 pt-space-8 border-t border-border-subtle flex items-center justify-between text-caption font-mono text-text-muted">
              <span>ACTIVE RUN: {activeRun?.run_id ?? "run-obs-003"}</span>
              <span className="text-status-safe">DETERMINISTIC VERIFIED</span>
            </div>
          </div>
        </section>

        {/* Tab content */}
        {activeTab === "architecture" && (
          <section className="bg-surface-panel border border-border-subtle p-space-16 flex flex-col gap-space-16">
            <div className="flex items-center justify-between border-b border-border-subtle pb-space-8">
              <div className="flex items-center gap-space-8">
                <span className="label-caps text-caption text-text-dim">AUTHORITATIVE PIPELINE (DETERMINISTIC)</span>
                <span className="text-caption font-mono text-text-muted">
                  · ALL SAFETY-CRITICAL DECISIONS FLOW THROUGH THIS PATH
                </span>
              </div>
              <div className="flex items-center gap-space-12 text-caption font-mono text-text-dim">
                <span className="flex items-center gap-space-4">
                  <span className="w-2 h-2 rounded-full bg-status-safe" />
                  <span>Deterministic (0 ML params)</span>
                </span>
                <span className="flex items-center gap-space-4">
                  <span className="w-2 h-2 rounded-full bg-status-warn" />
                  <span>Shadow ML (supplementary)</span>
                </span>
              </div>
            </div>

            {/* Deterministic pipeline stages */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-space-12">
              {DETERMINISTIC_STAGES.map((stage) => (
                <div key={stage.stage}
                  className="p-space-12 bg-surface-recessed border border-status-safe/30 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="px-space-6 py-space-2 text-[10px] font-mono font-bold bg-surface-panel border border-border-subtle text-status-safe">
                        STAGE 0{stage.stage}
                      </span>
                      <span className="flex items-center gap-space-4 text-[11px] font-mono text-status-safe">
                        <span className="w-2 h-2 rounded-full bg-status-safe" />
                        DETERMINISTIC
                      </span>
                    </div>
                    <h3 className="font-semibold text-body-md text-text-primary mt-space-8">{stage.name}</h3>
                    <p className="text-caption text-text-dim line-clamp-3 mt-space-2">{stage.desc}</p>
                  </div>
                  <div className="mt-space-12 pt-space-8 border-t border-border-subtle space-y-space-4 font-mono text-caption">
                    <div className="flex justify-between text-text-muted">
                      <span>INPUT:</span>
                      <span className="text-text-primary">{stage.input}</span>
                    </div>
                    <div className="flex justify-between text-text-muted">
                      <span>OUTPUT:</span>
                      <span className="text-status-safe font-bold">{stage.output}</span>
                    </div>
                    <div className="flex justify-between text-text-muted">
                      <span>PARAMS:</span>
                      <span className="text-text-dim">{stage.params}</span>
                    </div>
                    <div className="flex justify-between text-text-muted">
                      <span>LATENCY:</span>
                      <span className="text-status-safe font-bold">{stage.latencyMs} ms</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Flow path summary */}
            <div className="p-space-8 bg-surface-recessed border border-border-subtle font-mono text-caption flex items-center justify-between flex-wrap gap-space-8">
              <span className="text-text-dim">
                <span className="text-status-safe font-bold">AUTHORITATIVE PATH:</span> Sentinel-1 VV/VH → Calibration →
                Deterministic Change Mask → D8 Corridor → OSM Exposure → 5-Factor Hazard → Human Confirm → ≤250B Dispatch
              </span>
              <span className="text-text-muted">TOTAL: ~39.0 ms (deterministic only)</span>
            </div>

            {/* Shadow ML branch */}
            <div className="border-t border-border-subtle pt-space-16">
              <div className="flex items-center gap-space-8 mb-space-12">
                <span className="label-caps text-caption text-status-warn">SHADOW ML BRANCH (SUPPLEMENTARY ONLY)</span>
                <span className="text-caption font-mono text-text-muted">· DOES NOT ENTER THE AUTHORITATIVE PATH</span>
              </div>

              <div className="p-space-12 bg-surface-recessed border border-status-warn/40 flex flex-col gap-space-12">
                <div className="flex items-start justify-between gap-space-16">
                  <div className="flex-1">
                    <div className="flex items-center gap-space-8">
                      <span className="flex items-center gap-space-4 text-[11px] font-mono text-status-warn">
                        <span className="w-2 h-2 rounded-full bg-status-warn" />
                        SHADOW MODE
                      </span>
                      <h3 className="font-semibold text-body-md text-text-primary">{SHADOW_MODEL.name}</h3>
                    </div>
                    <p className="text-caption text-text-dim mt-space-4">
                      Single-date water segmentation from Sentinel-1 VV/VH. Change detection = deterministic set difference
                      between two independently segmented water masks. Output is displayed as supplementary evidence only.
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-space-4">
                    <span className="text-caption border border-status-warn text-status-warn px-space-4 py-space-1 bg-status-warn/10">
                      W_ML = 0.00
                    </span>
                    <span className="text-caption text-text-muted">Influence on H: NONE</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-space-8 font-mono text-caption">
                  <div className="bg-surface-panel border border-border-subtle p-space-6">
                    <div className="text-[9px] text-text-muted">PARAMETERS</div>
                    <div className="text-body-sm font-bold text-text-primary mt-space-2">{SHADOW_MODEL.params}</div>
                  </div>
                  <div className="bg-surface-panel border border-border-subtle p-space-6">
                    <div className="text-[9px] text-text-muted">CONTRACT</div>
                    <div className="text-body-sm font-bold text-text-primary mt-space-2">{SHADOW_MODEL.contract}</div>
                  </div>
                  <div className="bg-surface-panel border border-border-subtle p-space-6">
                    <div className="text-[9px] text-text-muted">OFFICIAL IoU</div>
                    <div className="text-body-sm font-bold text-text-primary mt-space-2">{SHADOW_MODEL.officialIou.toFixed(4)}</div>
                    <div className="text-[9px] text-text-muted">leakage-affected</div>
                  </div>
                  <div className="bg-surface-panel border border-status-warn/40 p-space-6">
                    <div className="text-[9px] text-text-muted">EVENT-HOLDOUT IoU</div>
                    <div className="text-body-sm font-bold text-status-danger mt-space-2">{SHADOW_MODEL.eventHoldoutIou.toFixed(4)}</div>
                    <div className="text-[9px] text-status-warn">below 0.65 gate</div>
                  </div>
                </div>

                <div className="bg-status-warn/10 border border-status-warn/30 px-space-8 py-space-6 text-caption font-mono text-text-dim">
                  <span className="text-status-warn font-bold">ADR-010 ENFORCEMENT:</span> WaterUNet output is displayed as
                  a visual overlay and agreement statistic. It cannot replace the deterministic change mask, cannot filter
                  rule-detected pixels, and has zero weight in the hazard score. The SegFormer mask-replacement violation
                  has been removed.
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Registry & Archive tab */}
        {activeTab === "registry" && (
          <section className="bg-surface-panel border border-border-subtle p-space-16 flex flex-col gap-space-16">
            <div className="flex items-center justify-between border-b border-border-subtle pb-space-8">
              <span className="label-caps text-caption text-text-dim">
                MODEL CHECKPOINT REGISTRY & PROVENANCE
              </span>
              <span className="text-caption font-mono text-text-muted">LOCATION: data/processed/*.pt</span>
            </div>

            {/* Active models */}
            <div>
              <div className="flex items-center gap-space-8 mb-space-8">
                <span className="w-2 h-2 rounded-full bg-status-safe" />
                <span className="text-caption font-mono text-status-safe font-medium">ACTIVE</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-caption border-collapse">
                  <thead>
                    <tr className="border-b border-border-strong text-text-muted text-[11px] bg-surface-recessed">
                      <th className="py-space-8 px-space-12">MODEL</th>
                      <th className="py-space-8 px-space-12">STATUS</th>
                      <th className="py-space-8 px-space-12">ARCHITECTURE</th>
                      <th className="py-space-8 px-space-12">SIZE</th>
                      <th className="py-space-8 px-space-12">CHECKPOINT</th>
                      <th className="py-space-8 px-space-12">DATASET</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border-subtle">
                    {models.filter((m) => m.status !== "archived_disqualified").map((m) => (
                      <tr key={m.name} className="hover:bg-surface-container transition-colors">
                        <td className="py-space-8 px-space-12 font-semibold text-text-primary">{m.name}</td>
                        <td className="py-space-8 px-space-12">
                          <span className="inline-flex items-center gap-space-4 px-space-6 py-space-1 text-[10px] border border-status-safe/40 text-status-safe bg-status-safe/10">
                            <span className="w-1.5 h-1.5 rounded-full bg-status-safe" />
                            {m.loaded ? "SHADOW" : "OFFLINE"}
                          </span>
                        </td>
                        <td className="py-space-8 px-space-12 text-text-dim">{m.description?.slice(0, 60) ?? "N/A"}</td>
                        <td className="py-space-8 px-space-12 text-text-dim">
                          {m.weights_size_mb > 0 ? `${m.weights_size_mb.toFixed(1)} MB` : "N/A"}
                        </td>
                        <td className="py-space-8 px-space-12 text-text-muted text-[11px] truncate max-w-[180px]">
                          {m.weights_path ? m.weights_path.split("/").pop() : "N/A"}
                        </td>
                        <td className="py-space-8 px-space-12 text-text-dim text-[11px] truncate max-w-[220px]">
                          {m.training_data ?? "N/A"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Archived models */}
            <div className="border-t border-border-subtle pt-space-16">
              <div className="flex items-center gap-space-8 mb-space-8">
                <span className="w-2 h-2 rounded-full bg-status-danger" />
                <span className="text-caption font-mono text-status-danger font-medium">ARCHIVED / DISQUALIFIED</span>
                <span className="text-caption font-mono text-text-muted">— retained for audit history only</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-space-8">
                {ARCHIVED_MODELS.map((model) => (
                  <div key={model.name} className="bg-surface-recessed border border-status-danger/30 p-space-8 flex flex-col gap-space-4">
                    <div className="flex items-center justify-between">
                      <span className="text-body-sm font-semibold text-text-dim line-through">{model.name}</span>
                      <span className="text-[10px] font-mono border border-status-danger text-status-danger px-space-2 py-space-1">
                        DISQUALIFIED
                      </span>
                    </div>
                    <p className="text-caption text-text-muted leading-snug">{model.reason}</p>
                    <span className="text-[10px] font-mono text-text-muted">{model.disqualified}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {/* ML Evaluation tab */}
        {activeTab === "evaluation" && mlEvaluation && (
          <section className="bg-surface-panel border border-border-subtle p-space-16 flex flex-col gap-space-16">
            <div className="flex items-center justify-between border-b border-border-subtle pb-space-8">
              <span className="label-caps text-caption text-text-dim">
                DUAL-SPLIT EVALUATION — HONEST GENERALIZATION REPORT
              </span>
              <span className="text-caption font-mono text-status-warn border border-status-warn px-space-6 py-space-2 bg-status-warn/10">
                ADR-010
              </span>
            </div>

            {/* Dual-split comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-space-12">
              {/* Official split */}
              <div className="bg-surface-recessed border border-border-subtle p-space-12 flex flex-col gap-space-8">
                <div className="flex items-center justify-between">
                  <span className="text-body-md text-text-dim font-medium">Official Chip-Level Split</span>
                  <span className="text-caption border border-text-dim text-text-dim px-space-2 py-space-1">Literature-comparable</span>
                </div>
                <div className="flex items-baseline gap-space-8">
                  <span className="data-val text-metric-display text-text-primary">{mlEvaluation.official.test_iou.toFixed(4)}</span>
                  <span className="text-body-sm text-text-dim">Test IoU</span>
                </div>
                <div className="grid grid-cols-3 gap-space-8 text-caption data-val text-text-dim">
                  <div className="bg-surface-panel border border-border-subtle p-space-4 text-center">
                    <div className="text-[9px] text-text-muted">PRECISION</div>
                    <div className="text-text-primary mt-space-2">{mlEvaluation.official.test_precision.toFixed(3)}</div>
                  </div>
                  <div className="bg-surface-panel border border-border-subtle p-space-4 text-center">
                    <div className="text-[9px] text-text-muted">RECALL</div>
                    <div className="text-text-primary mt-space-2">{mlEvaluation.official.test_recall.toFixed(3)}</div>
                  </div>
                  <div className="bg-surface-panel border border-border-subtle p-space-4 text-center">
                    <div className="text-[9px] text-text-muted">F1</div>
                    <div className="text-text-primary mt-space-2">{mlEvaluation.official.test_f1.toFixed(3)}</div>
                  </div>
                </div>
                <div className="text-caption text-text-muted font-mono">
                  {mlEvaluation.official.n_test} test chips · epoch {mlEvaluation.official.best_epoch} · val IoU {mlEvaluation.official.best_val_iou.toFixed(4)}
                </div>
                <div className="text-caption text-text-muted border-t border-border-subtle pt-space-4">
                  ⚠ {mlEvaluation.official.leakage_note}
                </div>
              </div>

              {/* Event-holdout split */}
              <div className="bg-surface-recessed border border-status-warn/40 p-space-12 flex flex-col gap-space-8">
                <div className="flex items-center justify-between">
                  <span className="text-body-md text-status-warn font-medium">Event-Holdout Split</span>
                  <span className="text-caption border border-status-warn text-status-warn px-space-2 py-space-1">Deployment Gate</span>
                </div>
                <div className="flex items-baseline gap-space-8">
                  <span className="data-val text-metric-display text-status-danger">{mlEvaluation.event_holdout.test_iou.toFixed(4)}</span>
                  <span className="text-body-sm text-text-dim">Test IoU</span>
                </div>
                <div className="grid grid-cols-3 gap-space-8 text-caption data-val text-text-dim">
                  <div className="bg-surface-panel border border-border-subtle p-space-4 text-center">
                    <div className="text-[9px] text-text-muted">PRECISION</div>
                    <div className="text-text-primary mt-space-2">{mlEvaluation.event_holdout.test_precision.toFixed(3)}</div>
                  </div>
                  <div className="bg-surface-panel border border-border-subtle p-space-4 text-center">
                    <div className="text-[9px] text-text-muted">RECALL</div>
                    <div className="text-text-primary mt-space-2">{mlEvaluation.event_holdout.test_recall.toFixed(3)}</div>
                  </div>
                  <div className="bg-surface-panel border border-border-subtle p-space-4 text-center">
                    <div className="text-[9px] text-text-muted">F1</div>
                    <div className="text-text-primary mt-space-2">{mlEvaluation.event_holdout.test_f1.toFixed(3)}</div>
                  </div>
                </div>
                <div className="text-caption text-text-muted font-mono">
                  {mlEvaluation.event_holdout.n_test} test chips · epoch {mlEvaluation.event_holdout.best_epoch} · val IoU {mlEvaluation.event_holdout.best_val_iou.toFixed(4)}
                </div>
                {mlEvaluation.event_holdout.test_events && (
                  <div className="text-caption text-text-dim font-mono">
                    Held-out test events: <span className="text-text-primary">{mlEvaluation.event_holdout.test_events.join(", ")}</span>
                  </div>
                )}
                <div className="text-caption text-status-safe border-t border-border-subtle pt-space-4">
                  ✓ {mlEvaluation.event_holdout.leakage_note}
                </div>
              </div>
            </div>

            {/* IoU comparison bar */}
            <div className="flex flex-col gap-space-8">
              <span className="label-caps text-caption text-text-dim">IoU COMPARISON (0.00 ─ 1.00)</span>
              <div className="flex flex-col gap-space-8">
                <div className="flex items-center gap-space-8">
                  <span className="text-caption text-text-dim w-32 shrink-0 font-mono">Official</span>
                  <div className="flex-1 h-[8px] bg-surface-recessed border border-border-subtle overflow-hidden">
                    <div className="h-full bg-primary" style={{ width: `${mlEvaluation.official.test_iou * 100}%` }} />
                  </div>
                  <span className="data-val text-body-sm text-text-primary w-12 text-right">{mlEvaluation.official.test_iou.toFixed(2)}</span>
                </div>
                <div className="flex items-center gap-space-8">
                  <span className="text-caption text-status-warn w-32 shrink-0 font-mono">Event-holdout</span>
                  <div className="flex-1 h-[8px] bg-surface-recessed border border-border-subtle overflow-hidden">
                    <div className="h-full bg-status-danger" style={{ width: `${mlEvaluation.event_holdout.test_iou * 100}%` }} />
                  </div>
                  <span className="data-val text-body-sm text-status-danger w-12 text-right">{mlEvaluation.event_holdout.test_iou.toFixed(2)}</span>
                </div>
                <div className="flex items-center gap-space-8">
                  <span className="text-caption text-status-safe w-32 shrink-0 font-mono">Load-bearing gate</span>
                  <div className="flex-1 h-[8px] bg-surface-recessed border border-border-subtle overflow-hidden relative">
                    <div className="h-full bg-status-safe/30" style={{ width: "100%" }} />
                    <div className="absolute top-0 bottom-0 w-[2px] bg-status-safe" style={{ left: "65%" }} />
                  </div>
                  <span className="data-val text-body-sm text-status-safe w-12 text-right">0.65</span>
                </div>
              </div>
            </div>

            {/* Shadow mode rationale */}
            <div className="bg-status-warn/10 border border-status-warn/30 px-space-12 py-space-8 flex items-start gap-space-8">
              <span className="text-status-warn text-body-md mt-space-1">⚠</span>
              <div className="flex flex-col gap-space-4">
                <span className="text-body-sm text-text-primary font-medium">Shadow Mode Rationale</span>
                <span className="text-body-sm text-text-dim">{mlEvaluation.shadow_mode_reason}</span>
                <span className="text-caption text-text-muted font-mono">
                  The 0.43 IoU gap between official (0.67) and event-holdout (0.24) is the leakage signal.
                  The model has not generalized to unseen flood events. Hazard score H uses 5 physical factors only (PRD §9.5).
                </span>
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
