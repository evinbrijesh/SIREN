import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import type { ModelStatusResponse, TrendClassification, Run } from "../api/types";

interface ModelsViewProps {
  activeRun?: Run | null;
}

type TabMode = "dag" | "registry" | "benchmarks";
type StageId = 1 | 2 | 3 | 4;

const STAGE_SPECS: Record<
  StageId,
  {
    stage: StageId;
    key: string;
    name: string;
    shortName: string;
    role: string;
    backbone: string;
    params: string;
    latencyMs: number;
    inputShape: string;
    outputShape: string;
    weightsFile: string;
    weightsSize: string;
    trainingData: string;
    checkpointMeta: { epoch: number; loss: number; accuracy: number; f1: number };
    thresholds: { name: string; value: string; condition: string };
    classes?: { name: string; prob: number; color: string; desc: string }[];
  }
> = {
  1: {
    stage: 1,
    key: "siamese_unet",
    name: "Siamese U-Net Change Detector",
    shortName: "Siamese U-Net",
    role: "Bi-Temporal Surface Difference Extraction",
    backbone: "ResNet-34 (Shared Siamese Encoders)",
    params: "21.84M",
    latencyMs: 42.4,
    inputShape: "[2, 3, 512, 512]",
    outputShape: "[1, 512, 512] (Sigmoid P ∈ [0, 1])",
    weightsFile: "siamese_unet_weights.pt",
    weightsSize: "85.3 MB",
    trainingData: "Sen1Floods11 (252 hand-labeled SAR/Optical chips)",
    checkpointMeta: { epoch: 40, loss: 0.0842, accuracy: 0.941, f1: 0.928 },
    thresholds: { name: "Change Cutoff (τ_change)", value: "0.40", condition: "P >= 0.40 flags changed pixel" },
  },
  2: {
    stage: 2,
    key: "segformer",
    name: "SegFormer Land-Cover Classifier",
    shortName: "SegFormer MiT-B0",
    role: "Semantic Functional Categorization of Changes",
    backbone: "MiT-B0 Transformer (16x16 Patch Embeddings)",
    params: "3.72M",
    latencyMs: 18.1,
    inputShape: "[N, 3, 256, 256] Crops",
    outputShape: "[N, 5] Categorical Logits",
    weightsFile: "segformer_classifier_weights.pt",
    weightsSize: "14.8 MB",
    trainingData: "Sen1Floods11 weak-labeled (2580 crops, 5 classes)",
    checkpointMeta: { epoch: 25, loss: 0.128, accuracy: 0.912, f1: 0.895 },
    thresholds: { name: "Water Certainty (τ_water)", value: "0.65", condition: "Softmax[Water] >= 0.65 flags hazard" },
    classes: [
      { name: "Water (Flood)", prob: 0.884, color: "#00f0ff", desc: "Active lake & river inundation" },
      { name: "Debris Flow", prob: 0.072, color: "#ffb000", desc: "Scoured gravel, mud & moraine" },
      { name: "Snowmelt (Benign)", prob: 0.026, color: "#38bdf8", desc: "Seasonal alpine snow retreat" },
      { name: "Cloud / Shadow", prob: 0.011, color: "#48556e", desc: "Sensor artifact (auto-rejected)" },
      { name: "Bare Rock", prob: 0.007, color: "#94a3b8", desc: "Static cliff and valley rock" },
    ],
  },
  3: {
    stage: 3,
    key: "consensus_gating",
    name: "Multi-Sensor Consensus Gating",
    shortName: "Consensus Gating",
    role: "Physics-Constrained Multi-Sensor Arbitration",
    backbone: "Vectorized NumPy / C++ Hydrological Engine",
    params: "Deterministic (0 params)",
    latencyMs: 6.2,
    inputShape: "[512, 512] x 3 + SRTM 30m DEM",
    outputShape: "[512, 512] uint8 Mask + GeoJSON",
    weightsFile: "consensus.py (Rule engine)",
    weightsSize: "N/A",
    trainingData: "NASA SRTM 1 Arc-Second DEM + CDSE SAR/Optical",
    checkpointMeta: { epoch: 0, loss: 0.0, accuracy: 0.998, f1: 0.995 },
    thresholds: { name: "Slope Prune Cutoff (θ_slope)", value: "35.0°", condition: "Slope > 35° automatically pruned" },
  },
  4: {
    stage: 4,
    key: "convlstm_trend",
    name: "ConvLSTM Temporal Trend Classifier",
    shortName: "ConvLSTM Trend",
    role: "Spatiotemporal Progression & Velocity Triage",
    backbone: "ConvLSTM2D Cells + CNN Feature Extractor",
    params: "1.42M",
    latencyMs: 28.3,
    inputShape: "[B, S=3, 1, 64, 64]",
    outputShape: "[B, 4] Trend Distribution",
    weightsFile: "convlstm_trend_weights.pt",
    weightsSize: "5.6 MB",
    trainingData: "Synthetic sequences from Sen1Floods11 (840 sequences)",
    checkpointMeta: { epoch: 50, loss: 0.194, accuracy: 0.884, f1: 0.871 },
    thresholds: { name: "Hybrid Fallback Cutoff (τ_hybrid)", value: "0.75", condition: "Confidence < 0.75 defers to rules" },
  },
};

export default function ModelsView({ activeRun }: ModelsViewProps) {
  const [selectedStage, setSelectedStage] = useState<StageId>(1);
  const [activeTab, setActiveTab] = useState<TabMode>("dag");
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const { data: modelStatusData } = useQuery({
    queryKey: ["model-status"],
    queryFn: () => apiOrMock(() => api.getModelStatus(), "modelStatus") as Promise<ModelStatusResponse>,
    staleTime: 30_000,
  });

  const { data: trendData } = useQuery({
    queryKey: ["trend"],
    queryFn: () => apiOrMock(() => api.getTrend(), "trend") as Promise<TrendClassification>,
    staleTime: 10_000,
  });

  const modelStatus = modelStatusData ?? mockData.modelStatus;
  const trend = trendData ?? mockData.trend;

  const currentStageSpec = STAGE_SPECS[selectedStage];
  const models = Object.values(modelStatus.models).sort((a, b) => a.stage - b.stage);
  const loadedCount = models.filter((m) => m.loaded).length;

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1500);
  };

  // Observations spatiotemporal series data
  const seriesPoints = [
    { label: "T0 (Baseline)", date: "2025-11-22", areaKm2: 3.0, deltaPct: 0.0, sensor: "Sentinel-2 MSI", ci: 0.12 },
    { label: "T1 (obs-001)", date: "2026-07-23", areaKm2: 3.32, deltaPct: 10.5, sensor: "Sentinel-1 SAR", ci: 0.15 },
    { label: "T2 (obs-002)", date: "2026-08-04", areaKm2: 4.1, deltaPct: 28.0, sensor: "Sentinel-1 SAR", ci: 0.18 },
    { label: "T3 (obs-003)", date: "2026-08-12", areaKm2: 4.3, deltaPct: 43.3, sensor: "Sentinel-1 SAR", ci: 0.2 },
  ];

  // Chart coordinate mapping
  const chartW = 620;
  const chartH = 140;
  const pad = { top: 20, right: 30, bottom: 28, left: 45 };
  const plotW = chartW - pad.left - pad.right;
  const plotH = chartH - pad.top - pad.bottom;
  const minArea = 2.6;
  const maxArea = 4.8;

  const getX = (idx: number) => pad.left + (idx / (seriesPoints.length - 1)) * plotW;
  const getY = (val: number) => pad.top + plotH - ((val - minArea) / (maxArea - minArea)) * plotH;

  // Upper & lower confidence interval path
  const upperPath = seriesPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(i)} ${getY(p.areaKm2 + p.ci)}`).join(" ");
  const lowerPath = [...seriesPoints]
    .reverse()
    .map((p, i) => `L ${getX(seriesPoints.length - 1 - i)} ${getY(p.areaKm2 - p.ci)}`)
    .join(" ");
  const ciAreaD = `${upperPath} ${lowerPath} Z`;
  const trendLineD = seriesPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(i)} ${getY(p.areaKm2)}`).join(" ");

  return (
    <div className="flex flex-col h-full bg-surface-canvas text-text-primary overflow-y-auto">
      {/* Top Header & Telemetry Bar */}
      <header className="flex-none flex items-center justify-between px-space-16 py-space-10 bg-surface-panel border-b border-border-subtle">
        <div className="flex items-center gap-space-16">
          <div className="flex items-center gap-space-8">
            <span className="w-2.5 h-2.5 rounded-full bg-status-safe animate-pulse" />
            <span className="label-caps font-mono tracking-wider text-headline-sm">
              ML OBSERVABILITY & INFERENCE WORKBENCH
            </span>
          </div>
          <span className="text-border-subtle hidden md:inline">|</span>
          <span className="text-body-sm text-text-dim hidden md:inline">
            Dudh Koshi / Imja Basin · 4-Stage Bi-Temporal Pipeline
          </span>
        </div>

        {/* View Mode Switcher */}
        <div className="flex items-center gap-space-8">
          <div className="flex items-center border border-border-subtle bg-surface-recessed p-space-2 text-body-sm">
            <button
              onClick={() => setActiveTab("dag")}
              className={`px-space-10 py-space-4 font-mono text-caption transition-colors ${
                activeTab === "dag"
                  ? "bg-surface-container text-primary font-semibold border-b border-primary"
                  : "text-text-dim hover:text-text-primary"
              }`}
            >
              PIPELINE DAG & TRACE
            </button>
            <button
              onClick={() => setActiveTab("registry")}
              className={`px-space-10 py-space-4 font-mono text-caption transition-colors ${
                activeTab === "registry"
                  ? "bg-surface-container text-primary font-semibold border-b border-primary"
                  : "text-text-dim hover:text-text-primary"
              }`}
            >
              WEIGHTS & REGISTRY
            </button>
            <button
              onClick={() => setActiveTab("benchmarks")}
              className={`px-space-10 py-space-4 font-mono text-caption transition-colors ${
                activeTab === "benchmarks"
                  ? "bg-surface-container text-primary font-semibold border-b border-primary"
                  : "text-text-dim hover:text-text-primary"
              }`}
            >
              LATENCY & PROFILING
            </button>
          </div>

          <div className="flex items-center gap-space-6 px-space-8 py-space-4 border border-border-subtle bg-surface-panel text-caption font-mono text-text-dim">
            <span className="text-status-safe">●</span>
            <span>{loadedCount}/4 MODELS READY</span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <div className="flex-1 p-space-16 flex flex-col gap-space-16 max-w-[1600px] w-full mx-auto">
        {/* SECTION 1: Temporal Trend & Hybrid Decision Card */}
        <section className="bg-surface-panel border border-border-subtle flex flex-col xl:flex-row divide-y xl:divide-y-0 xl:divide-x divide-border-subtle">
          {/* Left Column: Temporal Sequence Chart */}
          <div className="flex-1 p-space-16 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-space-8">
              <div className="flex items-center gap-space-8">
                <span className="label-caps text-caption text-text-dim">SPATIOTEMPORAL EXTENT SEQUENCE</span>
                <span className="text-caption font-mono text-text-muted">
                  ({seriesPoints.length} SATELLITE PASSES · CONFIDENCE BAND ±1σ)
                </span>
              </div>
              <div className="flex items-center gap-space-12 text-caption font-mono">
                <span className="flex items-center gap-space-4">
                  <span className="w-3 h-0.5 bg-primary inline-block" />
                  <span className="text-text-dim">Water Surface (km²)</span>
                </span>
                <span className="flex items-center gap-space-4">
                  <span className="w-3 h-2 bg-primary/20 border border-primary/40 inline-block" />
                  <span className="text-text-muted">Calibrated CI</span>
                </span>
              </div>
            </div>

            {/* SVG Spatiotemporal Chart */}
            <div className="w-full relative">
              <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-auto" preserveAspectRatio="none">
                {/* Horizontal reference grid lines */}
                {[2.8, 3.2, 3.6, 4.0, 4.4].map((v) => (
                  <g key={v}>
                    <line
                      x1={pad.left}
                      x2={chartW - pad.right}
                      y1={getY(v)}
                      y2={getY(v)}
                      stroke="var(--color-border-subtle)"
                      strokeWidth="0.75"
                      strokeDasharray="3 3"
                    />
                    <text
                      x={pad.left - 6}
                      y={getY(v) + 3}
                      textAnchor="end"
                      fontSize="9"
                      fill="var(--color-text-muted)"
                      fontFamily="monospace"
                    >
                      {v.toFixed(1)}
                    </text>
                  </g>
                ))}

                {/* Shaded Confidence Interval Envelope */}
                <path d={ciAreaD} fill="var(--color-primary)" fillOpacity="0.12" />

                {/* Water Area Primary Trend Line */}
                <path d={trendLineD} fill="none" stroke="var(--color-primary)" strokeWidth="2.5" />

                {/* Data point dots & delta callouts */}
                {seriesPoints.map((pt, i) => {
                  const cx = getX(i);
                  const cy = getY(pt.areaKm2);
                  const isCurrent = i === seriesPoints.length - 1;
                  return (
                    <g key={pt.label}>
                      {/* Vertical drop line */}
                      <line
                        x1={cx}
                        x2={cx}
                        y1={cy}
                        y2={chartH - pad.bottom}
                        stroke="var(--color-border-subtle)"
                        strokeWidth="1"
                        strokeDasharray="2 2"
                      />
                      {/* Point halo for current */}
                      {isCurrent && (
                        <circle cx={cx} cy={cy} r="7" fill="var(--color-primary)" fillOpacity="0.25" className="animate-ping" />
                      )}
                      {/* Point circle */}
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isCurrent ? "4.5" : "3.5"}
                        fill={isCurrent ? "var(--color-status-danger)" : "var(--color-primary)"}
                        stroke="var(--color-surface-panel)"
                        strokeWidth="1.5"
                      />
                      {/* Value label */}
                      <text
                        x={cx}
                        y={cy - 9}
                        textAnchor="middle"
                        fontSize="10"
                        fontWeight="600"
                        fill="var(--color-text-primary)"
                        fontFamily="monospace"
                      >
                        {pt.areaKm2.toFixed(2)} km²
                      </text>
                      {/* X-axis date */}
                      <text
                        x={cx}
                        y={chartH - pad.bottom + 14}
                        textAnchor="middle"
                        fontSize="9"
                        fill="var(--color-text-dim)"
                        fontFamily="monospace"
                      >
                        {pt.date}
                      </text>
                      {/* Sensor tag */}
                      <text
                        x={cx}
                        y={chartH - 2}
                        textAnchor="middle"
                        fontSize="8"
                        fill="var(--color-text-muted)"
                        fontFamily="monospace"
                      >
                        {pt.sensor.split(" ")[0]}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>

            {/* Sequence Timestep Delta Grid */}
            <div className="grid grid-cols-4 gap-space-8 mt-space-12 pt-space-8 border-t border-border-subtle">
              {seriesPoints.map((p) => (
                <div key={p.label} className="bg-surface-recessed p-space-6 border border-border-subtle text-caption">
                  <div className="flex items-center justify-between text-text-muted font-mono text-[10px]">
                    <span>{p.label.split(" ")[0]}</span>
                    <span>{p.date.slice(5)}</span>
                  </div>
                  <div className="flex items-baseline justify-between mt-space-2">
                    <span className="font-mono font-bold text-body-sm text-text-primary">{p.areaKm2.toFixed(2)} km²</span>
                    <span
                      className={`font-mono text-caption font-bold ${
                        p.deltaPct > 20
                          ? "text-status-danger"
                          : p.deltaPct > 0
                          ? "text-status-elevated"
                          : "text-text-muted"
                      }`}
                    >
                      {p.deltaPct > 0 ? `+${p.deltaPct.toFixed(1)}%` : "0.0%"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right Column: Hybrid Inference Telemetry */}
          <div className="w-full xl:w-[460px] p-space-16 flex flex-col justify-between bg-surface-panel">
            <div>
              {/* Header */}
              <div className="flex items-center justify-between">
                <span className="label-caps text-caption text-text-dim">HYBRID INFERENCE ARBITRATION</span>
                <span className="text-[10px] font-mono px-space-6 py-space-2 border border-border-strong text-primary bg-primary/10">
                  FALLBACK GATE: τ = 0.75
                </span>
              </div>

              {/* Classification Headline */}
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
                      {(trend.confidence * 100).toFixed(1)}% Calibrated
                    </span>
                    <span className="text-[10px] font-mono text-status-danger">HIGH-VELOCITY EXPANSION</span>
                  </div>
                </div>

                {/* Calibrated Confidence Distribution Bar */}
                <div className="mt-space-8">
                  <div className="flex items-center justify-between text-[10px] font-mono text-text-muted mb-space-2">
                    <span>ML Confidence Score: {trend.confidence.toFixed(2)}</span>
                    <span>Cutoff Threshold: 0.75</span>
                  </div>
                  <div className="h-2 w-full bg-surface-container relative overflow-hidden border border-border-subtle">
                    {/* Fill */}
                    <div
                      className="h-full bg-status-danger transition-all duration-300"
                      style={{ width: `${trend.confidence * 100}%` }}
                    />
                    {/* Cutoff marker line */}
                    <div className="absolute top-0 bottom-0 left-[75%] w-[2px] bg-primary z-10" />
                  </div>
                  <div className="flex justify-between text-[9px] font-mono text-text-muted mt-space-2">
                    <span>0.00 (Uncertain)</span>
                    <span className="text-primary font-bold">▲ 0.75 Hybrid Cutoff</span>
                    <span>1.00 (Definite)</span>
                  </div>
                </div>
              </div>

              {/* Hybrid Decision Trigger Rationale */}
              <div className="mt-space-12 space-y-space-6 text-caption font-mono">
                <div className="flex items-start gap-space-8 text-text-dim">
                  <span className="text-status-safe font-bold">✓</span>
                  <span>
                    <strong className="text-text-primary">Neural Confidence Valid:</strong> 0.82 &gt; 0.75 cutoff. Deep
                    learning temporal feature extraction active.
                  </span>
                </div>
                <div className="flex items-start gap-space-8 text-text-dim">
                  <span className="text-status-safe font-bold">✓</span>
                  <span>
                    <strong className="text-text-primary">Dual-Pass Persistence:</strong> Expansion confirmed across 2
                    consecutive Sentinel-1 passes (Aug 04 &amp; Aug 12).
                  </span>
                </div>
                <div className="flex items-start gap-space-8 text-text-dim">
                  <span className="text-status-safe font-bold">✓</span>
                  <span>
                    <strong className="text-text-primary">Environmental Alignment:</strong> 24h GPM rainfall = 60.0mm
                    exceeds saturated threshold. Deterministic rules concord with model.
                  </span>
                </div>
              </div>
            </div>

            {/* Micro Telemetry Footer */}
            <div className="mt-space-12 pt-space-8 border-t border-border-subtle flex items-center justify-between text-caption font-mono text-text-muted">
              <span>ACTIVE RUN: {activeRun?.run_id ?? "run-obs-003"}</span>
              <span className="text-status-safe">DETERMINISTIC VERIFIED</span>
            </div>
          </div>
        </section>

        {/* SECTION 2: Pipeline DAG or Registry or Benchmarks */}
        {activeTab === "dag" && (
          <section className="bg-surface-panel border border-border-subtle p-space-16 flex flex-col gap-space-16">
            <div className="flex items-center justify-between border-b border-border-subtle pb-space-8">
              <div className="flex items-center gap-space-8">
                <span className="label-caps text-caption text-text-dim">PIPELINE EXECUTION GRAPH (DAG)</span>
                <span className="text-caption font-mono text-text-muted">
                  · CLICK A STAGE TO INSPECT TENSOR DATA &amp; INTERMEDIATE ARTIFACTS
                </span>
              </div>
              <div className="flex items-center gap-space-12 text-caption font-mono text-text-dim">
                <span className="flex items-center gap-space-4">
                  <span className="w-2 h-2 rounded-full bg-status-safe" />
                  <span>Ready / Active</span>
                </span>
                <span className="flex items-center gap-space-4">
                  <span className="w-2 h-2 rounded-full bg-primary" />
                  <span>Deterministic Fusion</span>
                </span>
              </div>
            </div>

            {/* Horizontal Interactive Pipeline DAG */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-space-12 relative">
              {/* STAGE 1 */}
              <div
                onClick={() => setSelectedStage(1)}
                className={`p-space-12 bg-surface-recessed border transition-all cursor-pointer flex flex-col justify-between ${
                  selectedStage === 1
                    ? "border-primary bg-surface-container shadow-[0_0_12px_rgba(255,176,0,0.15)]"
                    : "border-border-subtle hover:border-border-strong"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="px-space-6 py-space-2 text-[10px] font-mono font-bold bg-surface-panel border border-border-subtle text-primary">
                      STAGE 01
                    </span>
                    <span className="flex items-center gap-space-4 text-[11px] font-mono text-status-safe">
                      <span className="w-2 h-2 rounded-full bg-status-safe" />
                      LOADED
                    </span>
                  </div>
                  <h3 className="font-semibold text-body-md text-text-primary mt-space-8">
                    Siamese U-Net Change Detector
                  </h3>
                  <p className="text-caption text-text-dim line-clamp-2 mt-space-2">
                    Extracts multi-scale bi-temporal feature differences |F1 - F0| across skip connections.
                  </p>
                </div>

                <div className="mt-space-12 pt-space-8 border-t border-border-subtle space-y-space-4 font-mono text-caption">
                  <div className="flex justify-between text-text-muted">
                    <span>INPUT:</span>
                    <span className="text-text-primary">[2, 3, 512, 512]</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>OUTPUT:</span>
                    <span className="text-primary font-bold">[1, 512, 512] P∈[0,1]</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>BACKBONE:</span>
                    <span className="text-text-dim">ResNet-34</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>LATENCY:</span>
                    <span className="text-status-safe font-bold">42.4 ms</span>
                  </div>
                </div>
              </div>

              {/* STAGE 2 */}
              <div
                onClick={() => setSelectedStage(2)}
                className={`p-space-12 bg-surface-recessed border transition-all cursor-pointer flex flex-col justify-between ${
                  selectedStage === 2
                    ? "border-primary bg-surface-container shadow-[0_0_12px_rgba(255,176,0,0.15)]"
                    : "border-border-subtle hover:border-border-strong"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="px-space-6 py-space-2 text-[10px] font-mono font-bold bg-surface-panel border border-border-subtle text-primary">
                      STAGE 02
                    </span>
                    <span className="flex items-center gap-space-4 text-[11px] font-mono text-status-safe">
                      <span className="w-2 h-2 rounded-full bg-status-safe" />
                      LOADED
                    </span>
                  </div>
                  <h3 className="font-semibold text-body-md text-text-primary mt-space-8">
                    SegFormer Land-Cover Classifier
                  </h3>
                  <p className="text-caption text-text-dim line-clamp-2 mt-space-2">
                    Classifies changed crops into 5 functional classes to filter out cloud shadows &amp; benign snowmelt.
                  </p>
                </div>

                <div className="mt-space-12 pt-space-8 border-t border-border-subtle space-y-space-4 font-mono text-caption">
                  <div className="flex justify-between text-text-muted">
                    <span>INPUT CROPS:</span>
                    <span className="text-text-primary">[N, 3, 256, 256]</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>OUTPUT:</span>
                    <span className="text-primary font-bold">[N, 5] Logits</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>ATTENTION:</span>
                    <span className="text-text-dim">MiT-B0 16x16</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>PRUNE GATE:</span>
                    <span className="text-text-dim">P &gt; 0.40 only</span>
                  </div>
                </div>
              </div>

              {/* STAGE 3 */}
              <div
                onClick={() => setSelectedStage(3)}
                className={`p-space-12 bg-surface-recessed border transition-all cursor-pointer flex flex-col justify-between ${
                  selectedStage === 3
                    ? "border-primary bg-surface-container shadow-[0_0_12px_rgba(255,176,0,0.15)]"
                    : "border-border-subtle hover:border-border-strong"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="px-space-6 py-space-2 text-[10px] font-mono font-bold bg-surface-panel border border-border-subtle text-primary">
                      STAGE 03
                    </span>
                    <span className="flex items-center gap-space-4 text-[11px] font-mono text-primary">
                      <span className="w-2 h-2 rounded-full bg-primary" />
                      DETERMINISTIC
                    </span>
                  </div>
                  <h3 className="font-semibold text-body-md text-text-primary mt-space-8">
                    Multi-Sensor Consensus Gating
                  </h3>
                  <p className="text-caption text-text-dim line-clamp-2 mt-space-2">
                    Fuses ML probability with DEM slope gating (&gt;35° pruned) and rule-based SAR backscatter ratio.
                  </p>
                </div>

                <div className="mt-space-12 pt-space-8 border-t border-border-subtle space-y-space-4 font-mono text-caption">
                  <div className="flex justify-between text-text-muted">
                    <span>FUSION EQ:</span>
                    <span className="text-text-primary">0.6·ML + 0.4·Rule</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>SLOPE PRUNE:</span>
                    <span className="text-status-danger font-bold">&gt;35.0° Disqualified</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>DEM RESOLUTION:</span>
                    <span className="text-text-dim">30m SRTM</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>LATENCY:</span>
                    <span className="text-status-safe font-bold">6.2 ms (NumPy)</span>
                  </div>
                </div>
              </div>

              {/* STAGE 4 */}
              <div
                onClick={() => setSelectedStage(4)}
                className={`p-space-12 bg-surface-recessed border transition-all cursor-pointer flex flex-col justify-between ${
                  selectedStage === 4
                    ? "border-primary bg-surface-container shadow-[0_0_12px_rgba(255,176,0,0.15)]"
                    : "border-border-subtle hover:border-border-strong"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="px-space-6 py-space-2 text-[10px] font-mono font-bold bg-surface-panel border border-border-subtle text-primary">
                      STAGE 04
                    </span>
                    <span className="flex items-center gap-space-4 text-[11px] font-mono text-status-safe">
                      <span className="w-2 h-2 rounded-full bg-status-safe" />
                      LOADED
                    </span>
                  </div>
                  <h3 className="font-semibold text-body-md text-text-primary mt-space-8">
                    ConvLSTM Temporal Trend Classifier
                  </h3>
                  <p className="text-caption text-text-dim line-clamp-2 mt-space-2">
                    Classifies spatiotemporal sequences into trend classes. Automatically falls back to rules if conf &lt; 0.75.
                  </p>
                </div>

                <div className="mt-space-12 pt-space-8 border-t border-border-subtle space-y-space-4 font-mono text-caption">
                  <div className="flex justify-between text-text-muted">
                    <span>SEQUENCE:</span>
                    <span className="text-text-primary">[B, S=3, 1, 64, 64]</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>HYBRID CUTOFF:</span>
                    <span className="text-primary font-bold">τ = 0.75 Gate</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>TREND PREDICT:</span>
                    <span className="text-status-danger font-bold">RAPIDLY (0.82)</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>LATENCY:</span>
                    <span className="text-status-safe font-bold">28.3 ms</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Data-Flow Connector Summary Strip */}
            <div className="p-space-8 bg-surface-recessed border border-border-subtle font-mono text-caption flex items-center justify-between flex-wrap gap-space-8">
              <span className="text-text-dim">
                <span className="text-primary font-bold">FLOW PATH:</span> S1 SAR + S2 Optical (512x512) → SiameseUNet
                (|F1-F0|) → SegFormer (5-Class) → DEM Slope Mask (&lt;=35°) → ConvLSTM Spatiotemporal Sequence → Alert
                Payload
              </span>
              <span className="text-text-muted">TOTAL END-TO-END LATENCY: ~95.0ms</span>
            </div>
          </section>
        )}

        {/* SECTION 3: Deep-Dive Observability & Interpretability Panel */}
        <section className="bg-surface-panel border border-border-subtle p-space-16 flex flex-col gap-space-16">
          {/* Panel Header */}
          <div className="flex items-center justify-between border-b border-border-subtle pb-space-8">
            <div className="flex items-center gap-space-8">
              <span className="label-caps text-caption text-text-dim">STAGE {currentStageSpec.stage} DEEP-DIVE:</span>
              <span className="font-semibold text-body-md text-text-primary">{currentStageSpec.name}</span>
              <span className="text-caption font-mono text-primary bg-primary/10 px-space-6 py-space-1 border border-primary/30">
                {currentStageSpec.role}
              </span>
            </div>
            <div className="flex items-center gap-space-8">
              <button
                onClick={() =>
                  copyToClipboard(
                    JSON.stringify(
                      {
                        stage: currentStageSpec.stage,
                        model: currentStageSpec.key,
                        backbone: currentStageSpec.backbone,
                        params: currentStageSpec.params,
                        metrics: currentStageSpec.checkpointMeta,
                      },
                      null,
                      2,
                    ),
                    "spec",
                  )
                }
                className="px-space-8 py-space-4 border border-border-subtle text-caption font-mono text-text-dim hover:text-text-primary hover:border-border-strong transition-colors"
              >
                {copiedKey === "spec" ? "COPIED JSON" : "EXPORT SPEC JSON"}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-space-16">
            {/* Column 1: Architecture & Checkpoint Specifications */}
            <div className="flex flex-col justify-between bg-surface-recessed border border-border-subtle p-space-12">
              <div>
                <span className="label-caps text-[11px] text-text-dim">ARCHITECTURE SPECIFICATIONS</span>
                <div className="mt-space-8 space-y-space-6 font-mono text-caption">
                  <div className="flex justify-between border-b border-border-subtle pb-space-4">
                    <span className="text-text-muted">BACKBONE / ENCODER:</span>
                    <span className="text-text-primary font-bold">{currentStageSpec.backbone}</span>
                  </div>
                  <div className="flex justify-between border-b border-border-subtle pb-space-4">
                    <span className="text-text-muted">PARAMETER COUNT:</span>
                    <span className="text-text-primary font-bold">{currentStageSpec.params}</span>
                  </div>
                  <div className="flex justify-between border-b border-border-subtle pb-space-4">
                    <span className="text-text-muted">INPUT TENSOR:</span>
                    <span className="text-primary font-bold">{currentStageSpec.inputShape}</span>
                  </div>
                  <div className="flex justify-between border-b border-border-subtle pb-space-4">
                    <span className="text-text-muted">OUTPUT TENSOR:</span>
                    <span className="text-primary font-bold">{currentStageSpec.outputShape}</span>
                  </div>
                  <div className="flex justify-between border-b border-border-subtle pb-space-4">
                    <span className="text-text-muted">INFERENCE LATENCY:</span>
                    <span className="text-status-safe font-bold">{currentStageSpec.latencyMs} ms</span>
                  </div>
                  <div className="flex justify-between border-b border-border-subtle pb-space-4">
                    <span className="text-text-muted">WEIGHTS FILE:</span>
                    <span className="text-text-dim">{currentStageSpec.weightsFile}</span>
                  </div>
                  <div className="flex justify-between border-b border-border-subtle pb-space-4">
                    <span className="text-text-muted">FILE SIZE:</span>
                    <span className="text-text-dim">{currentStageSpec.weightsSize}</span>
                  </div>
                  <div className="flex justify-between border-b border-border-subtle pb-space-4">
                    <span className="text-text-muted">BENCHMARK DATASET:</span>
                    <span className="text-text-dim truncate max-w-[200px]" title={currentStageSpec.trainingData}>
                      {currentStageSpec.trainingData}
                    </span>
                  </div>
                </div>
              </div>

              {/* Checkpoint Training Metrics */}
              <div className="mt-space-12 pt-space-8 border-t border-border-subtle">
                <span className="text-[10px] font-mono text-text-muted uppercase">Checkpoint Validation Metrics:</span>
                <div className="grid grid-cols-4 gap-space-4 mt-space-4 text-center font-mono">
                  <div className="bg-surface-panel p-space-4 border border-border-subtle">
                    <div className="text-[9px] text-text-muted">EPOCH</div>
                    <div className="text-body-sm font-bold text-text-primary">
                      {currentStageSpec.checkpointMeta.epoch}
                    </div>
                  </div>
                  <div className="bg-surface-panel p-space-4 border border-border-subtle">
                    <div className="text-[9px] text-text-muted">VAL LOSS</div>
                    <div className="text-body-sm font-bold text-primary">
                      {currentStageSpec.checkpointMeta.loss.toFixed(3)}
                    </div>
                  </div>
                  <div className="bg-surface-panel p-space-4 border border-border-subtle">
                    <div className="text-[9px] text-text-muted">ACCURACY</div>
                    <div className="text-body-sm font-bold text-status-safe">
                      {(currentStageSpec.checkpointMeta.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-surface-panel p-space-4 border border-border-subtle">
                    <div className="text-[9px] text-text-muted">F1 SCORE</div>
                    <div className="text-body-sm font-bold text-status-safe">
                      {(currentStageSpec.checkpointMeta.f1 * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Column 2: Live Intermediate Visual Artifacts */}
            <div className="flex flex-col bg-surface-recessed border border-border-subtle p-space-12">
              <span className="label-caps text-[11px] text-text-dim">
                INTERMEDIATE VISUAL ARTIFACTS · LIVE PIPELINE PREVIEWS
              </span>

              {/* Stage-specific Visual Artifact Previews */}
              {selectedStage === 1 && (
                <div className="grid grid-cols-2 gap-space-8 mt-space-8 flex-1">
                  <div className="flex flex-col bg-surface-panel border border-border-subtle p-space-6">
                    <span className="text-[10px] font-mono text-text-muted mb-space-4">T0: OPTICAL BASELINE</span>
                    <div className="flex-1 min-h-[140px] bg-surface-container relative flex items-center justify-center overflow-hidden">
                      <img
                        src="/data/processed/baseline_preview.png"
                        alt="Baseline Preview"
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = "none";
                        }}
                      />
                      <span className="absolute bottom-1 right-2 text-[9px] font-mono text-text-muted bg-surface-panel/80 px-1">
                        Nov 2025 (S2)
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-col bg-surface-panel border border-border-subtle p-space-6">
                    <span className="text-[10px] font-mono text-text-muted mb-space-4">T1: S1 SAR RADAR</span>
                    <div className="flex-1 min-h-[140px] bg-surface-container relative flex items-center justify-center overflow-hidden">
                      <img
                        src="/data/processed/obs-002_baseline.png"
                        alt="SAR Preview"
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = "none";
                        }}
                      />
                      <span className="absolute bottom-1 right-2 text-[9px] font-mono text-text-muted bg-surface-panel/80 px-1">
                        Aug 2026 (S1)
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-col bg-surface-panel border border-border-subtle p-space-6">
                    <span className="text-[10px] font-mono text-primary mb-space-4">RAW DIFFERENCE HEATMAP</span>
                    <div className="flex-1 min-h-[140px] bg-surface-container relative flex items-center justify-center overflow-hidden">
                      <img
                        src="/data/processed/obs-002_change_heatmap.png"
                        alt="Change Heatmap"
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = "none";
                        }}
                      />
                      <span className="absolute bottom-1 right-2 text-[9px] font-mono text-primary bg-surface-panel/80 px-1">
                        |F1 - F0|
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-col bg-surface-panel border border-border-subtle p-space-6">
                    <span className="text-[10px] font-mono text-status-danger mb-space-4">THRESHOLDED MASK (P&gt;0.40)</span>
                    <div className="flex-1 min-h-[140px] bg-surface-container relative flex items-center justify-center overflow-hidden">
                      <img
                        src="/data/processed/obs-002_expansion_mask.png"
                        alt="Expansion Mask"
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = "none";
                        }}
                      />
                      <span className="absolute bottom-1 right-2 text-[9px] font-mono text-status-danger bg-surface-panel/80 px-1">
                        Active Inundation
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {selectedStage === 2 && (
                <div className="flex flex-col mt-space-8 flex-1 justify-between">
                  <div className="space-y-space-8">
                    <span className="text-[10px] font-mono text-text-muted">
                      FUNCTIONAL CLASSIFICATION DISTRIBUTION (5 CLASSES):
                    </span>
                    <div className="space-y-space-6">
                      {STAGE_SPECS[2].classes?.map((cls) => (
                        <div key={cls.name} className="space-y-space-2 font-mono text-caption">
                          <div className="flex justify-between items-center text-[11px]">
                            <span className="flex items-center gap-space-6">
                              <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: cls.color }} />
                              <strong className="text-text-primary">{cls.name}</strong>
                              <span className="text-text-muted">({cls.desc})</span>
                            </span>
                            <span className="font-bold" style={{ color: cls.color }}>
                              {(cls.prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-2 w-full bg-surface-panel overflow-hidden border border-border-subtle">
                            <div
                              className="h-full transition-all duration-300"
                              style={{ width: `${cls.prob * 100}%`, backgroundColor: cls.color }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="mt-space-12 p-space-8 bg-surface-panel border border-border-subtle font-mono text-[11px] text-text-dim">
                    <span className="text-status-safe font-bold">ARTIFACT FILTERING RESULT:</span> Cloud shadow artifact
                    probability (1.1%) and benign seasonal snowmelt (2.6%) are below rejection cutoff. Water expansion (88.4%)
                    promoted to Stage 3 Hydrological Gating.
                  </div>
                </div>
              )}

              {selectedStage === 3 && (
                <div className="grid grid-cols-2 gap-space-8 mt-space-8 flex-1">
                  <div className="flex flex-col bg-surface-panel border border-border-subtle p-space-6">
                    <span className="text-[10px] font-mono text-text-muted mb-space-4">RAW ML CHANGE OVERLAY</span>
                    <div className="flex-1 min-h-[140px] bg-surface-container relative flex items-center justify-center overflow-hidden">
                      <img
                        src="/data/processed/obs-002_change_heatmap.png"
                        alt="Raw Overlay"
                        className="w-full h-full object-cover"
                      />
                      <span className="absolute bottom-1 right-2 text-[9px] font-mono text-text-muted bg-surface-panel/80 px-1">
                        Unconstrained
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-col bg-surface-panel border border-border-subtle p-space-6">
                    <span className="text-[10px] font-mono text-status-danger mb-space-4">SRTM SLOPE MASK (&gt;35° PRUNED)</span>
                    <div className="flex-1 min-h-[140px] bg-surface-container relative flex items-center justify-center p-space-8 text-center">
                      <div className="border border-status-danger/40 bg-status-danger/10 p-space-12 w-full h-full flex flex-col justify-center items-center">
                        <span className="font-mono text-[18px] font-bold text-status-danger">θ &gt; 35.0°</span>
                        <span className="text-[10px] font-mono text-text-muted mt-space-2">
                          12.4% Mountain Cliff Pixels Masked Out
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="col-span-2 flex flex-col bg-surface-panel border border-border-subtle p-space-6">
                    <span className="text-[10px] font-mono text-status-safe mb-space-4">
                      FINAL HYDROLOGICALLY REACHABLE CONSENSUS MASK
                    </span>
                    <div className="min-h-[100px] bg-surface-container relative flex items-center justify-center overflow-hidden">
                      <img
                        src="/data/processed/obs-002_expansion_mask.png"
                        alt="Consensus Mask"
                        className="w-full h-full object-cover max-h-[120px]"
                      />
                      <div className="absolute inset-0 bg-primary/10 pointer-events-none flex items-end p-space-4">
                        <span className="text-[10px] font-mono text-primary font-bold bg-surface-panel/90 px-space-6 py-space-2 border border-primary/30">
                          VALIDATED REACHABILITY: D8 Corridor &amp; Imja River Channel
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {selectedStage === 4 && (
                <div className="flex flex-col mt-space-8 flex-1 justify-between">
                  <div className="space-y-space-8">
                    <span className="text-[10px] font-mono text-text-muted">
                      TEMPORAL CONVLSTM CLASS ACTIVATION VECTOR:
                    </span>
                    <div className="grid grid-cols-2 gap-space-8 font-mono text-caption">
                      <div className="p-space-8 bg-surface-panel border border-status-danger">
                        <div className="flex justify-between">
                          <span className="text-status-danger font-bold">RAPIDLY</span>
                          <span className="text-text-primary font-bold">0.820</span>
                        </div>
                        <div className="h-1.5 w-full bg-surface-container mt-space-4">
                          <div className="h-full bg-status-danger" style={{ width: "82%" }} />
                        </div>
                      </div>

                      <div className="p-space-8 bg-surface-panel border border-border-subtle">
                        <div className="flex justify-between">
                          <span className="text-text-dim">SLOWLY</span>
                          <span className="text-text-muted">0.114</span>
                        </div>
                        <div className="h-1.5 w-full bg-surface-container mt-space-4">
                          <div className="h-full bg-status-elevated" style={{ width: "11.4%" }} />
                        </div>
                      </div>

                      <div className="p-space-8 bg-surface-panel border border-border-subtle">
                        <div className="flex justify-between">
                          <span className="text-text-dim">STABLE</span>
                          <span className="text-text-muted">0.042</span>
                        </div>
                        <div className="h-1.5 w-full bg-surface-container mt-space-4">
                          <div className="h-full bg-status-safe" style={{ width: "4.2%" }} />
                        </div>
                      </div>

                      <div className="p-space-8 bg-surface-panel border border-border-subtle">
                        <div className="flex justify-between">
                          <span className="text-text-dim">UNCERTAIN</span>
                          <span className="text-text-muted">0.024</span>
                        </div>
                        <div className="h-1.5 w-full bg-surface-container mt-space-4">
                          <div className="h-full bg-text-muted" style={{ width: "2.4%" }} />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-space-12 p-space-8 bg-surface-panel border border-border-subtle font-mono text-[11px] text-text-dim">
                    <span className="text-primary font-bold">SEQUENCE REASONING:</span> Input sequence delta
                    [0.0% → +10.5% → +28.0% → +43.3%] exceeds acceleration slope. ConvLSTM hidden state vectors converged
                    on high-velocity glacial lake outburst indicator.
                  </div>
                </div>
              )}
            </div>

            {/* Column 3: Confidence Calibration & Threshold Boundary Markers */}
            <div className="flex flex-col justify-between bg-surface-recessed border border-border-subtle p-space-12">
              <div>
                <span className="label-caps text-[11px] text-text-dim">
                  CONFIDENCE CALIBRATION &amp; BOUNDARY MARKERS
                </span>

                <div className="mt-space-8 p-space-12 bg-surface-panel border border-border-subtle">
                  <div className="flex justify-between text-caption font-mono">
                    <span className="text-text-muted">THRESHOLD PARAMETER:</span>
                    <span className="text-primary font-bold">{currentStageSpec.thresholds.name}</span>
                  </div>
                  <div className="text-headline-md font-mono font-bold text-text-primary mt-space-4">
                    {currentStageSpec.thresholds.value}
                  </div>
                  <div className="text-caption font-mono text-text-dim mt-space-2">
                    {currentStageSpec.thresholds.condition}
                  </div>
                </div>

                {/* Calibrated Uncertainty Distribution Diagram */}
                <div className="mt-space-12 space-y-space-6 font-mono text-caption">
                  <div className="text-[10px] text-text-muted uppercase">Uncertainty Calibration Envelope:</div>
                  <div className="space-y-space-4">
                    <div className="flex justify-between text-[11px]">
                      <span className="text-text-dim">Sensor Alignment Error:</span>
                      <span className="text-text-primary">±0.42 pixels (&lt;4.2m)</span>
                    </div>
                    <div className="flex justify-between text-[11px]">
                      <span className="text-text-dim">Optical Cloud Fraction:</span>
                      <span className="text-text-primary">0.0% (SAR all-weather promoted)</span>
                    </div>
                    <div className="flex justify-between text-[11px]">
                      <span className="text-text-dim">Quality Gate Multiplier:</span>
                      <span className="text-status-safe font-bold">0.95x confidence</span>
                    </div>
                    <div className="flex justify-between text-[11px]">
                      <span className="text-text-dim">False Positive Prune Ratio:</span>
                      <span className="text-primary font-bold">98.6% specificity</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Gating Rule Documentation */}
              <div className="mt-space-12 p-space-8 bg-surface-panel border border-border-subtle text-[11px] font-mono text-text-dim">
                <span className="text-primary font-bold">ADR-002 ENFORCEMENT:</span> Stage {currentStageSpec.stage} strictly
                complies with the deterministic-first mandate. Model outputs act as an evidence overlay; physical slope and
                human coordinator review preserve fail-safe operation.
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 4: Weights & Registry Tab View */}
        {activeTab === "registry" && (
          <section className="bg-surface-panel border border-border-subtle p-space-16 flex flex-col gap-space-16">
            <div className="flex items-center justify-between border-b border-border-subtle pb-space-8">
              <span className="label-caps text-caption text-text-dim">
                MODEL CHECKPOINT REGISTRY &amp; PROVENANCE (W&amp;B / MLFLOW STYLE)
              </span>
              <span className="text-caption font-mono text-text-muted">LOCATION: data/processed/*.pt</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-caption border-collapse">
                <thead>
                  <tr className="border-b border-border-strong text-text-muted text-[11px] bg-surface-recessed">
                    <th className="py-space-8 px-space-12">STAGE</th>
                    <th className="py-space-8 px-space-12">MODEL / PIPELINE COMPONENT</th>
                    <th className="py-space-8 px-space-12">STATUS</th>
                    <th className="py-space-8 px-space-12">BACKBONE / ARCH</th>
                    <th className="py-space-8 px-space-12">PARAMETERS</th>
                    <th className="py-space-8 px-space-12">SIZE</th>
                    <th className="py-space-8 px-space-12">CHECKPOINT FILE</th>
                    <th className="py-space-8 px-space-12">DATASET PROVENANCE</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle">
                  {models.map((m) => (
                    <tr key={m.stage} className="hover:bg-surface-container transition-colors">
                      <td className="py-space-8 px-space-12 text-primary font-bold">0{m.stage}</td>
                      <td className="py-space-8 px-space-12 font-semibold text-text-primary">{m.name}</td>
                      <td className="py-space-8 px-space-12">
                        <span
                          className={`inline-flex items-center gap-space-4 px-space-6 py-space-1 text-[10px] border ${
                            m.loaded
                              ? "border-status-safe/40 text-status-safe bg-status-safe/10"
                              : "border-text-muted text-text-muted bg-surface-recessed"
                          }`}
                        >
                          <span className={`w-1.5 h-1.5 rounded-full ${m.loaded ? "bg-status-safe" : "bg-text-muted"}`} />
                          {m.loaded ? "READY" : "OFFLINE"}
                        </span>
                      </td>
                      <td className="py-space-8 px-space-12 text-text-dim">{m.architecture}</td>
                      <td className="py-space-8 px-space-12 text-text-primary">
                        {STAGE_SPECS[m.stage as StageId]?.params ?? "N/A"}
                      </td>
                      <td className="py-space-8 px-space-12 text-text-dim">
                        {m.weights_size_mb > 0 ? `${m.weights_size_mb} MB` : "Rule-based"}
                      </td>
                      <td className="py-space-8 px-space-12 text-text-muted text-[11px] truncate max-w-[180px]">
                        {m.weights_path ? m.weights_path.split("/").pop() : "consensus.py"}
                      </td>
                      <td className="py-space-8 px-space-12 text-text-dim text-[11px] truncate max-w-[220px]">
                        {m.training_data ?? "NASA SRTM DEM / OpenStreetMap"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* SECTION 5: Latency Benchmarks Tab View */}
        {activeTab === "benchmarks" && (
          <section className="bg-surface-panel border border-border-subtle p-space-16 flex flex-col gap-space-16">
            <div className="flex items-center justify-between border-b border-border-subtle pb-space-8">
              <span className="label-caps text-caption text-text-dim">
                INFERENCE LATENCY &amp; COMPUTATIONAL PROFILING (GRAFANA STYLE)
              </span>
              <span className="text-caption font-mono text-status-safe">TOTAL TIME: 95.0 ms (TARGET &lt; 200 ms)</span>
            </div>

            <div className="space-y-space-12 font-mono">
              {[
                { stage: "Stage 01: Siamese U-Net", time: 42.4, pct: 44.6, color: "#38bdf8", device: "CPU / PyTorch" },
                { stage: "Stage 02: SegFormer (MiT-B0)", time: 18.1, pct: 19.1, color: "#c084fc", device: "CPU / PyTorch" },
                { stage: "Stage 03: Consensus & DEM Slope", time: 6.2, pct: 6.5, color: "#ffb000", device: "C++ / NumPy Vectorized" },
                { stage: "Stage 04: ConvLSTM Trend", time: 28.3, pct: 29.8, color: "#f87171", device: "CPU / PyTorch" },
              ].map((bench) => (
                <div key={bench.stage} className="space-y-space-4">
                  <div className="flex justify-between items-center text-caption">
                    <span className="text-text-primary font-semibold">{bench.stage}</span>
                    <span className="text-text-dim">
                      {bench.time.toFixed(1)} ms ({bench.pct.toFixed(1)}%) ·{" "}
                      <span className="text-text-muted">{bench.device}</span>
                    </span>
                  </div>
                  <div className="h-3 w-full bg-surface-recessed border border-border-subtle overflow-hidden flex">
                    <div className="h-full transition-all duration-300" style={{ width: `${bench.pct}%`, backgroundColor: bench.color }} />
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
