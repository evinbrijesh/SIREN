import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { canonicalBaseline, mockData } from "../api/mockData";
import { useSimulation, type SimStep } from "../simulation/SimulationContext";
import type { ObservationList, Observation } from "../api/types";

const SEVERITY_BADGE: Record<string, string> = {
  safe: "border-status-safe text-status-safe",
  advisory: "border-status-warn text-status-warn",
  elevated: "border-status-elevated text-status-elevated",
  critical: "border-status-danger text-status-danger",
};

const BASELINE: Observation = {
  observation_id: "baseline",
  basin_id: "dudh-koshi-demo-01",
  acquired_at: canonicalBaseline.acquired_at,
  source: canonicalBaseline.source,
  raster_uri: "data/processed/basemap.tif",
  crs: "EPSG:4326",
  quality_score: 0.95,
  cloud_fraction: canonicalBaseline.cloud_fraction,
  optical_cloud_fraction: canonicalBaseline.cloud_fraction,
  alignment_ok: true,
  usable: true,
  confidence_adjustment: 1,
  water_area_km2: canonicalBaseline.water_area_km2,
  water_area_change_percent: canonicalBaseline.water_area_change_percent,
  rainfall_24h_mm: canonicalBaseline.rainfall_24h_mm,
  rainfall_7d_mm: 0,
  mean_slope_degrees: 31,
  processing_version: "0.1.0",
  status: "processed",
};

function severityFromObs(obs: Observation): string {
  if (obs.observation_id === "baseline") return "safe";
  if (obs.observation_id === "obs-001") return "advisory";
  if (obs.observation_id === "obs-002") return "elevated";
  return "critical";
}

function Sparkline({ timeline, progress }: { timeline: Observation[]; progress: number }) {
  const visible = timeline.slice(0, progress + 1);
  if (visible.length < 2) return null;

  const width = 600;
  const height = 120;
  const padding = { top: 16, right: 16, bottom: 24, left: 36 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const areas = visible.map((o) => o.water_area_km2 ?? 0);
  const rains = visible.map((o) => o.rainfall_24h_mm ?? 0);
  const maxArea = Math.max(...areas, 4);
  const maxRain = Math.max(...rains, 1);
  const minArea = Math.min(...areas, 0);

  const xStep = visible.length > 1 ? plotW / (visible.length - 1) : 0;
  const x = (i: number) => padding.left + i * xStep;
  const yArea = (v: number) => padding.top + plotH - ((v - minArea) / (maxArea - minArea)) * plotH;
  const yRain = (v: number) => padding.top + plotH - (v / maxRain) * plotH;

  const areaPath = visible.map((o, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${yArea(o.water_area_km2 ?? 0)}`).join(" ");
  const barWidth = xStep * 0.4;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <line key={t} x1={padding.left} x2={width - padding.right} y1={padding.top + t * plotH} y2={padding.top + t * plotH} stroke="var(--color-border-subtle)" strokeWidth="0.5" />
      ))}
      {/* Rainfall bars (cyan) */}
      {visible.map((o, i) => {
        const rainVal = o.rainfall_24h_mm ?? 0;
        const barH = (rainVal / maxRain) * plotH;
        return <rect key={`rain-${i}`} x={x(i) - barWidth / 2} y={padding.top + plotH - barH} width={barWidth} height={barH} fill="var(--color-info)" opacity="0.35" />;
      })}
      {/* Water area line (amber) */}
      <path d={areaPath} fill="none" stroke="var(--color-primary)" strokeWidth="2" />
      {/* Data points */}
      {visible.map((o, i) => (
        <circle key={`pt-${i}`} cx={x(i)} cy={yArea(o.water_area_km2 ?? 0)} r="3" fill="var(--color-primary)" />
      ))}
      {/* X-axis labels */}
      {visible.map((o, i) => {
        const date = new Date(o.acquired_at).toISOString().slice(5, 10);
        return <text key={`x-${i}`} x={x(i)} y={height - 6} textAnchor="middle" fontSize="9" fill="var(--color-text-dim)" fontFamily="monospace">{date}</text>;
      })}
      {/* Y-axis labels */}
      <text x={padding.left - 4} y={padding.top + 4} textAnchor="end" fontSize="8" fill="var(--color-primary)" fontFamily="monospace">{maxArea.toFixed(1)}</text>
      <text x={padding.left - 4} y={padding.top + plotH} textAnchor="end" fontSize="8" fill="var(--color-primary)" fontFamily="monospace">{minArea.toFixed(1)}</text>
      <text x={width - padding.right} y={padding.top + 10} textAnchor="end" fontSize="8" fill="var(--color-info)" fontFamily="monospace">rain mm</text>
      <text x={width - padding.right} y={padding.top + 22} textAnchor="end" fontSize="8" fill="var(--color-primary)" fontFamily="monospace">area km²</text>
    </svg>
  );
}

export default function TimelineView() {
  const sim = useSimulation();
  const [error, setError] = useState<string | null>(null);

  const { data: obsData } = useQuery({
    queryKey: ["observations"],
    queryFn: () => apiOrMock(() => api.listObservations(), "observations") as Promise<ObservationList>,
  });

  const observations = obsData?.observations ?? mockData.observations.observations;
  const timeline = [BASELINE, ...[...observations].sort((a, b) => a.acquired_at.localeCompare(b.acquired_at))];

  const runSim = async () => {
    setError(null);
    if (sim.status === "complete") sim.reset();
    try {
      await sim.runAll();
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Simulation failed");
    }
  };

  const obs1 = timeline[1];
  const obs2 = timeline[2];
  const warningDays = obs1 && obs2
    ? Math.round((new Date(obs2.acquired_at).getTime() - new Date(obs1.acquired_at).getTime()) / 86_400_000)
    : null;
  const isComplete = sim.status === "complete";
  const isRunning = sim.status === "running";

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-space-16 py-space-8 border-b border-border-subtle bg-surface-panel">
        <h1 className="label-caps">Timeline</h1>
        <span className="data-val text-body-sm text-text-dim">{sim.progress}/3 observations</span>
      </div>

      <div className="bg-surface-panel border-b border-border-subtle px-space-16 py-space-12 flex items-center gap-space-16">
        <button
          onClick={runSim}
          disabled={isRunning}
          className="bg-surface-container text-text-primary text-body-md px-space-12 py-space-6 border border-border-strong hover:bg-surface-container-high transition-colors whitespace-nowrap disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {isComplete ? "Replay" : isRunning ? "Running..." : "Run simulation"}
        </button>
        <div className="flex-1 flex items-center gap-space-4">
          <span className="text-caption text-text-dim whitespace-nowrap">SEQUENCE</span>
          <div className="flex-1 h-[2px] bg-border-subtle overflow-hidden">
            <div className="h-full bg-primary-container transition-all duration-100" style={{ width: `${(sim.progress / 3) * 100}%` }} />
          </div>
        </div>
        {error && <span className="data-val text-body-sm text-status-danger">{error}</span>}
      </div>

      {sim.progress >= 2 && warningDays !== null && (
        <div className="bg-surface-panel border-l-2 border-l-status-safe border-b border-border-subtle px-space-16 py-space-12 flex items-center gap-space-12">
          <span className="text-status-safe text-body-md">★</span>
          <div className="flex items-center gap-space-8">
            <span className="text-body-md text-status-safe font-medium">Early warning window</span>
            <span className="text-body-md text-text-primary">
              <span className="data-val text-status-safe">{warningDays} days</span> between obs-01 and obs-02 — surge detected before critical expansion
            </span>
          </div>
        </div>
      )}

      {/* Sensor failover router strip — styled tactical chips */}
      {timeline.slice(1).map((obs, index) => {
        const stepNumber = index + 1;
        const opticalCloud = obs.optical_cloud_fraction ?? obs.cloud_fraction ?? 0;
        if (opticalCloud < 0.20 || stepNumber > sim.progress) return null;
        return (
          <div key={`router-${obs.observation_id}`} className="bg-surface-panel border-b border-border-subtle px-space-16 py-space-8 flex items-center gap-space-8">
            <span className="text-caption text-text-dim">Obs {stepNumber}</span>
            <span className="text-caption border border-status-danger text-status-danger px-space-6 py-space-2">
              Cloud {(opticalCloud * 100).toFixed(0)}%
            </span>
            <span className="text-caption text-text-dim">→</span>
            <span className="text-caption border border-primary text-primary px-space-6 py-space-2 bg-primary/10">
              SAR failover
            </span>
          </div>
        );
      })}

      <div className="flex-1 overflow-auto p-space-12 space-y-space-12">
        {/* Observation cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-space-8">
          {timeline.map((obs, index) => {
            const severity = severityFromObs(obs);
            const pct = obs.water_area_change_percent ?? 0;
            const isSAR = obs.source.includes("sentinel-1") || obs.source.includes("sar");
            const isActive = index === sim.progress;
            const isUnavailable = index > sim.progress;
            const cardStep: SimStep = index === 0 ? "before" : (`obs-${index}` as SimStep);
            const dateStr = new Date(obs.acquired_at).toISOString().slice(0, 10);
            const thumbUri = index > 0 ? `/data/processed/${obs.observation_id}_expansion_mask.png` : null;

            return (
              <button
                key={obs.observation_id}
                onClick={() => !isUnavailable && sim.scrubTo(cardStep)}
                disabled={isUnavailable}
                className={`text-left bg-surface-panel border flex flex-col transition-colors ${
                  isActive ? "border-primary" : "border-border-subtle hover:border-border-strong"
                } ${isUnavailable ? "opacity-40 cursor-not-allowed" : "cursor-pointer"}`}
              >
                <div className="w-full flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
                  <span className="data-val text-body-md text-text-primary font-medium">
                    {index === 0 ? "BASELINE" : `OBS-${String(index).padStart(2, "0")}`}
                  </span>
                  <span className={`data-val text-body-sm border px-space-4 py-space-1 ${SEVERITY_BADGE[severity]}`}>
                    {severity.toUpperCase()}
                  </span>
                </div>
                {/* Thumbnail */}
                {thumbUri && (
                  <div className="w-full h-[48px] bg-surface-canvas border-b border-border-subtle overflow-hidden flex items-center justify-center">
                    <img src={thumbUri} alt={`${obs.observation_id} water mask`} className="h-full w-full object-contain opacity-70" />
                  </div>
                )}
                <div className="w-full px-space-12 py-space-8 flex flex-col gap-space-2 data-val text-body-sm">
                  <Metric label="DATE" value={dateStr} />
                  <Metric label="SOURCE" value={isSAR ? "S1 SAR" : "S2 OPTICAL"} valueClass={isSAR ? "text-primary-container" : "text-status-info"} />
                  <Metric label="CLOUD" value={`${((obs.cloud_fraction ?? 0) * 100).toFixed(0)}%`} />
                  {obs.optical_cloud_fraction !== null && obs.optical_cloud_fraction !== obs.cloud_fraction && (
                    <Metric label="OPT CLOUD" value={`${(obs.optical_cloud_fraction * 100).toFixed(0)}%`} valueClass="text-status-warn" />
                  )}
                  <Metric label="RAIN 24H" value={`${(obs.rainfall_24h_mm ?? 0).toFixed(1)} mm`} />
                  <Metric label="AREA" value={`${(obs.water_area_km2 ?? 0).toFixed(2)} km²`} valueClass={isActive ? "text-primary-container" : undefined} />
                  <div className="flex items-center justify-between border-t border-border-subtle pt-space-2 mt-space-2">
                    <span className="text-text-dim">CHANGE</span>
                    <span className={pct > 0 ? "text-status-warn" : "text-text-primary"}>{pct > 0 ? "+" : ""}{pct.toFixed(1)}%</span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Dual-axis sparkline — fills the lower void */}
        {sim.progress >= 1 && (
          <div className="bg-surface-panel border border-border-subtle">
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h2 className="label-caps">Trend Analysis</h2>
              <div className="flex items-center gap-space-12 text-caption">
                <span className="flex items-center gap-space-4">
                  <span className="inline-block w-space-4 h-[2px] bg-primary" />
                  <span className="text-text-dim">Water area km²</span>
                </span>
                <span className="flex items-center gap-space-4">
                  <span className="inline-block w-space-4 h-space-4 bg-info opacity-40" />
                  <span className="text-text-dim">Rainfall mm</span>
                </span>
              </div>
            </div>
            <div className="p-space-12">
              <Sparkline timeline={timeline} progress={sim.progress} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Metric({ label, value, valueClass = "text-text-primary" }: { label: string; value: string; valueClass?: string }) {
  return <div className="flex items-center justify-between"><span className="text-text-dim">{label}</span><span className={valueClass}>{value}</span></div>;
}
