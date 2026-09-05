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
        <span className="data-val text-body-sm text-text-dim">{sim.progress}/3</span>
      </div>

      <div className="bg-surface-panel border-b border-border-subtle px-space-16 py-space-12 flex items-center gap-space-16">
        <button
          onClick={runSim}
          disabled={isRunning}
          className="bg-surface-container text-text-primary text-body-md px-space-12 py-space-6 border border-border-strong hover:bg-surface-container-high transition-colors whitespace-nowrap disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {isComplete ? "Replay" : isRunning ? "Running..." : "Run simulation"}
        </button>
        <div className="flex-1 h-[2px] bg-border-subtle overflow-hidden">
          <div className="h-full bg-primary-container transition-all duration-100" style={{ width: `${(sim.progress / 3) * 100}%` }} />
        </div>
        {error && <span className="data-val text-body-sm text-status-danger">{error}</span>}
      </div>

      {sim.progress >= 2 && warningDays !== null && (
        <div className="bg-surface-panel border-l-2 border-l-status-safe border-b border-border-subtle px-space-16 py-space-12 flex items-center justify-between">
          <div className="flex items-center gap-space-8">
            <span className="text-body-md text-status-safe">Early warning</span>
            <span className="text-body-md text-text-primary">
              <span className="data-val text-status-safe">{warningDays} days</span> between obs-01 and obs-02
            </span>
          </div>
        </div>
      )}

      {timeline.slice(1).map((obs, index) => {
        const stepNumber = index + 1;
        const opticalCloud = obs.optical_cloud_fraction ?? obs.cloud_fraction ?? 0;
        if (opticalCloud < 0.20 || stepNumber > sim.progress) return null;
        return (
          <div key={`router-${obs.observation_id}`} className="bg-surface-panel border-b border-border-subtle px-space-16 py-space-8 flex items-center gap-space-8">
            <span className="text-body-sm text-text-dim">Obs {stepNumber}:</span>
            <span className="text-body-sm text-text-primary">Cloud {(opticalCloud * 100).toFixed(0)}% → SAR path</span>
          </div>
        );
      })}

      <div className="flex-1 overflow-auto p-space-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-space-8">
          {timeline.map((obs, index) => {
            const severity = severityFromObs(obs);
            const pct = obs.water_area_change_percent ?? 0;
            const isSAR = obs.source.includes("sentinel-1") || obs.source.includes("sar");
            const isActive = index === sim.progress;
            const isUnavailable = index > sim.progress;
            const cardStep: SimStep = index === 0 ? "before" : (`obs-${index}` as SimStep);
            const dateStr = new Date(obs.acquired_at).toISOString().slice(0, 10);

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
      </div>
    </div>
  );
}

function Metric({ label, value, valueClass = "text-text-primary" }: { label: string; value: string; valueClass?: string }) {
  return <div className="flex items-center justify-between"><span className="text-text-dim">{label}</span><span className={valueClass}>{value}</span></div>;
}
