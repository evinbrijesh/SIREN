import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation, type SimStep } from "../simulation/SimulationContext";
import type { ObservationList, Observation } from "../api/types";

const SEVERITY_BADGE: Record<string, string> = {
  informational: "border-status-safe text-status-safe",
  watch: "border-status-warn text-status-warn",
  elevated: "border-status-elevated text-status-elevated",
  critical: "border-status-danger text-status-danger",
};

function severityFromObs(obs: Observation): string {
  const pct = obs.water_area_change_percent ?? 0;
  if (pct >= 30) return "critical";
  if (pct >= 10) return "elevated";
  if (pct > 0) return "watch";
  return "informational";
}

const STEP_TO_OBS: Record<SimStep, number> = { before: 0, "obs-1": 1, "obs-2": 2, "obs-3": 3 };

export default function TimelineView() {
  const sim = useSimulation();
  const [isRunning, setIsRunning] = useState(false);

  const { data: obsData } = useQuery({
    queryKey: ["observations"],
    queryFn: () => apiOrMock(() => api.listObservations(), "observations") as Promise<ObservationList>,
  });

  const observations = obsData?.observations ?? mockData.observations.observations;
  const timeline = [...observations].reverse();

  const runSim = async () => {
    if (sim.status === "complete") {
      sim.reset();
      return;
    }
    setIsRunning(true);
    for (let i = 0; i < 3; i++) {
      await sim.advance();
      await new Promise((r) => setTimeout(r, 600));
    }
    setIsRunning(false);
  };

  const obs1 = timeline[1];
  const obs2 = timeline[2];
  let warningDays: number | null = null;
  if (obs1 && obs2) {
    const d1 = new Date(obs1.acquired_at).getTime();
    const d2 = new Date(obs2.acquired_at).getTime();
    warningDays = Math.round((d2 - d1) / (1000 * 60 * 60 * 24));
  }

  const currentStepIdx = STEP_TO_OBS[sim.step];
  const isComplete = sim.status === "complete";
  const totalSteps = timeline.length - 1;

  return (
    <div className="flex flex-col gap-space-16 h-full">
      <div className="flex items-center justify-between">
        <h1 className="text-headline-lg text-text-primary font-medium tracking-tight">Timeline</h1>
        <div className="flex items-center gap-space-8">
          <span className="text-caption text-text-dim uppercase tracking-wider">Telemetry Sequence</span>
          <span className="font-mono text-code-sm text-primary-container">BASIN-DK-IMJA-04</span>
        </div>
      </div>

      {/* Simulation controller */}
      <div className="bg-surface-panel border border-border-subtle p-space-16 rounded-xl flex items-center gap-space-16">
        <button
          onClick={runSim}
          disabled={isRunning}
          className="bg-primary-container text-surface-canvas font-medium text-body-md rounded-lg px-space-16 py-space-8 hover:brightness-110 active:scale-[0.98] transition-all flex items-center gap-space-6 whitespace-nowrap select-none cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <span>{isComplete ? "↻" : isRunning ? "⏸" : "▶"}</span>
          <span>{isComplete ? "Replay" : isRunning ? "Running..." : "Run Simulation"}</span>
        </button>
        <span className="font-mono text-code-lg text-text-dim tracking-tight select-none">
          {currentStepIdx}/{totalSteps}
        </span>
        <div className="flex-1 h-[4px] bg-border-subtle rounded-full overflow-hidden relative">
          <div
            className="h-full bg-primary-container transition-all duration-300 rounded-full"
            style={{ width: `${(currentStepIdx / totalSteps) * 100}%` }}
          />
        </div>
      </div>

      {/* Prevention callout */}
      {currentStepIdx >= 2 && warningDays !== null && (
        <div className="bg-surface-panel border-l-[3px] border-l-status-safe border border-border-subtle p-space-16 rounded-xl flex items-center justify-between">
          <div className="flex items-center gap-space-8 text-body-lg">
            <span className="text-status-safe font-semibold leading-none select-none">★</span>
            <span className="text-text-primary">
              <span className="text-status-safe font-medium">{warningDays} days</span> of early warning between Obs 1 (
              <span className="text-status-safe font-medium">+{obs1?.water_area_change_percent?.toFixed(0)}%</span>) and Obs 2 (
              <span className="text-status-safe font-medium">+{obs2?.water_area_change_percent?.toFixed(0)}%</span>)
            </span>
          </div>
          <span className="font-mono text-code-sm text-text-dim hidden md:inline-block">
            DELTA: ΔT+{warningDays * 24}H
          </span>
        </div>
      )}

      {/* Router strip */}
      {timeline.map((obs, i) => {
        if (i === 0) return null;
        const isSAR = obs.source.includes("sentinel-1") || obs.source.includes("sar");
        if (isSAR && i <= currentStepIdx) {
          return (
            <div
              key={`router-${obs.observation_id}`}
              className="bg-surface-panel border border-border-subtle p-space-12 rounded-xl flex items-center gap-space-8 select-none"
            >
              <span className="text-body-lg text-text-dim font-medium">Obs {i}:</span>
              <span className="text-body-lg text-text-primary">{(obs.cloud_fraction ?? 1) * 100}% cloud</span>
              <span className="text-body-lg text-primary-container font-semibold">→</span>
              <span className="text-body-lg text-text-primary">SAR path</span>
              <span className="text-border-subtle mx-space-4">|</span>
              <span className="font-mono text-code-sm text-text-dim">C-Band Synthetic Aperture Radar fallback engaged</span>
            </div>
          );
        }
        return null;
      })}

      {/* Observation cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-space-12">
        {timeline.map((obs, i) => {
          const sev = severityFromObs(obs);
          const pct = obs.water_area_change_percent ?? 0;
          const isSAR = obs.source.includes("sentinel-1") || obs.source.includes("sar");
          const isActive = i === currentStepIdx;
          const isGreyed = i > currentStepIdx;
          const cardStep: SimStep = i === 0 ? "before" : (`obs-${i}` as SimStep);
          const dateStr = new Date(obs.acquired_at).toISOString().slice(0, 10);

          return (
            <div
              key={obs.observation_id}
              onClick={() => !isGreyed && sim.scrubTo(cardStep)}
              className={`bg-surface-panel border rounded-xl p-space-16 flex flex-col justify-between gap-space-16 transition-colors ${
                isActive ? "border-2 border-primary-container" : "border-border-subtle hover:border-text-dim/40"
              } ${isGreyed ? "opacity-40 pointer-events-none" : "cursor-pointer"}`}
            >
              <div className="flex flex-col gap-space-8">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-space-6">
                    <span className="text-headline-sm text-text-primary font-medium">
                      {i === 0 ? "Baseline" : `Obs ${i}`}
                    </span>
                    {isActive && <span className="w-2 h-2 rounded-full bg-primary-container animate-pulse" />}
                  </div>
                  <span className={`inline-block px-space-8 py-space-2 rounded text-caption uppercase tracking-wider bg-transparent border ${SEVERITY_BADGE[sev]}`}>
                    {sev}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-code-sm text-text-dim">{dateStr}</span>
                  <span className={`font-mono text-code-sm ${isSAR ? "text-primary-container font-medium" : "text-status-info"}`}>
                    {isSAR ? "S1 SAR" : "S2 Optical"}
                  </span>
                </div>
              </div>

              <div className="bg-surface-recessed border border-border-subtle rounded-lg p-space-12 flex flex-col gap-space-8 font-mono text-code-sm">
                <div className="flex items-center justify-between">
                  <span className="text-text-dim">Cloud</span>
                  <span className="text-text-primary">
                    {obs.cloud_fraction !== null ? `${(obs.cloud_fraction * 100).toFixed(0)}%` : "—"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-dim">Rain 24h</span>
                  <span className="text-text-primary">{obs.rainfall_24h_mm ?? "—"}mm</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-dim">Area</span>
                  <span className={`text-text-primary font-medium ${isActive ? "text-primary-container" : ""}`}>
                    {obs.water_area_km2 ?? "—"} km²
                  </span>
                </div>
                <div className="flex items-center justify-between border-t border-border-subtle/50 pt-space-6 mt-space-2">
                  <span className="text-text-dim">Change</span>
                  <span className={`font-medium ${pct > 0 ? "text-status-warn" : "text-text-primary"}`}>
                    {pct > 0 ? "▲ " : ""}+{pct.toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
