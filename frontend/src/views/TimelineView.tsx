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
    <div className="flex flex-col h-full">
      {/* Header bar */}
      <div className="flex items-center justify-between px-space-16 py-space-8 border-b border-border-subtle bg-surface-panel">
        <div className="flex items-center gap-space-12">
          <h1 className="label-caps">Telemetry Sequence</h1>
          <span className="data-val text-body-sm text-text-dim">BASIN-DK-IMJA-04</span>
        </div>
        <div className="flex items-center gap-space-8">
          <span className="data-val text-body-sm text-text-dim">STEP</span>
          <span className="data-val text-body-md text-text-primary">{currentStepIdx}/{totalSteps}</span>
        </div>
      </div>

      {/* Simulation controller */}
      <div className="bg-surface-panel border-b border-border-subtle px-space-16 py-space-12 flex items-center gap-space-16">
        <button
          onClick={runSim}
          disabled={isRunning}
          className="bg-surface-container text-text-primary data-val text-body-md px-space-12 py-space-6 border border-border-strong hover:bg-surface-container-high transition-colors flex items-center gap-space-6 whitespace-nowrap select-none cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <span>{isComplete ? "REPLAY" : isRunning ? "RUNNING" : "RUN"}</span>
        </button>
        <div className="flex-1 h-[2px] bg-border-subtle overflow-hidden relative">
          <div
            className="h-full bg-primary-container transition-all duration-300"
            style={{ width: `${(currentStepIdx / totalSteps) * 100}%` }}
          />
        </div>
      </div>

      {/* Prevention callout — semantic state only */}
      {currentStepIdx >= 2 && warningDays !== null && (
        <div className="bg-surface-panel border-l-2 border-l-status-safe border-b border-border-subtle px-space-16 py-space-12 flex items-center justify-between">
          <div className="flex items-center gap-space-8">
            <span className="data-val text-body-md text-status-safe">EARLY WARNING</span>
            <span className="text-body-md text-text-primary">
              <span className="data-val text-status-safe">{warningDays} days</span> between OBS-01 (
              <span className="data-val">+{obs1?.water_area_change_percent?.toFixed(1)}%</span>) and OBS-02 (
              <span className="data-val">+{obs2?.water_area_change_percent?.toFixed(1)}%</span>)
            </span>
          </div>
          <span className="data-val text-body-sm text-text-dim">
            DELTA T+{warningDays * 24}H
          </span>
        </div>
      )}

      {/* Router strip — SAR fallback indicator */}
      {timeline.map((obs, i) => {
        if (i === 0) return null;
        const isSAR = obs.source.includes("sentinel-1") || obs.source.includes("sar");
        if (isSAR && i <= currentStepIdx) {
          return (
            <div
              key={`router-${obs.observation_id}`}
              className="bg-surface-panel border-b border-border-subtle px-space-16 py-space-8 flex items-center gap-space-8 select-none"
            >
              <span className="data-val text-body-sm text-text-dim">OBS-{String(i).padStart(2, "0")}:</span>
              <span className="data-val text-body-sm text-text-primary">{((obs.cloud_fraction ?? 1) * 100).toFixed(0)}% cloud</span>
              <span className="data-val text-body-sm text-primary-container">-&gt;</span>
              <span className="data-val text-body-sm text-text-primary">SAR path</span>
              <span className="text-border-subtle mx-space-4">|</span>
              <span className="data-val text-body-sm text-text-dim">C-Band SAR fallback engaged</span>
            </div>
          );
        }
        return null;
      })}

      {/* Observation cards — dense data grid */}
      <div className="flex-1 overflow-auto p-space-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-space-8">
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
                className={`bg-surface-panel border flex flex-col transition-colors ${
                  isActive ? "border-primary" : "border-border-subtle hover:border-border-strong"
                } ${isGreyed ? "opacity-40 pointer-events-none" : "cursor-pointer"}`}
              >
                {/* Card header */}
                <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
                  <span className="data-val text-body-md text-text-primary font-medium">
                    {i === 0 ? "BASELINE" : `OBS-${String(i).padStart(2, "0")}`}
                  </span>
                  <span className={`data-val text-body-sm border px-space-4 py-space-1 ${SEVERITY_BADGE[sev]}`}>
                    {sev.toUpperCase()}
                  </span>
                </div>

                {/* Card body — telemetry table */}
                <div className="px-space-12 py-space-8 flex flex-col gap-space-2 data-val text-body-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-text-dim">DATE</span>
                    <span className="text-text-primary">{dateStr}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-dim">SOURCE</span>
                    <span className={isSAR ? "text-primary-container" : "text-status-info"}>
                      {isSAR ? "S1 SAR" : "S2 OPTICAL"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-dim">CLOUD</span>
                    <span className="text-text-primary">
                      {obs.cloud_fraction !== null ? `${(obs.cloud_fraction * 100).toFixed(0)}%` : "N/A"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-dim">RAIN 24H</span>
                    <span className="text-text-primary">
                      {obs.rainfall_24h_mm != null ? `${obs.rainfall_24h_mm.toFixed(1)} mm` : "N/A"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-dim">AREA</span>
                    <span className={`text-text-primary ${isActive ? "text-primary-container" : ""}`}>
                      {obs.water_area_km2 != null ? `${obs.water_area_km2.toFixed(2)} km²` : "N/A"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between border-t border-border-subtle pt-space-2 mt-space-2">
                    <span className="text-text-dim">CHANGE</span>
                    <span className={pct > 0 ? "text-status-warn" : "text-text-primary"}>
                      {pct > 0 ? "+" : ""}{pct.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
