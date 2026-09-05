import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation, type SimStep } from "../simulation/SimulationContext";
import type { ObservationList, Observation } from "../api/types";

const SEVERITY_BADGE: Record<string, string> = {
  informational: "badge-info",
  watch: "badge-warn",
  elevated: "badge-elevated",
  critical: "badge-danger",
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
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <h1 className="view-title">Timeline</h1>

      {/* Simulation controller */}
      <div className="card" style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <button className="btn btn-primary" onClick={runSim} disabled={isRunning}>
          {isComplete ? "↻ Replay" : isRunning ? "Running..." : "▶ Run Simulation"}
        </button>
        <span style={{ fontSize: 14, color: "var(--text-dim)", fontFamily: "JetBrains Mono, monospace" }}>
          {currentStepIdx}/{totalSteps}
        </span>
        <div style={{ flex: 1, maxWidth: 240, height: 4, background: "var(--panel-2)", borderRadius: 4, overflow: "hidden" }}>
          <div
            style={{
              width: `${(currentStepIdx / totalSteps) * 100}%`,
              height: "100%",
              background: "var(--accent)",
              transition: "width 0.4s",
            }}
          />
        </div>
      </div>

      {/* Prevention callout */}
      {currentStepIdx >= 2 && warningDays !== null && (
        <div className="callout">
          <span className="star">★</span>
          <span>
            <span className="highlight">{warningDays} days</span> of early warning between Obs 1 (+{obs1?.water_area_change_percent?.toFixed(0)}%) and Obs 2 (+{obs2?.water_area_change_percent?.toFixed(0)}%)
          </span>
        </div>
      )}

      {/* Router strip — only for SAR-routed observations */}
      {timeline.map((obs, i) => {
        if (i === 0) return null;
        const isSAR = obs.source.includes("sentinel-1") || obs.source.includes("sar");
        if (isSAR) {
          return (
            <div key={`router-${obs.observation_id}`} className="router-strip">
              <span style={{ color: "var(--text-dim)" }}>Obs {i}:</span>
              <span>95% cloud cover</span>
              <span className="arrow">→</span>
              <span>switched to radar (SAR)</span>
            </div>
          );
        }
        return null;
      })}

      {/* Observation cards */}
      <div className="obs-cards">
        {timeline.map((obs, i) => {
          const sev = severityFromObs(obs);
          const pct = obs.water_area_change_percent ?? 0;
          const isSAR = obs.source.includes("sentinel-1") || obs.source.includes("sar");
          const isActive = i === currentStepIdx;
          const isGreyed = i > currentStepIdx;
          const cardStep: SimStep = i === 0 ? "before" : `obs-${i}` as SimStep;
          const dateStr = new Date(obs.acquired_at).toISOString().slice(0, 10);

          return (
            <div
              key={obs.observation_id}
              className={`obs-card ${isActive ? "active" : ""} ${isGreyed ? "greyed" : ""}`}
              onClick={() => !isGreyed && sim.scrubTo(cardStep)}
            >
              <div className="obs-header">
                <span className="obs-label">{i === 0 ? "Baseline" : `Obs ${i}`}</span>
                <span className={`badge ${SEVERITY_BADGE[sev]}`}>{sev}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span className="obs-date">{dateStr}</span>
                <span className={`obs-sensor ${isSAR ? "sar" : "optical"}`}>
                  {isSAR ? "Radar (S1)" : "Optical (S2)"}
                </span>
              </div>
              <div className="obs-metrics">
                <div className="obs-metric">
                  <span className="label">cloud</span>
                  <span className="val">{obs.cloud_fraction !== null ? `${(obs.cloud_fraction * 100).toFixed(0)}%` : "—"}</span>
                </div>
                <div className="obs-metric">
                  <span className="label">rain 24h</span>
                  <span className="val">{obs.rainfall_24h_mm ?? "—"} mm</span>
                </div>
                <div className="obs-metric">
                  <span className="label">area</span>
                  <span className="val">{obs.water_area_km2 ?? "—"} km²</span>
                </div>
                <div className="obs-metric change">
                  <span className="label">change</span>
                  <span className="val" style={{ color: pct > 0 ? "var(--elevated)" : "var(--text)" }}>
                    {pct > 0 ? "▲" : "→"} {pct > 0 ? "+" : ""}{pct.toFixed(1)}%
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
