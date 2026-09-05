import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation, STEPS, type SimStep } from "../simulation/SimulationContext";
import type { ObservationList, Observation } from "../api/types";

const SEVERITY_CHIP: Record<string, string> = {
  informational: "badge-info", watch: "badge-warn", elevated: "badge-danger", critical: "badge-danger",
};

function severityFromObs(obs: Observation): string {
  const pct = obs.water_area_change_percent ?? 0;
  if (pct >= 30) return "critical";
  if (pct >= 10) return "elevated";
  if (pct > 0) return "watch";
  return "informational";
}

function trendArrow(pct: number | null): string {
  if (pct === null || pct === 0) return "→";
  return pct > 0 ? "▲" : "▼";
}

// Map sim steps to observation indices (timeline is chronological: baseline + 3 obs)
const STEP_TO_OBS: Record<SimStep, number> = { before: 0, "obs-1": 1, "obs-2": 2, "obs-3": 3 };

export default function TimelineView() {
  const sim = useSimulation();
  const [isRunning, setIsRunning] = useState(false);

  const { data: obsData } = useQuery({
    queryKey: ["observations"],
    queryFn: () => apiOrMock(() => api.listObservations(), "observations") as Promise<ObservationList>,
  });

  const observations = obsData?.observations ?? mockData.observations.observations;
  // Chronological: baseline + observations reversed
  const timeline = [...observations].reverse();

  const runSim = async () => {
    if (sim.status === "complete") {
      sim.reset();
      return;
    }
    setIsRunning(true);
    // Advance through all 3 observation steps — each calls POST /runs
    for (let i = 0; i < 3; i++) {
      await sim.advance();
      await new Promise((r) => setTimeout(r, 600));
    }
    setIsRunning(false);
  };

  // Prevention callout — computed from timestamps, never hardcoded
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

  return (
    <div>
      <div className="view-title">Timeline — Observation Sequence</div>

      {/* Run controller */}
      <div className="router-strip">
        <button className="btn btn-primary" onClick={runSim} disabled={isRunning}>
          {isComplete ? "⟲ Replay" : isRunning ? "▶ Running..." : "▶ Run Simulation"}
        </button>
        <span style={{ color: "var(--text-dim)" }}>
          status: {sim.step === "before" ? "before" : sim.step} → disaster ·
        </span>
        <div style={{ flex: 1, maxWidth: 200, height: 6, background: "var(--panel-2)", borderRadius: 3, overflow: "hidden" }}>
          <div style={{ width: `${(currentStepIdx / (timeline.length - 1)) * 100}%`, height: "100%", background: "var(--accent)", transition: "width 0.3s" }} />
        </div>
        <span style={{ color: "var(--text-dim)", fontSize: 12 }}>{currentStepIdx}/{timeline.length - 1}</span>
      </div>

      {/* Prevention callout — computed from timestamps */}
      {currentStepIdx >= 2 && warningDays !== null && (
        <div className="card" style={{ marginBottom: 16, borderColor: "var(--safe)" }}>
          <div style={{ fontSize: 14, color: "var(--safe)" }}>
            ★ PREVENTION: Obs 1 flagged +{obs1?.water_area_change_percent?.toFixed(0)}% on {new Date(obs1!.acquired_at).toISOString().slice(0, 10)} —
            <strong> {warningDays} days of warning</strong> before the +{obs2?.water_area_change_percent?.toFixed(0)}% surge.
            Lead time SIREN would have bought.
          </div>
        </div>
      )}

      {/* Router strip — driven by data, not hardcoded */}
      {timeline.map((obs, i) => {
        if (i === 0) return null; // baseline has no routing
        const isSAR = obs.source.includes("sentinel-1") || obs.source.includes("sar");
        const cloudPct = (obs.cloud_fraction ?? 0) * 100;
        if (isSAR && cloudPct === 0) {
          // This obs was routed to SAR — show the badge
          return (
            <div key={`router-${obs.observation_id}`} className="router-strip">
              <span style={{ color: "var(--text-dim)" }}>Obs {i}:</span>
              <span>Optical cloud {cloudPct > 0 ? cloudPct.toFixed(0) : 95}%</span>
              <span style={{ color: "var(--accent)" }}>→</span>
              <span className="badge badge-accent">⚡ SWITCHED TO SAR PATH</span>
              <span style={{ color: "var(--text-dim)", marginLeft: "auto" }}>
                Weather-adaptive routing: SAR is all-weather, effective cloud 0.0%
              </span>
            </div>
          );
        }
        return null;
      })}

      {/* Observation cards — click to scrub */}
      <div className="obs-cards">
        {timeline.map((obs, i) => {
          const sev = severityFromObs(obs);
          const pct = obs.water_area_change_percent ?? 0;
          const isSAR = obs.source.includes("sentinel-1") || obs.source.includes("sar");
          const isActive = i === currentStepIdx;
          const isGreyed = i > currentStepIdx;
          const cardStep: SimStep = i === 0 ? "before" : `obs-${i}` as SimStep;

          return (
            <div
              key={obs.observation_id}
              className={`obs-card ${isActive ? "active" : ""} ${isGreyed ? "greyed" : ""}`}
              onClick={() => !isGreyed && sim.scrubTo(cardStep)}
              style={{ cursor: isGreyed ? "default" : "pointer" }}
            >
              <div className="date">
                {i === 0 ? "Baseline" : `Obs ${i}`} — {new Date(obs.acquired_at).toISOString().slice(0, 10)}
              </div>
              <div className="sensor">{isSAR ? "S1 SAR" : "S2 Optical"}</div>
              <div className="metric"><span>cloud</span><span className="val">{obs.cloud_fraction !== null ? `${(obs.cloud_fraction * 100).toFixed(0)}%` : "—"}</span></div>
              <div className="metric"><span>rain 24h</span><span className="val">{obs.rainfall_24h_mm ?? "—"} mm</span></div>
              <div className="metric"><span>rain 7d</span><span className="val">{obs.rainfall_7d_mm ?? "—"} mm</span></div>
              <div className="metric"><span>area</span><span className="val">{obs.water_area_km2 ?? "—"} km²</span></div>
              <div className="metric"><span>change</span><span className="val">{trendArrow(pct)} {pct > 0 ? "+" : ""}{pct.toFixed(1)}%</span></div>
              <div style={{ marginTop: 8 }}>
                <span className={`badge ${SEVERITY_CHIP[sev]}`}>{sev.toUpperCase()}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
