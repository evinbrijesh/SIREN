import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "./api/client";
import { mockData } from "./api/mockData";
import { useSimulation, STEPS, type SimStep } from "./simulation/SimulationContext";
import type { BasinConfig, RunList } from "./api/types";
import MapView from "./views/MapView";
import TimelineView from "./views/TimelineView";
import ReviewView from "./views/ReviewView";
import AuditView from "./views/AuditView";

type ViewName = "map" | "timeline" | "review" | "audit";
const TAB_KEYS: ViewName[] = ["map", "timeline", "review", "audit"];
const TAB_LABELS: Record<ViewName, string> = {
  map: "Map",
  timeline: "Timeline",
  review: "Review",
  audit: "Audit",
};

export default function App() {
  const [view, setView] = useState<ViewName>("map");
  const [toast, setToast] = useState<{ msg: string; type: "error" | "info" | "success" } | null>(null);
  const sim = useSimulation();

  const { data: basin } = useQuery({
    queryKey: ["basin"],
    queryFn: () => apiOrMock(() => api.getBasin(), "basin") as Promise<BasinConfig>,
  });

  const { data: runsData } = useQuery({
    queryKey: ["runs"],
    queryFn: () => apiOrMock(() => api.listRuns(), "runs") as Promise<RunList>,
    refetchInterval: sim.status === "running" ? 1000 : false,
  });

  const currentRunId = sim.step !== "before" ? sim.runIds[sim.step] : null;
  const runs = runsData?.runs ?? mockData.runs.runs;
  const latestRun = (currentRunId ? runs.find((r) => r.run_id === currentRunId) : undefined) ?? runs[0] ?? mockData.runs.runs[0];
  const severity = latestRun?.score?.severity;
  const showBanner = (severity === "elevated" || severity === "critical") && !sim.reviewDecision;
  const expansionPct = (latestRun?.change_stats_json?.expansion_percent as number) ?? 0;

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key >= "1" && e.key <= "4") {
        setView(TAB_KEYS[parseInt(e.key) - 1]);
      } else if (e.key.toLowerCase() === "r" && view === "timeline") {
        sim.advance();
      } else if (e.key === "Escape") {
        setToast(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [view]);

  useEffect(() => {
    if (toast) {
      const t = setTimeout(() => setToast(null), 4000);
      return () => clearTimeout(t);
    }
  }, [toast]);

  const handleReset = useCallback(() => {
    sim.reset();
    setView("map");
    setToast({ msg: "Reset to baseline", type: "info" });
  }, [sim]);

  const basinName = basin?.name ?? "Dudh Koshi / Imja";

  return (
    <div className="app-shell">
      <nav className="nav-bar">
        <span className="nav-brand">SIREN</span>
        <span className="nav-basin">{basinName}</span>
        <div className="nav-tabs">
          {TAB_KEYS.map((v) => (
            <button
              key={v}
              className={`nav-tab ${view === v ? "active" : ""}`}
              onClick={() => setView(v)}
            >
              {TAB_LABELS[v]}
            </button>
          ))}
        </div>
        <button className="btn btn-ghost" onClick={handleReset} style={{ padding: "6px 12px", fontSize: 13 }}>
          Reset
        </button>
      </nav>

      {showBanner && (
        <div
          className={`alert-banner ${severity === "critical" ? "critical" : ""}`}
          onClick={() => setView("review")}
        >
          <span className={`sev-text ${severity === "critical" ? "critical" : "elevated"}`}>
            {severity === "elevated" ? "Elevated" : "Critical"}
          </span>
          <span>— water expansion +{expansionPct.toFixed(1)}% detected</span>
          <span className="review-link">Review →</span>
        </div>
      )}

      <div className="view-container">
        {view === "map" && <MapView />}
        {view === "timeline" && <TimelineView />}
        {view === "review" && <ReviewView run={latestRun} onToast={setToast} onJumpToMap={() => setView("map")} />}
        {view === "audit" && <AuditView onToast={setToast} />}
      </div>

      <footer className="app-footer">
        <span>Sentinel-2</span><span className="sep">·</span>
        <span>Sentinel-1</span><span className="sep">·</span>
        <span>SRTM</span><span className="sep">·</span>
        <span>Open-Meteo</span><span className="sep">·</span>
        <span>© OSM</span><span className="sep">·</span>
        <span>v0.1.0</span>
      </footer>

      {toast && (
        <div className={`toast ${toast.type}`} onClick={() => setToast(null)}>
          {toast.type === "error" && "⚠ "}{toast.type === "success" && "✓ "}{toast.msg}
        </div>
      )}
    </div>
  );
}
