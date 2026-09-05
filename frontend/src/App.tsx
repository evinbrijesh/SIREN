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

export default function App() {
  const [view, setView] = useState<ViewName>("map");
  const [toast, setToast] = useState<{ msg: string; type: "error" | "info" | "success" } | null>(null);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const sim = useSimulation();

  const { data: basin } = useQuery({
    queryKey: ["basin"],
    queryFn: () => apiOrMock(() => api.getBasin(), "basin") as Promise<BasinConfig>,
  });

  const { data: runsData } = useQuery({
    queryKey: ["runs"],
    queryFn: () => apiOrMock(() => api.listRuns(), "runs") as Promise<RunList>,
  });

  const latestRun = runsData?.runs?.[0] ?? mockData.runs.runs[0];
  const severity = latestRun?.score?.severity;
  const showBanner = (severity === "elevated" || severity === "critical") && !sim.reviewDecision;

  // Online/offline listener
  useEffect(() => {
    const on = () => setIsOnline(true);
    const off = () => setIsOnline(false);
    window.addEventListener("online", on);
    window.addEventListener("offline", off);
    return () => { window.removeEventListener("online", on); window.removeEventListener("offline", off); };
  }, []);

  // Keyboard shortcuts: 1-4 tabs, R run simulation, Esc close toast
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key >= "1" && e.key <= "4") {
        setView(TAB_KEYS[parseInt(e.key) - 1]);
      } else if (e.key.toLowerCase() === "r" && view === "timeline") {
        // R triggers run from timeline view
      } else if (e.key === "Escape") {
        setToast(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [view]);

  // Auto-dismiss toast
  useEffect(() => {
    if (toast) {
      const t = setTimeout(() => setToast(null), 4000);
      return () => clearTimeout(t);
    }
  }, [toast]);

  const handleReset = useCallback(() => {
    sim.reset();
    setView("map");
    setToast({ msg: "Console reset to before state", type: "info" });
  }, [sim]);

  return (
    <div className="app-shell">
      {/* NavBar */}
      <nav className="nav-bar">
        <span className="nav-brand">SIREN</span>
        <span className="nav-basin">Dudh Koshi/Imja</span>
        <span className="nav-live">LIVE</span>
        <div className="nav-tabs">
          {TAB_KEYS.map((v, i) => (
            <button
              key={v}
              className={`nav-tab ${view === v ? "active" : ""}`}
              onClick={() => setView(v)}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
              <span style={{ marginLeft: 4, opacity: 0.4, fontSize: 10 }}>{i + 1}</span>
            </button>
          ))}
        </div>
        {!isOnline && <span className="offline-badge">● OFFLINE — ALL SYSTEMS LOCAL</span>}
        <button className="nav-reset" onClick={handleReset} title="Reset console to before state">RESET ⟲</button>
      </nav>

      {/* Alert banner */}
      {showBanner && (
        <div className="alert-banner" onClick={() => setView("review")}>
          <span className="icon">⚠</span>
          <span className="sev">{severity?.toUpperCase()}</span>
          <span>— +14.3% · 2 villages · 1 bridge · 3 wells → Review</span>
        </div>
      )}

      {/* Active view */}
      <div className="view-container">
        {view === "map" && <MapView />}
        {view === "timeline" && <TimelineView />}
        {view === "review" && <ReviewView run={latestRun} onToast={setToast} onJumpToMap={() => setView("map")} />}
        {view === "audit" && <AuditView onToast={setToast} />}
      </div>

      {/* Provenance strip */}
      <footer className="provenance-strip">
        <span>Sentinel-2</span><span className="sep">·</span>
        <span>Sentinel-1</span><span className="sep">·</span>
        <span>SRTM</span><span className="sep">·</span>
        <span>Open-Meteo</span><span className="sep">·</span>
        <span>© OSM</span><span className="sep">·</span>
        <span>pipeline v0.1.0</span>
      </footer>

      {/* Toast */}
      {toast && (
        <div className={`toast ${toast.type}`} onClick={() => setToast(null)}>
          {toast.type === "error" && "⚠ "}{toast.type === "success" && "✓ "}{toast.msg}
        </div>
      )}
    </div>
  );
}
