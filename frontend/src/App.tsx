import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "./api/client";
import { mockData } from "./api/mockData";
import { useSimulation } from "./simulation/SimulationContext";
import type { BasinConfig, RunList } from "./api/types";
import MapView from "./views/MapView";
import TimelineView from "./views/TimelineView";
import ReviewView from "./views/ReviewView";
import AuditView from "./views/AuditView";

type ViewName = "map" | "timeline" | "review" | "audit";
const TABS: { key: ViewName; label: string }[] = [
  { key: "map", label: "Map" },
  { key: "timeline", label: "Timeline" },
  { key: "review", label: "Review" },
  { key: "audit", label: "Audit" },
];

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
  const latestRun =
    (currentRunId ? runs.find((r) => r.run_id === currentRunId) : undefined) ??
    runs[0] ??
    mockData.runs.runs[0];

  const severity = latestRun?.score?.severity;
  const showBanner = (severity === "elevated" || severity === "critical") && !sim.reviewDecision;
  const expansionPct = (latestRun?.change_stats_json?.expansion_percent as number) ?? 0;

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key >= "1" && e.key <= "4") {
        setView(TABS[parseInt(e.key, 10) - 1].key);
      } else if (e.key.toLowerCase() === "r" && view === "timeline") {
        sim.advance();
      } else if (e.key === "Escape") {
        setToast(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [view, sim]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(t);
  }, [toast]);

  const handleReset = useCallback(() => {
    sim.reset();
    setView("map");
    setToast({ msg: "Reset to baseline", type: "info" });
  }, [sim]);

  const basinName = basin?.name ?? "Dudh Koshi / Imja";

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-surface-canvas text-text-primary font-sans">
      {/* Navigation bar */}
      <nav className="h-nav-height flex-none flex items-center gap-space-16 px-space-16 bg-surface-panel border-b border-border-subtle">
        <div className="flex items-center gap-space-8">
          <span className="text-headline-sm font-medium text-primary flex items-center gap-space-8">
            <span className="w-2.5 h-2.5 rounded-full bg-primary" />
            SIREN
          </span>
          <span className="text-body-md text-text-dim">{basinName}</span>
        </div>

        <div className="ml-auto flex items-center h-full">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setView(tab.key)}
              className={`h-full px-space-16 text-body-md transition-colors border-b-2 ${
                view === tab.key
                  ? "text-primary border-primary"
                  : "text-text-dim border-transparent hover:text-text-primary"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <button
          onClick={handleReset}
          className="px-space-12 py-space-6 text-body-sm text-text-dim border border-border-subtle rounded hover:text-text-primary hover:border-text-dim transition-colors bg-transparent"
        >
          Reset
        </button>
      </nav>

      {/* Alert banner */}
      {showBanner && (
        <div
          onClick={() => setView("review")}
          className={`h-banner-height flex-none flex items-center gap-space-12 px-space-16 bg-surface-panel border-b border-border-subtle border-l-[3px] cursor-pointer ${
            severity === "critical" ? "border-l-status-danger" : "border-l-status-elevated"
          }`}
        >
          <span
            className={`text-body-md font-medium ${
              severity === "critical" ? "text-status-danger" : "text-status-elevated"
            }`}
          >
            ⚠ {severity === "critical" ? "Critical" : "Elevated"}
          </span>
          <span className="text-body-md text-text-primary">
            water expansion +{expansionPct.toFixed(1)}% detected
          </span>
          <span className="ml-auto text-body-md text-primary">Review →</span>
        </div>
      )}

      {/* View container */}
      <main className="flex-1 min-h-0 overflow-auto p-space-16">
        {view === "map" && <MapView onJumpToReview={() => setView("review")} />}
        {view === "timeline" && <TimelineView />}
        {view === "review" && (
          <ReviewView run={latestRun} onToast={setToast} onJumpToMap={() => setView("map")} />
        )}
        {view === "audit" && <AuditView onToast={setToast} />}
      </main>

      {/* Footer */}
      <footer className="h-footer-height flex-none flex items-center justify-center bg-surface-canvas border-t border-border-subtle">
        <div className="text-caption text-text-dim tracking-normal">
          Sentinel-2 · Sentinel-1 · SRTM · Open-Meteo · © OSM · pipeline v0.1.0
        </div>
      </footer>

      {/* Toast */}
      {toast && (
        <div
          onClick={() => setToast(null)}
          className={`fixed bottom-footer-height left-1/2 -translate-x-1/2 mb-space-16 px-space-20 py-space-12 rounded border bg-surface-panel z-50 text-body-md ${
            toast.type === "error"
              ? "border-status-danger"
              : toast.type === "success"
              ? "border-status-safe"
              : "border-border-subtle"
          }`}
        >
          {toast.type === "error" && "⚠ "}
          {toast.type === "success" && "✓ "}
          {toast.msg}
        </div>
      )}
    </div>
  );
}
