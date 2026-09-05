import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "./api/client";
import { mockData } from "./api/mockData";
import { useSimulation } from "./simulation/SimulationContext";
import { ThemeProvider } from "./theme/ThemeContext";
import ThemeToggle from "./theme/ThemeToggle";
import OfflineBadge from "./components/OfflineBadge";
import type { BasinConfig, RunList, Run } from "./api/types";
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
  const { data: activeRunData } = useQuery({
    queryKey: ["run", currentRunId],
    queryFn: () => api.getRun(currentRunId!),
    enabled: currentRunId !== null,
    refetchInterval: sim.status === "running" ? 1000 : false,
  });
  const activeRun = sim.step === "before"
    ? null
    : activeRunData ?? (currentRunId ? runs.find((run) => run.run_id === currentRunId) : undefined) ?? null;
  const severity = activeRun?.score?.severity;
  const decision = activeRun?.decision ?? sim.reviewDecision;
  const showBanner = sim.step !== "before" && activeRun !== null &&
    (severity === "elevated" || severity === "critical") && !decision;
  const expansionPct = (activeRun?.change_stats_json?.expansion_percent as number) ?? 0;

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA";
      if (!typing && e.key >= "1" && e.key <= "4") {
        setView(TABS[parseInt(e.key, 10) - 1].key);
      } else if (!typing && e.key.toLowerCase() === "r") {
        sim.runAll().catch((error) => setToast({ msg: error.message, type: "error" }));
      } else if (e.key === "Escape") {
        const escapeEvent = new CustomEvent("siren:escape", { cancelable: true });
        window.dispatchEvent(escapeEvent);
        if (!escapeEvent.defaultPrevented && sim.selectedAssetId) {
          sim.selectAsset(null);
        }
        setToast(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [sim]);

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
  const pipelineStatus = sim.status === "running" ? "PIPELINE: RUNNING" : sim.status === "complete" ? "PIPELINE: COMPLETE" : "PIPELINE: IDLE";

  return (
    <ThemeProvider>
      <div className="flex flex-col h-screen overflow-hidden bg-surface-canvas text-text-primary font-sans">
        {/* Top bar — compact operational header */}
        <nav className="h-nav-height flex-none flex items-stretch bg-surface-panel border-b border-border-subtle">
          {/* Brand + basin */}
          <div className="flex items-center gap-space-8 px-space-12 border-r border-border-subtle">
            <span className="text-headline-md font-headline text-text-primary tracking-wide">SIREN</span>
            <span className="text-body-md text-text-dim hidden sm:inline">{basinName}</span>
          </div>

          {/* Tabs */}
          <div className="flex items-stretch">
            {TABS.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setView(tab.key)}
                className={`px-space-16 text-body-md transition-colors border-b-2 ${
                  view === tab.key
                    ? "text-text-primary border-primary"
                    : "text-text-dim border-transparent hover:text-text-primary hover:bg-surface-container"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Status chips — system-level indicators */}
          <div className="flex items-center gap-space-8 px-space-12 ml-auto">
            <span className={`w-2.5 h-2.5 border ${sim.status === "running" ? "border-status-safe bg-status-safe animate-pulse" : "border-text-muted"}`} />
            <span className="text-body-sm text-text-dim">{pipelineStatus}</span>
            <span className="text-border-subtle">|</span>
            <OfflineBadge />
          </div>

          <div className="flex items-center gap-space-8 px-space-12 border-l border-border-subtle">
            <ThemeToggle />
            <button
              onClick={handleReset}
              className="px-space-8 py-space-4 text-body-sm text-text-dim border border-border-subtle hover:text-text-primary hover:border-border-strong transition-colors bg-transparent"
            >
              Reset
            </button>
          </div>
        </nav>

        {/* Alert banner — semantic state only */}
        {showBanner && (
          <div
            onClick={() => setView("review")}
            className={`h-banner-height flex-none flex items-center gap-space-12 px-space-16 bg-surface-panel border-b border-border-subtle border-l-2 cursor-pointer ${
              severity === "critical" ? "border-l-status-danger" : "border-l-status-elevated"
            }`}
          >
            <span
              className={`text-body-md font-medium ${
                severity === "critical" ? "text-status-danger" : "text-status-elevated"
              }`}
            >
              {severity === "critical" ? "Critical" : "Elevated"}
            </span>
            <span className="text-body-md text-text-primary">
              water expansion <span className="data-val">+{expansionPct.toFixed(1)}%</span> detected
            </span>
            <span className="ml-auto text-body-sm text-status-elevated border border-status-elevated px-space-8 py-space-2">
              Review pending →
            </span>
          </div>
        )}

        {/* View container — full-bleed, no padding */}
        <main key={view} className="flex-1 min-h-0 overflow-auto view-fade">
          {view === "map" && <MapView basin={basin ?? undefined} run={activeRun ?? undefined} onJumpToReview={() => setView("review")} />}
          {view === "timeline" && <TimelineView />}
          {view === "review" && (
            <ReviewView run={activeRun ?? undefined} onToast={setToast} onJumpToMap={() => setView("map")} />
          )}
          {view === "audit" && <AuditView run={activeRun} onToast={setToast} />}
        </main>

        {/* Footer — compact status bar */}
        <footer className="h-footer-height flex-none flex items-center px-space-16 bg-surface-panel border-t border-border-subtle">
          <span className="text-caption text-text-dim">
            Sentinel-2 / Sentinel-1 / SRTM / Open-Meteo / OSM
          </span>
          <span className="ml-auto data-val text-caption text-text-muted">
            build a5a1751
          </span>
        </footer>

        {/* Toast — sharp notification, no rounded pill */}
        {toast && (
          <div
            onClick={() => setToast(null)}
            className={`fixed bottom-footer-height left-space-16 mb-space-8 px-space-12 py-space-8 border bg-surface-panel z-50 text-body-md ${
              toast.type === "error"
                ? "border-status-danger text-status-danger"
                : toast.type === "success"
                ? "border-status-safe text-status-safe"
                : "border-border-subtle text-text-primary"
            }`}
          >
            {toast.msg}
          </div>
        )}
      </div>
    </ThemeProvider>
  );
}
