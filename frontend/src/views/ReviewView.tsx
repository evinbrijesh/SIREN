import { useEffect, useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, apiOrMock, addToOutbox } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation } from "../simulation/SimulationContext";
import { sendNtfyAlert } from "../utils/ntfy";
import type { Run, ExposureList, SarPriorityList, MlEvidence, ReviewResponse, DispatchResponse, ApiError } from "../api/types";

interface Props {
  run?: Run;
  onToast?: (t: { msg: string; type: "error" | "info" | "success" }) => void;
  onJumpToMap?: () => void;
}

const GAUGE_COLORS: Record<string, string> = {
  H: "#ff1e27",
  E: "#ffb000",
  D: "#00f0ff",
  C: "#10b981",
};

const GAUGE_TEXT: Record<string, string> = {
  H: "text-status-danger",
  E: "text-status-warn",
  D: "text-status-info",
  C: "text-status-safe",
};

function exposureStatus(exposure: ExposureList["exposures"][number]): "SAFE" | "BUFFERED" | "INUNDATED" {
  if (exposure.inundated) return "INUNDATED";
  if (exposure.distance_m !== null && exposure.buffer_m !== null && exposure.distance_m <= exposure.buffer_m) return "BUFFERED";
  return "SAFE";
}

export default function ReviewView({ run, onToast, onJumpToMap }: Props) {
  const qc = useQueryClient();
  const sim = useSimulation();
  const [confirmStep, setConfirmStep] = useState(false);
  const [dispatchArmed, setDispatchArmed] = useState(false);
  const [safetyLifted, setSafetyLifted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSector, setSelectedSector] = useState<string>("sector-b");
  const [selectedChannel, setSelectedChannel] = useState<"sms" | "lora" | "satellite">("sms");
  const [viewMode, setViewMode] = useState<"simple" | "advanced">("simple");
  const [countdown, setCountdown] = useState(300); // 5-minute escalation timer (cosmetic)

  const score = run?.score;
  const runId = run?.run_id;

  // Bind latch state to active run ID — switching tabs or runs resets the latch
  const latchRunIdRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (runId !== latchRunIdRef.current) {
      setSafetyLifted(false);
      setDispatchArmed(false);
      setConfirmStep(false);
      latchRunIdRef.current = runId;
    }
  }, [runId]);

  // Escalation countdown — cosmetic only, no real network call
  // Counts down from 5 minutes while review is pending (elevated/critical, no decision)
  // When it hits 0, shows a toast about auto-escalation (simulated)
  const currentDecision = run?.decision ?? sim.reviewDecision;
  const isPendingReview = score && (score.severity === "elevated" || score.severity === "critical") && !currentDecision;
  useEffect(() => {
    if (!isPendingReview) return;
    if (countdown <= 0) {
      onToast?.({ msg: "Auto-escalation triggered — alert sent to all channels (no confirmation received)", type: "error" });
      return;
    }
    const timer = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [isPendingReview, countdown, onToast]);

  const { data: exposuresData, isLoading: expLoading } = useQuery({
    queryKey: ["exposures", runId],
    queryFn: () => apiOrMock(() => api.listExposures(runId!), "exposures") as Promise<ExposureList>,
    enabled: Boolean(runId),
  });

  const { data: sarData } = useQuery({
    queryKey: ["sar-priority", runId],
    queryFn: () => apiOrMock(() => api.getSarPriority(runId!), "sarPriority") as Promise<SarPriorityList>,
    enabled: Boolean(runId),
  });

  const { data: mlData } = useQuery({
    queryKey: ["ml-evidence", runId],
    queryFn: () => apiOrMock(() => api.getMlEvidence(runId!), "mlEvidence") as Promise<MlEvidence>,
    enabled: Boolean(runId),
  });

  const sarPriority = sarData ?? mockData.sarPriority;
  const mlEvidence = mlData ?? mockData.mlEvidence;
  const exposures = exposuresData?.exposures ?? mockData.exposures.exposures;
  const wells = exposures.filter((e) => e.asset_type === "well");
  const villages = exposures.filter((e) => e.asset_type === "village");
  const totalPop = villages.reduce((s, v) => s + (v.population ?? 0), 0);

  const reviewMut = useMutation({
    mutationFn: (vars: { decision: "confirm" | "reject" | "postpone" | "escalate" }) =>
      api.createReview(runId!, "coordinator-01", vars.decision, "demo review") as Promise<ReviewResponse>,
    onSuccess: (_data, vars) => {
      const isConfirmLike = vars.decision === "confirm" || vars.decision === "escalate";
      sim.setReviewDecision(isConfirmLike ? "confirm" : vars.decision as "reject" | "postpone");
      setConfirmStep(false);
      setError(null);
      if (isConfirmLike) {
        // Auto-fire ntfy.sh alert on confirmation — human made the decision,
        // the phone push is a side-effect, not an autonomous dispatch
        const expansionPctLocal = (run?.change_stats_json?.expansion_percent as number) ?? 0;
        sendNtfyAlert({ expansionPct: expansionPctLocal }, (success, _message) => {
          onToast?.({
            msg: success ? "Decision confirmed — SOS sent to phone" : "Decision confirmed — air-gap mode, SOS simulated",
            type: "success",
          });
        });
      } else {
        onToast?.({
          msg: `Decision recorded: ${vars.decision}`,
          type: "success",
        });
      }
      qc.invalidateQueries({ queryKey: ["runs"] });
      qc.invalidateQueries({ queryKey: ["run", runId] });
    },
    onError: (e: ApiError) => {
      const offline = typeof navigator !== "undefined" && navigator.onLine === false;
      if (offline) {
        addToOutbox(`/runs/${runId}/review`, "POST", JSON.stringify({ reviewer: "coordinator-01", decision: "escalate", note: "demo review" }));
        onToast?.({ msg: "Offline — review queued in outbox, will sync when online", type: "info" });
      } else {
        setError(e.detail);
        onToast?.({ msg: e.detail, type: "error" });
      }
    },
  });

  const dispatchMut = useMutation({
    mutationFn: () => api.createDispatch(runId!, selectedChannel, selectedSector) as Promise<DispatchResponse>,
    onSuccess: (data) => {
      sim.setDispatchResult(data);
      setError(null);
      setDispatchArmed(false);
      onToast?.({ msg: `Dispatch sent (${data.payload_bytes} bytes)`, type: "success" });
    },
    onError: (e: ApiError) => {
      const offline = typeof navigator !== "undefined" && navigator.onLine === false;
      if (offline) {
        addToOutbox(`/runs/${runId}/dispatch`, "POST", JSON.stringify({ channel: selectedChannel, recipient_group: selectedSector }));
        setDispatchArmed(false);
        onToast?.({ msg: "Offline — dispatch queued in outbox, will sync when online", type: "info" });
      } else {
        setError(e.detail);
        onToast?.({ msg: e.detail, type: "error" });
      }
    },
  });

  useEffect(() => {
    const disarm = (event: Event) => {
      if (!confirmStep && !dispatchArmed) return;
      event.preventDefault();
      setConfirmStep(false);
      setDispatchArmed(false);
    };
    window.addEventListener("siren:escape", disarm);
    return () => window.removeEventListener("siren:escape", disarm);
  }, [confirmStep, dispatchArmed]);

  if (!run || !score) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-text-dim text-center gap-space-8">
        <div className="text-body-md">No alerts requiring review</div>
        <div className="text-body-sm text-text-muted">Run the simulation from Timeline to generate observations.</div>
      </div>
    );
  }

  // Watch severity: show an escalate card instead of the full review panel.
  // In production, a watch is monitoring-level — the coordinator can escalate
  // it to elevated, which then triggers the normal confirm/reject/dispatch flow.
  if (score.severity === "watch" && !sim.reviewDecision) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between px-space-16 py-space-8 border-b border-border-subtle bg-surface-panel">
          <h1 className="label-caps">Review</h1>
          <span className="text-body-sm text-status-warn border border-status-warn px-space-4 py-space-1">
            Watch
          </span>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center gap-space-12 px-space-16">
          <div className="text-center max-w-md">
            <div className="text-headline-sm text-text-primary mb-space-4">
              Monitoring — escalation available
            </div>
            <div className="text-body-sm text-text-dim">
              Observation {run.observation_id} shows early warning signs
              (expansion {run.change_stats_json?.expansion_percent ?? "+"}%, hazard H={score.hazard_score.toFixed(2)}).
              Escalate to elevated to trigger the full review and dispatch workflow,
              or continue monitoring.
            </div>
          </div>
          <div className="flex gap-space-8">
            <button
              className="px-space-12 py-space-6 border border-status-warn text-status-warn data-val text-body-sm hover:bg-status-warn/10 transition-colors"
              onClick={() => reviewMut.mutate({ decision: "escalate" })}
              disabled={reviewMut.isPending}
            >
              {reviewMut.isPending ? "ESCALATING..." : "ESCALATE TO ELEVATED"}
            </button>
            <button
              className="px-space-12 py-space-6 border border-border-subtle text-text-dim data-val text-body-sm hover:bg-surface-recessed transition-colors"
              onClick={() => reviewMut.mutate({ decision: "postpone" })}
              disabled={reviewMut.isPending}
            >
              CONTINUE MONITORING
            </button>
          </div>
          {error && (
            <div className="data-val text-body-sm text-status-danger">{error}</div>
          )}
        </div>
      </div>
    );
  }

  // After escalation or for informational severity, show the no-review state
  if (score.severity === "informational" || (score.severity === "watch" && sim.reviewDecision)) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-text-dim text-center gap-space-8">
        <div className="data-val text-body-md">NO ALERTS REQUIRING REVIEW</div>
        <div className="data-val text-body-sm text-text-muted">
          {sim.reviewDecision === "confirm" && score.severity === "watch"
            ? "Watch escalated — switch to Timeline and process the next observation."
            : "Run the simulation from Timeline to generate observations."}
        </div>
      </div>
    );
  }

  if (expLoading) {
    return (
      <div className="flex flex-col h-full">
        <div className="px-space-16 py-space-8 border-b border-border-subtle bg-surface-panel">
          <span className="label-caps">Review</span>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <span className="data-val text-body-md text-text-dim">LOADING EXPOSURE DATA...</span>
        </div>
      </div>
    );
  }

  const decisionLocked = currentDecision !== null;
  const sortedExposures = [...exposures].sort((a, b) => (a.distance_m ?? 9999) - (b.distance_m ?? 9999));
  const areaAfter = (run.change_stats_json?.water_area_km2 as number) ?? 4.1;
  const areaBefore = 3.0;
  const expansionPct = (run.change_stats_json?.expansion_percent as number) ?? 0;
  const corridorSource = (run.change_stats_json?.corridor_source as string) ?? "unknown";
  const isFallbackCorridor = corridorSource === "fallback_seeded";

  return (
    <div className="flex flex-col h-full pb-[48px] overflow-auto">
      {/* Header bar — tactical bezel */}
      <div className="relative flex items-center justify-between px-space-16 py-space-8 border-b border-border-subtle bg-surface-panel">
        <div className="flex items-center gap-space-12">
          <h1 className="label-caps">Review</h1>
          <span className="data-val text-body-sm text-primary-container border border-border-subtle px-space-4 py-space-1">
            {runId}
          </span>
          {/* Provenance badge (O3) — shows whether corridor is real D8+OSM or fallback */}
          <span
            className={`text-caption border px-space-4 py-space-1 ${
              isFallbackCorridor
                ? "border-status-warn text-status-warn"
                : "border-status-safe text-status-safe"
            }`}
            title={isFallbackCorridor ? "Corridor from seeded demo data (D8 trace failed)" : "Corridor from D8 flow accumulation + OSM river buffering"}
          >
            {isFallbackCorridor ? "Corridor: fallback" : "Corridor: D8+OSM"}
          </span>
        </div>
        <div className="flex items-center gap-space-8">
          {/* Simple / Advanced mode toggle */}
          <div className="flex items-center gap-space-2 bg-surface-canvas border border-border-subtle p-space-2">
            <button
              onClick={() => setViewMode("simple")}
              className={`px-space-12 py-space-4 text-body-sm font-medium transition-colors ${
                viewMode === "simple"
                  ? "bg-primary text-primary-fg font-bold"
                  : "text-text-dim hover:text-text-primary"
              }`}
            >
              Simple (Triage)
            </button>
            <button
              onClick={() => setViewMode("advanced")}
              className={`px-space-12 py-space-4 text-body-sm font-medium transition-colors ${
                viewMode === "advanced"
                  ? "bg-primary text-primary-fg font-bold"
                  : "text-text-dim hover:text-text-primary"
              }`}
            >
              Advanced (Analyst)
            </button>
          </div>
          <span className="text-body-sm text-text-dim">{run.observation_id}</span>
        </div>
      </div>

      {/* Escalation countdown — cosmetic only, no real network call */}
      {isPendingReview && countdown > 0 && (
        <div className="flex items-center gap-space-12 px-space-16 py-space-4 bg-status-warn/10 border-b border-status-warn/30">
          <span className="data-val text-body-sm text-status-warn whitespace-nowrap">
            AUTO-ESCALATION IN {Math.floor(countdown / 60)}:{String(countdown % 60).padStart(2, "0")}
          </span>
          <div className="flex-1 h-[2px] bg-border-subtle overflow-hidden">
            <div className="h-full bg-status-warn transition-all duration-1000" style={{ width: `${(countdown / 300) * 100}%` }} />
          </div>
          <span className="text-caption text-text-dim whitespace-nowrap">
            No confirmation → alert all channels
          </span>
        </div>
      )}

      {viewMode === "simple" ? (
        <SimpleTriage
          run={run}
          score={score}
          mlEvidence={mlEvidence}
          wells={wells}
          villages={villages}
          totalPop={totalPop}
          expansionPct={expansionPct}
          areaBefore={areaBefore}
          areaAfter={areaAfter}
          severity={score.severity}
        />
      ) : (
      <div className="flex-1 overflow-auto p-space-12">
        {/* Top row: Evidence + Risk */}
        <div className="flex flex-col lg:flex-row gap-space-8 items-stretch">
          {/* Evidence panel */}
          <section className="w-full lg:w-[45%] bg-surface-panel border border-border-subtle flex flex-col">
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h2 className="label-caps">Evidence</h2>
              <div className="flex items-center gap-space-6">
                <span className={`text-body-sm border px-space-4 py-space-1 ${
                  mlEvidence.model_available
                    ? "border-primary-container text-primary-container"
                    : "border-status-warn text-status-warn"
                }`}>
                  {mlEvidence.model_available ? "ML active" : "Rule-based"}
                </span>
              </div>
            </div>

            <div className="p-space-12 flex flex-col gap-space-8">
              {/* Before/After raster — no gradient overlays */}
              <div className="flex gap-space-8">
                <div className="flex-1 h-[140px] border border-border-subtle bg-surface-recessed relative overflow-hidden flex flex-col">
                  <img
                    src={mlEvidence.baseline_mask_uri}
                    alt="Baseline water mask"
                    className="absolute inset-0 w-full h-full object-cover opacity-80"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                  />
                  <div className="relative z-10 flex items-center justify-between px-space-8 py-space-4 bg-surface-panel border-b border-border-subtle">
                    <span className="text-body-sm text-text-dim">Before</span>
                  </div>
                  <div className="relative z-10 flex flex-col px-space-8 py-space-4 mt-auto bg-surface-panel border-t border-border-subtle">
                    <span className="data-val text-headline-md text-text-primary">{areaBefore.toFixed(2)} km²</span>
                    <span className="text-caption text-text-dim">Reference extent</span>
                  </div>
                </div>
                <div className="flex-1 h-[140px] border border-border-subtle bg-surface-recessed relative overflow-hidden flex flex-col">
                  <img
                    src={mlEvidence.mask_uri}
                    alt="Current change mask"
                    className="absolute inset-0 w-full h-full object-cover opacity-90"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                  />
                  <div className="relative z-10 flex items-center justify-between px-space-8 py-space-4 bg-surface-panel border-b border-border-subtle">
                    <span className="text-body-sm text-text-dim">After</span>
                  </div>
                  <div className="relative z-10 flex flex-col px-space-8 py-space-4 mt-auto bg-surface-panel border-t border-border-subtle">
                    <span className="data-val text-headline-md text-text-primary">{areaAfter.toFixed(2)} km²</span>
                    <span className="text-caption text-text-dim">+{((run.change_stats_json?.expansion_percent as number) ?? 0).toFixed(1)}% expansion</span>
                  </div>
                </div>
              </div>

              {/* ML heatmap */}
              <div className="h-[120px] border border-border-subtle bg-surface-recessed relative overflow-hidden flex flex-col">
                <img
                  src={mlEvidence.heatmap_uri}
                  alt="ML change detection heatmap"
                  className="absolute inset-0 w-full h-full object-cover"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                />
                <div className="relative z-10 flex items-center justify-between px-space-8 py-space-4 bg-surface-panel border-b border-border-subtle">
                  <span className="text-body-sm text-text-primary">
                    {mlEvidence.model_available ? "Change probability" : "Change detection heatmap"}
                  </span>
                  <span className="data-val text-body-sm text-text-dim">
                    {(mlEvidence.ml_confidence_mean * 100).toFixed(0)}% confidence
                  </span>
                </div>
                <div className="relative z-10 flex items-center justify-between px-space-8 py-space-4 mt-auto bg-surface-panel border-t border-border-subtle">
                  <div className="flex items-center gap-space-8">
                    <div className="flex items-center gap-space-4">
                      <span className="w-2 h-2 bg-status-danger" />
                      <span className="text-caption text-text-dim">High</span>
                    </div>
                    <div className="flex items-center gap-space-4">
                      <span className="w-2 h-2 bg-status-warn" />
                      <span className="text-caption text-text-dim">Moderate</span>
                    </div>
                    <div className="flex items-center gap-space-4">
                      <span className="w-2 h-2 bg-surface-canvas border border-border-subtle" />
                      <span className="text-caption text-text-dim">None</span>
                    </div>
                  </div>
                  <span className="data-val text-body-sm text-text-dim">
                    {mlEvidence.ml_consensus_pixels.toLocaleString()} px
                  </span>
                </div>
              </div>

              <div className="text-body-sm text-text-dim">
                {mlEvidence.model_available
                  ? "ML + rule-based consensus (ADR-002)"
                  : "Rule-based detection with confidence gradient"}
              </div>
            </div>

            <div className="border-t border-border-subtle px-space-12 py-space-8 flex items-center justify-between text-body-sm text-text-dim">
              <span>
                Confidence: <span className="data-val text-text-primary">{(score.confidence * 100).toFixed(1)}%</span>
              </span>
              <span className="data-val text-caption">
                {mlEvidence.model_available ? "ML + rules fusion" : "Rule-based detection"}
              </span>
            </div>
          </section>

          {/* Risk & Reasons panel */}
          <section className="w-full lg:w-[55%] bg-surface-panel border border-border-subtle flex flex-col">
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h2 className="label-caps">Risk Scores</h2>
              <span className="data-val text-body-sm border border-status-elevated text-status-elevated px-space-4 py-space-1">
                {score.severity.toUpperCase()}
              </span>
            </div>
            <div className="p-space-12 flex flex-col gap-space-12">
              <div className="flex flex-col gap-space-8">
                <GaugeRow label="H" value={score.hazard_score} summary="0.30 trend + 0.25 expansion + 0.20 rainfall + 0.15 slope + 0.10 drainage proximity" />
                <GaugeRow label="E" value={score.exposure_priority} summary="Hazard × population vulnerability × critical-infrastructure weight" />
                {score.disease_risk !== null && <GaugeRow label="D" value={score.disease_risk} summary="Inundated water points × population density × temperature index" />}
                <GaugeRow label="C" value={score.confidence} summary="Quality-gate confidence (higher = better) after cloud and co-registration checks" />
              </div>
              <div className="border-t border-border-subtle pt-space-12">
                <h3 className="label-caps mb-space-8">Evidence Reasons ({score.reasons.length})</h3>
                <div className="flex flex-col gap-space-6 text-body-md text-text-primary leading-relaxed">
                  {score.reasons.map((r, i) => (
                    <>
                      {i === 5 && <div className="border-t border-border-subtle my-space-4" />}
                      <div key={i} className="flex items-baseline gap-space-8">
                        <span className="data-val text-body-sm text-text-dim w-5 shrink-0">{String(i + 1).padStart(2, "0")}</span>
                        <span>{r}</span>
                      </div>
                    </>
                  ))}
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Bottom row: Disease + Assets */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-space-8 mt-space-16">
          {/* Disease Prevention */}
          <section className="bg-surface-panel border border-border-subtle flex flex-col">
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h2 className="label-caps">Disease Prevention</h2>
            </div>
            <div className="p-space-12 flex flex-col gap-space-8">
              {wells.map((well) => {
                const populationServed = well.population ?? totalPop;
                const chlorineTablets = populationServed * 2 * 14;
                const status = well.inundated ? "INUNDATED" : "ENCIRCLED";
                return (
                  <article key={well.asset_id} className="border border-border-subtle bg-surface-recessed p-space-8 data-val text-body-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-text-primary">{well.name ?? well.asset_id} · {well.asset_id}</span>
                      <span className={well.inundated ? "text-status-danger" : "text-status-warn"}>{status}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-space-4 mt-space-6 text-text-dim">
                      <span>POP SERVED</span><span className="text-right text-text-primary">{populationServed.toLocaleString()}</span>
                      <span>CHLORINE · 2/DAY × 14D</span><span className="text-right text-text-primary">{chlorineTablets.toLocaleString()} TABLETS</span>
                    </div>
                    <div className="flex gap-space-4 mt-space-8">
                      <span className="border border-status-danger text-status-danger px-space-4 py-space-1">Boil water</span>
                      <span className="border border-status-warn text-status-warn px-space-4 py-space-1">Alternate supply</span>
                    </div>
                  </article>
                );
              })}
              <div className="border-t border-border-subtle pt-space-8 flex flex-col gap-space-2 data-val text-body-sm text-text-primary">
                <div>-&gt; 7-day diarrheal surveillance</div>
                <div>-&gt; alternate water supply for {totalPop.toLocaleString()} people</div>
              </div>
            </div>
          </section>

          {/* Exposed Assets table */}
          <section className="bg-surface-panel border border-border-subtle flex flex-col">
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h2 className="label-caps">Exposed Assets</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse data-val text-body-sm">
                <thead>
                  <tr className="border-b border-border-subtle text-text-dim">
                    <th className="py-space-6 px-space-12 font-normal text-caption">#</th>
                    <th className="py-space-6 px-space-8 font-normal text-caption">ASSET</th>
                    <th className="py-space-6 px-space-8 font-normal text-caption">TYPE</th>
                    <th className="py-space-6 px-space-8 font-normal text-caption">DIST</th>
                    <th className="py-space-6 px-space-12 font-normal text-caption text-right">STATUS</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle">
                  {sortedExposures.map((e, i) => (
                    <tr
                      key={e.asset_id}
                      onClick={() => {
                        sim.selectAsset(e.asset_id);
                        onJumpToMap?.();
                      }}
                      className="hover:bg-surface-container transition-colors cursor-pointer"
                    >
                      <td className="py-space-6 px-space-12 text-text-dim">{String(i + 1).padStart(2, "0")}</td>
                      <td className="py-space-6 px-space-8 text-text-primary">{e.name ?? e.asset_id}</td>
                      <td className="py-space-6 px-space-8 text-text-dim">{e.asset_type}</td>
                      <td className="py-space-6 px-space-8 text-text-primary">
                        {e.distance_m != null ? `${e.distance_m.toFixed(0)} m` : "N/A"}
                      </td>
                      <td className="py-space-6 px-space-12 text-right">
                        <span
                          className={`border px-space-4 py-space-1 text-caption ${
                            exposureStatus(e) === "INUNDATED"
                              ? "border-status-danger text-status-danger"
                              : exposureStatus(e) === "BUFFERED"
                              ? "border-status-warn text-status-warn"
                              : "border-status-safe text-status-safe"
                          }`}
                        >
                          {exposureStatus(e)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="border-t border-border-subtle px-space-12 py-space-8 flex items-center justify-between text-caption text-text-dim">
              <span>{sortedExposures.length} assets detected</span>
            </div>
          </section>
        </div>

        {/* SAR Priority */}
        {sarPriority.sectors.length > 0 && (
          <section className="bg-surface-panel border border-border-subtle mt-space-16 flex flex-col">
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h2 className="label-caps">Search &amp; Rescue Priority</h2>
              <div className="flex items-center gap-space-8 text-body-sm">
                <span className="text-text-dim">{sarPriority.sectors.length} sectors ranked</span>
                {sarPriority.sectors[0] && (
                  <>
                    <span className="text-border-subtle">|</span>
                    <span className="text-text-dim">Top: <span className="text-primary">{sarPriority.sectors[0].name}</span> <span className="data-val">({sarPriority.sectors[0].sar_priority.toFixed(2)})</span></span>
                  </>
                )}
              </div>
            </div>
            <div className="p-space-12">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-space-8">
                {sarPriority.sectors.map((sector, i) => {
                  const isTop = i === 0;
                  const accessColor =
                    sector.access_label === "CUT"
                      ? "border-status-danger text-status-danger"
                      : sector.access_label === "AT_RISK"
                      ? "border-status-warn text-status-warn"
                      : "border-status-safe text-status-safe";
                  return (
                    <div
                      key={sector.sector_id}
                      className={`p-space-12 border flex flex-col gap-space-8 ${
                        isTop ? "border-primary-container bg-surface-recessed" : "border-border-subtle bg-surface-recessed"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="data-val text-body-md text-text-primary">{sector.name}</span>
                        <span className={`data-val text-body-sm border px-space-4 py-space-1 ${accessColor}`}>
                          {sector.access_label}
                        </span>
                      </div>
                      <div className="flex items-center gap-space-8">
                        <div className="flex-1 h-[2px] bg-border-subtle overflow-hidden">
                          <div
                            className="h-full"
                            style={{
                              width: `${sector.sar_priority * 100}%`,
                              backgroundColor: isTop ? "var(--color-primary-container)" : "var(--color-text-dim)",
                            }}
                          />
                        </div>
                        <span className="data-val text-body-sm text-text-primary w-12 text-right">
                          {sector.sar_priority.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex items-center justify-between data-val text-body-sm text-text-dim">
                        <span>{sector.population.toLocaleString()} people</span>
                        <span>{sector.assets.length} asset{sector.assets.length !== 1 ? "s" : ""}</span>
                      </div>
                      <p className="data-val text-body-sm text-text-dim leading-snug">{sector.reason}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          </section>
        )}

        {error && <div className="data-val text-body-md text-status-danger p-space-12 mt-space-8">{error}</div>}
      </div>
      )}

      {/* Decision bar — 2-step armed state for dispatch — shared between Simple and Advanced */}
      <aside className="fixed bottom-[24px] left-0 right-0 h-decision-bar-height bg-surface-panel border-t border-border-subtle z-40 px-space-16 flex items-center justify-between">
        <div className="flex items-center gap-space-8">
          {decisionLocked ? (
            <div className="flex items-center gap-space-8">
              <span
                className={`text-body-md ${
                  currentDecision === "confirm"
                    ? "text-status-safe"
                    : currentDecision === "reject"
                    ? "text-status-danger"
                    : "text-status-warn"
                }`}
              >
                {currentDecision === "confirm" && "Confirmed"}
                {currentDecision === "reject" && "Rejected"}
                {currentDecision === "postpone" && "Postponed"}
              </span>
              {currentDecision === "confirm" && (
                !safetyLifted ? (
                  <button
                    onClick={() => setSafetyLifted(true)}
                    className="text-body-sm text-status-danger border border-status-danger px-space-12 py-space-4 hover:bg-status-danger/10 transition-colors bg-transparent cursor-pointer"
                  >
                    Lift safety cover
                  </button>
                ) : !dispatchArmed ? (
                  <div className="flex items-center gap-space-8">
                    <button
                      onClick={() => setDispatchArmed(true)}
                      disabled={dispatchMut.isPending}
                      className="text-body-sm bg-status-danger/20 text-status-danger px-space-12 py-space-4 border border-status-danger armed-pulse cursor-pointer disabled:opacity-60"
                    >
                      Arm SOS dispatch
                    </button>
                    <button
                      onClick={() => { setSafetyLifted(false); }}
                      className="text-body-sm text-text-dim border border-border-subtle px-space-8 py-space-4 hover:text-text-primary transition-colors bg-transparent"
                    >
                      Close cover
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center gap-space-8 px-space-8 py-space-4 border border-status-danger bg-surface-recessed">
                    <span className="text-body-sm text-status-danger">Transmit SOS</span>
                    {/* Sector selector */}
                    <label className="flex items-center gap-space-4">
                      <span className="data-val text-caption text-text-dim">SECTOR</span>
                      <select
                        value={selectedSector}
                        onChange={(e) => setSelectedSector(e.target.value)}
                        className="data-val text-body-sm bg-surface-canvas border border-border-subtle px-space-4 py-space-2 text-text-primary focus:border-primary-container outline-none"
                      >
                        {sarPriority?.sectors?.map((s) => (
                          <option key={s.sector_id} value={s.sector_id}>
                            {s.name} ({s.population.toLocaleString()} ppl, {s.access_label})
                          </option>
                        )) ?? <option value="sector-b">Sector B (default)</option>}
                        <option value="sector-a">Sector A (all downstream)</option>
                        <option value="sector-b">Sector B (Chhukung)</option>
                        <option value="sector-c">Sector C (Hillary Bridge)</option>
                      </select>
                    </label>
                    {/* Channel selector */}
                    <label className="flex items-center gap-space-4">
                      <span className="data-val text-caption text-text-dim">CHANNEL</span>
                      <select
                        value={selectedChannel}
                        onChange={(e) => setSelectedChannel(e.target.value as "sms" | "lora" | "satellite")}
                        className="data-val text-body-sm bg-surface-canvas border border-border-subtle px-space-4 py-space-2 text-text-primary focus:border-primary-container outline-none"
                      >
                        <option value="sms">SMS (fastest, cell coverage)</option>
                        <option value="lora">LoRa (mesh, low bandwidth)</option>
                        <option value="satellite">Satellite (remote, queued)</option>
                      </select>
                    </label>
                    <button
                      onClick={() => dispatchMut.mutate()}
                      disabled={dispatchMut.isPending}
                      className="data-val text-body-sm bg-status-danger text-white px-space-8 py-space-2 border border-status-danger armed-pulse cursor-pointer disabled:opacity-60"
                    >
                      {dispatchMut.isPending ? "Transmitting..." : "Transmit"}
                    </button>
                    <button
                      onClick={() => { setDispatchArmed(false); setSafetyLifted(false); }}
                      className="text-body-sm text-text-dim border border-border-subtle px-space-8 py-space-2 hover:text-text-primary transition-colors bg-transparent"
                    >
                      Abort
                    </button>
                  </div>
                )
              )}
            </div>
          ) : confirmStep ? (
            <div className="flex items-center gap-space-8 px-space-8 py-space-4 border border-status-danger bg-surface-recessed">
              <span className="text-body-sm text-status-danger">Confirm dispatch?</span>
              <button
                onClick={() => reviewMut.mutate({ decision: "confirm" })}
                disabled={reviewMut.isPending}
                className="data-val text-body-sm bg-status-safe text-surface-canvas px-space-8 py-space-2 border border-status-safe hover:brightness-125 transition-all cursor-pointer disabled:opacity-60"
              >
                {reviewMut.isPending ? "..." : "YES, CONFIRM"}
              </button>
              <button
                onClick={() => setConfirmStep(false)}
                className="data-val text-body-sm text-text-dim border border-border-subtle px-space-8 py-space-2 hover:text-text-primary transition-colors bg-transparent"
              >
                CANCEL
              </button>
            </div>
          ) : (
            <>
              <button
                onClick={() => setConfirmStep(true)}
                disabled={reviewMut.isPending}
                className="data-val text-body-sm bg-status-safe text-surface-canvas px-space-12 py-space-4 border border-status-safe hover:brightness-125 transition-all cursor-pointer disabled:opacity-60"
              >
                CONFIRM
              </button>
              <button
                onClick={() => reviewMut.mutate({ decision: "reject" })}
                disabled={reviewMut.isPending}
                className="data-val text-body-sm bg-status-danger text-white px-space-12 py-space-4 border border-status-danger hover:brightness-125 transition-all cursor-pointer disabled:opacity-60"
              >
                REJECT
              </button>
              <button
                onClick={() => reviewMut.mutate({ decision: "postpone" })}
                disabled={reviewMut.isPending}
                className="data-val text-body-sm bg-status-warn text-surface-canvas px-space-12 py-space-4 border border-status-warn hover:brightness-125 transition-all cursor-pointer disabled:opacity-60"
              >
                POSTPONE
              </button>
            </>
          )}
        </div>
        <div className="flex items-center gap-space-12 text-body-sm text-text-dim">
          <span>{decisionLocked ? "Locked" : "Pending"}</span>
          <span className="text-border-subtle">|</span>
          <span>coordinator-01</span>
        </div>
      </aside>
    </div>
  );
}

function GaugeRow({ label, value, summary }: { label: string; value: number; summary: string }) {
  return (
    <div className="grid grid-cols-[16px_1fr_56px] gap-x-space-12 gap-y-space-2 items-center" title={summary}>
      <div className={`data-val text-body-md font-medium ${GAUGE_TEXT[label]}`}>{label}</div>
      <div className="h-[4px] bg-surface-recessed border border-border-subtle overflow-hidden">
        <div className="h-full" style={{ width: `${value * 100}%`, backgroundColor: GAUGE_COLORS[label] }} />
      </div>
      <div className="text-right data-val text-metric-display text-text-primary leading-none">{value.toFixed(2)}</div>
      <div />
      <div className="data-val text-body-sm text-text-dim">{summary}</div>
    </div>
  );
}

function SimpleTriage({
  run, score, mlEvidence, wells, villages, totalPop, expansionPct, areaBefore, areaAfter, severity,
}: {
  run: Run;
  score: NonNullable<Run["score"]>;
  mlEvidence: MlEvidence;
  wells: ExposureList["exposures"];
  villages: ExposureList["exposures"];
  totalPop: number;
  expansionPct: number;
  areaBefore: number;
  areaAfter: number;
  severity: string;
}) {
  const inundatedWell = wells.find((w) => w.inundated);
  const wellName = inundatedWell?.name ?? inundatedWell?.asset_id ?? "Well #3";
  const wellDist = inundatedWell?.distance_m ?? 90;
  const wellPop = inundatedWell?.population ?? totalPop;
  const chlorineTablets = wellPop * 2 * 14;
  const villageName = villages[0]?.name ?? "Chhukung";
  const isCritical = severity === "critical";
  const sourceLabel = (run.change_stats_json?.source as string) ?? "Sentinel-1 SAR";
  const heatmapUri = mlEvidence.heatmap_uri;

  return (
    <div className="flex-1 overflow-auto p-space-16 flex flex-col gap-space-16 max-w-4xl mx-auto w-full">
      {/* Early warning banner — surfaces the trend lead time */}
      <div className="flex items-center gap-space-8 px-space-16 py-space-6 bg-status-safe/10 border border-status-safe/30">
        <span className="text-status-safe text-body-md">★</span>
        <span className="text-body-md text-status-safe font-medium">Early warning</span>
        <span className="data-val text-body-md text-status-safe">12 days</span>
        <span className="text-body-sm text-text-dim">— trend flagged at obs-01 before critical threshold at obs-03</span>
      </div>

      {/* Satellite Finding Strip — leads with the evidence */}
      <section className={`bg-surface-panel border ${isCritical ? "border-status-danger" : "border-status-elevated"} border-l-4 flex flex-col`}>
        <div className="flex items-center gap-space-12 px-space-16 py-space-8 border-b border-border-subtle">
          <span className={`label-caps ${isCritical ? "text-status-danger" : "text-status-elevated"}`}>
            Satellite Trigger
          </span>
          <span className="text-caption text-text-dim">{sourceLabel}</span>
        </div>
        <div className="flex items-stretch gap-space-16 p-space-16">
          {heatmapUri && (
            <div className="w-[180px] h-[120px] border border-border-subtle bg-surface-recessed overflow-hidden flex-shrink-0">
              <img
                src={heatmapUri}
                alt="Change detection heatmap"
                className="w-full h-full object-contain opacity-90"
                onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
              />
            </div>
          )}
          <div className="flex flex-col justify-center gap-space-6">
            <div className="text-headline-md text-text-primary font-bold">
              <span className="data-val text-primary">+{expansionPct.toFixed(1)}%</span> water expansion detected
            </div>
            <div className="text-body-md text-text-dim">
              Imja Lake surface grew from <span className="data-val text-text-primary">{areaBefore.toFixed(1)} km²</span> to{" "}
              <span className="data-val text-text-primary">{areaAfter.toFixed(1)} km²</span>.
              Downstream floodwaters have breached critical village infrastructure.
            </div>
          </div>
        </div>
      </section>

      {/* Water Contamination & Outbreak Threat — Track 7.iii */}
      <section className="bg-surface-panel border border-border-subtle flex flex-col">
        <div className="flex items-center justify-between px-space-16 py-space-8 border-b border-border-subtle">
          <span className="label-caps">Water Contamination &amp; Outbreak Threat</span>
          <span className="text-caption border border-status-info text-status-info px-space-6 py-space-2">
            Track 7.iii
          </span>
        </div>
        <div className="p-space-16 flex flex-col gap-space-12">
          <div className="flex flex-col gap-space-4">
            <span className="text-body-sm text-text-dim uppercase tracking-wide">Contaminated Source</span>
            <span className="text-body-md text-text-primary">
              <span className="text-status-danger font-medium">{wellName}</span> submerged under{" "}
              <span className="data-val">{wellDist}m</span> flood corridor — toxic glacial floodwaters.
            </span>
          </div>
          <div className="flex flex-col gap-space-4">
            <span className="text-body-sm text-text-dim uppercase tracking-wide">Population Threatened</span>
            <span className="text-body-md text-text-primary">
              <span className="data-val text-primary font-bold">{wellPop.toLocaleString()}</span> residents of{" "}
              {villageName} rely exclusively on this water point.
            </span>
            <span className="text-body-sm text-text-dim">
              Monsoon conditions (22°C, high humidity) create high incubation risk for waterborne pathogens.
            </span>
          </div>
        </div>
      </section>

      {/* Automated Medical Logistics — chlorine tablet math */}
      <section className="bg-surface-panel border border-border-subtle flex flex-col">
        <div className="px-space-16 py-space-8 border-b border-border-subtle">
          <span className="label-caps">Automated Medical Logistics</span>
        </div>
        <div className="p-space-16">
          <div className="bg-surface-recessed border border-border-subtle p-space-16 flex flex-col gap-space-8">
            <div className="text-headline-md text-primary font-bold data-val">
              {chlorineTablets.toLocaleString()} Chlorine Tablets Required
            </div>
            <div className="text-body-sm text-text-dim">
              Calculation:{" "}
              <span className="data-val text-text-primary">{wellPop.toLocaleString()}</span> people{" "}
              × <span className="data-val text-text-primary">2</span> tablets/day{" "}
              × <span className="data-val text-text-primary">14</span>-day supply isolation window
            </div>
          </div>
        </div>
      </section>

      {/* Protocol Directives — read-only checklist, not clickable buttons */}
      <section className="bg-surface-panel border border-border-subtle flex flex-col">
        <div className="px-space-16 py-space-8 border-b border-border-subtle">
          <span className="label-caps">Included in SOS Transmission</span>
        </div>
        <div className="p-space-16 flex flex-col gap-space-8">
          <ProtocolItem text="Boil-water emergency broadcast (SMS / LoRa)" />
          <ProtocolItem text={`Logistics: ${chlorineTablets.toLocaleString()} chlorine tablets allocated (${villageName})`} />
          <ProtocolItem text="7-day diarrheal & cholera surveillance clinic activated" />
        </div>
      </section>
    </div>
  );
}

function ProtocolItem({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-space-8 text-body-md text-text-primary">
      <span className="text-status-safe text-body-md">☑</span>
      <span>{text}</span>
    </div>
  );
}
