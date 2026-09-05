import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation } from "../simulation/SimulationContext";
import type { Run, ExposureList, SarPriorityList, MlEvidence, ReviewResponse, DispatchResponse, ApiError } from "../api/types";

interface Props {
  run: Run;
  onToast?: (t: { msg: string; type: "error" | "info" | "success" }) => void;
  onJumpToMap?: () => void;
}

const GAUGE_COLORS: Record<string, string> = {
  H: "#EF4444",
  E: "#F59E0B",
  D: "#3B82F6",
  C: "#22C55E",
};

const GAUGE_TEXT: Record<string, string> = {
  H: "text-status-danger",
  E: "text-status-warn",
  D: "text-status-info",
  C: "text-status-safe",
};

export default function ReviewView({ run, onToast, onJumpToMap }: Props) {
  const qc = useQueryClient();
  const sim = useSimulation();
  const [confirmStep, setConfirmStep] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const score = run.score;
  const runId = run.run_id;

  const { data: exposuresData, isLoading: expLoading } = useQuery({
    queryKey: ["exposures", runId],
    queryFn: () => apiOrMock(() => api.listExposures(runId), "exposures") as Promise<ExposureList>,
  });

  const { data: sarData } = useQuery({
    queryKey: ["sar-priority", runId],
    queryFn: () => apiOrMock(() => api.getSarPriority(runId), "sarPriority") as Promise<SarPriorityList>,
  });

  const { data: mlData } = useQuery({
    queryKey: ["ml-evidence", runId],
    queryFn: () => apiOrMock(() => api.getMlEvidence(runId), "mlEvidence") as Promise<MlEvidence>,
  });

  const sarPriority = sarData ?? mockData.sarPriority;
  const mlEvidence = mlData ?? mockData.mlEvidence;
  const exposures = exposuresData?.exposures ?? mockData.exposures.exposures;
  const wells = exposures.filter((e) => e.asset_type === "well");
  const villages = exposures.filter((e) => e.asset_type === "village");
  const totalPop = villages.reduce((s, v) => s + (v.population ?? 0), 0);

  const reviewMut = useMutation({
    mutationFn: (vars: { decision: "confirm" | "reject" | "postpone" }) =>
      apiOrMock(() => api.createReview(runId, "coordinator-01", vars.decision, "demo review"), "dispatch" as any) as Promise<ReviewResponse>,
    onSuccess: (_data, vars) => {
      sim.setReviewDecision(vars.decision);
      setConfirmStep(false);
      setError(null);
      onToast?.({ msg: `Decision recorded: ${vars.decision}`, type: "success" });
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
    onError: (e: ApiError) => {
      setError(e.detail);
      onToast?.({ msg: e.detail, type: "error" });
    },
  });

  const dispatchMut = useMutation({
    mutationFn: () => apiOrMock(() => api.createDispatch(runId, "sms", "sector-b"), "dispatch") as Promise<DispatchResponse>,
    onSuccess: (data) => {
      sim.setDispatchResult(data);
      setError(null);
      onToast?.({ msg: `Dispatch sent (${data.payload_bytes} bytes)`, type: "success" });
    },
    onError: (e: ApiError) => {
      setError(e.detail);
      onToast?.({ msg: e.detail, type: "error" });
    },
  });

  if (!score || (score.severity !== "elevated" && score.severity !== "critical")) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-text-dim text-center gap-space-8">
        <div className="text-body-lg">No alerts requiring review</div>
        <div className="text-body-md">Run the simulation from the Timeline tab to generate observations.</div>
      </div>
    );
  }

  if (expLoading) {
    return (
      <div className="flex flex-col gap-space-16 h-full">
        <h1 className="text-headline-lg text-text-primary font-medium">Review</h1>
        <div className="bg-surface-panel border border-border-subtle rounded-xl p-space-16 space-y-space-12">
          <div className="h-5 bg-surface-container-high rounded animate-pulse" />
          <div className="h-5 bg-surface-container-high rounded animate-pulse" />
          <div className="h-5 bg-surface-container-high rounded animate-pulse" />
        </div>
      </div>
    );
  }

  const decisionLocked = sim.reviewDecision !== null;
  const sortedExposures = [...exposures].sort((a, b) => (a.distance_m ?? 9999) - (b.distance_m ?? 9999));
  const areaAfter = (run.change_stats_json?.water_area_km2 as number) ?? 4.1;
  const areaBefore = 3.0;

  return (
    <div className="flex flex-col h-full pb-[68px]">
      {/* Header */}
      <div className="flex items-center justify-between mb-space-16">
        <h1 className="text-headline-lg text-text-primary tracking-tight flex items-center gap-space-8">
          <span>Review — Run</span>
          <span className="font-mono text-code-lg text-primary-container px-space-6 py-space-2 bg-surface-recessed rounded border border-border-subtle">
            {runId}
          </span>
        </h1>
        <div className="flex items-center gap-space-8 font-mono text-code-sm text-text-dim">
          <span>TRIGGER: AUTOMATED_INGESTION_SENTINEL_2</span>
          <span>·</span>
          <span>UTC {new Date().toISOString().slice(0, 19).replace("T", " ")}</span>
        </div>
      </div>

      {/* Top row: Evidence + Risk & Reasons */}
      <div className="flex flex-col lg:flex-row gap-space-16 w-full items-stretch">
        {/* Evidence — real raster imagery + ML heatmap */}
        <section className="w-full lg:w-[45%] bg-surface-panel border border-border-subtle rounded-xl p-space-16 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-space-12">
              <h2 className="text-headline-md text-text-primary">Evidence</h2>
              <div className="flex items-center gap-space-6">
                <span className={`px-space-6 py-space-2 rounded text-caption uppercase tracking-wider border ${
                  mlEvidence.model_available
                    ? "border-primary-container text-primary-container"
                    : "border-status-warn text-status-warn"
                }`}>
                  {mlEvidence.model_available ? "ML ACTIVE" : "ML SCAFFOLD"}
                </span>
                <span className="font-mono text-code-sm text-text-dim">S2A_MSIL2A</span>
              </div>
            </div>

            {/* Before/After raster imagery */}
            <div className="flex gap-space-12 w-full mb-space-12">
              <div className="flex-1 h-[160px] rounded-xl border border-border-subtle bg-surface-recessed relative overflow-hidden flex flex-col">
                <img
                  src={mlEvidence.baseline_mask_uri}
                  alt="Baseline water mask"
                  className="absolute inset-0 w-full h-full object-cover opacity-80"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-surface-recessed via-transparent to-transparent" />
                <div className="relative z-10 flex items-center justify-between p-space-8">
                  <span className="text-body-sm text-text-dim uppercase tracking-wider bg-surface-canvas/60 px-space-4 rounded">Before</span>
                  <span className="font-mono text-code-sm text-text-dim bg-surface-canvas/60 px-space-4 rounded">T-14d</span>
                </div>
                <div className="relative z-10 flex flex-col p-space-8 mt-auto">
                  <span className="text-headline-md text-text-primary font-medium">{areaBefore.toFixed(1)} km²</span>
                  <span className="text-caption text-text-dim">Reference Extent</span>
                </div>
              </div>
              <div className="flex-1 h-[160px] rounded-xl border-2 border-primary-container/40 bg-surface-recessed relative overflow-hidden flex flex-col">
                <img
                  src={mlEvidence.mask_uri}
                  alt="Current change mask"
                  className="absolute inset-0 w-full h-full object-cover opacity-90"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-surface-recessed via-transparent to-transparent" />
                <div className="relative z-10 flex items-center justify-between p-space-8">
                  <span className="text-body-sm text-text-dim uppercase tracking-wider bg-surface-canvas/60 px-space-4 rounded">After</span>
                  <span className="font-mono text-code-sm text-primary-container font-medium bg-surface-canvas/60 px-space-4 rounded">CHANGE</span>
                </div>
                <div className="relative z-10 flex flex-col p-space-8 mt-auto">
                  <span className="text-headline-md text-primary-container font-medium">{areaAfter.toFixed(1)} km²</span>
                  <span className="text-caption text-primary-container/80">+{((run.change_stats_json?.expansion_percent as number) ?? 0).toFixed(1)}% Expansion</span>
                </div>
              </div>
            </div>

            {/* ML Change Detection Heatmap */}
            <div className="h-[140px] rounded-xl border border-border-subtle bg-surface-recessed relative overflow-hidden flex flex-col">
              <img
                src={mlEvidence.heatmap_uri}
                alt="ML change detection heatmap"
                className="absolute inset-0 w-full h-full object-cover"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-surface-recessed/90 via-transparent to-surface-recessed/40" />
              <div className="relative z-10 flex items-center justify-between p-space-8">
                <div className="flex items-center gap-space-6">
                  <span className="w-2 h-2 rounded-full bg-status-danger animate-pulse" />
                  <span className="text-body-sm text-text-primary font-medium bg-surface-canvas/60 px-space-4 rounded">
                    {mlEvidence.model_available ? "Siamese U-Net Change Probability" : "Change Detection Heatmap"}
                  </span>
                </div>
                <span className="font-mono text-code-sm text-text-dim bg-surface-canvas/60 px-space-4 rounded">
                  {(mlEvidence.ml_confidence_mean * 100).toFixed(0)}% conf
                </span>
              </div>
              <div className="relative z-10 flex items-center justify-between p-space-8 mt-auto">
                <div className="flex items-center gap-space-8">
                  <div className="flex items-center gap-space-4">
                    <span className="w-3 h-3 rounded-sm bg-status-danger" />
                    <span className="text-caption text-text-dim">High change</span>
                  </div>
                  <div className="flex items-center gap-space-4">
                    <span className="w-3 h-3 rounded-sm bg-status-warn" />
                    <span className="text-caption text-text-dim">Moderate</span>
                  </div>
                  <div className="flex items-center gap-space-4">
                    <span className="w-3 h-3 rounded-sm bg-surface-canvas border border-border-subtle" />
                    <span className="text-caption text-text-dim">No change</span>
                  </div>
                </div>
                <span className="font-mono text-code-sm text-text-dim">
                  {mlEvidence.ml_consensus_pixels.toLocaleString()} px
                </span>
              </div>
            </div>

            <div className="mt-space-8">
              <p className="font-mono text-code-sm text-text-dim break-all select-all">
                {mlEvidence.ml_source === "siamese_unet_consensus"
                  ? "ML: Siamese U-Net + rule-based consensus (ADR-002)"
                  : "ML: scaffold mode — rule-based detection with confidence gradient"}
              </p>
            </div>
          </div>
          <div className="mt-space-12 pt-space-12 border-t border-border-subtle grid grid-cols-2 gap-space-8 font-mono text-code-sm text-text-dim">
            <div>
              Projection: <span className="text-text-primary">WGS84 / UTM 45N</span>
            </div>
            <div>
              Confidence: <span className="text-text-primary">{(score.confidence * 100).toFixed(1)}%</span>
              <span className="text-border-subtle"> · </span>
              <span className={mlEvidence.model_available ? "text-primary-container" : "text-status-warn"}>
                {mlEvidence.model_available ? "ML+Rule" : "Rule-only"}
              </span>
            </div>
          </div>
        </section>

        {/* Risk & Reasons */}
        <section className="w-full lg:w-[55%] bg-surface-panel border border-border-subtle rounded-xl p-space-16 flex flex-col">
          <div className="flex items-center justify-between mb-space-12">
            <h2 className="text-headline-md text-text-primary">Risk Scores</h2>
            <span className="px-space-8 py-space-2 rounded border border-status-elevated text-status-elevated text-body-sm uppercase tracking-wider bg-transparent">
              ELEVATED
            </span>
          </div>
          <div className="flex flex-col gap-space-12 mb-space-12">
            <GaugeRow label="H" value={score.hazard_score} />
            <GaugeRow label="E" value={score.exposure_priority} />
            {score.disease_risk !== null && <GaugeRow label="D" value={score.disease_risk} />}
            <GaugeRow label="C" value={score.confidence} />
          </div>
          <div className="border-t border-border-subtle my-space-16" />
          <div className="flex flex-col">
            <h3 className="text-headline-md text-text-primary mb-space-8">Evidence Reasons ({score.reasons.length})</h3>
            <div className="flex flex-col gap-space-6 text-body-md text-text-primary">
              {score.reasons.map((r, i) => (
                <div key={i} className="flex items-baseline gap-space-6">
                  <span className="font-mono text-code-sm text-primary-container w-4 shrink-0">{i + 1}.</span>
                  <span>{r}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>

      {/* Bottom row: Disease + Assets */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-space-16 w-full items-stretch mt-space-16">
        <section className="bg-surface-panel border border-border-subtle rounded-xl p-space-16 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-space-12">
              <h2 className="text-headline-md text-text-primary">Disease Prevention</h2>
              <span className="text-caption text-text-dim uppercase tracking-wider">Protocol W-4</span>
            </div>
            <div className="flex flex-col gap-space-12">
              {wells.map((w) => {
                const isSubmerged = w.inundated;
                return (
                  <div key={w.asset_id} className="flex flex-col">
                    <div className={`text-body-md font-medium ${isSubmerged ? "text-status-danger" : "text-status-warn"}`}>
                      {w.name ?? w.asset_id} — {isSubmerged ? "submerged" : "encircled"}
                    </div>
                    <div className="flex flex-col gap-space-2 mt-space-2 pl-space-8 text-body-md text-text-primary">
                      {isSubmerged ? (
                        <>
                          <div>→ chlorine ×200 tablets</div>
                          <div>→ boil water notice</div>
                        </>
                      ) : (
                        <div>→ monitor for contamination</div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-space-12 pt-space-12 border-t border-border-subtle flex flex-col gap-space-4 text-body-md text-text-primary">
              <div>→ 7-day diarrheal surveillance</div>
              <div>→ alternate water supply for {totalPop.toLocaleString()} people</div>
            </div>
          </div>
          <div className="mt-space-16 text-caption text-text-dim">
            Direct dispatch will stage medical water sanitation packs at Chhukung depot.
          </div>
        </section>

        <section className="bg-surface-panel border border-border-subtle rounded-xl p-space-16 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-space-12">
              <h2 className="text-headline-md text-text-primary">Exposed Assets</h2>
              <span className="font-mono text-code-sm text-text-dim">Buffer: 250m contour</span>
            </div>
            <div className="w-full overflow-x-auto">
              <table className="w-full text-left border-collapse text-body-md text-text-primary">
                <thead>
                  <tr className="border-b border-border-subtle text-body-sm text-text-dim">
                    <th className="py-space-8 px-space-8 font-normal">#</th>
                    <th className="py-space-8 px-space-8 font-normal">Asset</th>
                    <th className="py-space-8 px-space-8 font-normal">Type</th>
                    <th className="py-space-8 px-space-8 font-normal">Dist</th>
                    <th className="py-space-8 px-space-8 font-normal text-right">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle font-mono text-code-sm">
                  {sortedExposures.map((e, i) => (
                    <tr
                      key={e.asset_id}
                      onClick={() => {
                        sim.selectAsset(e.asset_id);
                        onJumpToMap?.();
                      }}
                      className="hover:bg-surface-recessed/40 transition-colors cursor-pointer"
                    >
                      <td className="py-space-8 px-space-8 text-text-dim">{i + 1}</td>
                      <td className="py-space-8 px-space-8 text-body-md text-text-primary font-medium">{e.name ?? e.asset_id}</td>
                      <td className="py-space-8 px-space-8 text-text-dim">{e.asset_type}</td>
                      <td className="py-space-8 px-space-8">{e.distance_m?.toFixed(0)}m</td>
                      <td className="py-space-8 px-space-8 text-right">
                        <span
                          className={`inline-block px-space-8 py-space-2 rounded text-body-sm bg-transparent border ${
                            e.inundated
                              ? "border-status-danger text-status-danger"
                              : "border-status-warn text-status-warn"
                          }`}
                        >
                          {e.inundated ? "INUNDATED" : "BUFFERED"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="mt-space-16 pt-space-12 border-t border-border-subtle flex items-center justify-between text-caption text-text-dim">
            <span>Inundation vector model: Hydraulic-1D</span>
            <span>4 critical infra points detected</span>
          </div>
        </section>
      </div>

      {/* SAR Priority Layer (PRD §15 stretch goal) */}
      {sarPriority.sectors.length > 0 && (
        <section className="bg-surface-panel border border-border-subtle rounded-xl p-space-16 flex flex-col gap-space-12 mt-space-16">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-space-8">
              <h2 className="text-headline-md text-text-primary">Search &amp; Rescue Priority</h2>
              <span className="text-caption text-text-dim uppercase tracking-wider px-space-6 py-space-2 rounded border border-border-subtle">
                STRETCH
              </span>
            </div>
            <span className="font-mono text-code-sm text-text-dim">
              {sarPriority.sectors.length} sectors ranked
            </span>
          </div>
          <p className="text-body-md text-text-dim">{sarPriority.summary}</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-space-12">
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
                  className={`p-space-12 rounded-lg border flex flex-col gap-space-8 ${
                    isTop ? "border-primary-container bg-surface-recessed" : "border-border-subtle bg-surface-recessed"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-space-6">
                      {isTop && <span className="text-primary-container text-body-sm font-medium">★</span>}
                      <span className="text-body-md text-text-primary font-medium">{sector.name}</span>
                    </div>
                    <span className={`inline-block px-space-6 py-space-2 rounded text-caption uppercase tracking-wider bg-transparent border ${accessColor}`}>
                      {sector.access_label}
                    </span>
                  </div>
                  <div className="flex items-center gap-space-8">
                    <div className="flex-1 h-[4px] bg-border-subtle rounded overflow-hidden">
                      <div
                        className="h-full rounded"
                        style={{
                          width: `${sector.sar_priority * 100}%`,
                          backgroundColor: isTop ? "#06B6D4" : "#94A3B8",
                        }}
                      />
                    </div>
                    <span className="font-mono text-code-sm text-text-primary w-12 text-right">
                      {sector.sar_priority.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between font-mono text-code-sm text-text-dim">
                    <span>{sector.population.toLocaleString()} people</span>
                    <span>{sector.assets.length} asset{sector.assets.length !== 1 ? "s" : ""}</span>
                  </div>
                  <p className="text-body-sm text-text-dim leading-snug">{sector.reason}</p>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {error && <div className="text-status-danger text-body-md p-space-16 mt-space-8">{error}</div>}

      {/* Decision bar */}
      <aside className="fixed bottom-[28px] left-0 right-0 h-decision-bar-height bg-surface-panel border-t border-border-subtle z-40 px-space-16 flex items-center justify-between">
        <div className="flex items-center gap-space-8">
          {decisionLocked ? (
            <div className="flex items-center gap-space-8">
              <span
                className={`text-body-md font-medium ${
                  sim.reviewDecision === "confirm"
                    ? "text-status-safe"
                    : sim.reviewDecision === "reject"
                    ? "text-status-danger"
                    : "text-status-warn"
                }`}
              >
                {sim.reviewDecision === "confirm" && "✓ Confirmed"}
                {sim.reviewDecision === "reject" && "✗ Rejected"}
                {sim.reviewDecision === "postpone" && "⏸ Postponed"}
              </span>
              {sim.reviewDecision === "confirm" && (
                <button
                  onClick={() => dispatchMut.mutate()}
                  disabled={dispatchMut.isPending}
                  className="h-8 px-4 py-1.5 bg-primary-container text-surface-canvas text-body-md font-medium rounded hover:brightness-110 active:brightness-95 transition-all flex items-center gap-space-4 cursor-pointer disabled:opacity-60"
                >
                  {dispatchMut.isPending ? "Sending..." : "Send Dispatch"}
                </button>
              )}
            </div>
          ) : confirmStep ? (
            <div className="flex items-center gap-space-8 px-space-14 py-space-8 rounded border border-status-danger bg-status-danger/10">
              <span className="font-medium text-status-danger">Confirm SOS dispatch?</span>
              <button
                onClick={() => reviewMut.mutate({ decision: "confirm" })}
                disabled={reviewMut.isPending}
                className="h-8 px-4 py-1.5 bg-status-safe text-surface-canvas text-body-md font-medium rounded hover:brightness-110 active:brightness-95 transition-all cursor-pointer disabled:opacity-60"
              >
                Yes, confirm
              </button>
              <button
                onClick={() => setConfirmStep(false)}
                className="h-8 px-4 py-1.5 border border-border-subtle text-text-dim text-body-md rounded hover:text-text-primary hover:border-text-dim transition-colors bg-transparent"
              >
                Cancel
              </button>
            </div>
          ) : (
            <>
              <button
                onClick={() => setConfirmStep(true)}
                disabled={reviewMut.isPending}
                className="h-8 px-4 py-1.5 bg-status-safe text-surface-canvas text-body-md font-medium rounded hover:brightness-110 active:brightness-95 transition-all flex items-center gap-space-4 cursor-pointer disabled:opacity-60"
              >
                <span>✓</span>
                <span>Confirm</span>
              </button>
              <button
                onClick={() => reviewMut.mutate({ decision: "reject" })}
                disabled={reviewMut.isPending}
                className="h-8 px-4 py-1.5 bg-status-danger text-white text-body-md font-medium rounded hover:brightness-110 active:brightness-95 transition-all flex items-center gap-space-4 cursor-pointer disabled:opacity-60"
              >
                <span>✗</span>
                <span>Reject</span>
              </button>
              <button
                onClick={() => reviewMut.mutate({ decision: "postpone" })}
                disabled={reviewMut.isPending}
                className="h-8 px-4 py-1.5 bg-status-warn text-surface-canvas text-body-md font-medium rounded hover:brightness-110 active:brightness-95 transition-all flex items-center gap-space-4 cursor-pointer disabled:opacity-60"
              >
                <span>⏸</span>
                <span>Postpone</span>
              </button>
            </>
          )}
        </div>
        <div className="flex items-center gap-space-12 font-mono text-code-sm text-text-dim">
          <span className="hidden sm:inline">STATE: PENDING_SIG</span>
          <span className="hidden sm:inline">·</span>
          <span>reviewer: coordinator-01</span>
        </div>
      </aside>
    </div>
  );
}

function GaugeRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-space-12">
      <div className={`w-4 text-headline-sm font-medium ${GAUGE_TEXT[label]}`}>{label}</div>
      <div className="flex-1 h-[6px] bg-surface-recessed rounded border border-border-subtle overflow-hidden">
        <div className="h-full rounded" style={{ width: `${value * 100}%`, backgroundColor: GAUGE_COLORS[label] }} />
      </div>
      <div className="w-14 text-right font-mono text-metric-display text-text-primary leading-none">{value.toFixed(2)}</div>
    </div>
  );
}
