import { useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation } from "../simulation/SimulationContext";
import type { Run, ExposureList, SarPriorityList, MlEvidence, ReviewResponse, DispatchResponse, ApiError } from "../api/types";

interface Props {
  run?: Run;
  onToast?: (t: { msg: string; type: "error" | "info" | "success" }) => void;
  onJumpToMap?: () => void;
}

const GAUGE_COLORS: Record<string, string> = {
  H: "#ef4444",
  E: "#f59e0b",
  D: "#0ea5e9",
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
  const [error, setError] = useState<string | null>(null);

  const score = run?.score;
  const runId = run?.run_id;

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
    mutationFn: (vars: { decision: "confirm" | "reject" | "postpone" }) =>
      api.createReview(runId!, "coordinator-01", vars.decision, "demo review") as Promise<ReviewResponse>,
    onSuccess: (_data, vars) => {
      sim.setReviewDecision(vars.decision);
      setConfirmStep(false);
      setError(null);
      onToast?.({ msg: `Decision recorded: ${vars.decision}`, type: "success" });
      qc.invalidateQueries({ queryKey: ["runs"] });
      qc.invalidateQueries({ queryKey: ["run", runId] });
    },
    onError: (e: ApiError) => {
      setError(e.detail);
      onToast?.({ msg: e.detail, type: "error" });
    },
  });

  const dispatchMut = useMutation({
    mutationFn: () => api.createDispatch(runId!, "sms", "sector-b") as Promise<DispatchResponse>,
    onSuccess: (data) => {
      sim.setDispatchResult(data);
      setError(null);
      setDispatchArmed(false);
      onToast?.({ msg: `Dispatch sent (${data.payload_bytes} bytes)`, type: "success" });
    },
    onError: (e: ApiError) => {
      setError(e.detail);
      onToast?.({ msg: e.detail, type: "error" });
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

  if (!run || !score || (score.severity !== "elevated" && score.severity !== "critical")) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-text-dim text-center gap-space-8">
        <div className="data-val text-body-md">NO ALERTS REQUIRING REVIEW</div>
        <div className="data-val text-body-sm text-text-muted">Run the simulation from Timeline to generate observations.</div>
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

  const currentDecision = run.decision ?? sim.reviewDecision;
  const decisionLocked = currentDecision !== null;
  const sortedExposures = [...exposures].sort((a, b) => (a.distance_m ?? 9999) - (b.distance_m ?? 9999));
  const areaAfter = (run.change_stats_json?.water_area_km2 as number) ?? 4.1;
  const areaBefore = 3.0;

  return (
    <div className="flex flex-col h-full pb-[48px] overflow-auto">
      {/* Header bar */}
      <div className="flex items-center justify-between px-space-16 py-space-8 border-b border-border-subtle bg-surface-panel">
        <div className="flex items-center gap-space-12">
          <h1 className="label-caps">Review</h1>
          <span className="data-val text-body-sm text-text-dim">RUN</span>
          <span className="data-val text-body-sm text-primary-container border border-border-subtle px-space-4 py-space-1">
            {runId}
          </span>
        </div>
        <div className="flex items-center gap-space-8 data-val text-body-sm text-text-dim">
          <span>TRIGGER: AUTOMATED_INGESTION_SENTINEL_2</span>
          <span className="text-border-subtle">|</span>
          <span>UTC {new Date().toISOString().slice(0, 19).replace("T", " ")}</span>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-space-12">
        {/* Top row: Evidence + Risk */}
        <div className="flex flex-col lg:flex-row gap-space-8 items-stretch">
          {/* Evidence panel */}
          <section className="w-full lg:w-[45%] bg-surface-panel border border-border-subtle flex flex-col">
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h2 className="label-caps">Evidence</h2>
              <div className="flex items-center gap-space-6">
                <span className={`data-val text-body-sm border px-space-4 py-space-1 ${
                  mlEvidence.model_available
                    ? "border-primary-container text-primary-container"
                    : "border-status-warn text-status-warn"
                }`}>
                  {mlEvidence.model_available ? "ML ACTIVE" : "ML SCAFFOLD"}
                </span>
                <span className="data-val text-body-sm text-text-dim">S2A_MSIL2A</span>
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
                    <span className="data-val text-body-sm text-text-dim">BEFORE</span>
                    <span className="data-val text-body-sm text-text-dim">T-14d</span>
                  </div>
                  <div className="relative z-10 flex flex-col px-space-8 py-space-4 mt-auto bg-surface-panel border-t border-border-subtle">
                    <span className="data-val text-headline-md text-text-primary">{areaBefore.toFixed(2)} km²</span>
                    <span className="data-val text-caption text-text-dim">REFERENCE EXTENT</span>
                  </div>
                </div>
                <div className="flex-1 h-[140px] border border-status-elevated bg-surface-recessed relative overflow-hidden flex flex-col">
                  <img
                    src={mlEvidence.mask_uri}
                    alt="Current change mask"
                    className="absolute inset-0 w-full h-full object-cover opacity-90"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                  />
                  <div className="relative z-10 flex items-center justify-between px-space-8 py-space-4 bg-surface-panel border-b border-border-subtle">
                    <span className="data-val text-body-sm text-text-dim">AFTER</span>
                    <span className="data-val text-body-sm text-status-elevated">CHANGE</span>
                  </div>
                  <div className="relative z-10 flex flex-col px-space-8 py-space-4 mt-auto bg-surface-panel border-t border-border-subtle">
                    <span className="data-val text-headline-md text-status-elevated">{areaAfter.toFixed(2)} km²</span>
                    <span className="data-val text-caption text-text-dim">+{((run.change_stats_json?.expansion_percent as number) ?? 0).toFixed(1)}% EXPANSION</span>
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
                  <span className="data-val text-body-sm text-text-primary">
                    {mlEvidence.model_available ? "SIAMESE U-NET CHANGE PROBABILITY" : "CHANGE DETECTION HEATMAP"}
                  </span>
                  <span className="data-val text-body-sm text-text-dim">
                    CONF: {(mlEvidence.ml_confidence_mean * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="relative z-10 flex items-center justify-between px-space-8 py-space-4 mt-auto bg-surface-panel border-t border-border-subtle">
                  <div className="flex items-center gap-space-8">
                    <div className="flex items-center gap-space-4">
                      <span className="w-2 h-2 bg-status-danger" />
                      <span className="data-val text-caption text-text-dim">HIGH</span>
                    </div>
                    <div className="flex items-center gap-space-4">
                      <span className="w-2 h-2 bg-status-warn" />
                      <span className="data-val text-caption text-text-dim">MODERATE</span>
                    </div>
                    <div className="flex items-center gap-space-4">
                      <span className="w-2 h-2 bg-surface-canvas border border-border-subtle" />
                      <span className="data-val text-caption text-text-dim">NONE</span>
                    </div>
                  </div>
                  <span className="data-val text-body-sm text-text-dim">
                    {mlEvidence.ml_consensus_pixels.toLocaleString()} px
                  </span>
                </div>
              </div>

              <div className="data-val text-body-sm text-text-dim break-all select-all">
                {mlEvidence.ml_source === "siamese_unet_consensus"
                  ? "ML: Siamese U-Net + rule-based consensus (ADR-002)"
                  : "ML: scaffold mode — rule-based detection with confidence gradient"}
              </div>
            </div>

            <div className="border-t border-border-subtle px-space-12 py-space-8 grid grid-cols-2 gap-space-8 data-val text-body-sm text-text-dim">
              <div>
                PROJ: <span className="text-text-primary">WGS84 / UTM 45N</span>
              </div>
              <div>
                CONF: <span className="text-text-primary">{(score.confidence * 100).toFixed(1)}%</span>
                <span className="text-border-subtle"> | </span>
                <span className={mlEvidence.model_available ? "text-primary-container" : "text-status-warn"}>
                  {mlEvidence.model_available ? "ML+RULE" : "RULE-ONLY"}
                </span>
              </div>
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
                <GaugeRow label="C" value={score.confidence} summary="Quality-gate confidence after cloud and co-registration checks" />
              </div>
              <div className="border-t border-border-subtle pt-space-12">
                <h3 className="label-caps mb-space-8">Evidence Reasons ({score.reasons.length})</h3>
                <div className="flex flex-col gap-space-4 text-body-md text-text-primary">
                  {score.reasons.map((r, i) => (
                    <div key={i} className="flex items-baseline gap-space-8">
                      <span className="data-val text-body-sm text-text-dim w-4 shrink-0">{String(i + 1).padStart(2, "0")}</span>
                      <span>{r}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Bottom row: Disease + Assets */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-space-8 mt-space-8">
          {/* Disease Prevention */}
          <section className="bg-surface-panel border border-border-subtle flex flex-col">
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h2 className="label-caps">Disease Prevention</h2>
              <span className="data-val text-body-sm text-text-dim">PROTOCOL W-4</span>
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
                      <span className="border border-status-danger text-status-danger px-space-4 py-space-1">BOIL-WATER ADVISORY</span>
                      <span className="border border-status-warn text-status-warn px-space-4 py-space-1">ALTERNATE SUPPLY</span>
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
              <span className="data-val text-body-sm text-text-dim">BUFFER: 250m</span>
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
            <div className="border-t border-border-subtle px-space-12 py-space-8 flex items-center justify-between data-val text-caption text-text-dim">
              <span>INUNDATION MODEL: HYDRAULIC-1D</span>
              <span>{sortedExposures.length} ASSETS DETECTED</span>
            </div>
          </section>
        </div>

        {/* SAR Priority */}
        {sarPriority.sectors.length > 0 && (
          <section className="bg-surface-panel border border-border-subtle mt-space-8 flex flex-col">
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <div className="flex items-center gap-space-8">
                <h2 className="label-caps">Search &amp; Rescue Priority</h2>
                <span className="data-val text-body-sm text-text-dim border border-border-subtle px-space-4 py-space-1">
                  STRETCH
                </span>
              </div>
              <span className="data-val text-body-sm text-text-dim">
                {sarPriority.sectors.length} SECTORS RANKED
              </span>
            </div>
            <div className="p-space-12">
              <p className="data-val text-body-sm text-text-dim mb-space-12">{sarPriority.summary}</p>
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

      {/* Decision bar — 2-step armed state for dispatch */}
      <aside className="fixed bottom-[24px] left-0 right-0 h-decision-bar-height bg-surface-panel border-t border-border-subtle z-40 px-space-16 flex items-center justify-between">
        <div className="flex items-center gap-space-8">
          {decisionLocked ? (
            <div className="flex items-center gap-space-8">
              <span
                className={`data-val text-body-md ${
                  currentDecision === "confirm"
                    ? "text-status-safe"
                    : currentDecision === "reject"
                    ? "text-status-danger"
                    : "text-status-warn"
                }`}
              >
                {currentDecision === "confirm" && "STATUS: CONFIRMED BY COORDINATOR-01"}
                {currentDecision === "reject" && "STATUS: REJECTED BY COORDINATOR-01"}
                {currentDecision === "postpone" && "STATUS: POSTPONED BY COORDINATOR-01"}
              </span>
              {currentDecision === "confirm" && (
                dispatchArmed ? (
                  <div className="flex items-center gap-space-8 px-space-8 py-space-4 border border-status-danger bg-surface-recessed">
                    <span className="data-val text-body-sm text-status-danger">ARMED — CONFIRM DISPATCH?</span>
                    <button
                      onClick={() => dispatchMut.mutate()}
                      disabled={dispatchMut.isPending}
                      className="data-val text-body-sm bg-status-danger text-white px-space-8 py-space-2 border border-status-danger hover:brightness-125 transition-all cursor-pointer disabled:opacity-60"
                    >
                      {dispatchMut.isPending ? "SENDING..." : "SEND"}
                    </button>
                    <button
                      onClick={() => setDispatchArmed(false)}
                      className="data-val text-body-sm text-text-dim border border-border-subtle px-space-8 py-space-2 hover:text-text-primary transition-colors bg-transparent"
                    >
                      ABORT
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setDispatchArmed(true)}
                    disabled={dispatchMut.isPending}
                    className="data-val text-body-sm bg-status-danger text-white px-space-12 py-space-4 border border-status-danger hover:brightness-125 transition-all cursor-pointer disabled:opacity-60"
                  >
                    ARM DISPATCH
                  </button>
                )
              )}
            </div>
          ) : confirmStep ? (
            <div className="flex items-center gap-space-8 px-space-8 py-space-4 border border-status-danger bg-surface-recessed">
              <span className="data-val text-body-sm text-status-danger">CONFIRM SOS DISPATCH TO 3 SECTORS?</span>
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
        <div className="flex items-center gap-space-12 data-val text-body-sm text-text-dim">
          <span>STATE: {decisionLocked ? "LOCKED" : "PENDING_SIG"}</span>
          <span className="text-border-subtle">|</span>
          <span>REVIEWER: coordinator-01</span>
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
      <div className="data-val text-caption text-text-muted">{summary}</div>
    </div>
  );
}
