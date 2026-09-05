import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation } from "../simulation/SimulationContext";
import type { Run, ExposureList, ReviewResponse, DispatchResponse, ApiError } from "../api/types";

interface Props {
  run: Run;
  onToast?: (t: { msg: string; type: "error" | "info" | "success" }) => void;
  onJumpToMap?: () => void;
}

const SEV_COLOR: Record<string, string> = {
  informational: "#3B82F6", watch: "#F59E0B", elevated: "#F97316", critical: "#EF4444",
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

  const exposures = exposuresData?.exposures ?? mockData.exposures.exposures;
  const wells = exposures.filter((e) => e.asset_type === "well");
  const villages = exposures.filter((e) => e.asset_type === "village");

  const reviewMut = useMutation({
    mutationFn: (vars: { decision: "confirm" | "reject" | "postpone" }) =>
      apiOrMock(() => api.createReview(runId, "coordinator-01", vars.decision, "demo review"), "dispatch" as any) as Promise<ReviewResponse>,
    onSuccess: (_data, vars) => {
      sim.setReviewDecision(vars.decision);
      setConfirmStep(false);
      setError(null);
      onToast?.({ msg: `Review recorded: ${vars.decision}`, type: "success" });
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
    onError: (e: ApiError) => { setError(e.detail); onToast?.({ msg: e.detail, type: "error" }); },
  });

  const dispatchMut = useMutation({
    mutationFn: () => apiOrMock(() => api.createDispatch(runId, "sms", "sector-b"), "dispatch") as Promise<DispatchResponse>,
    onSuccess: (data) => { sim.setDispatchResult(data); setError(null); onToast?.({ msg: `Dispatch sent: ${data.payload_bytes} bytes`, type: "success" }); },
    onError: (e: ApiError) => { setError(e.detail); onToast?.({ msg: e.detail, type: "error" }); },
  });

  // Empty state — no elevated run
  if (!score || (score.severity !== "elevated" && score.severity !== "critical")) {
    return (
      <div className="empty-state">
        <div className="icon">✓</div>
        <div className="msg">No alerts requiring review</div>
        <div className="hint">Run the simulation from the Timeline tab to generate observations.</div>
      </div>
    );
  }

  if (expLoading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div className="view-title">Review — Loading...</div>
        <div className="card"><div className="skeleton skeleton-row" /><div className="skeleton skeleton-row" /><div className="skeleton skeleton-row" /></div>
      </div>
    );
  }

  const sevColor = SEV_COLOR[score.severity] ?? "#94A3B8";
  const decisionLocked = sim.reviewDecision !== null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div className="view-title">Review — Run {runId}</div>

      <div style={{ display: "flex", gap: 16, flex: 1, overflow: "auto" }}>
        {/* Evidence panel (left, 40%) */}
        <div className="card" style={{ flex: "0 0 40%" }}>
          <div className="card-title">Evidence — Before / After</div>
          <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
            <div style={{ flex: 1, height: 160, borderRadius: 6, background: "linear-gradient(135deg, #1a2a4a, #0a0f1e)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-dim)", fontSize: 12 }}>BEFORE</div>
            <div style={{ flex: 1, height: 160, borderRadius: 6, background: "linear-gradient(135deg, #1a3a5a, #0a1f3e)", display: "flex", alignItems: "center", justifyContent: "center", color: "#3B82F6", fontSize: 12 }}>AFTER (+14.3%)</div>
          </div>
          <div style={{ fontSize: 13, color: "var(--text-dim)" }}>Change mask: {run.change_mask_uri ?? "data/processed/obs-003_expansion_mask.tif"}</div>
        </div>

        {/* Risk gauge + reasons (center) */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card">
            <div className="card-title">Risk Scores</div>
            <GaugeRow label="H" value={score.hazard_score} color="#EF4444" title="Hazard: water expansion + rainfall" />
            <GaugeRow label="E" value={score.exposure_priority} color="#F59E0B" title="Exposure: villages + infrastructure in corridor" />
            {score.disease_risk !== null && <GaugeRow label="D" value={score.disease_risk} color="#3B82F6" title="Disease: wells submerged → water contamination" />}
            <GaugeRow label="conf" value={score.confidence} color="#22C55E" title="Confidence: quality-weighted sensor fusion" />
            <div style={{ marginTop: 12 }}>
              <span className="badge" style={{ background: `${sevColor}22`, color: sevColor, border: `1px solid ${sevColor}` }}>{score.severity.toUpperCase()}</span>
            </div>
          </div>

          {/* Reasons panel (≥3) */}
          <div className="card">
            <div className="card-title">Evidence Reasons ({score.reasons.length})</div>
            {score.reasons.map((r, i) => (
              <div key={i} style={{ display: "flex", gap: 8, marginBottom: 8, fontSize: 13 }}>
                <span style={{ color: "var(--accent)", fontWeight: 700 }}>{i + 1}.</span>
                <span>{r}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right dock — disease action sheet + assets table */}
        <div style={{ width: 320, flexShrink: 0, display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Disease action sheet — per-well actions */}
          <div className="card">
            <div className="card-title">Disease Prevention Actions</div>
            <div className="action-sheet">
              {wells.map((w) => {
                const isSubmerged = w.inundated;
                return (
                  <div key={w.asset_id} style={{ marginBottom: 12, padding: 8, borderRadius: 4, background: isSubmerged ? "rgba(239,68,68,0.08)" : "rgba(245,158,11,0.08)" }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
                      {w.name ?? w.asset_id}: {isSubmerged ? "submerged" : "encircled"}
                    </div>
                    {isSubmerged ? (
                      <>
                        <div className="action-item"><span className="dot" style={{ background: "#EF4444" }} /><div>→ chlorine ×200 tablets</div></div>
                        <div className="action-item"><span className="dot" style={{ background: "#EF4444" }} /><div>→ BOIL WATER NOW notice</div></div>
                      </>
                    ) : (
                      <div className="action-item"><span className="dot" style={{ background: "#F59E0B" }} /><div>→ monitor for contamination</div></div>
                    )}
                  </div>
                );
              })}
              <div className="action-item"><span className="dot" style={{ background: "#3B82F6" }} /><div>7-day diarrheal disease surveillance window</div></div>
              <div className="action-item"><span className="dot" style={{ background: "#22C55E" }} /><div>Safe water sources: identify alternate supply for {villages.reduce((s, v) => s + (v.population ?? 0), 0)} people</div></div>
            </div>
          </div>

          {/* Exposed assets table — row click → map flyTo */}
          <div className="card">
            <div className="card-title">Exposed Assets (ranked by distance)</div>
            <table className="table">
              <thead><tr><th>#</th><th>Asset</th><th>Type</th><th>Dist</th><th>Status</th></tr></thead>
              <tbody>
                {[...exposures].sort((a, b) => (a.distance_m ?? 9999) - (b.distance_m ?? 9999)).map((e, i) => (
                  <tr
                    key={e.asset_id}
                    style={{ cursor: "pointer" }}
                    onClick={() => { sim.selectAsset(e.asset_id); onJumpToMap?.(); }}
                  >
                    <td style={{ color: "var(--text-dim)" }}>{i + 1}</td>
                    <td>{e.name ?? e.asset_id}</td>
                    <td>{e.asset_type}</td>
                    <td>{e.distance_m?.toFixed(0)}m</td>
                    <td><span className={`badge ${e.inundated ? "badge-danger" : "badge-warn"}`}>{e.inundated ? "INUNDATED" : "BUFFERED"}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {error && <div className="error-msg" style={{ marginTop: 8 }}>{error}</div>}

      {/* Decision bar — two-step confirm + decision lock */}
      <div className="decision-bar">
        {decisionLocked ? (
          <div className={`decision-locked ${sim.reviewDecision}`}>
            ✓ Decision recorded: {sim.reviewDecision?.toUpperCase()}
            {sim.reviewDecision === "confirm" && (
              <button className="btn btn-ghost" style={{ marginLeft: 12, fontSize: 12, padding: "4px 10px" }} disabled={dispatchMut.isPending} onClick={() => dispatchMut.mutate()}>
                📤 Send Dispatch
              </button>
            )}
          </div>
        ) : confirmStep ? (
          <div className="confirm-step">
            <span style={{ fontWeight: 600, color: "var(--danger)" }}>Confirm SOS dispatch?</span>
            <button className="btn btn-primary" disabled={reviewMut.isPending} onClick={() => reviewMut.mutate({ decision: "confirm" })}>Yes, confirm</button>
            <button className="btn btn-ghost" onClick={() => setConfirmStep(false)}>Cancel</button>
          </div>
        ) : (
          <>
            <button className="btn btn-primary" disabled={reviewMut.isPending} onClick={() => setConfirmStep(true)}>✓ Confirm SOS</button>
            <button className="btn btn-danger" disabled={reviewMut.isPending} onClick={() => reviewMut.mutate({ decision: "reject" })}>✗ Reject</button>
            <button className="btn btn-warn" disabled={reviewMut.isPending} onClick={() => reviewMut.mutate({ decision: "postpone" })}>⏸ Postpone</button>
          </>
        )}
        <span className="reviewer">reviewer: coordinator-01</span>
      </div>
    </div>
  );
}

function GaugeRow({ label, value, color, title }: { label: string; value: number; color: string; title?: string }) {
  return (
    <div className="gauge-row" title={title}>
      <span className="gauge-label">{label}</span>
      <div className="gauge-bar"><div className="gauge-fill" style={{ width: `${value * 100}%`, background: color }} /></div>
      <span className="gauge-value">{value.toFixed(2)}</span>
    </div>
  );
}
