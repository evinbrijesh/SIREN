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

const SEV_BADGE: Record<string, string> = {
  elevated: "badge-elevated",
  critical: "badge-danger",
};

const GAUGE_COLORS = {
  H: "var(--danger)",
  E: "var(--warn)",
  D: "var(--info)",
  C: "var(--safe)",
};

const GAUGE_LABELS = {
  H: "Hazard",
  E: "Exposure",
  D: "Disease risk",
  C: "Confidence",
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
    onError: (e: ApiError) => { setError(e.detail); onToast?.({ msg: e.detail, type: "error" }); },
  });

  const dispatchMut = useMutation({
    mutationFn: () => apiOrMock(() => api.createDispatch(runId, "sms", "sector-b"), "dispatch") as Promise<DispatchResponse>,
    onSuccess: (data) => { sim.setDispatchResult(data); setError(null); onToast?.({ msg: `Dispatch sent (${data.payload_bytes} bytes)`, type: "success" }); },
    onError: (e: ApiError) => { setError(e.detail); onToast?.({ msg: e.detail, type: "error" }); },
  });

  if (!score || (score.severity !== "elevated" && score.severity !== "critical")) {
    return (
      <div className="empty-state">
        <div className="msg">No alerts requiring review</div>
        <div className="hint">Run the simulation from the Timeline tab to generate observations.</div>
      </div>
    );
  }

  if (expLoading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <h1 className="view-title">Review</h1>
        <div className="card"><div className="skeleton skeleton-row" /><div className="skeleton skeleton-row" /><div className="skeleton skeleton-row" /></div>
      </div>
    );
  }

  const decisionLocked = sim.reviewDecision !== null;
  const sortedExposures = [...exposures].sort((a, b) => (a.distance_m ?? 9999) - (b.distance_m ?? 9999));

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <h1 className="view-title" style={{ margin: 0 }}>
          Review — Run <span style={{ fontFamily: "JetBrains Mono, monospace", color: "var(--accent)" }}>{runId}</span>
        </h1>
        <span className={`badge ${SEV_BADGE[score.severity]}`}>{score.severity}</span>
      </div>

      {/* Top row: Evidence (left) + Risk & Reasons (right) */}
      <div style={{ display: "flex", gap: 16, flex: 1, overflow: "auto", minHeight: 0 }}>
        {/* Evidence panel */}
        <div className="card" style={{ flex: "0 0 45%", display: "flex", flexDirection: "column" }}>
          <div className="card-title">Evidence</div>
          <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
            <div style={{
              flex: 1, height: 140, borderRadius: 8,
              background: "var(--recessed)", border: "1px solid var(--panel-2)",
              display: "flex", flexDirection: "column", justifyContent: "space-between", padding: 12,
            }}>
              <span style={{ fontSize: 13, color: "var(--text-dim)" }}>Before</span>
              <div>
                <div style={{ fontSize: 16, fontWeight: 600 }}>3.0 km²</div>
                <div style={{ fontSize: 12, color: "var(--text-dim)" }}>Reference extent</div>
              </div>
            </div>
            <div style={{
              flex: 1, height: 140, borderRadius: 8,
              background: "rgba(0,66,79,0.3)", border: "1px solid rgba(6,182,212,0.3)",
              display: "flex", flexDirection: "column", justifyContent: "space-between", padding: 12,
            }}>
              <span style={{ fontSize: 13, color: "var(--accent)" }}>After</span>
              <div>
                <div style={{ fontSize: 16, fontWeight: 600, color: "var(--accent)" }}>4.1 km²</div>
                <div style={{ fontSize: 12, color: "var(--accent)", opacity: 0.8 }}>+14.3% expansion</div>
              </div>
            </div>
          </div>
          <div style={{ marginTop: "auto", fontSize: 12, color: "var(--text-dim)", fontFamily: "JetBrains Mono, monospace", wordBreak: "break-all" }}>
            {run.change_mask_uri ?? "data/processed/obs-003_expansion_mask.tif"}
          </div>
        </div>

        {/* Risk scores + reasons */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16, minWidth: 0 }}>
          <div className="card">
            <div className="card-title">Risk Scores</div>
            <GaugeRow label="H" value={score.hazard_score} />
            <GaugeRow label="E" value={score.exposure_priority} />
            {score.disease_risk !== null && <GaugeRow label="D" value={score.disease_risk} />}
            <GaugeRow label="C" value={score.confidence} />
          </div>

          <div className="card">
            <div className="card-title">Evidence Reasons ({score.reasons.length})</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {score.reasons.map((r, i) => (
                <div key={i} style={{ display: "flex", gap: 8, fontSize: 14 }}>
                  <span style={{ color: "var(--accent)", fontWeight: 600, flexShrink: 0 }}>{i + 1}.</span>
                  <span>{r}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom row: Disease actions + Exposed assets */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }}>
        <div className="card">
          <div className="card-title">Disease Prevention</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {wells.map((w) => {
              const isSubmerged = w.inundated;
              return (
                <div key={w.asset_id}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: isSubmerged ? "var(--danger)" : "var(--warn)", marginBottom: 4 }}>
                    {w.name ?? w.asset_id} — {isSubmerged ? "submerged" : "encircled"}
                  </div>
                  <div style={{ paddingLeft: 12, fontSize: 14, color: "var(--text)", display: "flex", flexDirection: "column", gap: 2 }}>
                    {isSubmerged ? (
                      <>
                        <span>→ chlorine ×200 tablets</span>
                        <span>→ boil water notice</span>
                      </>
                    ) : (
                      <span>→ monitor for contamination</span>
                    )}
                  </div>
                </div>
              );
            })}
            <div style={{ borderTop: "1px solid var(--panel-2)", paddingTop: 12, display: "flex", flexDirection: "column", gap: 4, fontSize: 14 }}>
              <span>→ 7-day diarrheal surveillance</span>
              <span>→ alternate water supply for {totalPop} people</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Exposed Assets</div>
          <table className="table">
            <thead>
              <tr><th>#</th><th>Asset</th><th>Type</th><th>Dist</th><th style={{ textAlign: "right" }}>Status</th></tr>
            </thead>
            <tbody>
              {sortedExposures.map((e, i) => (
                <tr
                  key={e.asset_id}
                  onClick={() => { sim.selectAsset(e.asset_id); onJumpToMap?.(); }}
                >
                  <td style={{ color: "var(--text-dim)" }}>{i + 1}</td>
                  <td style={{ fontWeight: 500 }}>{e.name ?? e.asset_id}</td>
                  <td style={{ color: "var(--text-dim)" }}>{e.asset_type}</td>
                  <td>{e.distance_m?.toFixed(0)}m</td>
                  <td style={{ textAlign: "right" }}>
                    <span className={`badge ${e.inundated ? "badge-danger" : "badge-warn"}`}>
                      {e.inundated ? "INUNDATED" : "BUFFERED"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {error && <div className="error-msg" style={{ marginTop: 8 }}>{error}</div>}

      {/* Decision bar */}
      <div className="decision-bar" style={{ marginTop: 16 }}>
        {decisionLocked ? (
          <div className={`decision-locked ${sim.reviewDecision}`}>
            <span>
              {sim.reviewDecision === "confirm" && "✓ Confirmed"}
              {sim.reviewDecision === "reject" && "✗ Rejected"}
              {sim.reviewDecision === "postpone" && "⏸ Postponed"}
            </span>
            {sim.reviewDecision === "confirm" && (
              <button
                className="btn btn-primary"
                style={{ marginLeft: 12, fontSize: 13, padding: "6px 14px" }}
                disabled={dispatchMut.isPending}
                onClick={() => dispatchMut.mutate()}
              >
                {dispatchMut.isPending ? "Sending..." : "Send Dispatch"}
              </button>
            )}
          </div>
        ) : confirmStep ? (
          <div className="confirm-step">
            <span style={{ fontWeight: 600, color: "var(--danger)" }}>Confirm SOS dispatch?</span>
            <button className="btn btn-safe" disabled={reviewMut.isPending} onClick={() => reviewMut.mutate({ decision: "confirm" })}>
              Yes, confirm
            </button>
            <button className="btn btn-ghost" onClick={() => setConfirmStep(false)}>Cancel</button>
          </div>
        ) : (
          <>
            <button className="btn btn-safe" disabled={reviewMut.isPending} onClick={() => setConfirmStep(true)}>
              ✓ Confirm
            </button>
            <button className="btn btn-danger" disabled={reviewMut.isPending} onClick={() => reviewMut.mutate({ decision: "reject" })}>
              ✗ Reject
            </button>
            <button className="btn btn-warn" disabled={reviewMut.isPending} onClick={() => reviewMut.mutate({ decision: "postpone" })}>
              ⏸ Postpone
            </button>
          </>
        )}
        <span className="reviewer">reviewer: coordinator-01</span>
      </div>
    </div>
  );
}

function GaugeRow({ label, value }: { label: keyof typeof GAUGE_LABELS; value: number }) {
  const color = GAUGE_COLORS[label];
  return (
    <div className="gauge-row">
      <span className="gauge-label" style={{ color }} title={GAUGE_LABELS[label]}>{label}</span>
      <div className="gauge-bar">
        <div className="gauge-fill" style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <span className="gauge-value">{value.toFixed(2)}</span>
    </div>
  );
}
