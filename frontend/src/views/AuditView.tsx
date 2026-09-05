import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation } from "../simulation/SimulationContext";
import type { AuditList, DispatchResponse } from "../api/types";

interface Props {
  onToast?: (t: { msg: string; type: "error" | "info" | "success" }) => void;
}

type ChannelStatus = "idle" | "sent" | "delivered" | "queued";

export default function AuditView({ onToast }: Props) {
  const alertId = "alert-0091";
  const sim = useSimulation();
  const [copied, setCopied] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [channels, setChannels] = useState<Record<string, ChannelStatus>>({
    sms: "idle", lora: "idle", satellite: "idle",
  });

  const { data: auditData } = useQuery({
    queryKey: ["audit", alertId],
    queryFn: () => apiOrMock(() => api.listAudit(alertId), "audit") as Promise<AuditList>,
  });

  const entries = auditData?.entries ?? mockData.audit.entries;
  const dispatch = (sim.dispatchResult ?? mockData.dispatch) as DispatchResponse;
  const payloadBytes = dispatch.payload_bytes;
  const maxBytes = 250;
  const bytePct = (payloadBytes / maxBytes) * 100;

  const copyPayload = () => {
    navigator.clipboard.writeText(dispatch.payload).then(() => {
      setCopied(true);
      onToast?.({ msg: "Payload copied to clipboard", type: "success" });
      setTimeout(() => setCopied(false), 2000);
    });
  };

  // Channel simulator — staged timing for demo drama
  const dispatchChannel = (channel: string) => {
    if (!sim.reviewDecision || sim.reviewDecision !== "confirm") {
      onToast?.({ msg: "409: Human gate — no confirm review exists. Dispatch blocked.", type: "error" });
      return;
    }
    setChannels((prev) => ({ ...prev, [channel]: "sent" }));
    setTimeout(() => setChannels((prev) => ({ ...prev, [channel]: channel === "satellite" ? "queued" : "delivered" })), 800);
  };

  // Decode payload for transmission preview
  const decodedPayload = (() => {
    try {
      const p = JSON.parse(dispatch.payload);
      const sevMap: Record<number, string> = { 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL" };
      return {
        alert_id: p.aid,
        sector: p.sec,
        hazard: p.haz,
        severity: sevMap[p.lvl] ?? "UNKNOWN",
        exposed_pop: p.exp_pop,
        critical_assets: p.crit,
        medical_action: p.med_act,
      };
    } catch {
      return null;
    }
  })();

  // Empty state
  if (!sim.dispatchResult && entries.length === 0) {
    return (
      <div className="empty-state">
        <div className="icon">📋</div>
        <div className="msg">No dispatches yet</div>
        <div className="hint">Confirm a review and send a dispatch to see the audit trail.</div>
      </div>
    );
  }

  return (
    <div>
      <div className="view-title">Audit — Lineage & Resilient Alerting</div>

      {/* Payload box with copy + byte meter */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span>Compressed Payload (Track 7.ii)</span>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="copy-btn" onClick={() => setShowPreview(true)}>👁 Preview</button>
            <button className="copy-btn" onClick={copyPayload}>{copied ? "✓ Copied" : "⧉ Copy"}</button>
          </div>
        </div>
        <div className="payload-box">{dispatch.payload}</div>
        <div className="payload-meta">
          <div className="byte-meter">
            <div className="byte-meter-bar">
              <div className="byte-meter-fill" style={{ width: `${bytePct}%`, background: payloadBytes <= maxBytes ? "var(--safe)" : "var(--danger)" }} />
            </div>
            <span className={`byte-counter ${payloadBytes <= maxBytes ? "ok" : ""}`}>
              {payloadBytes} / {maxBytes} bytes
            </span>
          </div>
          {payloadBytes <= maxBytes && <span className="badge badge-safe">✓ LoRa-compatible</span>}
          <span style={{ color: "var(--text-dim)" }}>alert_id: {dispatch.alert_id}</span>
        </div>
      </div>

      {/* Channel simulator */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Channel Simulator</div>
        <div style={{ color: "var(--text-dim)", fontSize: 13, marginBottom: 12 }}>→ sector-b geofence · 3 recipient groups</div>
        <div style={{ display: "flex", gap: 16 }}>
          {(["sms", "lora", "satellite"] as const).map((ch) => (
            <div key={ch} style={{ display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-start" }}>
              <button
                className={`btn ${channels[ch] !== "idle" ? "btn-ghost" : "btn-primary"}`}
                style={{ fontSize: 12, padding: "6px 12px" }}
                onClick={() => dispatchChannel(ch)}
                disabled={channels[ch] !== "idle"}
              >
                {ch === "sms" ? "SMS" : ch === "lora" ? "LoRa Mesh" : "Satellite"}
              </button>
              <ChannelStatusBadge status={channels[ch]} />
            </div>
          ))}
        </div>
        {!sim.reviewDecision && (
          <div style={{ marginTop: 12, fontSize: 12, color: "var(--warn)" }}>
            ⚠ Dispatch blocked by human gate — confirm a review first.
          </div>
        )}
      </div>

      {/* Audit trail table */}
      <div className="card">
        <div className="card-title">Audit Trail — Immutable Lineage (alert: {alertId})</div>
        <table className="table">
          <thead>
            <tr><th>Timestamp (UTC)</th><th>Actor</th><th>Action</th><th>Detail (JSON)</th></tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr key={e.entry_id}>
                <td className="mono" style={{ fontSize: 12 }}>{e.created_at}</td>
                <td>{e.actor}</td>
                <td>
                  <span className={`badge ${e.action === "dispatch" ? "badge-accent" : e.action === "review" ? "badge-info" : "badge-warn"}`}>
                    {e.action}
                  </span>
                </td>
                <td className="mono" style={{ fontSize: 12, color: "var(--text-dim)", maxWidth: 400, overflow: "hidden", textOverflow: "ellipsis" }}>
                  {e.detail_json}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-dim)" }}>
          ✓ Append-only — UPDATE and DELETE blocked by schema triggers (PRD §7.8). No edit/delete affordances exist.
        </div>
      </div>

      {/* Transmission preview modal */}
      {showPreview && decodedPayload && (
        <div className="modal-overlay" onClick={() => setShowPreview(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowPreview(false)}>×</button>
            <div className="modal-title">Transmission Preview — as recipient sees it</div>
            <div style={{ padding: 16, background: "#0a0f1e", borderRadius: 8, marginBottom: 16 }}>
              <div style={{ fontSize: 16, fontWeight: 700, color: "var(--danger)", marginBottom: 8 }}>
                ⚠ {decodedPayload.severity} ALERT — {decodedPayload.hazard}
              </div>
              <div style={{ fontSize: 14, marginBottom: 4 }}>Alert ID: {decodedPayload.alert_id}</div>
              <div style={{ fontSize: 14, marginBottom: 4 }}>Sector: {decodedPayload.sector}</div>
              <div style={{ fontSize: 14, marginBottom: 4 }}>Exposed population: {decodedPayload.exposed_pop}</div>
              <div style={{ fontSize: 14, marginBottom: 4 }}>Critical assets at risk: {decodedPayload.critical_assets.join(", ")}</div>
              <div style={{ fontSize: 14, marginBottom: 4, color: "var(--warn)" }}>Medical action: {decodedPayload.medical_action}</div>
            </div>
            <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
              This is the decoded view of the {payloadBytes}-byte compressed payload. The compact JSON on the left
              encodes the same information in a LoRa-compatible format.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ChannelStatusBadge({ status }: { status: ChannelStatus }) {
  if (status === "idle") return <span style={{ fontSize: 11, color: "var(--text-dim)" }}>ready</span>;
  const icon = status === "delivered" ? "✓" : status === "sent" ? "✓" : "⏳";
  const cls = status === "delivered" || status === "sent" ? "badge-safe" : "badge-warn";
  return <span className={`badge ${cls}`}>{icon} {status}</span>;
}
