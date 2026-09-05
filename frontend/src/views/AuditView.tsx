import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation } from "../simulation/SimulationContext";
import type { AuditList, DispatchResponse } from "../api/types";

interface Props {
  onToast?: (t: { msg: string; type: "error" | "info" | "success" }) => void;
}

type ChannelStatus = "idle" | "sent" | "delivered" | "queued";

const CHANNEL_LABELS: Record<string, string> = {
  sms: "SMS",
  lora: "LoRa",
  satellite: "Satellite",
};

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
      onToast?.({ msg: "Payload copied", type: "success" });
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const dispatchChannel = (channel: string) => {
    if (!sim.reviewDecision || sim.reviewDecision !== "confirm") {
      onToast?.({ msg: "Dispatch blocked — confirm a review first", type: "error" });
      return;
    }
    setChannels((prev) => ({ ...prev, [channel]: "sent" }));
    setTimeout(() => setChannels((prev) => ({ ...prev, [channel]: channel === "satellite" ? "queued" : "delivered" })), 800);
  };

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

  if (!sim.dispatchResult && entries.length === 0) {
    return (
      <div className="empty-state">
        <div className="msg">No dispatches yet</div>
        <div className="hint">Confirm a review and send a dispatch to see the audit trail.</div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <h1 className="view-title">Audit</h1>

      {/* Payload card */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <span className="card-title" style={{ margin: 0 }}>Compressed Payload</span>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-ghost" style={{ fontSize: 13, padding: "6px 12px" }} onClick={() => setShowPreview(true)}>
              Preview
            </button>
            <button className="btn btn-ghost" style={{ fontSize: 13, padding: "6px 12px" }} onClick={copyPayload}>
              {copied ? "✓ Copied" : "Copy"}
            </button>
          </div>
        </div>
        <div className="payload-box">{dispatch.payload}</div>
        <div className="payload-meta">
          <div className="byte-meter">
            <div className="byte-meter-bar">
              <div className="byte-meter-fill" style={{ width: `${bytePct}%`, background: payloadBytes <= maxBytes ? "var(--safe)" : "var(--danger)" }} />
            </div>
            <span style={{ fontFamily: "JetBrains Mono, monospace", color: payloadBytes <= maxBytes ? "var(--safe)" : "var(--danger)", fontWeight: 600 }}>
              {payloadBytes} / {maxBytes} bytes
            </span>
          </div>
          {payloadBytes <= maxBytes && <span style={{ color: "var(--safe)", fontSize: 13 }}>✓ LoRa-compatible</span>}
        </div>
      </div>

      {/* Channels */}
      <div className="card">
        <div className="card-title">Channels</div>
        <div className="channel-grid">
          {(["sms", "lora", "satellite"] as const).map((ch) => (
            <div key={ch} className="channel-card">
              <span className="ch-name">{CHANNEL_LABELS[ch]}</span>
              <button
                className={`btn ${channels[ch] !== "idle" ? "btn-ghost" : "btn-primary"}`}
                style={{ fontSize: 13, padding: "8px 14px" }}
                onClick={() => dispatchChannel(ch)}
                disabled={channels[ch] !== "idle"}
              >
                {channels[ch] === "idle" ? "Send" : channels[ch] === "sent" ? "Sending..." : "Sent"}
              </button>
              <span className={`ch-status ${channels[ch] === "delivered" ? "delivered" : channels[ch] === "queued" ? "queued" : "idle"}`}>
                {channels[ch] === "idle" && "ready"}
                {channels[ch] === "sent" && "sending..."}
                {channels[ch] === "delivered" && "✓ delivered"}
                {channels[ch] === "queued" && "⏳ queued"}
              </span>
            </div>
          ))}
        </div>
        {!sim.reviewDecision && (
          <div style={{ marginTop: 12, fontSize: 13, color: "var(--warn)" }}>
            ⚠ Dispatch blocked — confirm a review first.
          </div>
        )}
      </div>

      {/* Audit trail */}
      <div className="card">
        <div className="card-title">Audit Trail</div>
        <table className="table">
          <thead>
            <tr><th>Time (UTC)</th><th>Actor</th><th>Action</th><th>Detail</th></tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr key={e.entry_id}>
                <td className="mono" style={{ fontSize: 12, whiteSpace: "nowrap" }}>{e.created_at}</td>
                <td style={{ fontSize: 13 }}>{e.actor}</td>
                <td>
                  <span className={`badge ${e.action === "dispatch" ? "badge-accent" : e.action === "review" ? "badge-info" : "badge-warn"}`}>
                    {e.action}
                  </span>
                </td>
                <td className="mono" style={{ fontSize: 12, color: "var(--text-dim)", maxWidth: 400, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {e.detail_json}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ marginTop: 12, fontSize: 13, color: "var(--text-dim)" }}>
          Append-only — no edits, no deletes.
        </div>
      </div>

      {/* Transmission preview modal */}
      {showPreview && decodedPayload && (
        <div className="modal-overlay" onClick={() => setShowPreview(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowPreview(false)}>×</button>
            <div className="modal-title">Transmission Preview</div>
            <div className="modal-msg-box">
              <div style={{ fontSize: 16, fontWeight: 600, color: "var(--danger)", marginBottom: 12 }}>
                ⚠ {decodedPayload.severity} ALERT — {decodedPayload.hazard}
              </div>
              <div style={{ fontSize: 14, marginBottom: 6 }}>Alert: <span className="mono" style={{ color: "var(--accent)" }}>{decodedPayload.alert_id}</span></div>
              <div style={{ fontSize: 14, marginBottom: 6 }}>Sector: {decodedPayload.sector}</div>
              <div style={{ fontSize: 14, marginBottom: 6 }}>Population: {decodedPayload.exposed_pop}</div>
              <div style={{ fontSize: 14, marginBottom: 6 }}>Assets at risk: {decodedPayload.critical_assets.join(", ")}</div>
              <div style={{ fontSize: 14, color: "var(--warn)" }}>Action: {decodedPayload.medical_action}</div>
            </div>
            <div style={{ marginTop: 12, fontSize: 13, color: "var(--text-dim)" }}>
              Decoded view of the {payloadBytes}-byte payload.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
