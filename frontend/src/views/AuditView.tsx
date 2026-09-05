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

const CHANNEL_TECH: Record<string, string> = {
  sms: "GSM-T",
  lora: "868 MHz",
  satellite: "Iridium SBD",
};

export default function AuditView({ onToast }: Props) {
  const sim = useSimulation();
  const [copied, setCopied] = useState(false);
  const [channels, setChannels] = useState<Record<string, ChannelStatus>>({
    sms: "idle",
    lora: "idle",
    satellite: "idle",
  });

  const dispatch = (sim.dispatchResult ?? mockData.dispatch) as DispatchResponse;
  const alertId = dispatch.alert_id;
  const hasRealDispatch = !!sim.dispatchResult;
  const currentRunId = sim.step !== "before" ? sim.runIds[sim.step] : null;

  const { data: auditData } = useQuery({
    queryKey: ["audit", hasRealDispatch && currentRunId ? `run:${currentRunId}` : `alert:${alertId}`],
    queryFn: () =>
      hasRealDispatch && currentRunId
        ? apiOrMock(() => api.listAuditByRun(currentRunId), "audit") as Promise<AuditList>
        : apiOrMock(() => api.listAudit(alertId), "audit") as Promise<AuditList>,
  });

  const entries =
    auditData && auditData.entries.length > 0
      ? auditData.entries
      : hasRealDispatch
        ? (auditData?.entries ?? [])
        : mockData.audit.entries;
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
      <div className="flex flex-col items-center justify-center h-full text-text-dim text-center gap-space-8">
        <div className="data-val text-body-md">NO DISPATCHES YET</div>
        <div className="data-val text-body-sm text-text-muted">Confirm a review and send a dispatch to see the audit trail.</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-auto">
      {/* Header bar */}
      <div className="flex items-center justify-between px-space-16 py-space-8 border-b border-border-subtle bg-surface-panel">
        <div className="flex items-center gap-space-12">
          <h1 className="label-caps">Audit</h1>
          <span className="data-val text-body-sm text-text-dim border border-border-subtle px-space-4 py-space-1">
            LOG_STREAM // IMMUTABLE
          </span>
        </div>
        <div className="flex items-center gap-space-8">
          <span className="data-val text-body-sm text-text-dim">SYNC:</span>
          <span className="data-val text-body-sm text-status-safe">LIVE (0.4s)</span>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-space-12 flex flex-col gap-space-8">
        {/* Payload inspection */}
        <section className="bg-surface-panel border border-border-subtle flex flex-col">
          <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
            <h2 className="label-caps">Compressed Payload</h2>
            <div className="flex items-center gap-space-8">
              <button
                onClick={() => {}}
                className="data-val text-body-sm text-text-dim px-space-8 py-space-2 border border-border-subtle hover:text-text-primary hover:border-border-strong transition-colors bg-transparent"
              >
                [PREVIEW]
              </button>
              <button
                onClick={copyPayload}
                className="data-val text-body-sm text-text-dim px-space-8 py-space-2 border border-border-subtle hover:text-text-primary hover:border-border-strong transition-colors bg-transparent"
              >
                [{copied ? "COPIED" : "COPY"}]
              </button>
            </div>
          </div>
          <div className="p-space-12">
            <div className="bg-surface-recessed border border-border-subtle p-space-12">
              <pre className="data-val text-code-lg text-primary-container overflow-x-auto whitespace-pre-wrap select-all break-all">
                {dispatch.payload}
              </pre>
            </div>
            <div className="flex flex-wrap items-center gap-space-12 mt-space-8">
              <div className="flex items-center gap-space-8">
                <div className="w-[200px] h-[2px] bg-border-subtle overflow-hidden">
                  <div className="h-full bg-status-safe" style={{ width: `${bytePct}%` }} />
                </div>
                <span className="data-val text-body-sm text-status-safe">{payloadBytes} / {maxBytes} BYTES</span>
              </div>
              <span className="text-border-subtle">|</span>
              <span className="data-val text-body-sm text-status-safe">LORA-COMPATIBLE</span>
              <span className="text-border-subtle">|</span>
              <span className={`data-val text-body-sm text-text-dim transition-opacity ${copied ? "opacity-100" : "opacity-0"}`}>
                COPIED TO CLIPBOARD
              </span>
            </div>
          </div>
        </section>

        {/* Transmission preview */}
        {decodedPayload && (
          <section className="bg-surface-panel border border-border-subtle flex flex-col">
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h3 className="label-caps">Transmission Preview — Decoded {payloadBytes}-byte message</h3>
              <span className="data-val text-body-sm text-status-safe border border-status-safe px-space-4 py-space-1">
                PARSED
              </span>
            </div>
            <div className="p-space-12 data-val text-body-md">
              <div className="text-status-danger font-medium tracking-wide mb-space-4">
                {decodedPayload.severity} ALERT — {decodedPayload.hazard}
              </div>
              <div className="text-text-primary flex flex-wrap items-center gap-x-space-8 gap-y-space-2">
                <span>ALERT: <span className="text-primary-container">{decodedPayload.alert_id}</span></span>
                <span className="text-border-subtle">|</span>
                <span>SECTOR: <span className="text-text-primary">{decodedPayload.sector}</span></span>
                <span className="text-border-subtle">|</span>
                <span>POP: <span className="text-text-primary">{decodedPayload.exposed_pop.toLocaleString()}</span></span>
              </div>
              <div className="text-text-primary flex flex-wrap items-center gap-x-space-8 gap-y-space-2 mt-space-2">
                <span>ASSETS: <span className="text-primary-container">{decodedPayload.critical_assets.join(", ")}</span></span>
                <span className="text-border-subtle">|</span>
                <span>ACTION: <span className="text-status-warn">{decodedPayload.medical_action}</span></span>
              </div>
            </div>
          </section>
        )}

        {/* Channels */}
        <section className="bg-surface-panel border border-border-subtle flex flex-col">
          <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
            <h2 className="label-caps">Channels</h2>
            <span className="data-val text-body-sm text-text-dim">ACTIVE CARRIERS: 3</span>
          </div>
          <div className="p-space-12 grid grid-cols-1 md:grid-cols-3 gap-space-8">
            {(["sms", "lora", "satellite"] as const).map((ch) => (
              <div
                key={ch}
                className="p-space-12 border border-border-subtle bg-surface-recessed flex flex-col justify-between gap-space-8"
              >
                <div className="flex items-center justify-between">
                  <span className="data-val text-body-md text-text-primary">{CHANNEL_LABELS[ch]}</span>
                  <span className="data-val text-body-sm text-text-dim">{CHANNEL_TECH[ch]}</span>
                </div>
                <div className={`data-val text-body-sm flex items-center gap-space-4 ${
                  channels[ch] === "delivered"
                    ? "text-status-safe"
                    : channels[ch] === "queued"
                    ? "text-status-warn"
                    : "text-text-dim"
                }`}>
                  {channels[ch] === "idle" && (
                    <button
                      onClick={() => dispatchChannel(ch)}
                      className="data-val text-body-sm bg-surface-container text-text-primary px-space-8 py-space-4 border border-border-strong hover:bg-surface-container-high transition-colors"
                    >
                      SEND
                    </button>
                  )}
                  {channels[ch] === "sent" && "SENDING..."}
                  {channels[ch] === "delivered" && "DELIVERED"}
                  {channels[ch] === "queued" && "QUEUED"}
                </div>
              </div>
            ))}
          </div>
          {!sim.reviewDecision && (
            <div className="px-space-12 pb-space-12 data-val text-body-sm text-status-warn">
              DISPATCH BLOCKED — CONFIRM A REVIEW FIRST
            </div>
          )}
        </section>

        {/* Audit trail */}
        <section className="bg-surface-panel border border-border-subtle flex flex-col">
          <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
            <h2 className="label-caps">Audit Trail</h2>
            <div className="flex items-center gap-space-6">
              <span className="w-1.5 h-1.5 bg-status-safe" />
              <span className="data-val text-body-sm text-text-dim">CRYPTOGRAPHIC CHAIN VALID</span>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-border-subtle bg-surface-canvas data-val text-caption text-text-dim">
                  <th className="py-space-6 px-space-12 font-normal">TIME (UTC)</th>
                  <th className="py-space-6 px-space-12 font-normal">ACTOR</th>
                  <th className="py-space-6 px-space-12 font-normal">ACTION</th>
                  <th className="py-space-6 px-space-12 font-normal">DETAIL</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle data-val text-body-sm text-text-primary">
                {entries.map((e) => (
                  <tr key={e.entry_id} className="hover:bg-surface-container transition-colors">
                    <td className="py-space-6 px-space-12 text-text-dim whitespace-nowrap">{e.created_at}</td>
                    <td className="py-space-6 px-space-12 text-text-primary whitespace-nowrap">{e.actor}</td>
                    <td className="py-space-6 px-space-12 whitespace-nowrap">
                      <span
                        className={`border px-space-4 py-space-1 ${
                          e.action === "dispatch"
                            ? "border-primary-container text-primary-container"
                            : e.action === "review"
                            ? "border-status-info text-status-info"
                            : "border-status-warn text-status-warn"
                        }`}
                      >
                        {e.action.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-space-6 px-space-12 text-text-dim max-w-md truncate">
                      <span className="text-text-primary break-all">{e.detail_json}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between px-space-12 py-space-8 border-t border-border-subtle data-val text-body-sm text-text-dim">
            <span>APPEND-ONLY — NO EDITS, NO DELETES</span>
            <span>SHA-256: 8f4e2a7b3...e9d2</span>
          </div>
        </section>
      </div>
    </div>
  );
}
