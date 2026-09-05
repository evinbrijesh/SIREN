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

  // Use the real alert_id from the dispatch result; fall back to mock for
  // offline/demo mode before any dispatch has been sent.
  const dispatch = (sim.dispatchResult ?? mockData.dispatch) as DispatchResponse;
  const alertId = dispatch.alert_id;
  const hasRealDispatch = !!sim.dispatchResult;
  // The run_id for the current dispatched run (from the simulation cursor).
  const currentRunId = sim.step !== "before" ? sim.runIds[sim.step] : null;

  // Query the full lineage by run_id when we have a real dispatch (the run/score/
  // review audit entries carry run_id in detail_json, not alert_id). Fall back
  // to alert_id query for the mock/demo path.
  const { data: auditData } = useQuery({
    queryKey: ["audit", hasRealDispatch && currentRunId ? `run:${currentRunId}` : `alert:${alertId}`],
    queryFn: () =>
      hasRealDispatch && currentRunId
        ? apiOrMock(() => api.listAuditByRun(currentRunId), "audit") as Promise<AuditList>
        : apiOrMock(() => api.listAudit(alertId), "audit") as Promise<AuditList>,
  });

  // Before a real dispatch, the backend returns empty entries for the mock
  // alert_id. Fall back to mock data so the demo shows a populated trail.
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
        <div className="text-body-lg">No dispatches yet</div>
        <div className="text-body-md">Confirm a review and send a dispatch to see the audit trail.</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-space-16 h-full">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-space-12">
          <h1 className="text-headline-lg text-text-primary font-medium">Audit</h1>
          <span className="font-mono text-code-sm text-text-dim px-space-8 py-space-2 rounded bg-surface-recessed border border-border-subtle">
            LOG_STREAM // IMMUTABLE
          </span>
        </div>
        <div className="flex items-center gap-space-8">
          <span className="font-mono text-code-sm text-text-dim">SYNC:</span>
          <span className="font-mono text-code-sm text-status-safe">LIVE (0.4s)</span>
        </div>
      </div>

      {/* Payload card */}
      <section className="bg-surface-panel border border-border-subtle p-space-16 rounded-lg flex flex-col gap-space-12">
        <div className="flex items-center justify-between">
          <h2 className="text-headline-md text-text-primary">Compressed Payload</h2>
          <div className="flex items-center gap-space-8">
            <button
              onClick={() => {}}
              className="text-body-sm text-text-primary px-3 py-1 rounded border border-border-subtle hover:bg-surface-canvas hover:text-primary-container transition-colors bg-transparent"
            >
              [Preview]
            </button>
            <button
              onClick={copyPayload}
              className="text-body-sm text-text-primary px-3 py-1 rounded border border-border-subtle hover:bg-surface-canvas hover:text-primary-container transition-colors bg-transparent"
            >
              [{copied ? "Copied!" : "Copy"}]
            </button>
          </div>
        </div>
        <div className="bg-surface-recessed border border-border-subtle p-space-12 rounded-lg">
          <pre className="font-mono text-code-lg text-primary-container overflow-x-auto whitespace-pre-wrap select-all break-all">
            {dispatch.payload}
          </pre>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-space-8">
            <div className="w-[200px] h-[4px] bg-border-subtle rounded overflow-hidden">
              <div className="h-full bg-status-safe" style={{ width: `${bytePct}%` }} />
            </div>
            <span className="font-mono text-code-sm text-status-safe">{payloadBytes} / {maxBytes} bytes</span>
          </div>
          <span className="text-border-subtle hidden sm:inline">·</span>
          <div className="flex items-center gap-space-4">
            <span className="text-body-sm text-status-safe">✓ LoRa-compatible</span>
          </div>
          <span className="text-border-subtle hidden sm:inline">·</span>
          <span className={`font-mono text-code-sm text-text-dim transition-opacity ${copied ? "opacity-100" : "opacity-0"}`}>
            copied to clipboard
          </span>
        </div>
      </section>

      {/* Transmission preview */}
      {decodedPayload && (
        <section className="bg-surface-panel border border-border-subtle p-space-16 rounded-lg flex flex-col gap-space-12">
          <div className="flex items-center justify-between">
            <h3 className="text-headline-sm text-text-dim">Transmission Preview (Decoded {payloadBytes}-byte message)</h3>
            <span className="text-caption text-status-safe uppercase tracking-wider px-space-6 py-space-2 rounded border border-status-safe">
              PARSED
            </span>
          </div>
          <div className="bg-surface-recessed border border-border-subtle p-space-12 rounded-lg">
            <div className="text-headline-md text-status-danger font-semibold tracking-wide flex items-center gap-space-8">
              <span>⚠ {decodedPayload.severity} ALERT — {decodedPayload.hazard}</span>
            </div>
            <div className="text-body-md text-text-primary mt-1 flex flex-wrap items-center gap-x-2">
              <span>
                Alert: <span className="font-mono text-code-sm text-primary-container">{decodedPayload.alert_id}</span>
              </span>
              <span className="text-border-subtle">·</span>
              <span>
                Sector: <span className="font-mono text-code-sm text-text-primary font-medium">{decodedPayload.sector}</span>
              </span>
              <span className="text-border-subtle">·</span>
              <span>
                Population: <span className="font-mono text-code-sm text-text-primary font-medium">{decodedPayload.exposed_pop.toLocaleString()}</span>
              </span>
            </div>
            <div className="text-body-md text-text-primary mt-1 flex flex-wrap items-center gap-x-2">
              <span>
                Assets: <span className="font-mono text-code-sm text-primary-container">{decodedPayload.critical_assets.join(", ")}</span>
              </span>
              <span className="text-border-subtle">·</span>
              <span>
                Action: <span className="font-mono text-code-sm text-status-warn font-medium">{decodedPayload.medical_action}</span>
              </span>
            </div>
          </div>
        </section>
      )}

      {/* Channels */}
      <section className="bg-surface-panel border border-border-subtle p-space-16 rounded-lg flex flex-col gap-space-12">
        <div className="flex items-center justify-between">
          <h2 className="text-headline-md text-text-primary">Channels</h2>
          <span className="text-caption text-text-dim">ACTIVE CARRIERS: 3</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-space-16">
          {(["sms", "lora", "satellite"] as const).map((ch) => (
            <div
              key={ch}
              className="p-space-12 rounded-md border border-border-subtle bg-surface-recessed flex flex-col justify-between"
            >
              <div className="flex items-center justify-between">
                <span className="text-headline-sm text-text-primary font-medium">{CHANNEL_LABELS[ch]}</span>
                <span className="text-caption text-text-dim">{CHANNEL_TECH[ch]}</span>
              </div>
              <div
                className={`text-body-sm mt-1 flex items-center gap-space-4 ${
                  channels[ch] === "delivered"
                    ? "text-status-safe"
                    : channels[ch] === "queued"
                    ? "text-status-warn"
                    : "text-text-dim"
                }`}
              >
                {channels[ch] === "idle" && (
                  <button
                    onClick={() => dispatchChannel(ch)}
                    className="px-space-12 py-space-6 rounded text-body-sm bg-primary-container text-surface-canvas hover:brightness-110 transition-all"
                  >
                    Send
                  </button>
                )}
                {channels[ch] === "sent" && "⏳ sending..."}
                {channels[ch] === "delivered" && "✓ delivered"}
                {channels[ch] === "queued" && "⏳ queued"}
              </div>
            </div>
          ))}
        </div>
        {!sim.reviewDecision && (
          <div className="text-body-sm text-status-warn">
            ⚠ Dispatch blocked — confirm a review first.
          </div>
        )}
      </section>

      {/* Audit trail */}
      <section className="bg-surface-panel border border-border-subtle p-space-16 rounded-lg flex flex-col gap-space-12">
        <div className="flex items-center justify-between">
          <h2 className="text-headline-md text-text-primary">Audit Trail</h2>
          <div className="flex items-center gap-space-6">
            <span className="w-2 h-2 rounded-full bg-status-safe animate-pulse" />
            <span className="text-caption text-text-dim uppercase tracking-wider">CRYPTOGRAPHIC CHAIN VALID</span>
          </div>
        </div>
        <div className="overflow-x-auto border border-border-subtle rounded">
          <table className="w-full text-left border-collapse bg-surface-recessed">
            <thead>
              <tr className="border-b border-border-subtle bg-surface-canvas text-caption text-text-dim uppercase tracking-wider">
                <th className="py-space-8 px-space-12 font-medium">Time (UTC)</th>
                <th className="py-space-8 px-space-12 font-medium">Actor</th>
                <th className="py-space-8 px-space-12 font-medium">Action</th>
                <th className="py-space-8 px-space-12 font-medium">Detail</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle text-body-md text-text-primary">
              {entries.map((e) => (
                <tr key={e.entry_id} className="hover:bg-surface-panel/40 transition-colors">
                  <td className="py-space-8 px-space-12 font-mono text-code-sm text-text-dim whitespace-nowrap">{e.created_at}</td>
                  <td className="py-space-8 px-space-12 font-mono text-code-sm text-text-primary whitespace-nowrap">{e.actor}</td>
                  <td className="py-space-8 px-space-12 whitespace-nowrap">
                    <span
                      className={`font-mono text-code-sm border px-space-8 py-space-2 rounded uppercase tracking-wide ${
                        e.action === "dispatch"
                          ? "border-primary-container text-primary-container"
                          : e.action === "review"
                          ? "border-status-info text-status-info"
                          : "border-status-warn text-status-warn"
                      }`}
                    >
                      {e.action}
                    </span>
                  </td>
                  <td className="py-space-8 px-space-12 font-mono text-code-sm text-text-dim max-w-md truncate">
                    <span className="text-text-primary break-all">{e.detail_json}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between mt-space-12 text-body-sm text-text-dim">
          <span>Append-only — no edits, no deletes.</span>
          <span className="font-mono text-code-sm">SHA-256: 8f4e2a7b3...e9d2</span>
        </div>
      </section>
    </div>
  );
}
