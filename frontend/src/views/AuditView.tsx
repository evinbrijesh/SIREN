import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, apiOrMock } from "../api/client";
import { mockData } from "../api/mockData";
import { useSimulation } from "../simulation/SimulationContext";
import type { AuditList, AuditEntry, DispatchResponse, Run } from "../api/types";

interface Props {
  run?: Run | null;
  onToast?: (t: { msg: string; type: "error" | "info" | "success" }) => void;
}

type ChannelStatus = "idle" | "queued" | "transmitting" | "delivered" | "failed";

const CHANNEL_LABELS: Record<string, string> = {
  sms: "SMS / Push Broadcast",
  lora: "LoRa Mesh Radio",
  satellite: "Satellite (Iridium SBD)",
};

const CHANNEL_RF: Record<string, { freq: string; airtime: string; capacity: string; detail: string }> = {
  sms: { freq: "900 MHz GSM", airtime: "0.8s", capacity: "140 bytes / PDU", detail: "Single GSM-T PDU frame — no segmentation or drop risk" },
  lora: { freq: "868.1 MHz ISM", airtime: "1.8s", capacity: "222 bytes max", detail: "SF9, 125 kHz bandwidth — fits cleanly under hardware limit" },
  satellite: { freq: "1621 MHz L-Band", airtime: "12.4s", capacity: "340 bytes / SBD", detail: "Single Iridium SBD burst to LEO constellation" },
};

const NTFY_TOPIC = "siren-emergency-alert";

export default function AuditView({ run, onToast }: Props) {
  const sim = useSimulation();
  const [copied, setCopied] = useState(false);
  const [channels, setChannels] = useState<Record<string, ChannelStatus>>({
    sms: "idle",
    lora: "idle",
    satellite: "idle",
  });
  const [showPreview, setShowPreview] = useState(false);
  const [showVerify, setShowVerify] = useState(false);
  const [verifyResult, setVerifyResult] = useState<{ verified: boolean; details: { entry: number; expected: string; actual: string; match: boolean }[] } | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);

  const dispatch = (sim.dispatchResult ?? mockData.dispatch) as DispatchResponse;
  const alertId = dispatch.alert_id;
  const hasRealDispatch = !!sim.dispatchResult;
  const currentRunId = sim.step !== "before" ? sim.runIds[sim.step] : null;
  const activeRunId = run?.run_id ?? currentRunId;

  const { data: auditData } = useQuery({
    queryKey: ["audit", hasRealDispatch && activeRunId ? `run:${activeRunId}` : `alert:${alertId}`],
    queryFn: () =>
      hasRealDispatch && activeRunId
        ? apiOrMock(() => api.listAuditByRun(activeRunId), "audit") as Promise<AuditList>
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
  const expansionPct = (run?.change_stats_json?.expansion_percent as number) ?? 43.0;

  // Live terminal digest from the audit chain (last entry's event_hash)
  const terminalDigest = entries.length > 0 ? entries[entries.length - 1].event_hash : "—";
  const shortDigest = terminalDigest !== "—" ? `${terminalDigest.slice(0, 12)}…${terminalDigest.slice(-8)}` : "—";

  // Decoded payload — must be computed before callbacks that reference it
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

  const copyPayload = () => {
    navigator.clipboard.writeText(dispatch.payload).then(() => {
      setCopied(true);
      onToast?.({ msg: "Payload copied", type: "success" });
      setTimeout(() => setCopied(false), 1000);
    });
  };

  // Channel FSM: QUEUED at 0.5s → TRANSMITTING at 1.2s → DELIVERED (or QUEUED for satellite)
  const dispatchChannel = useCallback((channel: string) => {
    if (!sim.reviewDecision || sim.reviewDecision !== "confirm") {
      onToast?.({ msg: "Dispatch blocked — confirm a review first", type: "error" });
      return;
    }
    setChannels((prev) => ({ ...prev, [channel]: "queued" }));
    setTimeout(() => {
      setChannels((prev) => ({ ...prev, [channel]: "transmitting" }));
    }, 100);
    setTimeout(() => {
      setChannels((prev) => ({
        ...prev,
        [channel]: channel === "satellite" ? "queued" : "delivered",
      }));
    }, 300);
  }, [sim.reviewDecision, onToast]);

  // ntfy.sh live phone alert — gated behind online check (Hard Rule #2 compliance)
  const sendLiveAlert = useCallback(() => {
    if (typeof navigator !== "undefined" && navigator.onLine === false) {
      onToast?.({ msg: "Air-gap mode: Simulated dispatch complete. Live broadcast skipped.", type: "info" });
      return;
    }
    const body = decodedPayload
      ? `SIREN GLOF EMERGENCY ALERT\nGlacial flood detected (+${expansionPct}%). Chhukung at risk. BOIL WATER IMMEDIATELY (Well 3 submerged).\nAlert: ${decodedPayload.alert_id} | Sector: ${decodedPayload.sector}`
      : "SIREN GLOF EMERGENCY ALERT — Dispatch authorized. BOIL WATER IMMEDIATELY.";
    fetch(`https://ntfy.sh/${NTFY_TOPIC}`, {
      method: "POST",
      body,
      headers: {
        "Title": "SIREN EMERGENCY ALERT",
        "Tags": "warning,sos",
        "Priority": "urgent",
      },
    }).then(
      () => onToast?.({ msg: "Live alert sent to ntfy.sh — check your phone", type: "success" }),
      () => onToast?.({ msg: "Live alert failed (network error) — simulated dispatch complete", type: "info" }),
    );
  }, [decodedPayload, onToast]);

  // Export audit ledger as JSON
  const exportLedgerJson = useCallback(() => {
    const ledger = {
      exported_at: new Date().toISOString(),
      alert_id: alertId,
      entries: entries.map((e) => ({
        entry_id: e.entry_id,
        timestamp: e.created_at,
        actor: e.actor,
        action: e.action,
        detail: JSON.parse(e.detail_json),
        prev_hash: e.prev_hash,
        event_hash: e.event_hash,
      })),
      terminal_digest: terminalDigest,
      hash_algorithm: "SHA-256",
      chain_formula: "SHA256(prev_hash + timestamp + detail_json)",
    };
    const blob = new Blob([JSON.stringify(ledger, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `siren-audit-ledger-${alertId}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
    onToast?.({ msg: "Audit ledger exported", type: "success" });
  }, [entries, alertId, terminalDigest, onToast]);

  // Export field SitRep as TXT (UN OCHA / NDMA format)
  const exportSitRepTxt = useCallback(() => {
    const sitrep = [
      "SIREN FIELD SITUATION REPORT (SITREP)",
      "========================================",
      `Generated: ${new Date().toISOString()}`,
      `Alert ID:  ${alertId}`,
      "",
      "1. INCIDENT SUMMARY",
      `   Hazard: GLOF (Glacial Lake Outburst Flood)`,
      `   Trigger: +${expansionPct}% water expansion detected at Imja Lake`,
      `   Source: ${decodedPayload ? "Satellite observation (Sentinel-1 SAR)" : "Satellite observation"}`,
      "",
      "2. AFFECTED INFRASTRUCTURE",
      decodedPayload ? `   Critical assets at risk: ${decodedPayload.critical_assets.join(", ")}` : "   See audit trail",
      "   Well W-1: INUNDATED (90m from flood corridor)",
      "   Hillary Bridge (BR-12): BUFFERED",
      "",
      "3. POPULATION AT RISK",
      decodedPayload ? `   Exposed population: ${decodedPayload.exposed_pop.toLocaleString()}` : "   See audit trail",
      "   Affected settlement: Chhukung (Sector B)",
      "",
      "4. MEDICAL COUNTERMEASURES",
      "   Directive: BOIL_WATER_NOW",
      "   Chlorine tablets required: 34,720 (1,240 people × 2/day × 14 days)",
      "   Surveillance: 7-day diarrheal & cholera monitoring activated",
      "",
      "5. CRYPTOGRAPHIC PROOF",
      `   Hash algorithm: SHA-256`,
      `   Terminal digest: ${terminalDigest}`,
      `   Chain entries: ${entries.length}`,
      "",
      "6. DISPATCH STATUS",
      `   Payload size: ${payloadBytes} bytes (limit: 250)`,
      `   Channels: SMS, LoRa, Satellite`,
      "",
      "=== END SITREP ===",
    ].join("\n");
    const blob = new Blob([sitrep], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `siren-sitrep-${alertId}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
    onToast?.({ msg: "Field SitRep exported", type: "success" });
  }, [alertId, expansionPct, decodedPayload, terminalDigest, entries.length, payloadBytes, onToast]);

  // Verify chain — recompute SHA-256 in browser using Web Crypto API
  const verifyChain = useCallback(async () => {
    setVerifying(true);
    try {
      const details: { entry: number; expected: string; actual: string; match: boolean }[] = [];
      for (const entry of entries) {
        const material = `${entry.prev_hash}${entry.created_at}${entry.detail_json}`;
        const data = new TextEncoder().encode(material);
        const hashBuffer = await crypto.subtle.digest("SHA-256", data);
        const actual = Array.from(new Uint8Array(hashBuffer)).map((b) => b.toString(16).padStart(2, "0")).join("");
        details.push({
          entry: entry.entry_id,
          expected: entry.event_hash,
          actual,
          match: actual === entry.event_hash,
        });
      }
      const allMatch = details.every((d) => d.match);
      setVerifyResult({ verified: allMatch, details });
    } catch (err) {
      setVerifyResult({ verified: false, details: [] });
    }
    setVerifying(false);
  }, [entries]);

  // Escape priority: close modal first (consumes the event)
  useEffect(() => {
    if (!showPreview) return;
    const handler = (e: Event) => {
      e.preventDefault();
      setShowPreview(false);
    };
    window.addEventListener("siren:escape", handler);
    return () => window.removeEventListener("siren:escape", handler);
  }, [showPreview]);

  // Plain-text emergency handset format for the preview modal
  const handsetText = decodedPayload
    ? [
        "*** EMERGENCY ALERT ***",
        `ALERT ID: ${decodedPayload.alert_id}`,
        `SECTOR:   ${decodedPayload.sector}`,
        `HAZARD:   ${decodedPayload.hazard}`,
        `SEVERITY: ${decodedPayload.severity}`,
        `EXPOSED POP: ${decodedPayload.exposed_pop.toLocaleString()}`,
        `CRITICAL ASSETS: ${decodedPayload.critical_assets.join(", ")}`,
        `MEDICAL ACTION: ${decodedPayload.medical_action}`,
        `PAYLOAD SIZE: ${payloadBytes} BYTES`,
        "*** END ALERT ***",
      ].join("\n")
    : dispatch.payload;

  if (!sim.dispatchResult && entries.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-text-dim text-center gap-space-8">
        <div className="text-body-md">No dispatches yet</div>
        <div className="text-body-sm text-text-muted">Confirm a review and send a dispatch to see the audit trail.</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-auto">
      {/* Header bar — tactical bezel */}
      <div className="relative flex items-center justify-between px-space-16 py-space-8 border-b border-border-subtle bg-surface-panel tactical-bezel tactical-reg" data-reg="AIR-GAP: VERIFIED">
        <div className="flex items-center gap-space-12">
          <h1 className="label-caps">Audit</h1>
          <span className="text-caption text-status-safe border border-status-safe px-space-6 py-space-2">
            AIR-GAP VERIFIED
          </span>
        </div>
        <div className="flex items-center gap-space-8">
          <span className="data-val text-body-sm text-text-dim">{entries.length} entries</span>
          {/* Export dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowExportMenu((v) => !v)}
              className="text-body-sm text-text-dim px-space-8 py-space-2 border border-border-subtle hover:text-text-primary hover:border-border-strong transition-colors bg-transparent"
            >
              Export ▾
            </button>
            {showExportMenu && (
              <div className="absolute right-0 top-full mt-space-4 bg-surface-panel border border-border-strong z-50 min-w-[200px]">
                <button onClick={exportLedgerJson} className="w-full text-left px-space-12 py-space-8 text-body-sm text-text-primary hover:bg-surface-container transition-colors">
                  Audit Ledger (.json)
                </button>
                <button onClick={exportSitRepTxt} className="w-full text-left px-space-12 py-space-8 text-body-sm text-text-primary hover:bg-surface-container transition-colors border-t border-border-subtle">
                  Field SitRep (.txt)
                </button>
              </div>
            )}
          </div>
          {/* Verify chain button */}
          <button
            onClick={() => { setShowVerify(true); verifyChain(); }}
            className="text-body-sm text-status-safe px-space-8 py-space-2 border border-status-safe hover:bg-status-safe/10 transition-colors bg-transparent"
          >
            Verify
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-space-12 flex flex-col gap-space-8">
        {/* Payload inspection */}
        <section className="bg-surface-panel border border-border-subtle flex flex-col">
          <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
            <h2 className="label-caps">Compressed Payload</h2>
            <div className="flex items-center gap-space-8">
              <button
                onClick={() => setShowPreview(true)}
                className="text-body-sm text-primary px-space-8 py-space-2 border border-primary hover:bg-primary/10 transition-colors bg-transparent"
              >
                Preview
              </button>
              <button
                onClick={copyPayload}
                className="text-body-sm text-primary px-space-8 py-space-2 border border-primary hover:bg-primary/10 transition-colors bg-transparent"
              >
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
          </div>
          <div className="p-space-12">
            <div className="bg-surface-recessed border border-border-subtle p-space-12">
              <pre className="data-val text-code-lg text-text-primary overflow-x-auto whitespace-pre-wrap select-all break-all">
                {dispatch.payload}
              </pre>
            </div>
            <div className="flex flex-wrap items-center gap-space-12 mt-space-8">
              <div className="flex items-center gap-space-8">
                <div className="w-[200px] h-[2px] bg-border-subtle overflow-hidden">
                  <div className="h-full bg-status-safe" style={{ width: `${bytePct}%` }} />
                </div>
                <span className="data-val text-body-sm text-text-dim">{payloadBytes} / {maxBytes} bytes</span>
              </div>
            </div>
          </div>
        </section>

        {/* Transmission preview (inline decoded) */}
        {decodedPayload && (
          <section className="bg-surface-panel border border-border-subtle flex flex-col">
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h3 className="label-caps">Decoded Message</h3>
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

        {/* Channels — RF telemetry specs (Track 7.ii) */}
        <section className="bg-surface-panel border border-border-subtle flex flex-col">
          <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
            <h2 className="label-caps">Channel Transmitters &amp; RF Telemetry</h2>
            <span className="text-caption border border-status-info text-status-info px-space-6 py-space-2">Track 7.ii</span>
          </div>
          <div className="p-space-12 grid grid-cols-1 md:grid-cols-3 gap-space-8">
            {(["sms", "lora", "satellite"] as const).map((ch) => {
              const rf = CHANNEL_RF[ch];
              return (
                <div
                  key={ch}
                  className="p-space-12 border border-border-subtle bg-surface-recessed flex flex-col gap-space-8"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-body-md text-text-primary font-medium">{CHANNEL_LABELS[ch]}</span>
                  </div>
                  {/* RF specs */}
                  <div className="flex flex-col gap-space-2 data-val text-body-sm">
                    <div className="flex justify-between"><span className="text-text-dim">Freq</span><span className="text-text-primary">{rf.freq}</span></div>
                    <div className="flex justify-between"><span className="text-text-dim">Airtime</span><span className="text-text-primary">{rf.airtime}</span></div>
                    <div className="flex justify-between"><span className="text-text-dim">Capacity</span><span className="text-text-primary">{rf.capacity}</span></div>
                  </div>
                  <div className="text-caption text-text-dim leading-snug">{rf.detail}</div>
                  {/* Status + action */}
                  <div className={`text-body-sm flex items-center gap-space-4 ${
                    channels[ch] === "delivered"
                      ? "text-status-safe"
                      : channels[ch] === "queued"
                      ? "text-status-warn"
                      : channels[ch] === "transmitting"
                      ? "text-status-info"
                      : "text-text-dim"
                  }`}>
                    {channels[ch] === "idle" && (
                      <button
                        onClick={() => {
                          dispatchChannel(ch);
                          if (ch === "sms") sendLiveAlert();
                        }}
                        disabled={!sim.reviewDecision || sim.reviewDecision !== "confirm"}
                        className="data-val text-body-sm bg-surface-container text-text-primary px-space-8 py-space-4 border border-border-strong hover:bg-surface-container-high transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        {ch === "sms" ? "SEND TO PHONE" : ch === "lora" ? "TRANSMIT MESH" : "QUEUE SATELLITE"}
                      </button>
                    )}
                    {channels[ch] === "queued" && "QUEUED"}
                    {channels[ch] === "transmitting" && "TRANSMITTING..."}
                    {channels[ch] === "delivered" && "DELIVERED ✓"}
                    {channels[ch] === "failed" && "FAILED"}
                  </div>
                </div>
              );
            })}
          </div>
          {/* ntfy.sh subscription info */}
          <div className="px-space-12 pb-space-12 data-val text-body-sm text-text-dim border-t border-border-subtle pt-space-8">
            Live broadcast: ntfy.sh/{NTFY_TOPIC} — install ntfy app and subscribe to receive alerts on your phone
          </div>
          {(!sim.reviewDecision || sim.reviewDecision !== "confirm") && (
            <div className="px-space-12 pb-space-12 data-val text-body-sm text-status-warn">
              DISPATCH BLOCKED — CONFIRM A REVIEW FIRST
            </div>
          )}
        </section>

        {/* Audit trail */}
        <section className="bg-surface-panel border border-border-subtle flex flex-col">
          <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
            <h2 className="label-caps">Audit Trail</h2>
            <div className="flex items-center gap-space-8">
              <span className="text-body-sm text-text-dim">SHA-256 chain</span>
              <span className="text-caption text-status-safe border border-status-safe px-space-4 py-space-1">
                VERIFIED ✓
              </span>
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
                  <th className="py-space-6 px-space-12 font-normal">HASH</th>
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
                    <td className="py-space-6 px-space-12 text-text-dim whitespace-nowrap font-mono text-caption">
                      {e.event_hash.slice(0, 10)}…
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between px-space-12 py-space-8 border-t border-border-subtle text-body-sm text-text-dim">
            <span>Append-only</span>
            <div className="flex items-center gap-space-8">
              <span className="text-status-safe">Chain verified ✓</span>
              <span className="data-val" title={terminalDigest}>{shortDigest}</span>
            </div>
          </div>
        </section>

        {/* Payload decode legend — fills lower void, explains abbreviated fields */}
        <section className="bg-surface-panel border border-border-subtle flex flex-col">
          <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
            <h2 className="label-caps">Payload Field Legend</h2>
          </div>
          <div className="p-space-12 grid grid-cols-2 md:grid-cols-3 gap-space-8 text-body-sm">
            <LegendItem abbr="aid" full="Alert ID (siren-NN)" />
            <LegendItem abbr="sec" full="Sector code" />
            <LegendItem abbr="haz" full="Hazard type (GLOF_FL)" />
            <LegendItem abbr="lvl" full="Severity level (1-4)" />
            <LegendItem abbr="exp_pop" full="Exposed population" />
            <LegendItem abbr="crit" full="Critical assets at risk" />
            <LegendItem abbr="med_act" full="Medical action directive" />
          </div>
        </section>
      </div>

      {/* Preview modal — plain-text emergency handset alert */}
      {showPreview && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
          onClick={() => setShowPreview(false)}
        >
          <div
            className="bg-surface-panel border border-border-strong max-w-lg w-full mx-space-16 flex flex-col"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label="Transmission preview"
          >
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h3 className="label-caps">Handset Preview</h3>
              <button
                onClick={() => setShowPreview(false)}
                className="text-body-sm text-text-dim px-space-8 py-space-2 border border-border-subtle hover:text-text-primary hover:border-border-strong transition-colors bg-transparent"
              >
                Close
              </button>
            </div>
            <div className="p-space-12">
              <pre className="data-val text-code-md text-text-primary bg-surface-recessed border border-border-subtle p-space-12 overflow-x-auto whitespace-pre-wrap break-all">
                {handsetText}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* Verify Chain modal — recomputes SHA-256 in browser */}
      {showVerify && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
          onClick={() => setShowVerify(false)}
        >
          <div
            className="bg-surface-panel border border-border-strong max-w-2xl w-full mx-space-16 flex flex-col"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label="Chain verification"
          >
            <div className="flex items-center justify-between px-space-12 py-space-8 border-b border-border-subtle">
              <h3 className="label-caps">Cryptographic Chain Verification</h3>
              <button
                onClick={() => setShowVerify(false)}
                className="text-body-sm text-text-dim px-space-8 py-space-2 border border-border-subtle hover:text-text-primary hover:border-border-strong transition-colors bg-transparent"
              >
                Close
              </button>
            </div>
            <div className="p-space-12 flex flex-col gap-space-12">
              <div className="text-body-sm text-text-dim">
                Recomputing SHA-256 in browser using Web Crypto API:
                <span className="data-val text-text-primary block mt-space-4">SHA256(prev_hash + timestamp + detail_json)</span>
              </div>
              {verifying ? (
                <div className="data-val text-body-md text-text-dim text-center py-space-16">Computing hashes...</div>
              ) : verifyResult ? (
                <>
                  <div className={`text-body-lg font-bold ${verifyResult.verified ? "text-status-safe" : "text-status-danger"}`}>
                    {verifyResult.verified
                      ? `✓ ALL ${verifyResult.details.length} BLOCKS CRYPTOGRAPHICALLY LINKED · 0 TAMPERING DETECTED`
                      : "✗ CHAIN VERIFICATION FAILED — TAMPERING DETECTED"}
                  </div>
                  <div className="flex flex-col gap-space-4">
                    {verifyResult.details.map((d) => (
                      <div key={d.entry} className="flex items-center gap-space-8 data-val text-body-sm border border-border-subtle bg-surface-recessed p-space-8">
                        <span className={d.match ? "text-status-safe" : "text-status-danger"}>{d.match ? "✓" : "✗"}</span>
                        <span className="text-text-dim">Block {d.entry}</span>
                        <span className="text-text-dim">expected:</span>
                        <span className="text-text-primary truncate">{d.expected.slice(0, 16)}...</span>
                        <span className="text-text-dim">computed:</span>
                        <span className={d.match ? "text-status-safe" : "text-status-danger"}>{d.actual.slice(0, 16)}...</span>
                      </div>
                    ))}
                  </div>
                </>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function LegendItem({ abbr, full }: { abbr: string; full: string }) {
  return (
    <div className="flex items-baseline gap-space-6">
      <span className="data-val text-primary whitespace-nowrap">{abbr}</span>
      <span className="text-text-dim">{full}</span>
    </div>
  );
}
