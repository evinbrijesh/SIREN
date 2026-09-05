// ntfy.sh live phone alert utility — gated behind online check (Hard Rule #2)
// Shared between ReviewView (auto-fire on CONFIRM) and AuditView (manual send)

export const NTFY_TOPIC = "siren-emergency-alert";

interface NtfyPayload {
  alertId?: string;
  sector?: string;
  expansionPct?: number;
  decoded?: {
    alert_id: string;
    sector: string;
    hazard: string;
    severity: string;
    exposed_pop: number;
    critical_assets: string[];
    medical_action: string;
  } | null;
}

export function sendNtfyAlert(
  payload: NtfyPayload,
  onResult?: (success: boolean, message: string) => void,
) {
  if (typeof navigator !== "undefined" && navigator.onLine === false) {
    onResult?.(false, "Air-gap mode: Simulated dispatch complete. Live broadcast skipped.");
    return;
  }

  const { decoded, expansionPct } = payload;
  const body = decoded
    ? `SIREN GLOF EMERGENCY ALERT\nGlacial flood detected (+${expansionPct ?? 0}%). Chhukung at risk. BOIL WATER IMMEDIATELY (Well 3 submerged).\nAlert: ${decoded.alert_id} | Sector: ${decoded.sector}`
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
    () => onResult?.(true, "Live alert sent to ntfy.sh — check your phone"),
    () => onResult?.(false, "Live alert failed (network error) — simulated dispatch complete"),
  );
}
