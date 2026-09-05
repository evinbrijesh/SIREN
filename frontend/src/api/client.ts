// SIREN typed API client — matches docs/API_CONTRACT.md.
// Uses /api proxy in dev (vite.config.ts), falls back to mock data when the
// backend is not reachable (offline demo-safe).

import type {
  BasinConfig,
  ObservationList,
  Observation,
  RunList,
  Run,
  ExposureList,
  ReviewResponse,
  DispatchResponse,
  AuditList,
  SarPriorityList,
  MlEvidence,
  ApiError,
} from "./types";
import { mockData } from "./mockData";

const BASE = "/api";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ error: "network", detail: resp.statusText }));
    const err: ApiError = body;
    throw err;
  }
  return resp.json() as Promise<T>;
}

export const api = {
  getBasin: () => fetchJson<BasinConfig>("/basin"),
  listObservations: () => fetchJson<ObservationList>("/observations"),
  getObservation: (id: string) => fetchJson<Observation>(`/observations/${id}`),
  listRuns: () => fetchJson<RunList>("/runs"),
  createRun: (observationId: string) =>
    fetchJson<{ run_id: string; observation_id: string; status: string; started_at: string }>(
      "/runs",
      { method: "POST", body: JSON.stringify({ observation_id: observationId }) },
    ),
  listExposures: (runId: string) => fetchJson<ExposureList>(`/runs/${runId}/exposures`),
  getSarPriority: (runId: string) => fetchJson<SarPriorityList>(`/runs/${runId}/sar-priority`),
  getMlEvidence: (runId: string) => fetchJson<MlEvidence>(`/runs/${runId}/ml-evidence`),
  createReview: (runId: string, reviewer: string, decision: "confirm" | "reject" | "postpone", note?: string) =>
    fetchJson<ReviewResponse>(`/runs/${runId}/review`, {
      method: "POST",
      body: JSON.stringify({ reviewer, decision, note }),
    }),
  createDispatch: (runId: string, channel: string, recipientGroup: string) =>
    fetchJson<DispatchResponse>(`/runs/${runId}/dispatch`, {
      method: "POST",
      body: JSON.stringify({ channel, recipient_group: recipientGroup }),
    }),
  listAudit: (alertId: string) => fetchJson<AuditList>(`/audit?alert_id=${alertId}`),
  processAll: () =>
    fetchJson<{ runs: Run[]; count: number }>("/runs/process-all", { method: "POST" }),
};

// Offline-safe wrapper: if the backend is down, return mock data so the
// frontend renders for demo purposes.
export async function apiOrMock<T>(call: () => Promise<T>, mockKey: keyof typeof mockData): Promise<T> {
  try {
    return await call();
  } catch {
    return mockData[mockKey] as T;
  }
}
