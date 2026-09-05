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

export class ApiRequestError extends Error implements ApiError {
  status: number;
  error: string;
  detail: string;

  constructor(status: number, body: ApiError) {
    super(body.detail);
    this.name = "ApiRequestError";
    this.status = status;
    this.error = body.error;
    this.detail = body.detail;
  }
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ error: "http_error", detail: resp.statusText }));
    throw new ApiRequestError(resp.status, body);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  getBasin: () => fetchJson<BasinConfig>("/basin"),
  listObservations: () => fetchJson<ObservationList>("/observations"),
  getObservation: (id: string) => fetchJson<Observation>(`/observations/${id}`),
  listRuns: () => fetchJson<RunList>("/runs"),
  getRun: (runId: string) => fetchJson<Run>(`/runs/${runId}`),
  createRun: (observationId: string) =>
    fetchJson<{ run_id: string; observation_id: string; status: string; started_at: string }>(
      "/runs",
      { method: "POST", body: JSON.stringify({ observation_id: observationId }) },
    ),
  listExposures: (runId: string) => fetchJson<ExposureList>(`/runs/${runId}/exposures`),
  getSarPriority: (runId: string) => fetchJson<SarPriorityList>(`/runs/${runId}/sar-priority`),
  getMlEvidence: (runId: string) => fetchJson<MlEvidence>(`/runs/${runId}/ml-evidence`),
  createReview: (runId: string, reviewer: string, decision: "confirm" | "reject" | "postpone" | "escalate", note?: string) =>
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
  listAuditByRun: (runId: string) => fetchJson<AuditList>(`/audit?run_id=${runId}`),
  processAll: () =>
    fetchJson<{ runs: Run[]; count: number }>("/runs/process-all", { method: "POST" }),
};

// Offline-safe wrapper: HTTP errors remain operational errors; only a genuine
// network failure may use deterministic local demo data.
export async function apiOrMock<T>(call: () => Promise<T>, mockKey: keyof typeof mockData): Promise<T> {
  try {
    return await call();
  } catch (error) {
    const offline = typeof navigator !== "undefined" && navigator.onLine === false;
    if (error instanceof TypeError || offline) {
      return mockData[mockKey] as T;
    }
    throw error;
  }
}
