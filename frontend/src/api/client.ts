// SIREN typed API client — matches docs/spec/API_CONTRACT.md.
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

// --- Offline cache + staleness + outbox (O6) ---
// Last successful GET responses are cached in localStorage with a timestamp.
// When the backend is unreachable (any error, not just TypeError), the cached
// version is returned with a staleness indicator. If no cache exists, falls
// back to mock data.

const CACHE_PREFIX = "siren:cache:";
const OUTBOX_KEY = "siren:outbox";
const STALE_THRESHOLD_MS = 5 * 60 * 1000; // 5 minutes

interface CachedResponse<T> {
  data: T;
  cached_at: number;
}

function getCached<T>(key: string): CachedResponse<T> | null {
  try {
    const raw = localStorage.getItem(CACHE_PREFIX + key);
    if (!raw) return null;
    return JSON.parse(raw) as CachedResponse<T>;
  } catch {
    return null;
  }
}

function setCached<T>(key: string, data: T): void {
  try {
    const entry: CachedResponse<T> = { data, cached_at: Date.now() };
    localStorage.setItem(CACHE_PREFIX + key, JSON.stringify(entry));
  } catch {
    // localStorage full or unavailable — non-critical
  }
}

export interface StalenessInfo {
  isStale: boolean;
  cachedAt: number | null;
  ageMs: number;
}

let lastStaleness: StalenessInfo = { isStale: false, cachedAt: null, ageMs: 0 };

export function getLastStaleness(): StalenessInfo {
  return lastStaleness;
}

// Outbox for failed POST requests (review, dispatch) while offline.
// These are retried when the backend becomes reachable again.
interface OutboxEntry {
  id: string;
  path: string;
  method: string;
  body: string;
  queued_at: number;
}

function getOutbox(): OutboxEntry[] {
  try {
    const raw = localStorage.getItem(OUTBOX_KEY);
    return raw ? (JSON.parse(raw) as OutboxEntry[]) : [];
  } catch {
    return [];
  }
}

function saveOutbox(entries: OutboxEntry[]): void {
  try {
    localStorage.setItem(OUTBOX_KEY, JSON.stringify(entries));
  } catch {
    // non-critical
  }
}

export function addToOutbox(path: string, method: string, body: string): void {
  const entries = getOutbox();
  entries.push({
    id: `obx-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    path,
    method,
    body,
    queued_at: Date.now(),
  });
  saveOutbox(entries);
}

export function getOutboxCount(): number {
  return getOutbox().length;
}

export async function flushOutbox(): Promise<number> {
  const entries = getOutbox();
  if (entries.length === 0) return 0;
  let flushed = 0;
  const remaining: OutboxEntry[] = [];
  for (const entry of entries) {
    try {
      const resp = await fetch(`${BASE}${entry.path}`, {
        method: entry.method,
        headers: { "Content-Type": "application/json" },
        body: entry.body,
      });
      if (resp.ok) {
        flushed++;
      } else {
        remaining.push(entry);
      }
    } catch {
      remaining.push(entry);
    }
  }
  saveOutbox(remaining);
  return flushed;
}

// Offline-safe wrapper: caches successful GETs, falls back to cache on any
// error, then to mock data. Tracks staleness for the UI to display.
export async function apiOrMock<T>(call: () => Promise<T>, mockKey: keyof typeof mockData): Promise<T> {
  try {
    const data = await call();
    // Cache successful responses for offline fallback
    setCached(mockKey as string, data);
    lastStaleness = { isStale: false, cachedAt: null, ageMs: 0 };
    return data;
  } catch (error) {
    const offline = typeof navigator !== "undefined" && navigator.onLine === false;
    // Fall back on any error when offline, or on network errors when online
    const isNetworkError = error instanceof TypeError || error instanceof ApiRequestError;
    if (offline || isNetworkError) {
      // Try cached version first
      const cached = getCached<T>(mockKey as string);
      if (cached) {
        const ageMs = Date.now() - cached.cached_at;
        lastStaleness = {
          isStale: ageMs > STALE_THRESHOLD_MS,
          cachedAt: cached.cached_at,
          ageMs,
        };
        return cached.data;
      }
      // No cache — fall back to mock data
      lastStaleness = { isStale: true, cachedAt: null, ageMs: 0 };
      return mockData[mockKey] as T;
    }
    throw error;
  }
}
