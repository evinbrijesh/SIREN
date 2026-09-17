import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import ReviewView from "./ReviewView";
import { SimulationProvider } from "../simulation/SimulationContext";
import { mockData } from "../api/mockData";
import { api } from "../api/client";

const REVIEW_RESPONSE = {
  review_id: "rev-1",
  score_id: "score-1",
  reviewer: "coordinator-01",
  decision: "confirm" as const,
  decided_at: "2026-08-04T12:10:00Z",
};

const DISPATCH_RESPONSE = {
  dispatch_id: "disp-1",
  alert_id: "siren-0001",
  geofence_id: "geo-1",
  payload: "...",
  payload_bytes: 210,
  channel: "sms",
  status: "sent",
  sent_at: "2026-08-04T12:11:00Z",
};

vi.mock("../api/client", async () => {
  const { mockData } = await import("../api/mockData");
  return {
  api: {
    createReview: vi.fn(),
    createDispatch: vi.fn(),
    listExposures: vi.fn(),
    getSarPriority: vi.fn(),
    getMlEvidence: vi.fn(),
    getMlEvaluation: vi.fn(),
    getPersonnelRegistry: vi.fn(),
    createRun: vi.fn(),
    getRun: vi.fn(),
  },
  apiOrMock: async (_call: () => Promise<unknown>, key: keyof typeof mockData) =>
    mockData[key],
  addToOutbox: vi.fn(),
  };
});

vi.mock("../utils/ntfy", () => ({
  NTFY_TOPIC: "siren-emergency-alert",
  sendNtfyAlert: vi.fn((_p: unknown, cb?: (ok: boolean, msg: string) => void) =>
    cb?.(false, "air-gap")),
}));

const run = mockData.runs.runs[0];

function renderView() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const onToast = vi.fn();
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={qc}>
      <SimulationProvider>{children}</SimulationProvider>
    </QueryClientProvider>
  );
  const utils = render(<ReviewView run={run} onToast={onToast} />, { wrapper });
  return { onToast, ...utils };
}

describe("ReviewView human gate (DoD chain)", () => {
  beforeEach(() => {
    // Re-apply implementations per test: clearing also strips
    // mockResolvedValue, and call history must not leak between tests.
    vi.mocked(api.createReview).mockReset().mockResolvedValue(REVIEW_RESPONSE);
    vi.mocked(api.createDispatch).mockReset().mockResolvedValue(DISPATCH_RESPONSE);
  });

  it("shows decision buttons and no dispatch controls before a decision", async () => {
    renderView();
    expect(await screen.findByText("CONFIRM")).toBeInTheDocument();
    expect(screen.getByText("REJECT")).toBeInTheDocument();
    expect(screen.getByText("POSTPONE")).toBeInTheDocument();
    expect(screen.queryByText("Transmit")).not.toBeInTheDocument();
    expect(screen.queryByText("Arm SOS dispatch")).not.toBeInTheDocument();
    expect(screen.getByText("Pending")).toBeInTheDocument();
  });

  it("walks confirm → safety cover → arm → transmit and calls dispatch", async () => {
    const { onToast } = renderView();

    // Step 1: two-step decision confirm
    fireEvent.click(await screen.findByText("CONFIRM"));
    fireEvent.click(await screen.findByText("YES, CONFIRM"));

    // Step 2: decision locked, safety cover over the arm button
    await screen.findByText("Lift safety cover");
    expect(api.createReview).toHaveBeenCalledWith(
      run.run_id, "coordinator-01", "confirm", "demo review",
    );
    expect(screen.queryByText("Transmit")).not.toBeInTheDocument();

    // Step 3: lift cover → arm
    fireEvent.click(screen.getByText("Lift safety cover"));
    fireEvent.click(await screen.findByText("Arm SOS dispatch"));

    // Step 4: armed console → transmit
    expect(screen.getByText("Transmit SOS")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Transmit"));

    await waitFor(() =>
      expect(api.createDispatch).toHaveBeenCalledWith(
        run.run_id, "sms", "sector-b",
      ),
    );
    await waitFor(() =>
      expect(onToast).toHaveBeenCalledWith(
        expect.objectContaining({ msg: expect.stringContaining("210 bytes") }),
      ),
    );
  });

  it("locks the bar as Rejected and never exposes dispatch controls", async () => {
    renderView();
    fireEvent.click(await screen.findByText("REJECT"));

    await screen.findByText("Rejected");
    expect(api.createReview).toHaveBeenCalledWith(
      run.run_id, "coordinator-01", "reject", "demo review",
    );
    expect(api.createDispatch).not.toHaveBeenCalled();
    expect(screen.queryByText("Lift safety cover")).not.toBeInTheDocument();
    expect(screen.queryByText("Transmit")).not.toBeInTheDocument();
    expect(screen.getAllByText("Locked").length).toBeGreaterThan(0);
  });
});
