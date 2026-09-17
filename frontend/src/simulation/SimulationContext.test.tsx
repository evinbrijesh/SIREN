import { describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { SimulationProvider, useSimulation } from "./SimulationContext";

vi.mock("../api/client", () => ({
  api: {
    createRun: vi.fn(),
    getRun: vi.fn(),
  },
}));

const wrapper = ({ children }: { children: ReactNode }) => (
  <SimulationProvider>{children}</SimulationProvider>
);

describe("SimulationContext", () => {
  it("starts idle at the before step with no decision", () => {
    const { result } = renderHook(() => useSimulation(), { wrapper });
    expect(result.current.step).toBe("before");
    expect(result.current.status).toBe("idle");
    expect(result.current.reviewDecision).toBeNull();
    expect(result.current.dispatchResult).toBeNull();
  });

  it("records a review decision", () => {
    const { result } = renderHook(() => useSimulation(), { wrapper });
    act(() => result.current.setReviewDecision("confirm"));
    expect(result.current.reviewDecision).toBe("confirm");
  });

  it("clears decision and dispatch when scrubbing to a new step", () => {
    const { result } = renderHook(() => useSimulation(), { wrapper });
    act(() => {
      result.current.setReviewDecision("confirm");
      result.current.setDispatchResult({
        dispatch_id: "d1",
        alert_id: "siren-1",
        geofence_id: "g1",
        payload: "x",
        payload_bytes: 200,
        channel: "sms",
        status: "sent",
        sent_at: "2026-01-01T00:00:00Z",
      });
    });
    act(() => result.current.scrubTo("obs-2"));
    expect(result.current.step).toBe("obs-2");
    expect(result.current.reviewDecision).toBeNull();
    expect(result.current.dispatchResult).toBeNull();
  });

  it("resets to initial state", () => {
    const { result } = renderHook(() => useSimulation(), { wrapper });
    act(() => {
      result.current.setReviewDecision("reject");
      result.current.selectAsset("asset-9");
    });
    act(() => result.current.reset());
    expect(result.current.step).toBe("before");
    expect(result.current.reviewDecision).toBeNull();
    expect(result.current.selectedAssetId).toBeNull();
  });
});
