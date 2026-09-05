// SimulationContext — single source of truth for the demo cursor.
// Backend is source of truth for run data; this context is only a cursor + selection.
// When advancing, calls POST /runs to trigger the real pipeline.
// Falls back to mock advancement if the backend is unreachable (offline-safe).

import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import { api } from "../api/client";

export type SimStep = "before" | "obs-1" | "obs-2" | "obs-3";
export type SimStatus = "idle" | "running" | "complete";

// Map sim steps to observation IDs (obs-001, obs-002, obs-003)
const STEP_TO_OBS: Record<Exclude<SimStep, "before">, string> = {
  "obs-1": "obs-001",
  "obs-2": "obs-002",
  "obs-3": "obs-003",
};

export interface SimulationState {
  step: SimStep;
  status: SimStatus;
  progress: number; // 0..4 (number of steps advanced)
  selectedAssetId: string | null;
  // run IDs populated as simulation advances
  runIds: Record<Exclude<SimStep, "before">, string | null>;
  // review decision for current run
  reviewDecision: "confirm" | "reject" | "postpone" | null;
  // dispatch result
  dispatchResult: import("../api/types").DispatchResponse | null;
}

interface SimulationContextValue extends SimulationState {
  advance: () => Promise<void>;
  reset: () => void;
  scrubTo: (step: SimStep) => void;
  selectAsset: (assetId: string | null) => void;
  setReviewDecision: (d: SimulationState["reviewDecision"]) => void;
  setDispatchResult: (d: SimulationState["dispatchResult"]) => void;
}

const STEPS: SimStep[] = ["before", "obs-1", "obs-2", "obs-3"];

const initialState: SimulationState = {
  step: "before",
  status: "idle",
  progress: 0,
  selectedAssetId: null,
  runIds: { "obs-1": null, "obs-2": null, "obs-3": null },
  reviewDecision: null,
  dispatchResult: null,
};

const SimulationContext = createContext<SimulationContextValue | null>(null);

export function SimulationProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SimulationState>(initialState);

  const advance = useCallback(async () => {
    const currentIdx = STEPS.indexOf(state.step);
    if (currentIdx >= STEPS.length - 1) return;

    const nextStep = STEPS[currentIdx + 1] as Exclude<SimStep, "before">;
    const nextIdx = currentIdx + 1;

    // Mark as running
    setState((prev) => ({
      ...prev,
      step: nextStep,
      progress: nextIdx,
      status: "running",
      reviewDecision: null,
      dispatchResult: null,
    }));

    // Call the real backend to trigger the pipeline
    try {
      const obsId = STEP_TO_OBS[nextStep];
      const result = await api.createRun(obsId);
      setState((prev) => ({
        ...prev,
        runIds: { ...prev.runIds, [nextStep]: result.run_id },
        status: nextIdx === STEPS.length - 1 ? "complete" : "running",
      }));
    } catch {
      // Backend unreachable — still advance the cursor (offline demo-safe)
      setState((prev) => ({
        ...prev,
        status: nextIdx === STEPS.length - 1 ? "complete" : "running",
      }));
    }
  }, [state.step]);

  const reset = useCallback(() => {
    setState({ ...initialState });
  }, []);

  const scrubTo = useCallback((step: SimStep) => {
    setState((prev) => ({
      ...prev,
      step,
      progress: STEPS.indexOf(step),
      status: step === "before" ? "idle" : prev.status === "complete" ? "complete" : "running",
    }));
  }, []);

  const selectAsset = useCallback((assetId: string | null) => {
    setState((prev) => ({ ...prev, selectedAssetId: assetId }));
  }, []);

  const setReviewDecision = useCallback((d: SimulationState["reviewDecision"]) => {
    setState((prev) => ({ ...prev, reviewDecision: d }));
  }, []);

  const setDispatchResult = useCallback((d: SimulationState["dispatchResult"]) => {
    setState((prev) => ({ ...prev, dispatchResult: d }));
  }, []);

  return (
    <SimulationContext.Provider
      value={{ ...state, advance, reset, scrubTo, selectAsset, setReviewDecision, setDispatchResult }}
    >
      {children}
    </SimulationContext.Provider>
  );
}

export function useSimulation(): SimulationContextValue {
  const ctx = useContext(SimulationContext);
  if (!ctx) throw new Error("useSimulation must be used within SimulationProvider");
  return ctx;
}

export { STEPS };
