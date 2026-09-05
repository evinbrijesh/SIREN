// SimulationContext — single source of truth for the demo cursor.
// Backend is source of truth for run data; this context is only a cursor + selection.
// No new deps — pure React context.

import { createContext, useContext, useState, useCallback, type ReactNode } from "react";

export type SimStep = "before" | "obs-1" | "obs-2" | "obs-3";
export type SimStatus = "idle" | "running" | "complete";

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
  advance: () => void;
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

  const advance = useCallback(() => {
    setState((prev) => {
      const currentIdx = STEPS.indexOf(prev.step);
      if (currentIdx >= STEPS.length - 1) return prev;
      const nextStep = STEPS[currentIdx + 1];
      const nextIdx = currentIdx + 1;
      return {
        ...prev,
        step: nextStep,
        progress: nextIdx,
        status: nextIdx === STEPS.length - 1 ? "complete" : "running",
        // clear review when advancing to a new step
        reviewDecision: prev.reviewDecision,
        dispatchResult: prev.dispatchResult,
      };
    });
  }, []);

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
