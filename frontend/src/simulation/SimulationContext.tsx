import { createContext, useContext, useState, useCallback, useRef, type ReactNode } from "react";
import { api } from "../api/client";
import type { Run, DispatchResponse } from "../api/types";

export type SimStep = "before" | "obs-1" | "obs-2" | "obs-3";
export type SimStatus = "idle" | "running" | "complete";
type ObservationStep = Exclude<SimStep, "before">;

const RUN_SEQUENCE: { step: ObservationStep; observationId: string }[] = [
  { step: "obs-1", observationId: "obs-001" },
  { step: "obs-2", observationId: "obs-002" },
  { step: "obs-3", observationId: "obs-003" },
];

const STEP_TO_OBS: Record<ObservationStep, string> = {
  "obs-1": "obs-001",
  "obs-2": "obs-002",
  "obs-3": "obs-003",
};

export interface SimulationState {
  step: SimStep;
  status: SimStatus;
  progress: number;
  selectedAssetId: string | null;
  runIds: Record<ObservationStep, string | null>;
  runData: Record<ObservationStep, Run | null>;
  reviewDecision: "confirm" | "reject" | "postpone" | null;
  dispatchResult: DispatchResponse | null;
}

interface SimulationContextValue extends SimulationState {
  advance: () => Promise<void>;
  runAll: () => Promise<void>;
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
  runData: { "obs-1": null, "obs-2": null, "obs-3": null },
  reviewDecision: null,
  dispatchResult: null,
};

const SimulationContext = createContext<SimulationContextValue | null>(null);

async function waitForProcessed(runId: string): Promise<Run> {
  const deadline = Date.now() + 10_000;
  while (Date.now() <= deadline) {
    const run = await api.getRun(runId);
    if (run.status === "processed") return run;
    if (run.status === "failed") throw new Error(`Pipeline run ${runId} failed`);
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error(`Pipeline run ${runId} did not complete within 10 seconds`);
}

export function SimulationProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SimulationState>(initialState);
  const runningRef = useRef(false);

  const processObservation = useCallback(async (step: ObservationStep, observationId: string, index: number) => {
    setState((prev) => ({
      ...prev,
      step,
      progress: index,
      status: "running",
      reviewDecision: null,
      dispatchResult: null,
      selectedAssetId: null,
    }));
    const result = await api.createRun(observationId);
    const run = await waitForProcessed(result.run_id);
    setState((prev) => ({
      ...prev,
      step,
      progress: index,
      runIds: { ...prev.runIds, [step]: result.run_id },
      runData: { ...prev.runData, [step]: run },
    }));
  }, []);

  const advance = useCallback(async () => {
    if (runningRef.current) return;
    const currentIndex = STEPS.indexOf(state.step);
    if (currentIndex >= RUN_SEQUENCE.length) return;
    const next = RUN_SEQUENCE[currentIndex];
    runningRef.current = true;
    try {
      await processObservation(next.step, next.observationId, currentIndex + 1);
      setState((prev) => ({
        ...prev,
        status: currentIndex + 1 === RUN_SEQUENCE.length ? "complete" : "idle",
      }));
    } catch (error) {
      setState((prev) => ({ ...prev, status: "idle" }));
      throw error;
    } finally {
      runningRef.current = false;
    }
  }, [processObservation, state.step]);

  const runAll = useCallback(async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    setState({ ...initialState, status: "running" });
    try {
      for (let index = 0; index < RUN_SEQUENCE.length; index += 1) {
        const item = RUN_SEQUENCE[index];
        await processObservation(item.step, item.observationId, index + 1);
      }
      setState((prev) => ({ ...prev, step: "obs-3", progress: 3, status: "complete" }));
    } catch (error) {
      setState((prev) => ({ ...prev, status: "idle" }));
      throw error;
    } finally {
      runningRef.current = false;
    }
  }, [processObservation]);

  const reset = useCallback(() => {
    runningRef.current = false;
    setState({ ...initialState });
  }, []);

  const scrubTo = useCallback((step: SimStep) => {
    setState((prev) => ({
      ...prev,
      step,
      progress: STEPS.indexOf(step),
      reviewDecision: null,
      dispatchResult: null,
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
      value={{ ...state, advance, runAll, reset, scrubTo, selectAsset, setReviewDecision, setDispatchResult }}
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

export { STEPS, RUN_SEQUENCE, STEP_TO_OBS };
