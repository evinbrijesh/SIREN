import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type { ModelStatus, TrendClassification } from "../api/types";

const TREND_COLORS: Record<string, string> = {
  stable: "#4ade80",
  slowly: "#facc15",
  rapidly: "#ef4444",
  uncertain: "#94a3b8",
};

const STAGE_COLORS: Record<number, string> = {
  1: "#3b82f6",
  2: "#8b5cf6",
  3: "#06b6d4",
  4: "#f59e0b",
};

function ModelCard({ model }: { model: ModelStatus }) {
  const meta = model.metadata;
  return (
    <div
      style={{
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: "8px",
        padding: "12px 14px",
        background: "rgba(255,255,255,0.03)",
        borderLeft: `3px solid ${STAGE_COLORS[model.stage] || "#666"}`,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span
            style={{
              fontSize: "10px",
              fontWeight: 700,
              color: STAGE_COLORS[model.stage] || "#666",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Stage {model.stage}
          </span>
          <span style={{ fontSize: "13px", fontWeight: 600, color: "#e2e8f0" }}>
            {model.name}
          </span>
        </div>
        <span
          style={{
            fontSize: "10px",
            fontWeight: 600,
            padding: "2px 8px",
            borderRadius: "4px",
            background: model.loaded ? "rgba(74,222,128,0.15)" : "rgba(239,68,68,0.15)",
            color: model.loaded ? "#4ade80" : "#ef4444",
          }}
        >
          {model.loaded ? "● LOADED" : "○ OFFLINE"}
        </span>
      </div>

      <p style={{ fontSize: "11px", color: "#94a3b8", margin: "6px 0 8px", lineHeight: 1.4 }}>
        {model.description}
      </p>

      <div style={{ display: "flex", gap: "16px", fontSize: "10px", color: "#64748b" }}>
        <span>Arch: {model.architecture}</span>
        {model.weights_size_mb > 0 && <span>Weights: {model.weights_size_mb}MB</span>}
      </div>

      {meta && (
        <div
          style={{
            display: "flex",
            gap: "12px",
            marginTop: "6px",
            fontSize: "10px",
            color: "#94a3b8",
            fontFamily: "monospace",
          }}
        >
          {meta.epoch && <span>epoch={meta.epoch}</span>}
          {meta.loss != null && <span>loss={meta.loss.toFixed(4)}</span>}
          {meta.accuracy != null && <span>acc={(meta.accuracy * 100).toFixed(1)}%</span>}
          {meta.dataset && <span>data={meta.dataset}</span>}
        </div>
      )}
    </div>
  );
}

function TrendDisplay({ trend }: { trend: TrendClassification }) {
  const color = TREND_COLORS[trend.trend_class] || "#666";
  return (
    <div
      style={{
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: "8px",
        padding: "14px",
        background: "rgba(255,255,255,0.03)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: "12px", fontWeight: 600, color: "#e2e8f0" }}>
          Temporal Trend Classification
        </span>
        <span
          style={{
            fontSize: "10px",
            color: trend.ml_model_available ? "#4ade80" : "#94a3b8",
          }}
        >
          {trend.source}
        </span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "10px" }}>
        <div
          style={{
            fontSize: "20px",
            fontWeight: 700,
            color,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          {trend.trend_class}
        </div>
        <div style={{ fontSize: "12px", color: "#94a3b8" }}>
          {(trend.confidence * 100).toFixed(0)}% confidence
        </div>
      </div>

      {/* Expansion progression bar */}
      <div style={{ marginTop: "12px" }}>
        <div style={{ fontSize: "10px", color: "#64748b", marginBottom: "4px" }}>
          Water area progression ({trend.sequence_length} timesteps)
        </div>
        <div style={{ display: "flex", gap: "4px", alignItems: "flex-end", height: "40px" }}>
          {trend.water_areas.map((area, i) => {
            const maxArea = Math.max(...trend.water_areas, 1);
            const heightPct = (area / maxArea) * 100;
            return (
              <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div
                  style={{
                    width: "100%",
                    height: `${heightPct}%`,
                    background: `linear-gradient(to top, ${color}, ${color}88)`,
                    borderRadius: "2px 2px 0 0",
                    minHeight: "2px",
                  }}
                />
                <span style={{ fontSize: "8px", color: "#64748b", marginTop: "2px" }}>
                  T{i}
                </span>
              </div>
            );
          })}
        </div>
        <div style={{ display: "flex", gap: "4px", marginTop: "2px" }}>
          {trend.expansion_pcts.map((pct, i) => (
            <span key={i} style={{ flex: 1, textAlign: "center", fontSize: "8px", color: "#64748b", fontFamily: "monospace" }}>
              {pct > 0 ? "+" : ""}{pct.toFixed(1)}%
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export function MlPanel() {
  const { data: modelStatus } = useQuery({
    queryKey: ["model-status"],
    queryFn: () => api.getModelStatus(),
    staleTime: 30_000,
  });

  const { data: trend } = useQuery({
    queryKey: ["trend"],
    queryFn: () => api.getTrend(),
    staleTime: 10_000,
  });

  if (!modelStatus) return null;

  const models = Object.values(modelStatus.models).sort((a, b) => a.stage - b.stage);
  const loadedCount = models.filter((m) => m.loaded).length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "8px 0",
        }}
      >
        <span style={{ fontSize: "14px", fontWeight: 700, color: "#e2e8f0" }}>
          Deep Learning Pipeline
        </span>
        <span style={{ fontSize: "11px", color: "#64748b" }}>
          {loadedCount}/{models.length} models loaded
        </span>
      </div>

      {/* Trend classification */}
      {trend && <TrendDisplay trend={trend} />}

      {/* Model cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {models.map((model) => (
          <ModelCard key={model.stage} model={model} />
        ))}
      </div>
    </div>
  );
}
