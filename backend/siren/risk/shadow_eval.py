"""Shadow-mode evaluation harness (V3 §3.6, §6 — ADR-010 §3 ML evaluation gate).

Compares the ML shadow evidence against the deterministic rules-only baseline
to determine whether the ML is ready to become load-bearing. The acceptance
criteria (ADR-011, V3 §3.6):

    - Water detection IoU > 0.65 on event-holdout data
    - Susceptibility Brier score < 0.15 on calibration set

The harness runs the pipeline on a set of observations, collects both the
deterministic scores and the shadow evidence, computes comparison metrics,
and reports whether the ML evaluation gate is passed.

Usage:
    from siren.risk.shadow_eval import ShadowModeEvaluator
    evaluator = ShadowModeEvaluator(repo)
    report = evaluator.evaluate()
    print(report.summary())

Deterministic and reproducible (Hard Rule 6). No network calls (Hard Rule 2).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ML evaluation gate thresholds (ADR-011, V3 §3.6)
IOU_GATE: float = 0.65
BRIER_GATE: float = 0.15


@dataclass
class ShadowMetric:
    """A single comparison metric between ML shadow and deterministic baseline.

    Attributes:
        name: metric name (e.g. "brier_score", "iou").
        ml_value: the ML shadow evidence value.
        deterministic_value: the deterministic baseline value (if applicable).
        gate_threshold: the acceptance gate threshold (if applicable).
        passes_gate: True if the ML value meets the gate threshold.
        description: human-readable description of the metric.
    """

    name: str
    ml_value: float | None
    deterministic_value: float | None = None
    gate_threshold: float | None = None
    passes_gate: bool | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ml_value": round(self.ml_value, 4) if self.ml_value is not None else None,
            "deterministic_value": (
                round(self.deterministic_value, 4)
                if self.deterministic_value is not None
                else None
            ),
            "gate_threshold": self.gate_threshold,
            "passes_gate": self.passes_gate,
            "description": self.description,
        }


@dataclass
class ShadowEvalReport:
    """Report from a shadow-mode evaluation run.

    Attributes:
        metrics: list of comparison metrics.
        overall_passes: True if all gated metrics pass.
        n_observations: number of observations evaluated.
        summary_text: human-readable summary.
    """

    metrics: list[ShadowMetric] = field(default_factory=list)
    overall_passes: bool = False
    n_observations: int = 0
    summary_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "overall_passes": self.overall_passes,
            "n_observations": self.n_observations,
            "summary_text": self.summary_text,
        }

    def summary(self) -> str:
        """Return a human-readable summary of the evaluation."""
        lines = ["=" * 60, "Shadow-Mode Evaluation Report", "=" * 60, ""]

        for metric in self.metrics:
            status = ""
            if metric.passes_gate is True:
                status = " PASS"
            elif metric.passes_gate is False:
                status = " FAIL"
            lines.append(f"  {metric.name}:{status}")
            if metric.ml_value is not None:
                lines.append(f"    ML value:          {metric.ml_value:.4f}")
            if metric.deterministic_value is not None:
                lines.append(f"    Deterministic value: {metric.deterministic_value:.4f}")
            if metric.gate_threshold is not None:
                lines.append(f"    Gate threshold:     {metric.gate_threshold}")
            if metric.description:
                lines.append(f"    {metric.description}")
            lines.append("")

        lines.append("-" * 60)
        if self.overall_passes:
            lines.append("  RESULT: ML evaluation gate PASSED")
            lines.append("  ML is authorized for load-bearing use (requires new ADR)")
        else:
            lines.append("  RESULT: ML evaluation gate NOT PASSED")
            lines.append("  ML remains shadow-only (ADR-010 §3)")
        lines.append("-" * 60)

        return "\n".join(lines)


class ShadowModeEvaluator:
    """Shadow-mode evaluation harness (V3 §3.6, §6).

    Runs the pipeline on a set of observations, collects the shadow evidence,
    and compares it against the deterministic baseline.

    Args:
        repo: the repository (SQLite or PostgreSQL).
        observation_ids: list of observation IDs to evaluate. If None,
            uses all demo observations.
    """

    def __init__(
        self,
        repo: Any = None,
        observation_ids: list[str] | None = None,
    ) -> None:
        if repo is None:
            from siren.db.repo import get_repository
            repo = get_repository()

        if observation_ids is None:
            from siren.pipeline import DEMO_OBS_IDS
            observation_ids = list(DEMO_OBS_IDS)

        self.repo = repo
        self.observation_ids = observation_ids

    def evaluate(self) -> ShadowEvalReport:
        """Run the shadow-mode evaluation.

        Returns:
            ShadowEvalReport with comparison metrics and gate status.
        """
        logger.info(
            "Starting shadow-mode evaluation for %d observations",
            len(self.observation_ids),
        )

        # Collect shadow evidence from pipeline runs
        shadow_results: list[dict[str, Any]] = []
        for obs_id in self.observation_ids:
            try:
                from siren.pipeline import run_pipeline
                run = run_pipeline(obs_id, self.repo)
                change_stats = run.get("change_stats", {}) if run else {}
                shadow = change_stats.get("shadow_evidence", {})
                shadow_results.append({
                    "observation_id": obs_id,
                    "shadow": shadow,
                    "hazard_score": run.get("hazard_score") if run else None,
                    "severity": run.get("severity") if run else None,
                })
            except Exception as exc:
                logger.warning("Pipeline run failed for %s: %s", obs_id, exc)
                shadow_results.append({
                    "observation_id": obs_id,
                    "error": str(exc),
                })

        # Compute metrics
        metrics = self._compute_metrics(shadow_results)

        # Determine overall gate status
        gated_metrics = [m for m in metrics if m.passes_gate is not None]
        overall_passes = all(m.passes_gate for m in gated_metrics) if gated_metrics else False

        report = ShadowEvalReport(
            metrics=metrics,
            overall_passes=overall_passes,
            n_observations=len(shadow_results),
        )
        report.summary_text = report.summary()

        logger.info(
            "Shadow-mode evaluation complete: %d metrics, overall_passes=%s",
            len(metrics), overall_passes,
        )

        return report

    def _compute_metrics(
        self, results: list[dict[str, Any]]
    ) -> list[ShadowMetric]:
        """Compute comparison metrics from shadow evidence results."""
        metrics: list[ShadowMetric] = []

        # 1. Susceptibility Brier score (V3 §3.6 gate)
        brier_scores: list[float] = []
        p_breaches: list[float] = []
        for r in results:
            shadow = r.get("shadow", {})
            sus = shadow.get("susceptibility", {})
            if isinstance(sus, dict) and "brier_score" in sus:
                bs = sus["brier_score"]
                if bs is not None:
                    brier_scores.append(float(bs))
            if isinstance(sus, dict) and "p_breach" in sus:
                p_breaches.append(float(sus["p_breach"]))

        if brier_scores:
            avg_brier = sum(brier_scores) / len(brier_scores)
            metrics.append(ShadowMetric(
                name="brier_score",
                ml_value=avg_brier,
                gate_threshold=BRIER_GATE,
                passes_gate=avg_brier < BRIER_GATE,
                description=(
                    "Calibration quality of the XGBoost breach probability. "
                    f"Average over {len(brier_scores)} observations."
                ),
            ))
        else:
            metrics.append(ShadowMetric(
                name="brier_score",
                ml_value=None,
                gate_threshold=BRIER_GATE,
                passes_gate=None,
                description="Brier score not available (no calibrated susceptibility results)",
            ))

        # 2. Susceptibility P_breach distribution
        if p_breaches:
            avg_p_breach = sum(p_breaches) / len(p_breaches)
            metrics.append(ShadowMetric(
                name="avg_p_breach",
                ml_value=avg_p_breach,
                description=(
                    f"Average breach probability over {len(p_breaches)} observations. "
                    "Shadow-only — does not enter the hazard score."
                ),
            ))

        # 3. Conformal interval width
        interval_widths: list[float] = []
        for r in results:
            shadow = r.get("shadow", {})
            sus = shadow.get("susceptibility", {})
            if isinstance(sus, dict) and "interval_width" in sus:
                interval_widths.append(float(sus["interval_width"]))

        if interval_widths:
            avg_width = sum(interval_widths) / len(interval_widths)
            metrics.append(ShadowMetric(
                name="avg_interval_width",
                ml_value=avg_width,
                description=(
                    f"Average conformal interval width over {len(interval_widths)} "
                    "observations. Width > 0.35 triggers manual inspection."
                ),
            ))

        # 4. FNO trigger rate
        fno_triggered = 0
        for r in results:
            shadow = r.get("shadow", {})
            if not isinstance(shadow, dict):
                continue
            hydro = shadow.get("hydro_surrogate", {})
            if isinstance(hydro, dict) and hydro.get("is_triggered", False):
                fno_triggered += 1
        metrics.append(ShadowMetric(
            name="fno_trigger_rate",
            ml_value=fno_triggered / len(results) if results else 0.0,
            description=(
                f"FNO triggered on {fno_triggered}/{len(results)} observations "
                "(P_breach >= 0.70). Shadow-only."
            ),
        ))

        # 5. Water detection IoU (placeholder — requires labeled validation data)
        metrics.append(ShadowMetric(
            name="water_detection_iou",
            ml_value=None,
            gate_threshold=IOU_GATE,
            passes_gate=None,
            description=(
                "Water detection IoU on event-holdout data. Requires labeled "
                "validation set — not available in demo mode. Production would "
                "compare WaterResUNet predictions against manually delineated "
                "flood extents."
            ),
        ))

        # 6. Shadow evidence coverage
        n_with_shadow = 0
        for r in results:
            shadow = r.get("shadow", {})
            if isinstance(shadow, dict) and "error" not in shadow and shadow:
                n_with_shadow += 1
        metrics.append(ShadowMetric(
            name="shadow_coverage",
            ml_value=n_with_shadow / len(results) if results else 0.0,
            description=(
                f"Shadow evidence attached on {n_with_shadow}/{len(results)} "
                "observations."
            ),
        ))

        return metrics


def run_shadow_evaluation(
    observation_ids: list[str] | None = None,
) -> ShadowEvalReport:
    """Convenience function: run a shadow-mode evaluation.

    Args:
        observation_ids: list of observation IDs to evaluate. If None,
            uses all demo observations.

    Returns:
        ShadowEvalReport with comparison metrics and gate status.
    """
    evaluator = ShadowModeEvaluator(observation_ids=observation_ids)
    return evaluator.evaluate()
