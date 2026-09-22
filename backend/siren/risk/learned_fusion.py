"""Learned risk-fusion scorer — P(event | morphology, weather, prior).

Level-6 component (DL_PRIMARY_ROADMAP §9): the fused event-probability
model trained by ``siren.ml.train_risk_fusion`` on the 887-window
HMAGLOFDB/NASA POWER corpus. Where ``dynamic_escalation`` (Tier-2)
consumes statics + weather, this scorer additionally consumes the
deployed susceptibility prior (``p_susceptibility``) and the Jun–Sep
monsoon flag — the "fusion" of the two gate-passed learned components
into one calibrated probability.

Runtime contract (fail-closed, advisory evidence):
    - No valid checkpoint → ``is_available=False``, no fabricated score.
    - Missing features → NaN (XGBoost-native) + ``degraded`` flag.
    - Never enters the canonical hazard score; attaches to
      ``change_stats["shadow_evidence"]["learned_risk_fusion"]``.

Gate result (2026-09-22, ``risk_fusion_eval_report.json``):
    spatio-temporal holdout on 887 windows — mean ROC-AUC 0.835 vs the
    deterministic five-factor baseline's 0.504 on identical folds; mean
    raw Brier 0.142 < 0.15. Both gate legs pass. Caveat: statistically
    identical to Tier-2 alone (AUC 0.835 vs 0.838) — the fused scorer's
    value is the unified contract and the honest baseline comparison,
    not marginal accuracy.

The deterministic five-factor severity stays authoritative; this score
is advisory evidence on the review card (promotion record in
``ml/promotion.py``), and ``SIREN_ML_DEMOTE`` reverses at runtime.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "models" / "checkpoints" / "xgboost_risk_fusion.json"
)

# Advisory flag threshold — same convention as dynamic_escalation's
# WARNING_THRESHOLD (calibrated event probability above which the score
# surfaces as an advisory reason). Policy, not a learned threshold.
ADVISORY_THRESHOLD = 0.65


class LearnedFusionScorer:
    """Booster + calibration-sidecar fused risk scorer (advisory)."""

    def __init__(self) -> None:
        self._model: Any = None
        self._cal_x: np.ndarray | None = None
        self._cal_y: np.ndarray | None = None
        self._platt: tuple[float, float] | None = None
        self._is_loaded = False
        self._meta: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_checkpoint(
        self, model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> bool:
        """Load booster + calibration sidecar; reject disqualified meta."""
        model_path = Path(model_path)
        meta_path = model_path.with_suffix(".meta.json")
        cal_path = model_path.with_suffix(".calibration.json")

        if not model_path.exists():
            logger.warning("Risk-fusion checkpoint missing: %s", model_path)
            return False

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (meta.get("evaluation_valid") is False
                    or meta.get("status") == "disqualified"
                    or meta.get("inference_allowed") is False):
                logger.warning("Risk-fusion checkpoint rejected: %s",
                               model_path.name)
                return False
            self._meta = meta

        try:
            import xgboost as xgb

            self._model = xgb.XGBClassifier()
            self._model.load_model(str(model_path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Risk-fusion load failed: %s", exc)
            return False

        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            if "platt" in cal or "isotonic" in cal:
                if "platt" in cal:
                    self._platt = (
                        float(cal["platt"]["coef"]),
                        float(cal["platt"]["intercept"]),
                    )
                iso = cal.get("isotonic")
                if iso:
                    self._cal_x = np.asarray(iso["x_thresholds"])
                    self._cal_y = np.asarray(iso["y_thresholds"])
            else:
                self._cal_x = np.asarray(cal["x_thresholds"])
                self._cal_y = np.asarray(cal["y_thresholds"])

        self._is_loaded = True
        return True

    def predict(self, features: dict[str, float | None]) -> dict[str, Any]:
        """Score one observation. Missing features → NaN."""
        if not self._is_loaded or self._model is None:
            raise RuntimeError("LearnedFusionScorer: no checkpoint loaded")

        from siren.ml.train_risk_fusion import FUSED_FEATURE_NAMES

        x = np.array(
            [[features.get(f) if features.get(f) is not None else np.nan
              for f in FUSED_FEATURE_NAMES]],
            dtype=np.float32,
        )
        p_raw = float(self._model.predict_proba(x)[0, 1])
        if self._platt is not None:
            coef, intercept = self._platt
            p_cal = float(1.0 / (1.0 + np.exp(-(coef * p_raw + intercept))))
        elif self._cal_x is not None:
            p_cal = float(np.interp(p_raw, self._cal_x, self._cal_y))
        else:
            p_cal = p_raw

        missing = [f for f in FUSED_FEATURE_NAMES
                   if features.get(f) is None]
        contributions = self._feature_contributions(x[0])
        reasons = self._build_reasons(features, p_cal, contributions)

        return {
            "is_available": True,
            "p_fused": round(p_cal, 4),
            "p_fused_raw": round(p_raw, 4),
            "calibrated": self._platt is not None or self._cal_x is not None,
            "degraded": bool(missing),
            "features_missing": missing,
            "feature_contributions": contributions,
            "reasons": reasons,
            "method": "xgboost_risk_fusion_v1",
        }

    def _feature_contributions(self, x: np.ndarray) -> dict[str, float]:
        try:
            import shap

            from siren.ml.train_risk_fusion import FUSED_FEATURE_NAMES

            sv = np.asarray(
                shap.TreeExplainer(self._model).shap_values(
                    x.reshape(1, -1))
            )
            if isinstance(sv, list):
                sv = sv[1]
            return {
                f: float(v) for f, v in zip(FUSED_FEATURE_NAMES, sv.flatten())
            }
        except Exception:  # noqa: BLE001 — SHAP optional at runtime
            return {}

    def _build_reasons(
        self,
        features: dict[str, float | None],
        p_cal: float,
        contributions: dict[str, float],
    ) -> list[str]:
        reasons: list[str] = []
        if contributions:
            top = sorted(contributions.items(),
                         key=lambda kv: abs(kv[1]), reverse=True)[:3]
            for name, val in top:
                reasons.append(f"{name}: {'+' if val >= 0 else '-'}"
                               f"{abs(val):.3f} to log-odds")
        else:
            for f in ("precip_30d_mm", "mdd_30", "p_susceptibility"):
                v = features.get(f)
                if v is not None:
                    reasons.append(f"{f}={v:.2f}")
        reasons.append(f"P_event_fused={p_cal:.3f}")
        return reasons


def score_imja_observation(
    obs_config: dict[str, Any],
    change_stats: dict[str, Any],
    p_static: float | None = None,
) -> dict[str, Any]:
    """Runtime entry point — fused advisory score for one observation.

    Builds the 13-feature window vector identically to the Tier-2
    scorer (``dynamic_escalation._imja_window_features`` — committed
    NASA POWER asset, offline), then appends the deployed
    susceptibility prior and the monsoon-window flag.

    Returns an explicit ``is_available=False`` result (never a
    fabricated probability) when no valid checkpoint exists.
    """
    scorer = LearnedFusionScorer()
    if not scorer.load_checkpoint():
        return {
            "is_available": False,
            "reason": (
                "No valid risk-fusion checkpoint. Train via "
                "siren.ml.train_risk_fusion --save-model. Advisory-only; "
                "no probability is fabricated (PRD v4.7 §17.3)."
            ),
        }

    from siren.risk.dynamic_escalation import _imja_window_features
    from siren.scope import in_operational_window

    obs_date = str(obs_config.get("acquired_at", ""))[:10]
    features = _imja_window_features(obs_date, obs_config)
    features["p_susceptibility"] = p_static
    try:
        features["in_monsoon_window"] = float(
            in_operational_window(obs_date))
    except Exception:  # noqa: BLE001 — unparseable date stays missing
        features["in_monsoon_window"] = None

    result = scorer.predict(features)

    # Advisory flag: calibrated fused probability above the escalation
    # convention threshold. The deterministic severity is echoed for the
    # review card's side-by-side comparison — never modified.
    result["advisory_threshold"] = ADVISORY_THRESHOLD
    result["elevated_event_probability"] = bool(
        result["p_fused"] >= ADVISORY_THRESHOLD
    )
    result["deterministic_severity"] = change_stats.get("severity")
    result["expansion_pct"] = float(
        change_stats.get("expansion_percent") or 0.0)

    from siren.ml.promotion import is_promoted, promotion_record

    promoted = is_promoted("learned_risk_fusion")
    result["is_shadow"] = not promoted
    result["promoted"] = promoted
    if promoted:
        result["promotion"] = promotion_record("learned_risk_fusion")
    return result
