"""Dynamic escalation scorer — P(escalation | morphometrics, weather window).

Tier-2 early-warning component: the antecedent-trigger classifier trained
by ``siren.ml.train_dynamic_escalation`` on HMAGLOFDB event windows +
ERA5/ERA5-Land reanalysis. Where the static susceptibility scorer answers
"is this the *kind* of lake that breaches", this answers "is the weather
right now pushing it toward breach" — the forecasting layer.

Runtime contract (fail-closed, ADR-010 shadow evidence):
    - No valid checkpoint → ``is_available=False``, no fabricated score.
    - Missing weather coverage → features stay NaN (XGBoost handles
      missing natively) and the result is flagged ``degraded`` with the
      list of absent feature names. No synthetic weather is invented.
    - Never enters the canonical hazard score; attaches to
      ``change_stats["shadow_evidence"]["dynamic_escalation"]``.

Prior combination:
    p_dynamic already conditions on static morphometrics + weather. When
    the static susceptibility prior is also available, we expose an
    odds-update cross-check:

        posterior_odds = prior_odds * (p_dyn_odds / base_rate_odds)

    i.e. the dynamic model's likelihood ratio against its own training
    prevalence updates the static prior. Diagnostic only — p_dynamic is
    the primary output; the posterior is recorded for auditability.

Warning rule (Phase-3 spec): ``pre_breach_warning`` fires when a detected
expansion co-occurs with p_dynamic >= WARNING_THRESHOLD — a pre-breach
dynamic warning giving hours-to-days of lead before moraine compromise.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from siren.detect.thermal_state import (
    IMJA_ELEV_M,
    LAPSE_RATE_C_PER_KM,
    SERIES_PATH,
)

# NOTE: ml constants are imported inside functions — pandas/xgboost are not
# core runtime deps (they live under [ml]/[production] extras), so risk/
# must not pull them at import time.
_WINDOW_DAYS = 30

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "models" / "checkpoints" / "xgboost_dynamic_escalation.json"
)

WARNING_THRESHOLD = 0.65          # p_dynamic → pre-breach dynamic warning
DEFAULT_BASE_RATE = 0.30          # training prevalence (from dataset meta)

# Imja Tsho static morphometrics — measured offline via
# train_susceptibility_spatial._glacier_features on RGI v7 (regions 13–15):
#   dist_to_nearest_glacier = 275.0 m, glacier area within 10 km = 136.2 km²
IMJA_STATIC = {
    "lake_elev_m": IMJA_ELEV_M,
    "log_lake_area_km2": float(np.log1p(1.28)),
    "log_dist_glacier_m": float(np.log1p(275.0)),
    "log_glacier_area_10km": float(np.log1p(136.2)),
}


class DynamicEscalationScorer:
    """Booster + isotonic-head escalation scorer (shadow evidence)."""

    def __init__(self, base_rate: float = DEFAULT_BASE_RATE) -> None:
        self._model: Any = None
        self._cal_x: np.ndarray | None = None
        self._cal_y: np.ndarray | None = None
        self._platt: tuple[float, float] | None = None
        self._is_loaded = False
        self._base_rate = base_rate
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
            logger.warning("Dynamic escalation checkpoint missing: %s",
                           model_path)
            return False

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (meta.get("evaluation_valid") is False
                    or meta.get("status") == "disqualified"
                    or meta.get("inference_allowed") is False):
                logger.warning("Dynamic escalation checkpoint rejected: %s",
                               model_path.name)
                return False
            self._meta = meta

        try:
            import xgboost as xgb

            self._model = xgb.XGBClassifier()
            self._model.load_model(str(model_path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Dynamic escalation load failed: %s", exc)
            return False

        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            # New sidecar: {"platt": {...}, "isotonic": {...}, "prefer"}
            # legacy flat: {"x_thresholds": [...], "y_thresholds": [...]}
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
            self._base_rate = float(cal.get("base_rate", self._base_rate))

        self._is_loaded = True
        return True

    def predict(self, features: dict[str, float | None]) -> dict[str, Any]:
        """Score one window. Missing features → NaN (XGBoost-native)."""
        if not self._is_loaded or self._model is None:
            raise RuntimeError("DynamicEscalationScorer: no checkpoint loaded")

        from siren.ml.dataset_dynamic_escalation import FEATURE_NAMES

        x = np.array(
            [[features.get(f) if features.get(f) is not None else np.nan
              for f in FEATURE_NAMES]],
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

        missing = [f for f in FEATURE_NAMES
                   if features.get(f) is None]
        contributions = self._feature_contributions(x[0])
        reasons = self._build_reasons(features, p_cal, contributions)

        return {
            "is_available": True,
            "p_dynamic": round(p_cal, 4),
            "p_dynamic_raw": round(p_raw, 4),
            "calibrated": self._cal_x is not None,
            "degraded": bool(missing),
            "features_missing": missing,
            "feature_contributions": contributions,
            "reasons": reasons,
            "method": "xgboost_escalation_v1",
        }

    def combine_with_static(
        self, p_dynamic: float, p_static: float | None,
    ) -> dict[str, float | None]:
        """Odds-update the static prior by the dynamic likelihood ratio."""
        if p_static is None or not (0.0 < p_static < 1.0):
            return {"p_posterior": None}
        eps = 1e-4
        p_dyn = float(np.clip(p_dynamic, eps, 1 - eps))
        p_sta = float(np.clip(p_static, eps, 1 - eps))
        base = float(np.clip(self._base_rate, eps, 1 - eps))

        lr = (p_dyn / (1 - p_dyn)) / (base / (1 - base))
        post_odds = (p_sta / (1 - p_sta)) * lr
        return {
            "p_posterior": round(post_odds / (1 + post_odds), 4),
            "likelihood_ratio": round(lr, 3),
            "base_rate": base,
        }

    def _feature_contributions(self, x: np.ndarray) -> dict[str, float]:
        try:
            import shap

            from siren.ml.dataset_dynamic_escalation import FEATURE_NAMES

            sv = np.asarray(
                shap.TreeExplainer(self._model).shap_values(
                    x.reshape(1, -1))
            )
            if isinstance(sv, list):
                sv = sv[1]
            return {
                f: float(v) for f, v in zip(FEATURE_NAMES, sv.flatten())
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
            for f in ("precip_30d_mm", "mdd_30", "api_30"):
                v = features.get(f)
                if v is not None:
                    reasons.append(f"{f}={v:.2f}")
        reasons.append(f"P_escalation={p_cal:.3f}")
        return reasons


def _imja_window_features(obs_date: str, obs_config: dict) -> dict:
    """Assemble Imja's trailing-30d feature row from committed assets.

    Temperature-derived features come from the committed ERA5-Land series
    (lake_thermal_series.json); precipitation uses the observation's
    recorded 7-day rainfall. Features with no offline source stay None —
    flagged ``degraded`` by the scorer, never fabricated.
    """
    feats: dict[str, float | None] = {
        "precip_30d_mm": None,
        "precip_7d_mm": obs_config.get("rainfall_7d_mm"),
        "max_daily_precip_mm": obs_config.get("rainfall_24h_mm"),
        "heavy_rain_days": None,
        "api_30": None,
        "rain_anom_30d_mm": None,
        "mdd_30": None,
        "mdd_anom_30": None,
        "ft_cycles_14": None,
        **IMJA_STATIC,
    }

    try:
        series = json.loads(Path(SERIES_PATH).read_text())
        days = series.get("days", {})
        station_elev = series.get("station_elev_m")
        end = date.fromisoformat(obs_date[:10])
        lapse_c = (
            LAPSE_RATE_C_PER_KM * (IMJA_ELEV_M - station_elev) / 1000.0
            if station_elev is not None else 0.0
        )
        t_lake = [
            days[(end - timedelta(days=i)).isoformat()].get("mean_c")
            for i in range(_WINDOW_DAYS)
            if (end - timedelta(days=i)).isoformat() in days
            and days[(end - timedelta(days=i)).isoformat()].get("mean_c")
            is not None
        ]
        t_lake = [t - lapse_c for t in t_lake]
        if len(t_lake) >= _WINDOW_DAYS // 2:
            feats["mdd_30"] = round(
                sum(max(0.0, t) for t in t_lake), 2)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Imja window features degraded: %s", exc)

    return feats


def score_imja_observation(
    obs_config: dict[str, Any],
    change_stats: dict[str, Any],
    p_static: float | None = None,
) -> dict[str, Any]:
    """Runtime entry point — score the current Imja observation.

    Returns an explicit ``is_available=False`` result (never a fabricated
    probability) when no valid checkpoint exists.
    """
    scorer = DynamicEscalationScorer()
    if not scorer.load_checkpoint():
        return {
            "is_available": False,
            "reason": (
                "No valid dynamic-escalation checkpoint. Train via "
                "siren.ml.train_dynamic_escalation on the antecedent-window "
                "dataset (requires the weather fetch stage). Shadow-only; "
                "no probability is fabricated (PRD v4.7 §17.3)."
            ),
        }

    obs_date = str(obs_config.get("acquired_at", ""))[:10]
    features = _imja_window_features(obs_date, obs_config)
    result = scorer.predict(features)

    # Odds-update cross-check against the static prior when present
    result["prior_combination"] = scorer.combine_with_static(
        result["p_dynamic"], p_static,
    )

    # Pre-breach warning: detected expansion co-occurring with elevated
    # dynamic escalation probability (Phase-3 spec).
    expansion_pct = float(change_stats.get("expansion_percent") or 0.0)
    result["expansion_pct"] = expansion_pct
    result["pre_breach_warning"] = bool(
        expansion_pct > 0.0 and result["p_dynamic"] >= WARNING_THRESHOLD
    )
    result["warning_threshold"] = WARNING_THRESHOLD
    result["is_shadow"] = True
    return result
