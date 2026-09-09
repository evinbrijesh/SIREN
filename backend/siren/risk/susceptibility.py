"""Glacial lake breach susceptibility scorer (V3 §3.2, §3.4).

An XGBoost gradient-boosted tree classifier trained on ICIMOD/HMA historical
glacial-lake attributes to predict calibrated breach probability P_breach ∈ [0, 1].

Features (V3 §3.2):
    - Lake expansion rate (ΔArea/Δt)
    - Moraine dam width / height
    - Rain anomaly (7d vs climatology)
    - Mean slope upstream
    - Lake area (absolute)

Output: calibrated P_breach with a split conformal prediction interval at
95% coverage. When the interval width exceeds 0.35, the system flags high
epistemic uncertainty and refuses autonomous recommendation (V3 §3.4).

Shadow-only (V3 §3.3): the susceptibility score does NOT enter the five-factor
hazard score until §6 acceptance (Brier < 0.15, IoU > 0.65). It is recorded
as separate evidence with TreeSHAP reasons (Sprint 2.7).

Acceptance gates (V3 §3.6):
    - Brier score < 0.15 (calibration quality)
    - Conformal interval width ≤ 0.35 for autonomous recommendation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Feature names in canonical order (V3 §3.2)
FEATURE_NAMES: tuple[str, ...] = (
    "lake_expansion_rate",    # ΔArea/Δt (fraction per year)
    "moraine_dam_width_m",    # dam width in metres
    "moraine_dam_height_m",   # dam freeboard height in metres
    "rain_anomaly_7d",        # 7d rainfall vs climatology (z-score)
    "mean_upstream_slope_deg", # mean slope upstream of lake (degrees)
    "lake_area_km2",          # absolute lake area (km²)
)

# Conformal prediction parameters (V3 §3.4)
DEFAULT_ALPHA: float = 0.05          # 95% coverage
INTERVAL_WIDTH_THRESHOLD: float = 0.35  # > this → manual inspection required

# Brier score gate for load-bearing use (V3 §3.6)
BRIER_SCORE_GATE: float = 0.15


@dataclass
class SusceptibilityResult:
    """Output of the susceptibility scorer.

    Attributes:
        p_breach: calibrated breach probability ∈ [0, 1].
        interval_low: conformal prediction lower bound.
        interval_high: conformal prediction upper bound.
        interval_width: interval_high - interval_low.
        requires_manual_inspection: True if interval_width > 0.35.
        brier_score: calibration Brier score (None if not evaluated).
        is_calibrated: True if the model has been calibrated on a holdout set.
        feature_contributions: TreeSHAP contributions per feature (Sprint 2.7).
        reasons: human-readable explanation strings (≥3 on elevated+).
    """

    p_breach: float
    interval_low: float
    interval_high: float
    interval_width: float
    requires_manual_inspection: bool
    brier_score: float | None = None
    is_calibrated: bool = False
    feature_contributions: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_breach": round(self.p_breach, 4),
            "interval_low": round(self.interval_low, 4),
            "interval_high": round(self.interval_high, 4),
            "interval_width": round(self.interval_width, 4),
            "requires_manual_inspection": self.requires_manual_inspection,
            "brier_score": round(self.brier_score, 4) if self.brier_score is not None else None,
            "is_calibrated": self.is_calibrated,
            "feature_contributions": {
                k: round(v, 4) for k, v in self.feature_contributions.items()
            },
            "reasons": self.reasons,
        }


class SusceptibilityScorer:
    """XGBoost-based glacial lake breach susceptibility scorer.

    The scorer is trained on historical glacial-lake attributes and produces
    a calibrated breach probability with a conformal prediction interval.

    Shadow-only: the output does not enter the five-factor hazard score until
    the acceptance gates (Brier < 0.15) are met (V3 §3.3, §3.6).
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        interval_width_threshold: float = INTERVAL_WIDTH_THRESHOLD,
        random_state: int = 42,
    ) -> None:
        self.alpha = alpha
        self.interval_width_threshold = interval_width_threshold
        self.random_state = random_state
        self._model: Any = None  # xgboost.XGBClassifier
        self._calibration_q: float | None = None  # conformal quantile
        self._brier_score: float | None = None
        self._is_trained: bool = False
        self._is_calibrated: bool = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def brier_score(self) -> float | None:
        return self._brier_score

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
    ) -> float:
        """Train the XGBoost classifier and optionally calibrate conformal interval.

        Args:
            X: training features (n_samples, n_features).
            y: training labels {0, 1} (0 = no breach, 1 = breach).
            X_cal: calibration features for split conformal prediction.
                If None, conformal interval is not available.
            y_cal: calibration labels for split conformal prediction.

        Returns:
            Brier score on the calibration set (or 0.0 if no calibration set).
        """
        import xgboost as xgb
        from sklearn.metrics import brier_score_loss

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self._model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        self._model.fit(X, y)
        self._is_trained = True

        # Calibrate conformal interval on a disjoint calibration set
        if X_cal is not None and y_cal is not None:
            X_cal = np.asarray(X_cal, dtype=np.float32)
            y_cal = np.asarray(y_cal, dtype=np.float32)
            self._calibrate(X_cal, y_cal)
            # Brier score on calibration set
            p_cal = self._model.predict_proba(X_cal)[:, 1]
            self._brier_score = float(brier_score_loss(y_cal, p_cal))
            self._is_calibrated = True
            logger.info(
                "Susceptibility model calibrated: Brier=%.4f, conformal_q=%.4f",
                self._brier_score, self._calibration_q,
            )
            return self._brier_score

        logger.info("Susceptibility model trained (no calibration set)")
        return 0.0

    def _calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Compute the conformal prediction quantile from calibration scores.

        Split conformal (V3 §3.4):
            s_i = |y_i - ŷ_i|  (nonconformity score)
            q = ⌈(1-α)(n+1)⌉-th order statistic of {s_i}
        """
        p_cal = self._model.predict_proba(X_cal)[:, 1]
        scores = np.abs(y_cal - p_cal)
        n = len(scores)
        # ⌈(1-α)(n+1)⌉-th order statistic (1-indexed → 0-indexed)
        q_idx = int(np.ceil((1 - self.alpha) * (n + 1))) - 1
        q_idx = max(0, min(q_idx, n - 1))
        self._calibration_q = float(np.sort(scores)[q_idx])

    def predict(self, X: np.ndarray) -> SusceptibilityResult:
        """Predict breach probability with conformal interval.

        Args:
            X: features (n_samples, n_features). If n_samples > 1, only the
                first sample is scored (the susceptibility scorer is per-lake).

        Returns:
            SusceptibilityResult with p_breach, conformal interval, and reasons.

        Raises:
            RuntimeError: if the model has not been trained.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("SusceptibilityScorer has not been trained")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        p_breach = float(self._model.predict_proba(X[:1])[0, 1])

        # Conformal interval
        if self._calibration_q is not None:
            q = self._calibration_q
            interval_low = max(0.0, p_breach - q)
            interval_high = min(1.0, p_breach + q)
        else:
            # No calibration → wide interval (force manual inspection)
            interval_low = 0.0
            interval_high = 1.0

        interval_width = interval_high - interval_low
        requires_manual = interval_width > self.interval_width_threshold

        # TreeSHAP feature contributions (V3 §3.3)
        feature_contributions = self.explain(X[:1])

        # Build reasons with SHAP contributions (Hard Rule 5: ≥3 on elevated+)
        reasons = self._build_reasons(
            X[0], p_breach, interval_width, requires_manual, feature_contributions
        )

        return SusceptibilityResult(
            p_breach=p_breach,
            interval_low=interval_low,
            interval_high=interval_high,
            interval_width=interval_width,
            requires_manual_inspection=requires_manual,
            brier_score=self._brier_score,
            is_calibrated=self._is_calibrated,
            feature_contributions=feature_contributions,
            reasons=reasons,
        )

    def explain(self, X: np.ndarray) -> dict[str, float]:
        """Compute TreeSHAP feature contributions for a single sample.

        Uses shap.TreeExplainer on the XGBoost model to decompose the
        log-odds prediction into per-feature contributions (V3 §3.3).
        The top contributions are wired into the review card reasons array.

        Args:
            X: features (1, n_features) or (n_features,).

        Returns:
            Dict mapping feature name → SHAP contribution value.
            Positive values increase P_breach; negative values decrease it.

        Raises:
            RuntimeError: if the model has not been trained.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("SusceptibilityScorer has not been trained")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        import shap

        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X[:1])

        # shap_values shape: (1, n_features) for binary classification
        if isinstance(shap_values, list):
            # XGBoost binary: shap returns [class0, class1] — use class1
            shap_values = shap_values[1]
        shap_values = np.asarray(shap_values)

        # Flatten to (n_features,)
        sv = shap_values.flatten()

        contributions: dict[str, float] = {}
        for i, name in enumerate(FEATURE_NAMES):
            if i < len(sv):
                contributions[name] = float(sv[i])
        return contributions

    def _build_reasons(
        self,
        features: np.ndarray,
        p_breach: float,
        interval_width: float,
        requires_manual: bool,
        feature_contributions: dict[str, float] | None = None,
    ) -> list[str]:
        """Build human-readable reasons for the susceptibility score.

        Hard Rule 5: ≥3 entries on elevated+ (p_breach >= 0.5).

        When TreeSHAP contributions are available (V3 §3.3), the top-k
        SHAP contributions are included as reasons — these decompose the
        log-odds prediction into per-feature effects, giving the coordinator
        actionable explanations (e.g. "Lake area expansion rate: +0.32 to
        log-odds").
        """
        reasons: list[str] = []

        # TreeSHAP feature contributions (V3 §3.3) — top 3 by absolute value
        if feature_contributions:
            sorted_contribs = sorted(
                feature_contributions.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:3]
            for name, shap_val in sorted_contribs:
                direction = "+" if shap_val >= 0 else "-"
                reasons.append(
                    f"{name}: {direction}{abs(shap_val):.3f} to log-odds"
                )
        else:
            # Fallback: raw feature values (pre-SHAP behavior)
            for i, name in enumerate(FEATURE_NAMES):
                val = float(features[i]) if i < len(features) else 0.0
                reasons.append(f"{name}={val:.3f}")

        # Probability statement
        reasons.append(f"P_breach={p_breach:.3f}")

        # Interval statement
        if requires_manual:
            reasons.append(
                f"Conformal interval width={interval_width:.3f} > {self.interval_width_threshold} "
                f"— manual inspection required"
            )
        else:
            reasons.append(
                f"Conformal interval width={interval_width:.3f} (95% coverage)"
            )

        # Brier score if available
        if self._brier_score is not None:
            gate_status = "PASS" if self._brier_score < BRIER_SCORE_GATE else "FAIL"
            reasons.append(
                f"Brier score={self._brier_score:.4f} ({gate_status} gate < {BRIER_SCORE_GATE})"
            )

        # Ensure ≥3 reasons on elevated+ (p_breach >= 0.5)
        if p_breach >= 0.5 and len(reasons) < 3:
            reasons.append("Elevated breach susceptibility detected")

        return reasons

    def passes_acceptance_gate(self) -> bool:
        """Check if the model passes the Brier score gate for load-bearing use.

        Returns:
            True if Brier < 0.15 and the model is calibrated.
        """
        return (
            self._is_calibrated
            and self._brier_score is not None
            and self._brier_score < BRIER_SCORE_GATE
        )
