"""Bayesian neural uncertainty via MC Dropout (ADR-013 §9.7.4).

Replaces hardcoded severity thresholds with spatially-resolved epistemic
uncertainty estimation. Monte Carlo Dropout (Gal & Ghahramani 2016) runs
T stochastic forward passes with dropout active at inference, producing
a per-pixel mean (the prediction) and variance (the uncertainty).

The uncertainty map σ²(x, y) identifies regions where the model is
uncertain — OOD terrain, cloud-affected pixels, unusual lake shapes.
Split conformal prediction (Angelopoulos & Bates 2021) calibrates the
uncertainty to guarantee distribution-free coverage of the 90% confidence
interval.

Contract:
    Input:  (B, C, H, W) model input + trained model with dropout
    Output: MCDropoutResult with mean, variance, and confidence bounds

Human gate (Hard Rule 3) is preserved — uncertainty maps inform the
coordinator's review decision but do not auto-dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MCDropoutResult:
    """Output of MC Dropout inference.

    Attributes:
        mean: (B, 1, H, W) mean prediction (water probability, sigmoid).
        variance: (B, 1, H, W) per-pixel epistemic variance.
        std: (B, 1, H, W) per-pixel standard deviation.
        lower_bound: (B, 1, H, W) lower bound of the confidence interval.
        upper_bound: (B, 1, H, W) upper bound of the confidence interval.
        n_samples: number of MC forward passes.
        confidence_level: nominal coverage (e.g., 0.90 for 90% CI).
        conformal_quantile: calibrated quantile used for the interval, or None
            if conformal calibration was not applied.
    """

    mean: np.ndarray
    variance: np.ndarray
    std: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    n_samples: int
    confidence_level: float
    conformal_quantile: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist() if self.mean.ndim <= 2 else None,
            "mean_shape": list(self.mean.shape),
            "variance_mean": float(self.variance.mean()),
            "std_mean": float(self.std.mean()),
            "std_max": float(self.std.max()),
            "n_samples": self.n_samples,
            "confidence_level": self.confidence_level,
            "conformal_quantile": self.conformal_quantile,
        }


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable dropout layers while keeping BatchNorm in eval mode.

    MC Dropout requires dropout to be active at inference time, but
    BatchNorm should use running statistics (eval mode). This function
    iterates over all modules and sets the appropriate mode.

    Args:
        model: the neural network to configure for MC inference.
    """
    model.eval()  # Set everything to eval first
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()  # Re-enable dropout


def mc_dropout_inference(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 30,
    confidence_level: float = 0.90,
    conformal_quantile: float | None = None,
    apply_sigmoid: bool = True,
) -> MCDropoutResult:
    """Run MC Dropout inference and compute uncertainty estimates.

    Runs T stochastic forward passes with dropout active (Gal & Ghahramani
    2016). The per-pixel mean is the prediction; the per-pixel variance
    is the epistemic uncertainty. The confidence interval is computed
    from the empirical quantile of the MC samples.

    Args:
        model: trained model with dropout layers (WaterResUNet with dropout > 0).
        x: (B, C, H, W) input tensor.
        n_samples: number of MC forward passes (T). Default 30.
        confidence_level: nominal coverage level (e.g., 0.90 for 90% CI).
        conformal_quantile: calibrated quantile from split conformal prediction.
            If None, uses the nominal quantile (confidence_level / 2 and
            1 - confidence_level / 2).
        apply_sigmoid: if True, apply sigmoid to logits before computing
            statistics (for binary segmentation). If False, use raw logits.

    Returns:
        MCDropoutResult with mean, variance, std, and confidence bounds.
    """
    enable_mc_dropout(model)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            if apply_sigmoid:
                logits = torch.sigmoid(logits)
            samples.append(logits.cpu().numpy())

    samples_array = np.stack(samples, axis=0)  # (T, B, 1, H, W)

    mean = samples_array.mean(axis=0)  # (B, 1, H, W)
    variance = samples_array.var(axis=0, ddof=1) if n_samples > 1 else np.zeros_like(mean)
    std = np.sqrt(variance)

    # Confidence interval from empirical quantiles
    if conformal_quantile is not None:
        # Conformal-calibrated: use the calibrated quantile
        alpha = 1.0 - confidence_level
        lower_q = conformal_quantile
        upper_q = 1.0 - conformal_quantile
    else:
        # Nominal (uncalibrated) quantiles
        alpha = 1.0 - confidence_level
        lower_q = alpha / 2
        upper_q = 1.0 - alpha / 2

    lower_bound = np.quantile(samples_array, lower_q, axis=0)
    upper_bound = np.quantile(samples_array, upper_q, axis=0)

    return MCDropoutResult(
        mean=mean,
        variance=variance,
        std=std,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n_samples=n_samples,
        confidence_level=confidence_level,
        conformal_quantile=conformal_quantile,
    )


def calibrate_conformal(
    model: nn.Module,
    calibration_inputs: list[torch.Tensor],
    calibration_targets: list[np.ndarray],
    n_samples: int = 30,
    confidence_level: float = 0.90,
    apply_sigmoid: bool = True,
    valid_masks: list[np.ndarray] | None = None,
) -> float:
    """Split conformal calibration for distribution-free coverage.

    Computes the conformal quantile q* that guarantees distribution-free
    coverage of the nominal confidence level (Angelopoulos & Bates 2021).
    The calibrated quantile is used by mc_dropout_inference() to construct
    the confidence interval.

    The nonconformity score is the absolute deviation between the MC mean
    prediction and the ground truth. The conformal quantile is the
    ceil((n+1)(1-alpha)/n)-th quantile of the nonconformity scores.

    Args:
        model: trained model with dropout layers.
        calibration_inputs: list of input tensors for the calibration set.
        calibration_targets: list of ground truth arrays (binary masks).
        n_samples: number of MC forward passes per calibration sample.
        confidence_level: nominal coverage level (e.g., 0.90).
        apply_sigmoid: if True, apply sigmoid to logits.
        valid_masks: optional list of per-sample valid-pixel masks (1=valid).
            When provided, nonconformity scores are computed only on valid
            pixels — datasets like Kuro Siwo carry nodata regions whose
            pixels must not contaminate the score distribution.

    Returns:
        The calibrated conformal quantile q* ∈ [0, 1].
    """
    enable_mc_dropout(model)

    nonconformity_scores = []

    with torch.no_grad():
        for i, (x, target) in enumerate(zip(calibration_inputs, calibration_targets)):
            # Run MC inference
            samples = []
            for _ in range(n_samples):
                logits = model(x)
                if apply_sigmoid:
                    logits = torch.sigmoid(logits)
                samples.append(logits.cpu().numpy())

            samples_array = np.stack(samples, axis=0)  # (T, B, 1, H, W)
            mean_pred = samples_array.mean(axis=0)  # (B, 1, H, W)

            # Nonconformity: absolute deviation from ground truth
            target_arr = np.asarray(target)
            if target_arr.ndim == 3:
                target_arr = target_arr[np.newaxis, ...]  # add batch dim

            # Per-pixel nonconformity
            nonconformity = np.abs(mean_pred - target_arr)
            if valid_masks is not None:
                vm = np.asarray(valid_masks[i])
                while vm.ndim < nonconformity.ndim:
                    vm = vm[np.newaxis, ...]
                nonconformity = nonconformity[vm > 0]
            nonconformity_scores.extend(nonconformity.flatten().tolist())

    n = len(nonconformity_scores)
    alpha = 1.0 - confidence_level

    # Conformal quantile: ceil((n+1)(1-alpha)/n) / n
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # cap at 1.0

    conformal_quantile = float(np.quantile(nonconformity_scores, q_level))

    logger.info(
        "Conformal calibration: n=%d samples, q_level=%.4f, "
        "conformal_quantile=%.4f, nominal_level=%.2f",
        n, q_level, conformal_quantile, confidence_level,
    )

    return conformal_quantile


def evaluate_coverage(
    model: nn.Module,
    test_inputs: list[torch.Tensor],
    test_targets: list[np.ndarray],
    conformal_quantile: float,
    n_samples: int = 30,
    confidence_level: float = 0.90,
    apply_sigmoid: bool = True,
    valid_masks: list[np.ndarray] | None = None,
) -> dict[str, float]:
    """Evaluate empirical coverage of the conformal confidence interval.

    Computes the fraction of ground-truth pixels that fall within the
    conformal-calibrated confidence interval. The target is coverage
    within ±5% of the nominal level (PRD §17.2 Bayesian uncertainty gate).

    Args:
        model: trained model with dropout layers.
        test_inputs: list of input tensors for the test set.
        test_targets: list of ground truth arrays (binary masks).
        conformal_quantile: calibrated quantile from calibrate_conformal().
        n_samples: number of MC forward passes.
        confidence_level: nominal coverage level.
        apply_sigmoid: if True, apply sigmoid to logits.
        valid_masks: optional list of per-sample valid-pixel masks (1=valid).
            Coverage is computed only on valid pixels.

    Returns:
        Dict with 'empirical_coverage', 'nominal_level', 'coverage_error'.
    """
    enable_mc_dropout(model)

    total_pixels = 0
    covered_pixels = 0

    with torch.no_grad():
        for i, (x, target) in enumerate(zip(test_inputs, test_targets)):
            result = mc_dropout_inference(
                model, x, n_samples=n_samples,
                confidence_level=confidence_level,
                conformal_quantile=conformal_quantile,
                apply_sigmoid=apply_sigmoid,
            )

            target_arr = np.asarray(target)
            if target_arr.ndim == 3:
                target_arr = target_arr[np.newaxis, ...]

            # Check if target falls within [mean - q*, mean + q*]
            lower = result.mean - conformal_quantile
            upper = result.mean + conformal_quantile

            covered = (target_arr >= lower) & (target_arr <= upper)
            if valid_masks is not None:
                vm = np.asarray(valid_masks[i]) > 0
                while vm.ndim < covered.ndim:
                    vm = vm[np.newaxis, ...]
                covered_pixels += int((covered & vm).sum())
                total_pixels += int(vm.sum())
            else:
                covered_pixels += int(covered.sum())
                total_pixels += target_arr.size

    empirical_coverage = covered_pixels / total_pixels if total_pixels > 0 else 0.0
    coverage_error = abs(empirical_coverage - confidence_level)

    return {
        "empirical_coverage": empirical_coverage,
        "nominal_level": confidence_level,
        "coverage_error": coverage_error,
        "gate_passed": coverage_error <= 0.05,
    }
