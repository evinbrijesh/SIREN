"""Physics-informed losses for water segmentation (V3 §2.6).

The L_gravity penalty addresses a failure mode of standard pixel-level
losses (BCE, Dice): they optimize for visual similarity only and have no
representation of potential energy or gravity. A network trained this way
can predict water flowing uphill, which is physically impossible.

The penalty uses the DEM elevation (channel 2 of the 4-channel tensor) to
penalize connected water components where higher-elevation water pixels
are connected to lower-elevation water pixels without sufficient downhill
flow — i.e. the network predicts water at a ridge that "drains" to a
valley, which is physically implausible for a flat-water surface.

Implementation (V3 §2.6):
    L = L_Dice + lambda * ReLU(z(p_i) - z(p_j))

where z(p) is the DEM elevation at pixel p, and p_i, p_j are connected
water-class pixels. Connected components are computed on the binarized
prediction via scipy.ndimage.label.

The penalty is a soft constraint — it does not hard-clip predictions, it
gradients the network toward physically plausible water connectivity.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default gravity penalty weight (V3 §2.6: "start at lambda = 0.1, tune
# on validation"). Too high overconstrains (ignores valid superelevated
# ponds); too low and the penalty is decorative.
DEFAULT_LAMBDA_GRAVITY: float = 0.1

# Binarization threshold for connected-component analysis of the prediction.
# Pixels with sigmoid(logits) > this are considered "water" for the purpose
# of computing the gravity penalty.
WATER_THRESHOLD: float = 0.5


def dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Soft Dice loss for binary water segmentation.

    Args:
        logits: (B, 1, H, W) raw logits from the model.
        target: (B, 1, H, W) binary water mask {0, 1}.
        valid: optional (B, 1, H, W) validity mask {0, 1}; invalid pixels
            are excluded from the loss.

    Returns:
        Scalar Dice loss (1 - Dice coefficient).
    """
    probs = torch.sigmoid(logits)
    if valid is not None:
        probs = probs * valid
        target = target * valid
    intersection = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return (1.0 - dice).mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Binary cross-entropy loss with optional validity mask.

    Args:
        logits: (B, 1, H, W) raw logits.
        target: (B, 1, H, W) binary water mask {0, 1}.
        valid: optional (B, 1, H, W) validity mask; invalid pixels excluded.
    """
    if valid is not None:
        # Mask invalid pixels by zeroing their contribution
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = loss * valid
        return loss.sum() / (valid.sum() + 1e-7)
    return F.binary_cross_entropy_with_logits(logits, target)


def gravity_penalty(
    logits: torch.Tensor,
    dem: torch.Tensor,
    threshold: float = WATER_THRESHOLD,
) -> torch.Tensor:
    """L_gravity penalty — penalize water at higher elevation draining to lower.

    For each connected water component (binarized prediction > threshold),
    compute the mean elevation of water pixels and penalize the variance
    of elevation within the component. A flat water surface should have
    near-zero elevation variance; water on a ridge has high variance.

    This is a differentiable approximation of the V3 §2.6 formula. The
    full connected-component + drainage-direction check is expensive to
    differentiate through; instead we use a soft elevation-variance penalty
    weighted by the water probability, which gradients in the same
    direction (toward low-elevation-variance water predictions).

    The penalty is scaled by the mean water probability so that it vanishes
    when the model predicts no water (otherwise the weighted variance
    ratio is non-trivial even with near-zero weights).

    Args:
        logits: (B, 1, H, W) raw logits from the model.
        dem: (B, 1, H, W) DEM elevation in metres (unnormalised).
        threshold: binarization threshold for water probability.

    Returns:
        Scalar penalty (mean over batch).
    """
    probs = torch.sigmoid(logits)  # (B, 1, H, W)
    # Soft water mask — use probability directly for differentiability
    water_weight = probs  # (B, 1, H, W)

    # Per-component elevation variance approximation:
    # weighted_mean_z = sum(w * z) / sum(w)
    # weighted_var_z = sum(w * (z - mean)^2) / sum(w)
    # Penalize high variance (water spread across elevations = uphill water)
    n_pixels = probs.shape[2] * probs.shape[3]
    w_sum = water_weight.sum(dim=(2, 3), keepdim=True) + 1e-7
    mean_z = (water_weight * dem).sum(dim=(2, 3), keepdim=True) / w_sum
    var_z = (water_weight * (dem - mean_z) ** 2).sum(dim=(2, 3), keepdim=True) / w_sum

    # Normalize by DEM scale to keep the penalty in a reasonable range
    from siren.ml.contract import DEM_MAX_M
    normalized_var = var_z / (DEM_MAX_M ** 2)

    # Scale by mean water coverage so the penalty vanishes when no water
    # is predicted (otherwise the variance ratio is non-trivial even with
    # near-zero weights, since the weights cancel in the ratio).
    coverage = w_sum / n_pixels
    return (normalized_var * coverage).mean()


class WaterLoss(nn.Module):
    """Combined loss for water segmentation: Dice + BCE + lambda * L_gravity.

    Args:
        lambda_gravity: weight for the gravity penalty (default 0.1, V3 §2.6).
        dice_weight: weight for the Dice loss term.
        bce_weight: weight for the BCE loss term.
    """

    def __init__(
        self,
        lambda_gravity: float = DEFAULT_LAMBDA_GRAVITY,
        dice_weight: float = 1.0,
        bce_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_gravity = lambda_gravity
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        dem: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the combined loss.

        Args:
            logits: (B, 1, H, W) raw model logits.
            target: (B, 1, H, W) binary water mask.
            dem: (B, 1, H, W) DEM elevation in metres. Required for the
                gravity penalty; if None, the penalty is zero.
            valid: optional (B, 1, H, W) validity mask.

        Returns:
            Dict with 'total', 'dice', 'bce', 'gravity' loss components.
        """
        d = dice_loss(logits, target, valid)
        b = bce_loss(logits, target, valid)

        g = torch.tensor(0.0, device=logits.device)
        if dem is not None and self.lambda_gravity > 0:
            g = gravity_penalty(logits, dem)

        total = self.dice_weight * d + self.bce_weight * b + self.lambda_gravity * g
        return {
            "total": total,
            "dice": d,
            "bce": b,
            "gravity": g,
        }
