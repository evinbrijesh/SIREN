"""Consensus mask — fuses ML change probability with the deterministic mask.

Per ADR-002, the ML model is an evidence layer, not a replacement. The
consensus mask intersects the ML prediction with the physical hydrological
slope from the DEM, preventing false-positive water predictions on mountain
ridges or cloud shadows.

  consensus = ML_mask AND (rule_based_mask OR slope_valid)

Where slope_valid marks pixels where the terrain slope allows water flow
(not a ridge or cliff face). This ensures the ML model cannot predict water
expansion on physically impossible terrain.
"""

from __future__ import annotations

import numpy as np


def compute_consensus_mask(
    ml_mask: np.ndarray,
    rule_based_mask: np.ndarray,
    dem_slope: np.ndarray | None = None,
    slope_threshold_deg: float = 35.0,
    ml_weight: float = 0.6,
    rule_weight: float = 0.4,
) -> dict[str, np.ndarray]:
    """Fuse ML and rule-based masks into a consensus change mask.

    Args:
        ml_mask: Binary ML change mask (H, W) — from SiameseUNet
        rule_based_mask: Binary rule-based mask (H, W) — from NDWI/backscatter
        dem_slope: Slope in degrees (H, W), or None to skip slope gating
        slope_threshold_deg: Pixels steeper than this are excluded from ML
        ml_weight: Weight for the ML mask in the fusion (0-1)
        rule_weight: Weight for the rule-based mask (0-1)

    Returns:
        Dict with:
          - consensus: Binary fused mask (H, W)
          - confidence: Float confidence map (H, W) in [0, 1]
          - agreement: Where both masks agree (H, W) bool
    """
    # Ensure same shape
    h, w = rule_based_mask.shape[:2]
    ml_resized = _resize_mask(ml_mask, (h, w))

    # Slope gating: exclude steep terrain from ML predictions
    if dem_slope is not None:
        slope_valid = (dem_slope <= slope_threshold_deg).astype(np.uint8)
        ml_gated = ml_resized & slope_valid
    else:
        ml_gated = ml_resized
        slope_valid = np.ones_like(ml_resized)

    # Weighted fusion
    ml_f = ml_gated.astype(np.float32) * ml_weight
    rule_f = rule_based_mask.astype(np.float32) * rule_weight
    fused = ml_f + rule_f

    # Consensus: rule-based detections are always included (Hard Rule 1 — the
    # deterministic mask is the trusted physical method and the ML layer only
    # adds evidence, never suppresses it). ML-only detections pass when their
    # weighted fusion score clears the threshold on slope-valid terrain.
    agreement = (ml_gated == 1) & (rule_based_mask == 1)
    consensus = (
        rule_based_mask.astype(bool) | (fused >= 0.5) | agreement
    ).astype(np.uint8)

    # Confidence: how much the two sources agree
    confidence = np.zeros((h, w), dtype=np.float32)
    both = agreement.astype(np.float32)
    either = ((ml_gated == 1) | (rule_based_mask == 1)).astype(np.float32)
    # High confidence where both agree, medium where either fires
    confidence = np.where(
        agreement,
        0.95,  # both agree → high confidence
        np.where(
            ml_gated & ~rule_based_mask.astype(bool),
            0.60,  # ML only → medium confidence (could be false positive)
            np.where(
                rule_based_mask.astype(bool) & ~ml_gated.astype(bool),
                0.70,  # rule-based only → medium-high (physical method)
                0.0,  # neither → no change
            ),
        ),
    )

    return {
        "consensus": consensus,
        "confidence": confidence,
        "agreement": agreement,
        "ml_gated": ml_gated,
        "slope_valid": slope_valid,
    }


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to target shape using nearest-neighbor."""
    h, w = target_shape
    if mask.shape[:2] == (h, w):
        return mask

    # Simple nearest-neighbor resize (no cv2 dependency)
    src_h, src_w = mask.shape[:2]
    row_idx = (np.arange(h) * src_h / h).astype(int)
    col_idx = (np.arange(w) * src_w / w).astype(int)
    return mask[np.ix_(row_idx, col_idx)]
