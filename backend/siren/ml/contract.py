"""SAR input contract — single source of truth for model input format.

This module defines the exact tensor format that the ML water-segmentation
model expects during BOTH training and inference. Any code that feeds data
to a trained model must use ``normalize_sar()`` from this module.

Contract (ADR-010 Stage 1):
    - Input shape:  (C, H, W) or (B, C, H, W) where C = 2
    - Channel 0:    VV polarisation in decibels (sigma0 dB)
    - Channel 1:    VH polarisation in decibels (sigma0 dB)
    - Clamping:     [-30.0, 0.0] dB before normalisation
    - Normalised:   linear map to [0.0, 1.0]
    - Water pixels:  low backscatter → near 0.0 after normalisation
    - Land/terrain:  high backscatter → near 1.0 after normalisation

Rationale for the [-30, 0] dB clamp:
    - Open water:   -25 to -30 dB (specular reflection, very dark)
    - Bare soil:    -15 to -10 dB
    - Urban/rock:   -5 to 0 dB (volume/double-bounce scattering)
    - Values below -30 dB are typically noise/layover artefacts.
    - Values above 0 dB are rare for C-band GRD and indicate bright targets.

This contract is FROZEN once a checkpoint is trained against it. Changing
the clamp range or channel order invalidates all existing weights.
"""

from __future__ import annotations

import numpy as np

# --- Contract constants (do not change after weights are trained) ---

SAR_DB_MIN: float = -30.0
SAR_DB_MAX: float = 0.0
SAR_CHANNELS: int = 2  # VV, VH
CHANNEL_NAMES: tuple[str, ...] = ("VV", "VH")


def normalize_sar(arr: np.ndarray) -> np.ndarray:
    """Clamp and linearly normalise a SAR sigma0-dB array to [0, 1].

    Args:
        arr: array of shape (C, H, W) or (B, C, H, W) containing
             sigma0 values in decibels.  NaNs are replaced with the
             channel median before normalisation.

    Returns:
        float32 array of the same shape, values in [0.0, 1.0].
        Water (low dB) maps toward 0.0; terrain (high dB) toward 1.0.
    """
    arr = arr.astype(np.float32, copy=True)

    # Replace NaN with per-channel median (computed on valid pixels only)
    if np.any(np.isnan(arr)):
        if arr.ndim == 3:
            for c in range(arr.shape[0]):
                ch = arr[c]
                valid = ch[~np.isnan(ch)]
                if len(valid) > 0:
                    ch[np.isnan(ch)] = float(np.median(valid))
                else:
                    ch[np.isnan(ch)] = SAR_DB_MIN
        elif arr.ndim == 4:
            for b in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    ch = arr[b, c]
                    valid = ch[~np.isnan(ch)]
                    if len(valid) > 0:
                        ch[np.isnan(ch)] = float(np.median(valid))
                    else:
                        ch[np.isnan(ch)] = SAR_DB_MIN

    # Clamp to the contract dB range
    arr = np.clip(arr, SAR_DB_MIN, SAR_DB_MAX)

    # Linear normalisation to [0, 1]
    arr = (arr - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN)

    return arr.astype(np.float32)


def denormalize_sar(arr: np.ndarray) -> np.ndarray:
    """Inverse of ``normalize_sar`` — map [0, 1] back to dB.

    Useful for visualisation and debugging.
    """
    return (arr.astype(np.float32) * (SAR_DB_MAX - SAR_DB_MIN) + SAR_DB_MIN)
