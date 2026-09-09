"""SAR + terrain input contract — single source of truth for model input format.

This module defines the exact tensor format that the ML water-segmentation
model expects during BOTH training and inference. Any code that feeds data
to a trained model must use the normalize functions from this module.

Contract (ADR-011 / V3 §2.1 — 4-channel terrain-aware):
    - Input shape:  (C, H, W) or (B, C, H, W) where C = 4
    - Channel 0:    VV polarisation in decibels (sigma0 dB)
    - Channel 1:    VH polarisation in decibels (sigma0 dB)
    - Channel 2:    DEM elevation in metres
    - Channel 3:    Slope angle in degrees
    - Clamping:     SAR [-30.0, 0.0] dB; DEM [0, 8848] m; Slope [0, 90] deg
    - Normalised:   each channel linearly mapped to [0.0, 1.0]
    - Water pixels:  low backscatter -> near 0.0 after normalisation
    - Land/terrain:  high backscatter -> near 1.0 after normalisation

Rationale for the [-30, 0] dB clamp (SAR channels):
    - Open water:   -25 to -30 dB (specular reflection, very dark)
    - Bare soil:    -15 to -10 dB
    - Urban/rock:   -5 to 0 dB (volume/double-bounce scattering)
    - Values below -30 dB are typically noise/layover artefacts.
    - Values above 0 dB are rare for C-band GRD and indicate bright targets.

Rationale for DEM/Slope normalization (terrain channels):
    - DEM_MAX_M = 8848.0 (Everest summit — upper bound for Himalayan basins)
    - SLOPE_MAX_DEG = 90.0 (vertical cliff — physical upper bound)
    - Elevation / 8848.0 and slope_deg / 90.0 both map to [0, 1].

This contract is FROZEN once a checkpoint is trained against it. Changing
the clamp range or channel order invalidates all existing weights.

Backward compatibility: the 2-channel ``normalize_sar`` is preserved for the
legacy 2-channel WaterUNet (ADR-010 Stage 1 shadow model). The 4-channel
``normalize_tensor`` is the production path (ADR-011 / V3 §2.1).
"""

from __future__ import annotations

import numpy as np

# --- Contract constants (do not change after weights are trained) ---

SAR_DB_MIN: float = -30.0
SAR_DB_MAX: float = 0.0
DEM_MAX_M: float = 8848.0
SLOPE_MAX_DEG: float = 90.0

# 4-channel terrain-aware contract (ADR-011 / V3 §2.1)
SAR_CHANNELS: int = 4  # VV, VH, DEM, Slope
CHANNEL_NAMES: tuple[str, ...] = ("VV", "VH", "DEM", "Slope")

# Legacy 2-channel contract (ADR-010 Stage 1, preserved for shadow model)
SAR_CHANNELS_LEGACY: int = 2
CHANNEL_NAMES_LEGACY: tuple[str, ...] = ("VV", "VH")


# ---------------------------------------------------------------------------
# SAR-only normalization (legacy 2-channel path, ADR-010 Stage 1)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Terrain channel normalization (DEM + Slope, V3 §2.1)
# ---------------------------------------------------------------------------

def normalize_dem(elevation_m: np.ndarray) -> np.ndarray:
    """Normalise DEM elevation in metres to [0, 1] via ``elevation / 8848``.

    Args:
        elevation_m: array of elevation values in metres. Negative values
            (below sea level) clamp to 0.0; values above 8848 m clamp to 1.0.
            NaNs are replaced with 0.0 (sea level) before normalisation.

    Returns:
        float32 array of the same shape, values in [0.0, 1.0].
    """
    arr = elevation_m.astype(np.float32, copy=True)
    arr = np.where(np.isnan(arr), 0.0, arr)
    arr = np.clip(arr, 0.0, DEM_MAX_M)
    return (arr / DEM_MAX_M).astype(np.float32)


def denormalize_dem(arr: np.ndarray) -> np.ndarray:
    """Inverse of ``normalize_dem`` — map [0, 1] back to metres."""
    return (arr.astype(np.float32) * DEM_MAX_M)


def normalize_slope(slope_deg: np.ndarray) -> np.ndarray:
    """Normalise slope angle in degrees to [0, 1] via ``slope_deg / 90``.

    Args:
        slope_deg: array of slope values in degrees [0, 90]. Negative values
            clamp to 0.0; values above 90 clamp to 1.0. NaNs -> 0.0 (flat).

    Returns:
        float32 array of the same shape, values in [0.0, 1.0].
    """
    arr = slope_deg.astype(np.float32, copy=True)
    arr = np.where(np.isnan(arr), 0.0, arr)
    arr = np.clip(arr, 0.0, SLOPE_MAX_DEG)
    return (arr / SLOPE_MAX_DEG).astype(np.float32)


def denormalize_slope(arr: np.ndarray) -> np.ndarray:
    """Inverse of ``normalize_slope`` — map [0, 1] back to degrees."""
    return (arr.astype(np.float32) * SLOPE_MAX_DEG)


# ---------------------------------------------------------------------------
# 4-channel tensor normalization (production path, ADR-011 / V3 §2.1)
# ---------------------------------------------------------------------------

def normalize_tensor(arr: np.ndarray) -> np.ndarray:
    """Normalise a 4-channel (VV, VH, DEM, Slope) tensor per-channel to [0, 1].

    Each channel is normalised with its own contract:
        - Channel 0 (VV):  clamp [-30, 0] dB -> [0, 1]  (normalize_sar)
        - Channel 1 (VH):  clamp [-30, 0] dB -> [0, 1]  (normalize_sar)
        - Channel 2 (DEM): clamp [0, 8848] m -> [0, 1]  (normalize_dem)
        - Channel 3 (Slope): clamp [0, 90] deg -> [0, 1] (normalize_slope)

    Args:
        arr: array of shape (4, H, W) or (B, 4, H, W). Channels must be in
             the order (VV, VH, DEM, Slope). SAR channels are in dB, DEM in
             metres, Slope in degrees.

    Returns:
        float32 array of the same shape, all values in [0.0, 1.0].

    Raises:
        ValueError: if the channel dimension is not 4.
    """
    arr = arr.astype(np.float32, copy=True)
    if arr.ndim == 3:
        if arr.shape[0] != 4:
            raise ValueError(
                f"normalize_tensor expects 4 channels, got {arr.shape[0]}"
            )
        arr[0:2] = normalize_sar(arr[0:2])
        arr[2] = normalize_dem(arr[2])
        arr[3] = normalize_slope(arr[3])
    elif arr.ndim == 4:
        if arr.shape[1] != 4:
            raise ValueError(
                f"normalize_tensor expects 4 channels, got {arr.shape[1]}"
            )
        for b in range(arr.shape[0]):
            arr[b, 0:2] = normalize_sar(arr[b, 0:2])
            arr[b, 2] = normalize_dem(arr[b, 2])
            arr[b, 3] = normalize_slope(arr[b, 3])
    else:
        raise ValueError(
            f"normalize_tensor expects 3D or 4D array, got {arr.ndim}D"
        )
    return arr


def denormalize_tensor(arr: np.ndarray) -> np.ndarray:
    """Inverse of ``normalize_tensor`` — map [0, 1] back to physical units.

    Returns an array where channels 0-1 are in dB, channel 2 in metres,
    channel 3 in degrees.
    """
    arr = arr.astype(np.float32, copy=True)
    if arr.ndim == 3:
        arr[0:2] = denormalize_sar(arr[0:2])
        arr[2] = denormalize_dem(arr[2])
        arr[3] = denormalize_slope(arr[3])
    elif arr.ndim == 4:
        for b in range(arr.shape[0]):
            arr[b, 0:2] = denormalize_sar(arr[b, 0:2])
            arr[b, 2] = denormalize_dem(arr[b, 2])
            arr[b, 3] = denormalize_slope(arr[b, 3])
    return arr
