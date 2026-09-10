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

# 6-channel multi-temporal contract (V3 §2.8 — multi-temporal upgrade)
# Replaces raw DEM with HAND and adds temporal difference channels Δσ⁰.
# This is the target architecture for passing the IoU > 0.65 gate.
SAR_CHANNELS_MULTITEMPORAL: int = 6
CHANNEL_NAMES_MULTITEMPORAL: tuple[str, ...] = (
    "VV_post", "VH_post", "dVV", "dVH", "HAND", "Slope",
)

# Δσ⁰ clamp range: flood water drops 6–10 dB; dry soil ~0 dB; new build +2–5 dB
DELTA_SAR_DB_MIN: float = -15.0
DELTA_SAR_DB_MAX: float = 5.0

# HAND clamp: 0 m (in-channel) to 200 m (hilltop). Preserves resolution in
# the critical 0–10 m flood-exposure range.
HAND_MAX_M: float = 200.0

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


# ---------------------------------------------------------------------------
# 6-channel multi-temporal normalization (V3 §2.8 — multi-temporal upgrade)
# ---------------------------------------------------------------------------

def normalize_delta_sar(arr: np.ndarray) -> np.ndarray:
    """Clamp and normalise a Δσ⁰ (temporal difference) array to [0, 1].

    Δσ⁰ = σ⁰_post - σ⁰_pre. Flood water shows a large negative drop
    (rough vegetation → smooth water, -6 to -10 dB). Dry soil and
    permanent water show ~0 dB change. New construction shows +2 to +5 dB.

    Args:
        arr: array of Δσ⁰ values in dB. Shape (C, H, W) or (B, C, H, W).

    Returns:
        float32 array in [0, 1]. Flood drop → near 0.0; no change → ~0.75;
        increase → near 1.0.
    """
    arr = arr.astype(np.float32, copy=True)
    arr = np.where(np.isnan(arr), 0.0, arr)  # NaN → no change
    arr = np.clip(arr, DELTA_SAR_DB_MIN, DELTA_SAR_DB_MAX)
    return ((arr - DELTA_SAR_DB_MIN) / (DELTA_SAR_DB_MAX - DELTA_SAR_DB_MIN)).astype(np.float32)


def denormalize_delta_sar(arr: np.ndarray) -> np.ndarray:
    """Inverse of ``normalize_delta_sar`` — map [0, 1] back to Δσ⁰ dB."""
    return (arr.astype(np.float32) * (DELTA_SAR_DB_MAX - DELTA_SAR_DB_MIN) + DELTA_SAR_DB_MIN)


def normalize_hand(hand_m: np.ndarray) -> np.ndarray:
    """Normalise HAND (Height Above Nearest Drainage) in metres to [0, 1].

    HAND is scale-invariant: a floodplain at HAND ∈ [0, 3] m maps to
    [0, 0.015] whether in Bangladesh at sea level or Nepal at 4,500 m.
    This prevents the network from memorising absolute elevation.

    Args:
        hand_m: array of HAND values in metres [0, 200+]. NaN → 0 (in-channel).

    Returns:
        float32 array in [0, 1]. In-channel → 0.0; hilltop → 1.0.
    """
    arr = hand_m.astype(np.float32, copy=True)
    arr = np.where(np.isnan(arr), 0.0, arr)
    arr = np.clip(arr, 0.0, HAND_MAX_M)
    return (arr / HAND_MAX_M).astype(np.float32)


def denormalize_hand(arr: np.ndarray) -> np.ndarray:
    """Inverse of ``normalize_hand`` — map [0, 1] back to metres."""
    return (arr.astype(np.float32) * HAND_MAX_M)


def normalize_tensor_multitemporal(arr: np.ndarray) -> np.ndarray:
    """Normalise a 6-channel multi-temporal tensor per-channel to [0, 1].

    Channel order (V3 §2.8):
        - Channel 0 (VV_post):  clamp [-30, 0] dB → [0, 1]  (normalize_sar)
        - Channel 1 (VH_post):  clamp [-30, 0] dB → [0, 1]  (normalize_sar)
        - Channel 2 (dVV):      clamp [-15, 5] dB → [0, 1]  (normalize_delta_sar)
        - Channel 3 (dVH):      clamp [-15, 5] dB → [0, 1]  (normalize_delta_sar)
        - Channel 4 (HAND):     clamp [0, 200] m → [0, 1]   (normalize_hand)
        - Channel 5 (Slope):    clamp [0, 90] deg → [0, 1]  (normalize_slope)

    Args:
        arr: array of shape (6, H, W) or (B, 6, H, W). Channels must be in
             the order (VV_post, VH_post, dVV, dVH, HAND, Slope).

    Returns:
        float32 array of the same shape, all values in [0.0, 1.0].

    Raises:
        ValueError: if the channel dimension is not 6.
    """
    arr = arr.astype(np.float32, copy=True)
    if arr.ndim == 3:
        if arr.shape[0] != 6:
            raise ValueError(
                f"normalize_tensor_multitemporal expects 6 channels, got {arr.shape[0]}"
            )
        arr[0:2] = normalize_sar(arr[0:2])
        arr[2:4] = normalize_delta_sar(arr[2:4])
        arr[4] = normalize_hand(arr[4])
        arr[5] = normalize_slope(arr[5])
    elif arr.ndim == 4:
        if arr.shape[1] != 6:
            raise ValueError(
                f"normalize_tensor_multitemporal expects 6 channels, got {arr.shape[1]}"
            )
        for b in range(arr.shape[0]):
            arr[b, 0:2] = normalize_sar(arr[b, 0:2])
            arr[b, 2:4] = normalize_delta_sar(arr[b, 2:4])
            arr[b, 4] = normalize_hand(arr[b, 4])
            arr[b, 5] = normalize_slope(arr[b, 5])
    else:
        raise ValueError(
            f"normalize_tensor_multitemporal expects 3D or 4D array, got {arr.ndim}D"
        )
    return arr


def denormalize_tensor_multitemporal(arr: np.ndarray) -> np.ndarray:
    """Inverse of ``normalize_tensor_multitemporal`` — map [0, 1] back to physical units.

    Returns an array where channels 0-1 are post-event dB, channels 2-3
    are Δσ⁰ dB, channel 4 is HAND in metres, channel 5 is slope in degrees.
    """
    arr = arr.astype(np.float32, copy=True)
    if arr.ndim == 3:
        arr[0:2] = denormalize_sar(arr[0:2])
        arr[2:4] = denormalize_delta_sar(arr[2:4])
        arr[4] = denormalize_hand(arr[4])
        arr[5] = denormalize_slope(arr[5])
    elif arr.ndim == 4:
        for b in range(arr.shape[0]):
            arr[b, 0:2] = denormalize_sar(arr[b, 0:2])
            arr[b, 2:4] = denormalize_delta_sar(arr[b, 2:4])
            arr[b, 4] = denormalize_hand(arr[b, 4])
            arr[b, 5] = denormalize_slope(arr[b, 5])
    return arr
