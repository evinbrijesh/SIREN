"""Radiometric Terrain Correction (RTC) — gamma-nought γ⁰ for Sentinel-1 (Sprint 1 Step 6).

Sentinel-1 GRD products are calibrated to sigma-nought (σ⁰), which is
normalised to the ellipsoid. In steep Himalayan terrain the local slope
aspect modulates the illuminated area per pixel, producing systematic
brightness variations (foreshortening, layover, shadow) that are the
biggest source of false-positive water classifications in gorges.

Radiometric Terrain Correction flattens these variations by normalising
to the local terrain area projected onto the plane perpendicular to the
radar look direction. Per the Production Roadmap §2.3 the correction is:

    γ⁰ = σ⁰ / cos(θ_local)

where θ_local is the local incidence angle — the angle between the
incoming radar beam and the local surface normal — derived from the DEM
and the satellite look vector.

In decibels (the unit used throughout the SIREN SAR pipeline):

    γ⁰_dB = σ⁰_dB − 10·log10(cos(θ_local))

The corrected γ⁰ replaces σ⁰ as channels 0/1 of the 4-channel tensor
(Sprint 2, ADR-011).

Conventions
-----------
* ``dem`` is a 2D float array in metres (e.g. Copernicus GLO-30) with the
  standard GeoTIFF row orientation: row 0 = north, rows increase
  southward; column 0 = west, columns increase eastward.
* ``look_vector`` is the ground→satellite unit vector in local ENU
  (east, north, up) coordinates. Build it with
  :func:`look_vector_from_angles` from the scene's incidence angle and
  look azimuth, both available in the Sentinel-1 product annotations.
* ``pixel_size_m`` is the DEM pixel spacing in metres. The Copernicus
  GLO-30 default is 30 m. Pass a ``(dx, dy)`` tuple for non-square pixels.

All functions are pure and deterministic (Hard Rule 6). No network, no
disk, no randomness.
"""

from __future__ import annotations

import logging
import math
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

# Copernicus GLO-30 DEM pixel spacing (metres). Used as the default for
# gradient scaling when the caller does not supply pixel dimensions.
DEFAULT_PIXEL_SIZE_M: float = 30.0

# cos(θ_local) is clamped to this minimum before the division to avoid
# division-by-zero / negative values in radar-shadow and layover regions
# (where the terrain faces away from the sensor and the geometric model
# breaks down). A small positive floor keeps γ⁰ finite and bounded.
COS_THETA_MIN: float = 1e-3

# Sentinel-1 IW nominal incidence angle range (degrees). Used only for
# documentation / sanity checks; the actual per-scene value comes from
# the product annotations and is passed by the caller.
S1_IW_INCIDENCE_MIN_DEG: float = 29.0
S1_IW_INCIDENCE_MAX_DEG: float = 46.0

PixelSize = Union[float, tuple[float, float]]
LookVector = tuple[float, float, float]


def look_vector_from_angles(
    incidence_deg: float,
    look_azimuth_deg: float,
) -> LookVector:
    """Build the ground→satellite unit look vector in local ENU coords.

    Args:
        incidence_deg: Radar incidence angle at the scene centre, in
            degrees from vertical (0 = nadir, 90 = grazing). Sentinel-1
            IW scenes range ~29–46°.
        look_azimuth_deg: Compass azimuth (degrees clockwise from north)
            of the look direction from ground to satellite. For
            Sentinel-1 ascending passes over the Himalaya this is ~282°
            (right-looking, westward); descending passes ~108° (eastward).

    Returns:
        ``(lx, ly, lz)`` unit vector with ``lx``=east, ``ly``=north,
        ``lz``=up.
    """
    if not (0.0 <= incidence_deg <= 90.0):
        raise ValueError(
            f"incidence_deg must be in [0, 90], got {incidence_deg}"
        )
    inc = math.radians(incidence_deg)
    az = math.radians(look_azimuth_deg)
    # ENU: east=lx, north=ly, up=lz
    lx = math.sin(inc) * math.sin(az)   # east component
    ly = math.sin(inc) * math.cos(az)   # north component
    lz = math.cos(inc)                  # up component
    return (float(lx), float(ly), float(lz))


def _resolve_pixel_size(pixel_size_m: PixelSize) -> tuple[float, float]:
    """Normalise pixel size to a (dx_east, dy_north) tuple in metres."""
    if isinstance(pixel_size_m, (tuple, list)):
        if len(pixel_size_m) != 2:
            raise ValueError(
                f"pixel_size_m tuple must have length 2, got {len(pixel_size_m)}"
            )
        return float(pixel_size_m[0]), float(pixel_size_m[1])
    p = float(pixel_size_m)
    if p <= 0.0:
        raise ValueError(f"pixel_size_m must be > 0, got {p}")
    return p, p


def surface_normal(
    dem: np.ndarray,
    pixel_size_m: PixelSize = DEFAULT_PIXEL_SIZE_M,
) -> np.ndarray:
    """Per-pixel upward surface normal unit vector from a DEM.

    Args:
        dem: 2D float array of elevations in metres, standard GeoTIFF
            orientation (row 0 = north, rows increase southward).
        pixel_size_m: Pixel spacing in metres (scalar or ``(dx, dy)``).

    Returns:
        float32 array of shape ``(3, H, W)`` with channels
        ``(nx_east, ny_north, nz_up)`` — unit vectors pointing outward
        from the terrain surface.
    """
    dem = np.asarray(dem, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError(f"dem must be 2D, got shape {dem.shape}")

    dx_east, dy_north = _resolve_pixel_size(pixel_size_m)

    # np.gradient(dem, d_axis0, d_axis1) → [dZ/d(row), dZ/d(col)]
    # Rows increase southward, so dZ/d(row) = -dZ/d(north).
    # Columns increase eastward, so dZ/d(col) = dZ/d(east).
    grad_row, grad_col = np.gradient(dem, dy_north, dx_east)
    dz_dx_east = grad_col            # dZ/d(east)
    dz_dy_north = -grad_row          # dZ/d(north) = -dZ/d(row)

    # Upward normal (unnormalised): (-dZ/dx, -dZ/dy, 1)
    nx = -dz_dx_east
    ny = -dz_dy_north
    nz = np.ones_like(dem, dtype=np.float32)

    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    norm = np.where(norm > 0, norm, 1.0)
    return np.stack([nx / norm, ny / norm, nz / norm], axis=0).astype(np.float32)


def slope_degrees(
    dem: np.ndarray,
    pixel_size_m: PixelSize = DEFAULT_PIXEL_SIZE_M,
) -> np.ndarray:
    """Terrain slope angle in degrees from a DEM.

    Convenience wrapper over :func:`surface_normal` — the slope is the
    angle between the surface normal and vertical::

        slope = arccos(nz)

    Sprint 2 reuses this for channel 3 of the 4-channel tensor.
    """
    n = surface_normal(dem, pixel_size_m)
    nz = np.clip(n[2], -1.0, 1.0)
    return np.degrees(np.arccos(nz)).astype(np.float32)


def local_incidence_cos(
    dem: np.ndarray,
    look_vector: LookVector,
    pixel_size_m: PixelSize = DEFAULT_PIXEL_SIZE_M,
) -> np.ndarray:
    """cos(θ_local) — the local incidence angle cosine per pixel.

    θ_local is the angle between the incoming radar beam (satellite→ground,
    i.e. ``-look_vector``) and the upward surface normal. Equivalently,
    for a flat surface cos(θ_local) = cos(θ_incidence).

    Returns a 2D float32 array in ``[-1, 1]``. Values ≤ 0 indicate radar
    shadow / layover (terrain facing away from the sensor); callers
    should clamp before dividing.
    """
    dem = np.asarray(dem, dtype=np.float32)
    lx, ly, lz = look_vector
    # Validate unit vector (tolerate small drift, re-normalise).
    mag = math.sqrt(lx * lx + ly * ly + lz * lz)
    if mag <= 0.0:
        raise ValueError(f"look_vector must be non-zero, got {look_vector}")
    lx, ly, lz = lx / mag, ly / mag, lz / mag

    n = surface_normal(dem, pixel_size_m)  # (3, H, W)
    # cos(θ_local) = L · n  (ground→sat dot upward normal)
    cos_t = lx * n[0] + ly * n[1] + lz * n[2]
    return cos_t.astype(np.float32)


def gamma_nought(
    sigma0_db: np.ndarray,
    dem: np.ndarray,
    look_vector: LookVector,
    pixel_size_m: PixelSize = DEFAULT_PIXEL_SIZE_M,
) -> np.ndarray:
    """Apply Radiometric Terrain Correction: σ⁰_dB → γ⁰_dB.

    Implements the Production Roadmap §2.3 formula::

        γ⁰ = σ⁰ / cos(θ_local)        (linear)
        γ⁰_dB = σ⁰_dB − 10·log10(cos(θ_local))   (dB)

    Args:
        sigma0_db: 2D float array of σ⁰ in decibels (e.g. from
            :func:`siren.preprocess.sar_calibrate.calibrate_s1_to_db`).
        dem: 2D float array of DEM elevations in metres, co-registered
            to the σ⁰ grid. If shapes differ the DEM is bilinearly
            resampled to the σ⁰ grid.
        look_vector: Ground→satellite unit vector in ENU, from
            :func:`look_vector_from_angles`.
        pixel_size_m: DEM pixel spacing in metres.

    Returns:
        2D float32 array of γ⁰ in decibels, same shape as ``sigma0_db``.
        Radar-shadow pixels (cos ≤ ``COS_THETA_MIN``) are clamped so γ⁰
        stays finite; the input σ⁰ is preserved there (no amplification).
    """
    sigma0_db = np.asarray(sigma0_db, dtype=np.float32)
    if sigma0_db.ndim != 2:
        raise ValueError(f"sigma0_db must be 2D, got shape {sigma0_db.shape}")

    dem = np.asarray(dem, dtype=np.float32)
    if dem.shape != sigma0_db.shape:
        dem = _resample_dem(dem, sigma0_db.shape)

    cos_t = local_incidence_cos(dem, look_vector, pixel_size_m)

    # Clamp to a small positive floor in shadow / layover regions.
    cos_t_safe = np.where(cos_t > COS_THETA_MIN, cos_t, COS_THETA_MIN)

    # γ⁰_dB = σ⁰_dB − 10·log10(cos(θ_local))
    correction_db = 10.0 * np.log10(cos_t_safe)
    gamma_db = sigma0_db - correction_db

    # In true shadow (cos ≤ 0) preserve the input σ⁰ rather than
    # amplifying it with the floor — those pixels are unreliable either
    # way and should not be brightened.
    shadow_mask = cos_t <= 0.0
    gamma_db = np.where(shadow_mask, sigma0_db, gamma_db)

    logger.info(
        "RTC γ⁰ applied: cos(θ_local) range=[%.3f, %.3f], shadow px=%d (%.1f%%)",
        float(cos_t.min()), float(cos_t.max()),
        int(shadow_mask.sum()), 100.0 * float(shadow_mask.mean()),
    )
    return gamma_db.astype(np.float32)


def gamma_nought_stack(
    sigma0_db_stack: np.ndarray,
    dem: np.ndarray,
    look_vector: LookVector,
    pixel_size_m: PixelSize = DEFAULT_PIXEL_SIZE_M,
) -> np.ndarray:
    """Apply RTC to a (C, H, W) σ⁰_dB stack (e.g. VV+VH).

    The DEM and look vector are shared across channels (the terrain and
    geometry are identical for co-polarised and cross-polarised bands of
    the same scene). Returns a float32 array of the same shape.
    """
    sigma0_db_stack = np.asarray(sigma0_db_stack, dtype=np.float32)
    if sigma0_db_stack.ndim != 3:
        raise ValueError(
            f"sigma0_db_stack must be 3D (C, H, W), got {sigma0_db_stack.shape}"
        )
    out = np.empty_like(sigma0_db_stack)
    for c in range(sigma0_db_stack.shape[0]):
        out[c] = gamma_nought(
            sigma0_db_stack[c], dem, look_vector, pixel_size_m
        )
    return out


def _resample_dem(dem: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a DEM to a target (H, W) grid.

    Uses scipy.ndimage.zoom for a deterministic, dependency-light
    resample. scipy is already a core dependency (sar_calibrate.py).
    """
    from scipy.ndimage import zoom

    th, tw = target_shape
    zh = th / dem.shape[0]
    zw = tw / dem.shape[1]
    return zoom(dem, (zh, zw), order=1).astype(np.float32)
