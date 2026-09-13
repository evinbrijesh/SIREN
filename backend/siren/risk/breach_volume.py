"""Breach volume estimation from segmentation delta + DEM hypsometry.

Contract-preserving V_breach estimator for the FNO-2D hydrodynamic surrogate
(ADR-012). The deployed FNO checkpoint ``fno_hydro_surrogate_v1.pt`` has a
frozen 2-channel input contract: ``[normalised DEM, log1p(V_breach)/20.0]``.
This module estimates the scalar ``V_breach`` (m³) that feeds channel 1,
replacing the hardcoded placeholder ``obs_config.get("v_breach_m3", 45e6)``
in ``shadow_evidence.py::_compute_shadow_hydro``.

Method:
    1. Extract pre/post-event lake footprints from binary water masks.
    2. Determine the pre-event water surface elevation ``z_pre`` from the DEM
       at the lake rim — the minimum elevation of land pixels immediately
       adjacent to the lake (the outlet/spillway elevation).
    3. Build the lake basin hypsometric volume ``V(z)`` from the DEM inside the
       pre-event footprint::

           V(z) = Σ_i max(0, z − z_bed_i) · pixel_area

       This is the exact hypsometric integral for a discrete pixel grid.
    4. If the lake contracted (post < pre): determine ``z_post`` from the
       post-event rim, ``V_breach = V(z_pre) − V(z_post)`` (observed drainage).
    5. If the lake expanded or is stable: ``V_breach = V(z_pre)`` (potential
       full-drain breach volume — the entire lake would release if the
       moraine dam fails).

No silent fallbacks: missing data, empty masks, or non-positive volumes raise
``ValueError`` (CLAUDE.md: "No silent fallbacks that hide data gaps").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BreachVolumeResult:
    """Output of the breach volume estimator.

    Attributes:
        v_breach_m3: estimated breach volume in cubic metres (scalar).
        z_pre_m: pre-event water surface elevation (metres).
        z_post_m: post-event water surface elevation (metres), or None if
            the lake did not contract.
        lake_area_pre_m2: pre-event lake surface area (m²).
        lake_area_post_m2: post-event lake surface area (m²).
        delta_area_m2: surface area change (pre − post). Positive = contraction.
        lake_volume_pre_m3: total lake volume at z_pre (m³).
        lake_volume_post_m3: remaining lake volume at z_post (m³), or None.
        mode: ``"observed_drainage"`` (contraction) or
            ``"potential_full_drain"`` (expansion/stable).
        provenance: estimation provenance tag.
    """

    v_breach_m3: float
    z_pre_m: float
    z_post_m: float | None
    lake_area_pre_m2: float
    lake_area_post_m2: float
    delta_area_m2: float
    lake_volume_pre_m3: float
    lake_volume_post_m3: float | None
    mode: str
    provenance: str = "breach_volume_hypsometric_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "v_breach_m3": round(self.v_breach_m3, 1),
            "z_pre_m": round(self.z_pre_m, 2),
            "z_post_m": round(self.z_post_m, 2) if self.z_post_m is not None else None,
            "lake_area_pre_m2": round(self.lake_area_pre_m2, 1),
            "lake_area_post_m2": round(self.lake_area_post_m2, 1),
            "delta_area_m2": round(self.delta_area_m2, 1),
            "lake_volume_pre_m3": round(self.lake_volume_pre_m3, 1),
            "lake_volume_post_m3": (
                round(self.lake_volume_post_m3, 1)
                if self.lake_volume_post_m3 is not None
                else None
            ),
            "mode": self.mode,
            "provenance": self.provenance,
        }


def _lake_rim(dem: np.ndarray, water: np.ndarray) -> np.ndarray:
    """Find the lake rim: non-water pixels immediately adjacent to the lake.

    The rim is the set of land pixels that have at least one 4-connected
    water neighbour. The minimum DEM value on the rim approximates the
    outlet/spillway elevation — the lowest point where water can escape.

    Args:
        dem: (H, W) elevation grid (used only for shape).
        water: (H, W) boolean water mask.

    Returns:
        (H, W) boolean array — True at rim pixels.
    """
    w = water.astype(bool)
    # Dilate the water mask by one pixel (4-connectivity), then subtract
    # the water mask itself to get the non-water ring around the lake.
    padded = np.pad(w, 1, mode="constant", constant_values=False)
    dilated = (
        padded[:-2, 1:-1] | padded[2:, 1:-1]
        | padded[1:-1, :-2] | padded[1:-1, 2:]
    )
    return dilated & ~w


def _shoreline_elevation(dem: np.ndarray, water: np.ndarray) -> float:
    """Estimate water surface elevation from the lake rim.

    The water surface elevation equals the elevation of the lowest point on
    the lake rim (the outlet). This is the standard hypsometric convention:
    the lake is filled up to the level of its lowest rim point.

    Args:
        dem: (H, W) elevation grid in metres.
        water: (H, W) binary water mask.

    Returns:
        Water surface elevation in metres.

    Raises:
        ValueError: if the water mask is empty or the lake fills the entire
            grid (no rim pixels).
    """
    if not water.astype(bool).any():
        raise ValueError("water mask is empty — no lake to analyse")

    rim = _lake_rim(dem, water)
    if not rim.any():
        raise ValueError(
            "water mask has no rim pixels (lake fills the entire grid) — "
            "cannot determine outlet elevation; supply a larger DEM extent"
        )
    return float(dem[rim].min())


def _hypsometric_volume(
    bed_elevations: np.ndarray,
    z_surface: float,
    pixel_area: float,
) -> float:
    """Compute lake volume at a given surface elevation.

    ``V(z) = Σ_i max(0, z_surface − z_bed_i) · pixel_area``

    Each pixel contributes its water depth (surface elevation minus bed
    elevation, clamped at 0) times the pixel area.

    Args:
        bed_elevations: 1D array of DEM elevations inside the lake footprint.
        z_surface: water surface elevation in metres.
        pixel_area: area of one pixel in m².

    Returns:
        Volume in m³.
    """
    depths = np.maximum(0.0, z_surface - bed_elevations)
    return float(depths.sum() * pixel_area)


def estimate_breach_volume(
    pre_water_mask: np.ndarray,
    post_water_mask: np.ndarray,
    dem: np.ndarray,
    pixel_area_m2: float,
) -> BreachVolumeResult:
    """Estimate breach volume from segmentation delta + DEM hypsometry.

    Contract-preserving: outputs a scalar ``V_breach`` (m³) that feeds the
    FNO-2D channel 1 as ``log1p(V_breach) / 20.0`` (ADR-012 frozen input
    contract).

    Args:
        pre_water_mask: (H, W) binary array — pre-event water mask.
        post_water_mask: (H, W) binary array — post-event water mask.
        dem: (H, W) bare-earth DEM in metres (same grid as masks).
        pixel_area_m2: area of one pixel in square metres.

    Returns:
        BreachVolumeResult with V_breach and diagnostics.

    Raises:
        ValueError: if inputs are invalid (empty masks, shape mismatch, NaN
            in DEM, non-positive V_breach). No silent fallbacks.
    """
    pre = np.asarray(pre_water_mask).astype(bool)
    post = np.asarray(post_water_mask).astype(bool)
    dem_arr = np.asarray(dem, dtype=np.float64)

    # --- Input validation (no silent fallbacks) ---
    if pre.shape != post.shape or pre.shape != dem_arr.shape:
        raise ValueError(
            f"shape mismatch: pre={pre.shape}, post={post.shape}, "
            f"dem={dem_arr.shape}"
        )
    if not pre.any():
        raise ValueError("pre-event water mask is empty — no lake to breach")
    if pixel_area_m2 <= 0:
        raise ValueError(f"pixel_area_m2 must be positive, got {pixel_area_m2}")

    dem_in_pre_lake = dem_arr[pre]
    if np.any(np.isnan(dem_in_pre_lake)):
        raise ValueError(
            "DEM contains NaN values inside the pre-event lake footprint — "
            "cannot compute hypsometric volume"
        )

    # --- Surface areas ---
    area_pre = float(pre.sum() * pixel_area_m2)
    area_post = float(post.sum() * pixel_area_m2)
    delta_area = area_pre - area_post  # positive = contraction

    # --- Shoreline elevation (outlet) ---
    z_pre = _shoreline_elevation(dem_arr, pre)

    # --- Hypsometric volume at z_pre (total lake volume) ---
    v_pre = _hypsometric_volume(dem_in_pre_lake, z_pre, pixel_area_m2)

    # --- Determine mode: observed drainage vs potential full drain ---
    if delta_area > 0:
        # Lake contracted — observed drainage
        if not post.any():
            # Fully drained: no post-event lake remains. V_post = 0,
            # z_post = minimum bed elevation (the deepest point of the basin).
            z_post = float(dem_in_pre_lake.min())
            v_post = 0.0
        else:
            z_post = _shoreline_elevation(dem_arr, post)
            dem_in_post_lake = dem_arr[post]
            if np.any(np.isnan(dem_in_post_lake)):
                raise ValueError(
                    "DEM contains NaN values inside the post-event "
                    "lake footprint"
                )
            v_post = _hypsometric_volume(
                dem_in_post_lake, z_post, pixel_area_m2
            )
        v_breach = v_pre - v_post
        mode = "observed_drainage"

        if v_breach <= 0:
            raise ValueError(
                f"computed V_breach is non-positive ({v_breach:.1f} m³) — "
                f"z_pre={z_pre:.2f}, z_post={z_post:.2f}, "
                f"v_pre={v_pre:.1f}, v_post={v_post:.1f}; "
                "check mask alignment and DEM consistency"
            )
    else:
        # Lake expanded or stable — potential full-drain breach
        z_post = None
        v_post = None
        v_breach = v_pre
        mode = "potential_full_drain"

        if v_breach <= 0:
            raise ValueError(
                f"computed V_breach is non-positive ({v_breach:.1f} m³) — "
                f"z_pre={z_pre:.2f}; the DEM may not resolve the lake basin "
                "bathymetry (all bed elevations >= shoreline)"
            )

    logger.info(
        "Breach volume estimated: V_breach=%.1f m³, mode=%s, z_pre=%.2f, "
        "z_post=%s, area_pre=%.1f m², area_post=%.1f m²",
        v_breach,
        mode,
        z_pre,
        f"{z_post:.2f}" if z_post is not None else "N/A",
        area_pre,
        area_post,
    )

    return BreachVolumeResult(
        v_breach_m3=v_breach,
        z_pre_m=z_pre,
        z_post_m=z_post,
        lake_area_pre_m2=area_pre,
        lake_area_post_m2=area_post,
        delta_area_m2=delta_area,
        lake_volume_pre_m3=v_pre,
        lake_volume_post_m3=v_post,
        mode=mode,
    )
