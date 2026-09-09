"""Tests for Radiometric Terrain Correction (preprocess/rtc.py).

Sprint 1 Step 6. All tests use synthetic DEMs and σ⁰ arrays — no real
SAFE archives or DEM tiles required. Verifies the Production Roadmap §2.3
formula γ⁰ = σ⁰ / cos(θ_local) and the supporting geometry helpers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from siren.preprocess.rtc import (
    COS_THETA_MIN,
    gamma_nought,
    gamma_nought_stack,
    local_incidence_cos,
    look_vector_from_angles,
    slope_degrees,
    surface_normal,
)


# --------------------------------------------------------------------------- #
# look_vector_from_angles
# --------------------------------------------------------------------------- #
def test_look_vector_flat_nadir():
    """Incidence 0° (nadir) → look vector is straight up (0, 0, 1)."""
    lv = look_vector_from_angles(0.0, 0.0)
    assert lv == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)


def test_look_vector_unit_magnitude():
    """The look vector is always a unit vector for valid angles."""
    for inc in (10.0, 30.0, 45.0, 89.0):
        for az in (0.0, 90.0, 180.0, 270.0, 360.0):
            lv = look_vector_from_angles(inc, az)
            mag = math.sqrt(sum(c * c for c in lv))
            assert mag == pytest.approx(1.0, abs=1e-6), (inc, az, lv)


def test_look_vector_s1_iw_typical():
    """A typical S1 IW descending pass: ~37° incidence, ~108° azimuth."""
    lv = look_vector_from_angles(37.0, 108.0)
    lx, ly, lz = lv
    # east component positive (looking east-ish), up component = cos(37°)
    assert lx > 0.0
    assert lz == pytest.approx(math.cos(math.radians(37.0)), abs=1e-6)


def test_look_vector_rejects_invalid_incidence():
    with pytest.raises(ValueError):
        look_vector_from_angles(-1.0, 0.0)
    with pytest.raises(ValueError):
        look_vector_from_angles(91.0, 0.0)


# --------------------------------------------------------------------------- #
# surface_normal
# --------------------------------------------------------------------------- #
def test_surface_normal_flat_dem():
    """A flat DEM has normals pointing straight up: (0, 0, 1)."""
    dem = np.zeros((10, 10), dtype=np.float32)
    n = surface_normal(dem, pixel_size_m=30.0)
    assert n.shape == (3, 10, 10)
    assert np.allclose(n[0], 0.0)
    assert np.allclose(n[1], 0.0)
    assert np.allclose(n[2], 1.0)


def test_surface_normal_unit_vectors():
    """Every normal is a unit vector."""
    dem = np.array([
        [0, 0, 0, 0],
        [0, 50, 50, 0],
        [0, 50, 50, 0],
        [0, 0, 0, 0],
    ], dtype=np.float32)
    n = surface_normal(dem, pixel_size_m=30.0)
    mag = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    assert np.allclose(mag, 1.0, atol=1e-5)


def test_surface_normal_east_facing_slope():
    """A slope rising to the east (dZ/dx > 0) tilts the normal westward (nx < 0)."""
    # Elevation increases with column index (west→east): rising to the east.
    dem = np.tile(np.arange(10, dtype=np.float32) * 10.0, (10, 1))
    n = surface_normal(dem, pixel_size_m=30.0)
    # Interior pixels: dZ/dx > 0 → nx = -dZ/dx < 0 (normal tilts west)
    assert np.all(n[0, 5, 5] < 0.0)
    assert n[2, 5, 5] > 0.0  # still pointing up


def test_surface_normal_rejects_non_2d():
    with pytest.raises(ValueError):
        surface_normal(np.zeros((3, 10, 10)), 30.0)


# --------------------------------------------------------------------------- #
# slope_degrees
# --------------------------------------------------------------------------- #
def test_slope_flat_is_zero():
    dem = np.zeros((8, 8), dtype=np.float32)
    s = slope_degrees(dem, 30.0)
    assert s.shape == (8, 8)
    assert np.allclose(s, 0.0)


def test_slope_known_gradient():
    """A 100% grade (45°) slope: rise == run over one pixel."""
    # dz/dx = 1.0 (rise 30 m over run 30 m) → slope = 45°
    dem = np.tile(np.arange(8, dtype=np.float32) * 30.0, (8, 1))
    s = slope_degrees(dem, 30.0)
    # Interior pixels should be ~45°
    assert abs(s[4, 4] - 45.0) < 1.0


# --------------------------------------------------------------------------- #
# local_incidence_cos
# --------------------------------------------------------------------------- #
def test_local_incidence_flat_equals_cos_incidence():
    """For flat terrain cos(θ_local) == cos(θ_incidence) everywhere."""
    dem = np.zeros((12, 12), dtype=np.float32)
    lv = look_vector_from_angles(37.0, 108.0)
    cos_t = local_incidence_cos(dem, lv, 30.0)
    expected = math.cos(math.radians(37.0))
    assert np.allclose(cos_t, expected, atol=1e-5)


def test_local_incidence_slope_facing_radar_increases_cos():
    """A slope facing the radar (toward the satellite) increases cos(θ_local).

    Radar to the east (azimuth 90°). A slope *facing* east has downhill to
    the east, i.e. elevation decreases eastward (``arange[::-1]``).
    """
    lv = look_vector_from_angles(37.0, 90.0)  # satellite to the east
    # Downhill to the east → faces the eastward radar.
    dem_facing = np.tile(np.arange(12, dtype=np.float32)[::-1] * 20.0, (12, 1))
    dem_flat = np.zeros((12, 12), dtype=np.float32)
    cos_facing = local_incidence_cos(dem_facing, lv, 30.0)
    cos_flat = local_incidence_cos(dem_flat, lv, 30.0)
    # Interior pixel: facing slope → larger cos (smaller incidence angle)
    assert cos_facing[6, 6] > cos_flat[6, 6]


def test_local_incidence_slope_away_decreases_cos():
    """A slope facing away from the radar decreases cos(θ_local).

    Radar to the east. A slope facing away has downhill to the west, i.e.
    elevation increases eastward (``arange``) — it rises to the east.
    """
    lv = look_vector_from_angles(37.0, 90.0)  # satellite to the east
    # Downhill to the west → faces away from the eastward radar.
    dem_away = np.tile(np.arange(12, dtype=np.float32) * 20.0, (12, 1))
    dem_flat = np.zeros((12, 12), dtype=np.float32)
    cos_away = local_incidence_cos(dem_away, lv, 30.0)
    cos_flat = local_incidence_cos(dem_flat, lv, 30.0)
    assert cos_away[6, 6] < cos_flat[6, 6]


def test_local_incidence_rejects_zero_look_vector():
    dem = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        local_incidence_cos(dem, (0.0, 0.0, 0.0), 30.0)


# --------------------------------------------------------------------------- #
# gamma_nought
# --------------------------------------------------------------------------- #
def test_gamma_nought_flat_dem_is_uniform_correction():
    """Flat DEM → γ⁰_dB = σ⁰_dB − 10·log10(cos θ_i), constant offset."""
    sigma = np.full((10, 10), -15.0, dtype=np.float32)
    dem = np.zeros((10, 10), dtype=np.float32)
    lv = look_vector_from_angles(37.0, 108.0)
    gamma = gamma_nought(sigma, dem, lv, 30.0)

    expected = -15.0 - 10.0 * math.log10(math.cos(math.radians(37.0)))
    assert gamma.shape == sigma.shape
    assert np.allclose(gamma, expected, atol=1e-4)


def test_gamma_nought_flat_matches_linear_formula():
    """Verify γ⁰ = σ⁰/cos(θ) in linear space matches the dB implementation."""
    sigma_linear = 0.05  # linear σ⁰
    sigma_db = 10.0 * math.log10(sigma_linear)
    inc = 37.0
    cos_t = math.cos(math.radians(inc))
    expected_gamma_linear = sigma_linear / cos_t
    expected_gamma_db = 10.0 * math.log10(expected_gamma_linear)

    sigma = np.full((6, 6), sigma_db, dtype=np.float32)
    dem = np.zeros((6, 6), dtype=np.float32)
    lv = look_vector_from_angles(inc, 108.0)
    gamma = gamma_nought(sigma, dem, lv, 30.0)
    assert np.allclose(gamma, expected_gamma_db, atol=1e-3)


def test_gamma_nought_facing_slope_smaller_correction():
    """A slope facing the radar has larger cos(θ_local) → smaller RTC offset.

    RTC subtracts ``10·log10(cos(θ_local))`` from σ⁰. Facing the radar
    (cos → 1) drives that term toward 0, so γ⁰ stays close to σ⁰. Facing
    away (cos → 0) makes the term more negative, brightening γ⁰.
    """
    lv = look_vector_from_angles(37.0, 90.0)  # satellite to the east
    # Downhill east → faces the radar.
    dem_facing = np.tile(np.arange(12, dtype=np.float32)[::-1] * 20.0, (12, 1))
    dem_flat = np.zeros((12, 12), dtype=np.float32)
    sigma = np.full((12, 12), -15.0, dtype=np.float32)

    gamma_facing = gamma_nought(sigma, dem_facing, lv, 30.0)
    gamma_flat = gamma_nought(sigma, dem_flat, lv, 30.0)
    # offset = σ⁰ − γ⁰ = 10·log10(cos). Facing → cos larger → offset closer to 0.
    offset_facing = float(sigma[6, 6] - gamma_facing[6, 6])
    offset_flat = float(sigma[6, 6] - gamma_flat[6, 6])
    assert abs(offset_facing) < abs(offset_flat)
    # Facing γ⁰ stays closer to the input σ⁰ than the flat case.
    assert abs(gamma_facing[6, 6] - sigma[6, 6]) < abs(gamma_flat[6, 6] - sigma[6, 6])


def test_gamma_nought_shadow_preserves_sigma0():
    """In radar shadow (cos ≤ 0) the input σ⁰ is preserved, not amplified."""
    # Steep slope facing away from a grazing radar → cos ≤ 0 somewhere.
    lv = look_vector_from_angles(60.0, 90.0)
    # Slope rising steeply to the west (away from eastward radar).
    dem = np.tile(np.arange(20, dtype=np.float32)[::-1] * 200.0, (20, 1))
    sigma = np.full((20, 20), -15.0, dtype=np.float32)
    gamma = gamma_nought(sigma, dem, lv, 5.0)

    cos_t = local_incidence_cos(dem, lv, 5.0)
    shadow = cos_t <= 0.0
    if shadow.any():
        # Shadow pixels must equal the input σ⁰ exactly.
        assert np.allclose(gamma[shadow], sigma[shadow])
    # And no pixel should be NaN / inf.
    assert np.all(np.isfinite(gamma))


def test_gamma_nought_resamples_mismatched_dem():
    """A DEM with a different shape is bilinearly resampled to the σ⁰ grid."""
    sigma = np.full((10, 10), -15.0, dtype=np.float32)
    dem = np.zeros((20, 20), dtype=np.float32)
    lv = look_vector_from_angles(37.0, 108.0)
    gamma = gamma_nought(sigma, dem, lv, 30.0)
    assert gamma.shape == sigma.shape
    expected = -15.0 - 10.0 * math.log10(math.cos(math.radians(37.0)))
    assert np.allclose(gamma, expected, atol=1e-2)


def test_gamma_nought_is_deterministic():
    """Same inputs → identical outputs (Hard Rule 6)."""
    dem = np.tile(np.arange(12, dtype=np.float32) * 20.0, (12, 1))
    sigma = np.full((12, 12), -15.0, dtype=np.float32)
    lv = look_vector_from_angles(37.0, 90.0)
    g1 = gamma_nought(sigma, dem, lv, 30.0)
    g2 = gamma_nought(sigma, dem, lv, 30.0)
    assert np.array_equal(g1, g2)


def test_gamma_nought_rejects_non_2d_sigma():
    dem = np.zeros((4, 4), dtype=np.float32)
    lv = look_vector_from_angles(37.0, 108.0)
    with pytest.raises(ValueError):
        gamma_nought(np.zeros((2, 4, 4), dtype=np.float32), dem, lv, 30.0)


def test_gamma_nought_output_dtype_is_float32():
    sigma = np.full((6, 6), -15.0, dtype=np.float32)
    dem = np.zeros((6, 6), dtype=np.float32)
    lv = look_vector_from_angles(37.0, 108.0)
    gamma = gamma_nought(sigma, dem, lv, 30.0)
    assert gamma.dtype == np.float32


# --------------------------------------------------------------------------- #
# gamma_nought_stack
# --------------------------------------------------------------------------- #
def test_gamma_nought_stack_vv_vh():
    """RTC on a (2, H, W) VV+VH stack matches per-channel application."""
    dem = np.tile(np.arange(10, dtype=np.float32) * 15.0, (10, 1))
    vv = np.full((10, 10), -12.0, dtype=np.float32)
    vh = np.full((10, 10), -20.0, dtype=np.float32)
    stack = np.stack([vv, vh], axis=0)
    lv = look_vector_from_angles(37.0, 90.0)

    out = gamma_nought_stack(stack, dem, lv, 30.0)
    assert out.shape == stack.shape
    assert out.dtype == np.float32

    expected_vv = gamma_nought(vv, dem, lv, 30.0)
    expected_vh = gamma_nought(vh, dem, lv, 30.0)
    assert np.allclose(out[0], expected_vv)
    assert np.allclose(out[1], expected_vh)


def test_gamma_nought_stack_rejects_non_3d():
    dem = np.zeros((4, 4), dtype=np.float32)
    lv = look_vector_from_angles(37.0, 108.0)
    with pytest.raises(ValueError):
        gamma_nought_stack(np.zeros((4, 4), dtype=np.float32), dem, lv, 30.0)


# --------------------------------------------------------------------------- #
# COS_THETA_MIN sanity
# --------------------------------------------------------------------------- #
def test_cos_theta_min_is_small_positive():
    """The shadow floor must be a small positive value to keep γ⁰ finite."""
    assert 0.0 < COS_THETA_MIN < 0.01
