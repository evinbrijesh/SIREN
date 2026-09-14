"""Tests for risk/breach_volume.py — contract-preserving V_breach estimator.

Tests cover:
    - Observed drainage (lake contraction): V_breach = V(z_pre) − V(z_post)
    - Potential full drain (lake expansion/stable): V_breach = V(z_pre)
    - Analytically verifiable basin geometries (flat bed, sloping bed)
    - FNO channel-1 normalization round-trip (log1p(V_breach)/20.0)
    - Error handling: empty masks, shape mismatch, NaN DEM, non-positive
      volume, lake filling entire grid — all raise ValueError (no silent
      fallbacks)
"""

from __future__ import annotations

import numpy as np
import pytest

from siren.risk.breach_volume import (
    BreachVolumeResult,
    estimate_breach_volume,
    _lake_rim,
    _shoreline_elevation,
    _hypsometric_volume,
    _huggel_volume,
    HUGGEL_ALPHA,
    HUGGEL_GAMMA,
    MIN_BATHYMETRIC_MEAN_DEPTH_M,
)


# --------------------------------------------------------------------------- #
# Helper: build a synthetic DEM with a lake basin
# --------------------------------------------------------------------------- #

def _flat_basin_dem(
    grid_size: int = 12,
    lake_rows: tuple[int, int] = (1, 11),
    lake_cols: tuple[int, int] = (1, 11),
    bed_elev: float = 0.0,
    terrain_elev: float = 5.0,
) -> np.ndarray:
    """DEM with a flat-bed lake basin surrounded by flat terrain."""
    dem = np.full((grid_size, grid_size), terrain_elev, dtype=np.float64)
    dem[lake_rows[0]:lake_rows[1], lake_cols[0]:lake_cols[1]] = bed_elev
    return dem


def _sloping_basin_dem(
    grid_size: int = 12,
    lake_rows: tuple[int, int] = (1, 11),
    lake_cols: tuple[int, int] = (1, 11),
    bed_start: float = 0.0,
    bed_end: float = 9.0,
    terrain_elev: float = 12.0,
) -> np.ndarray:
    """DEM with a lake bed sloping linearly along rows (deep → shallow)."""
    dem = np.full((grid_size, grid_size), terrain_elev, dtype=np.float64)
    r0, r1 = lake_rows
    c0, c1 = lake_cols
    for i in range(r0, r1):
        frac = (i - r0) / max(r1 - r0 - 1, 1)
        dem[i, c0:c1] = bed_start + frac * (bed_end - bed_start)
    return dem


# --------------------------------------------------------------------------- #
# _lake_rim
# --------------------------------------------------------------------------- #

def test_lake_rim_finds_adjacent_land_pixels():
    """The rim is the set of non-water pixels touching the lake."""
    water = np.zeros((5, 5), dtype=bool)
    water[1:4, 1:4] = True  # 3x3 lake
    rim = _lake_rim(np.zeros((5, 5)), water)
    # Rim should be the ring around the 3x3 lake (not the lake itself)
    assert not rim[2, 2]  # center of lake — not rim
    assert rim[0, 1] and rim[1, 0]  # land pixels adjacent to lake
    assert rim[4, 2] and rim[2, 4]
    assert not rim[0, 0]  # corner — diagonal, not 4-connected


def test_lake_rim_empty_for_full_grid():
    """If the lake fills the entire grid, there are no rim pixels."""
    water = np.ones((5, 5), dtype=bool)
    rim = _lake_rim(np.zeros((5, 5)), water)
    assert not rim.any()


# --------------------------------------------------------------------------- #
# _shoreline_elevation
# --------------------------------------------------------------------------- #

def test_shoreline_elevation_is_min_rim_elev():
    """Shoreline elevation = minimum DEM on the lake rim (outlet)."""
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    water = np.zeros((12, 12), dtype=bool)
    water[1:11, 1:11] = True
    z = _shoreline_elevation(dem, water)
    assert z == 5.0  # rim is terrain at 5m


def test_shoreline_elevation_uses_outlet_not_mean():
    """Shoreline uses the min rim elevation (outlet), not the mean."""
    dem = np.full((10, 10), 20.0, dtype=np.float64)
    dem[0, 4:6] = 8.0  # low outlet on the north rim
    water = np.zeros((10, 10), dtype=bool)
    water[1:9, 1:9] = True
    z = _shoreline_elevation(dem, water)
    assert z == 8.0  # the outlet, not the mean (20)


def test_shoreline_elevation_raises_on_empty_mask():
    """Empty water mask raises ValueError."""
    dem = np.zeros((5, 5), dtype=np.float64)
    water = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="empty"):
        _shoreline_elevation(dem, water)


def test_shoreline_elevation_raises_on_full_grid():
    """Lake filling the entire grid raises ValueError (no rim)."""
    dem = np.zeros((5, 5), dtype=np.float64)
    water = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="no rim"):
        _shoreline_elevation(dem, water)


# --------------------------------------------------------------------------- #
# _hypsometric_volume
# --------------------------------------------------------------------------- #

def test_hypsometric_volume_flat_bed():
    """Flat bed: V = n_pixels * depth * pixel_area."""
    bed = np.array([0.0, 0.0, 0.0, 0.0])
    v = _hypsometric_volume(bed, z_surface=5.0, pixel_area=100.0)
    assert v == pytest.approx(4 * 5.0 * 100.0)


def test_hypsometric_volume_sloping_bed():
    """Sloping bed: V = Σ max(0, z - bed_i) * pixel_area."""
    bed = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    v = _hypsometric_volume(bed, z_surface=5.0, pixel_area=1.0)
    # depths: 5, 4, 3, 2, 0  (last pixel is above water)
    assert v == pytest.approx(5 + 4 + 3 + 2 + 0)


def test_hypsometric_volume_zero_when_bed_above_surface():
    """If all bed elevations are above the water surface, V = 0."""
    bed = np.array([10.0, 20.0, 30.0])
    v = _hypsometric_volume(bed, z_surface=5.0, pixel_area=1.0)
    assert v == 0.0


# --------------------------------------------------------------------------- #
# estimate_breach_volume — observed drainage (contraction)
# --------------------------------------------------------------------------- #

def test_observed_drainage_flat_bed():
    """Full drainage of a flat-bed lake: V_breach = V(z_pre).

    Basin: 12x12, 1 m² pixels.
    Pre lake: 10x10 (rows 1-10, cols 1-10), bed = 0m flat.
    Post lake: empty (fully drained).
    Terrain: 5m flat.
    z_pre = 5m, V_pre = 100 * 5 * 1 = 500 m³.
    V_breach = 500 m³ (full drain).
    """
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = np.zeros((12, 12), dtype=bool)  # fully drained

    result = estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)

    assert result.mode == "observed_drainage"
    assert result.z_pre_m == 5.0
    assert result.z_post_m == 0.0  # empty lake → z_post = bed minimum
    assert result.lake_volume_pre_m3 == pytest.approx(500.0)
    assert result.lake_volume_post_m3 == pytest.approx(0.0)
    assert result.v_breach_m3 == pytest.approx(500.0)
    assert result.delta_area_m2 == pytest.approx(100.0)


def test_observed_drainage_sloping_bed():
    """Partial drainage of a sloping-bed lake (analytically verified).

    Basin: 12x12, 1 m² pixels.
    Pre lake: rows 1-10, cols 1-10 (10x10 = 100 px).
    Bed slopes: bed[i] = i-1 (0m at row 1, 9m at row 10).
    Terrain: 12m flat.

    z_pre = 12 (rim = terrain).
    V_pre = 10 * (12+11+10+9+8+7+6+5+4+3) = 750 m³.

    Post lake: rows 1-5, cols 1-10 (5x10 = 50 px, deep end remains).
    Post rim: exposed bed at row 6 (5m) + terrain (12m).
    z_post = 5.
    V_post = 10 * (5+4+3+2+1) = 150 m³.

    V_breach = 750 - 150 = 600 m³.
    """
    dem = _sloping_basin_dem(
        grid_size=12, lake_rows=(1, 11), lake_cols=(1, 11),
        bed_start=0.0, bed_end=9.0, terrain_elev=12.0,
    )
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = np.zeros((12, 12), dtype=bool)
    post[1:6, 1:11] = True  # rows 1-5 remain (deep end)

    result = estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)

    assert result.mode == "observed_drainage"
    assert result.z_pre_m == 12.0
    assert result.z_post_m == 5.0
    assert result.lake_volume_pre_m3 == pytest.approx(750.0, abs=0.5)
    assert result.lake_volume_post_m3 == pytest.approx(150.0, abs=0.5)
    assert result.v_breach_m3 == pytest.approx(600.0, abs=1.0)
    assert result.delta_area_m2 > 0  # contraction


def test_observed_drainage_partial_flat_bed():
    """Partial drainage of a flat-bed lake with a lowered outlet.

    Basin: 12x12, 1 m² pixels.
    Pre lake: 10x10, bed = 0m, terrain = 10m.
    Post lake: 8x8 (rows 2-9, cols 2-9), same bed.
    Post rim: exposed bed at 0m → z_post = 0.
    V_post = 0 (water surface at bed level).
    V_breach = V_pre = 100 * 10 * 1 = 1000 m³.

    This is effectively a full drain (z_post = bed level).
    """
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=10.0)
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = np.zeros((12, 12), dtype=bool)
    post[2:10, 2:10] = True  # 8x8

    result = estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)

    assert result.mode == "observed_drainage"
    assert result.z_pre_m == 10.0
    assert result.v_breach_m3 == pytest.approx(1000.0)


# --------------------------------------------------------------------------- #
# estimate_breach_volume — potential full drain (expansion/stable)
# --------------------------------------------------------------------------- #

def test_potential_full_drain_lake_expansion():
    """Expanding lake: V_breach = V(z_pre) (potential full-drain).

    Basin: 12x12, 1 m² pixels.
    Pre lake: 8x8 (rows 2-9, cols 2-9), bed = 0m.
    Post lake: 10x10 (rows 1-10, cols 1-10) — expanded.
    Terrain: 5m.

    z_pre = 5 (rim = terrain).
    V_pre = 64 * (5-0) * 1 = 320 m³.
    V_breach = 320 m³ (potential full drain).
    """
    dem = np.full((12, 12), 5.0, dtype=np.float64)
    dem[2:10, 2:10] = 0.0  # bed only under the pre-event lake

    pre = np.zeros((12, 12), dtype=bool)
    pre[2:10, 2:10] = True  # 8x8
    post = np.zeros((12, 12), dtype=bool)
    post[1:11, 1:11] = True  # 10x10 (expanded)

    result = estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)

    assert result.mode == "potential_full_drain"
    assert result.z_pre_m == 5.0
    assert result.z_post_m is None
    assert result.lake_volume_pre_m3 == pytest.approx(320.0)
    assert result.v_breach_m3 == pytest.approx(320.0)
    assert result.delta_area_m2 < 0  # expansion


def test_potential_full_drain_stable_lake():
    """Stable lake (no change): V_breach = V(z_pre) (potential full-drain)."""
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = pre.copy()  # no change

    result = estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)

    assert result.mode == "potential_full_drain"
    assert result.v_breach_m3 == pytest.approx(500.0)
    assert result.delta_area_m2 == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# FNO channel-1 normalization round-trip
# --------------------------------------------------------------------------- #

def test_v_breach_produces_valid_fno_channel_value():
    """The estimated V_breach produces a valid FNO channel-1 value.

    FNO channel 1 = log1p(V_breach) / 20.0 (ADR-012 frozen contract).
    The value should be finite, positive, and in a reasonable range
    (0, 1) for typical Himalayan lake volumes (1e4 – 1e8 m³).
    """
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = np.zeros((12, 12), dtype=bool)

    result = estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)
    v_norm = float(np.log1p(result.v_breach_m3) / 20.0)

    assert np.isfinite(v_norm)
    assert v_norm > 0.0
    assert v_norm < 1.0  # log1p(500)/20 ≈ 0.31


def test_v_breach_scales_with_pixel_area():
    """V_breach scales linearly with pixel area (volume = depth * area)."""
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = np.zeros((12, 12), dtype=bool)

    r1 = estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)
    r2 = estimate_breach_volume(pre, post, dem, pixel_area_m2=100.0)

    assert r2.v_breach_m3 == pytest.approx(r1.v_breach_m3 * 100.0)


def test_v_breach_scales_with_basin_depth():
    """Deeper basin → larger V_breach (volume proportional to depth)."""
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = np.zeros((12, 12), dtype=bool)

    shallow = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    deep = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=20.0)

    r_shallow = estimate_breach_volume(pre, post, shallow, pixel_area_m2=1.0)
    r_deep = estimate_breach_volume(pre, post, deep, pixel_area_m2=1.0)

    assert r_deep.v_breach_m3 > r_shallow.v_breach_m3
    assert r_deep.v_breach_m3 == pytest.approx(r_shallow.v_breach_m3 * 4.0)


# --------------------------------------------------------------------------- #
# Error handling — no silent fallbacks
# --------------------------------------------------------------------------- #

def test_empty_pre_mask_raises():
    """Empty pre-event mask raises ValueError."""
    dem = np.zeros((5, 5), dtype=np.float64)
    pre = np.zeros((5, 5), dtype=bool)
    post = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="pre-event water mask is empty"):
        estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)


def test_shape_mismatch_raises():
    """Shape mismatch between masks and DEM raises ValueError."""
    dem = np.zeros((10, 10), dtype=np.float64)
    pre = np.zeros((5, 5), dtype=bool)
    post = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="shape mismatch"):
        estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)


def test_nan_in_dem_raises():
    """NaN in DEM within the lake footprint raises ValueError."""
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    dem[5, 5] = np.nan  # NaN inside the lake
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = np.zeros((12, 12), dtype=bool)
    with pytest.raises(ValueError, match="NaN"):
        estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)


def test_non_positive_volume_raises():
    """Flat DEM (bed ≥ shoreline) with method='hypsometric' raises ValueError.

    If the bed elevation equals the rim elevation everywhere, the lake has
    zero depth and V_breach = 0. With the default strict hypsometric method,
    this should raise, not silently return 0.
    """
    dem = np.full((10, 10), 5.0, dtype=np.float64)  # flat everywhere
    pre = np.zeros((10, 10), dtype=bool)
    pre[3:7, 3:7] = True
    post = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="does not resolve submerged lake bathymetry"):
        estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)


def test_zero_pixel_area_raises():
    """Non-positive pixel area raises ValueError."""
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = np.zeros((12, 12), dtype=bool)
    with pytest.raises(ValueError, match="pixel_area_m2 must be positive"):
        estimate_breach_volume(pre, post, dem, pixel_area_m2=0.0)


def test_lake_fills_entire_grid_raises():
    """Lake filling the entire grid (no rim) raises ValueError."""
    dem = _flat_basin_dem(grid_size=5, bed_elev=0.0, terrain_elev=5.0)
    pre = np.ones((5, 5), dtype=bool)  # fills entire grid
    post = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="no rim"):
        estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #

def test_result_to_dict():
    """BreachVolumeResult.to_dict produces a serialisable dict."""
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = np.zeros((12, 12), dtype=bool)

    result = estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)
    d = result.to_dict()

    assert d["v_breach_m3"] == pytest.approx(500.0, abs=0.1)
    assert d["z_pre_m"] == 5.0
    assert d["z_post_m"] == 0.0
    assert d["mode"] == "observed_drainage"
    assert d["provenance"] == "breach_volume_hypsometric_v1"
    assert isinstance(d["lake_area_pre_m2"], float)


def test_result_provenance_tag():
    """Result carries the hypsometric provenance tag."""
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = pre.copy()

    result = estimate_breach_volume(pre, post, dem, pixel_area_m2=1.0)
    assert result.provenance == "breach_volume_hypsometric_v1"


# --------------------------------------------------------------------------- #
# Huggel et al. (2002) empirical area-volume scaling
# --------------------------------------------------------------------------- #

def test_huggel_volume_formula():
    """Huggel volume formula: V = 0.104 * A^1.421."""
    # Known: A = 1e6 m² (1 km²) → V = 0.104 * (1e6)^1.421
    expected = 0.104 * (1e6 ** 1.421)
    assert _huggel_volume(1e6) == pytest.approx(expected, rel=1e-6)


def test_huggel_volume_zero_area():
    """Zero or negative area returns 0.0."""
    assert _huggel_volume(0.0) == 0.0
    assert _huggel_volume(-1.0) == 0.0


def test_huggel_volume_scales_with_area():
    """Larger area → larger volume (non-linear power law)."""
    v1 = _huggel_volume(1e5)
    v2 = _huggel_volume(1e6)
    v3 = _huggel_volume(1e7)
    assert v1 < v2 < v3
    # Power law: V(10A) / V(A) = 10^gamma = 10^1.421 ≈ 26.3
    assert v2 / v1 == pytest.approx(10 ** HUGGEL_GAMMA, rel=1e-4)


def test_huggel_fallback_on_flat_surface_dem():
    """Auto mode falls back to Huggel 2002 when DEM bathymetry is unresolved.

    Simulates SRTM over a lake: flat water surface (5000 m) with no submerged
    bathymetry. The hypsometric integral is 0, so auto mode falls back to
    Huggel empirical scaling and records the method in provenance.
    """
    H, W = 50, 50
    dem = np.full((H, W), 5000.0, dtype=np.float32)  # flat water surface (DSM)
    dem[0, :] = 5010.0  # rim elevated

    pre_mask = np.zeros((H, W), dtype=np.float32)
    pre_mask[10:40, 10:40] = 1.0  # 900 px

    post_mask = np.zeros((H, W), dtype=np.float32)
    post_mask[15:35, 15:35] = 1.0  # 400 px

    res = estimate_breach_volume(
        pre_mask, post_mask, dem, pixel_area_m2=100.0, method="auto"
    )

    assert res.method == "empirical_huggel"
    assert res.bathymetry_resolved is False
    assert res.v_breach_m3 > 0.0
    assert res.provenance == "breach_volume_huggel_2002"
    # Verify the Huggel formula was applied
    area_pre = 900 * 100.0  # 90000 m²
    area_post = 400 * 100.0  # 40000 m²
    expected_v = _huggel_volume(area_pre) - _huggel_volume(area_post)
    assert res.v_breach_m3 == pytest.approx(expected_v, rel=1e-3)


def test_hypsometric_strict_failure():
    """method='hypsometric' fails fast without silent fallback if bathymetry missing."""
    H, W = 30, 30
    dem = np.full((H, W), 5000.0, dtype=np.float32)
    pre_mask = np.zeros((H, W), dtype=np.float32)
    pre_mask[5:25, 5:25] = 1.0

    with pytest.raises(ValueError, match="does not resolve submerged lake bathymetry"):
        estimate_breach_volume(
            pre_mask, np.zeros_like(pre_mask), dem,
            pixel_area_m2=100.0, method="hypsometric"
        )


def test_empirical_huggel_mode_explicit():
    """method='empirical_huggel' uses Huggel scaling directly."""
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True  # 100 px
    post = np.zeros((12, 12), dtype=bool)
    post[2:10, 2:10] = True  # 64 px

    res = estimate_breach_volume(
        pre, post, dem, pixel_area_m2=100.0, method="empirical_huggel"
    )

    assert res.method == "empirical_huggel"
    assert res.bathymetry_resolved is False
    assert res.provenance == "breach_volume_huggel_2002"
    # Verify Huggel formula
    area_pre = 100 * 100.0  # 10000 m²
    area_post = 64 * 100.0  # 6400 m²
    expected = _huggel_volume(area_pre) - _huggel_volume(area_post)
    assert res.v_breach_m3 == pytest.approx(expected, rel=1e-3)


def test_auto_uses_hypsometric_when_bathymetry_available():
    """Auto mode uses hypsometric when the DEM resolves bathymetry."""
    dem = _flat_basin_dem(grid_size=12, bed_elev=0.0, terrain_elev=5.0)
    pre = np.zeros((12, 12), dtype=bool)
    pre[1:11, 1:11] = True
    post = np.zeros((12, 12), dtype=bool)

    res = estimate_breach_volume(
        pre, post, dem, pixel_area_m2=1.0, method="auto"
    )

    assert res.method == "hypsometric"
    assert res.bathymetry_resolved is True
    assert res.provenance == "breach_volume_hypsometric_v1"
    assert res.v_breach_m3 == pytest.approx(500.0)


def test_huggel_result_to_dict_includes_method():
    """to_dict includes method and bathymetry_resolved fields."""
    H, W = 30, 30
    dem = np.full((H, W), 5000.0, dtype=np.float32)
    dem[0, :] = 5010.0
    pre = np.zeros((H, W), dtype=np.float32)
    pre[5:25, 5:25] = 1.0

    res = estimate_breach_volume(
        pre, np.zeros_like(pre), dem,
        pixel_area_m2=100.0, method="auto"
    )
    d = res.to_dict()

    assert d["method"] == "empirical_huggel"
    assert d["bathymetry_resolved"] is False
    assert d["provenance"] == "breach_volume_huggel_2002"
    assert "v_breach_m3" in d
    assert d["v_breach_m3"] > 0.0


def test_huggel_fno_channel_normalization():
    """Huggel-derived V_breach produces a valid FNO channel-1 value."""
    H, W = 30, 30
    dem = np.full((H, W), 5000.0, dtype=np.float32)
    dem[0, :] = 5010.0
    pre = np.zeros((H, W), dtype=np.float32)
    pre[5:25, 5:25] = 1.0

    res = estimate_breach_volume(
        pre, np.zeros_like(pre), dem,
        pixel_area_m2=900.0, method="auto"
    )
    v_norm = float(np.log1p(res.v_breach_m3) / 20.0)

    assert np.isfinite(v_norm)
    assert v_norm > 0.0
    assert v_norm < 1.0  # should be in a reasonable range for FNO input


# --------------------------------------------------------------------------- #
# Noisy DSM (SRTM-over-water) auto fallback — mean-depth criterion
# --------------------------------------------------------------------------- #

def test_auto_falls_back_on_noisy_dsm_small_positive_vpre():
    """Auto mode falls back to Huggel when a DSM produces a small but nonzero
    V_pre from elevation noise (the real SRTM/Imja case).

    SRTM is a surface model: over a lake it returns the flat water surface
    (~5003 m), not the bed. Tiny per-pixel noise yields a small positive
    hypsometric V_pre, but the mean depth is far below the bathymetric
    threshold. The old ``v_pre > 0`` check wrongly treated this as resolved
    and then raised on V_breach=0 (z_pre == z_post). The mean-depth criterion
    detects the DSM and falls back to Huggel with full provenance.
    """
    rng = np.random.default_rng(42)
    H, W = 50, 50
    # Flat water surface at 5003 m with sub-metre noise (SRTM DSM signature).
    dem = np.full((H, W), 5003.0, dtype=np.float64) + rng.normal(0, 0.05, (H, W))
    dem[0, :] = 5010.0  # elevated rim row so a shoreline exists

    pre = np.zeros((H, W), dtype=bool)
    pre[10:40, 10:40] = True   # 900 px
    post = np.zeros((H, W), dtype=bool)
    post[15:35, 15:35] = True  # 400 px (contraction → observed drainage)

    res = estimate_breach_volume(
        pre, post, dem, pixel_area_m2=900.0, method="auto",
    )

    # Must fall back to Huggel, not raise.
    assert res.method == "empirical_huggel"
    assert res.bathymetry_resolved is False
    assert res.provenance == "breach_volume_huggel_2002"
    assert res.mode == "observed_drainage"
    # Huggel formula on the area delta.
    area_pre = 900 * 900.0
    area_post = 400 * 900.0
    expected = _huggel_volume(area_pre) - _huggel_volume(area_post)
    assert res.v_breach_m3 == pytest.approx(expected, rel=1e-3)
    assert res.v_breach_m3 > 0.0


def test_hypsometric_strict_fails_on_noisy_dsm():
    """method='hypsometric' fails fast on the noisy DSM (no silent fallback)."""
    rng = np.random.default_rng(42)
    H, W = 50, 50
    dem = np.full((H, W), 5003.0, dtype=np.float64) + rng.normal(0, 0.05, (H, W))
    dem[0, :] = 5010.0
    pre = np.zeros((H, W), dtype=bool)
    pre[10:40, 10:40] = True

    with pytest.raises(ValueError, match="does not resolve submerged lake bathymetry"):
        estimate_breach_volume(
            pre, np.zeros_like(pre), dem,
            pixel_area_m2=900.0, method="hypsometric",
        )


def test_min_bathymetric_mean_depth_constant():
    """The threshold constant is exposed and is a conservative 1.0 m floor."""
    assert MIN_BATHYMETRIC_MEAN_DEPTH_M == 1.0
