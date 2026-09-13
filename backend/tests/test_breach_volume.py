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
    """Non-positive V_breach (flat DEM, bed ≥ shoreline) raises ValueError.

    If the bed elevation equals the rim elevation everywhere, the lake has
    zero depth and V_breach = 0. This should raise, not silently return 0.
    """
    dem = np.full((10, 10), 5.0, dtype=np.float64)  # flat everywhere
    pre = np.zeros((10, 10), dtype=bool)
    pre[3:7, 3:7] = True
    post = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="non-positive"):
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
