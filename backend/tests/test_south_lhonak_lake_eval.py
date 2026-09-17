"""Tests for the South Lhonak Pleiades lake diagnostic helpers."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from siren.ml.south_lhonak_lake_eval import (
    _largest_flat_region,
    evaluate,
    measure_lake,
)

# UTM 45N corner near South Lhonak — the box must contain the synthetic
# rasters entirely so the window read is not clipped.
_ORIGIN = (620500.0, 3087000.0)  # west, north
_BOX = (620500.0, 3084200.0, 622200.0, 3087000.0)


def _write_dem(path, arr, res=20.0):
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": arr.shape[1],
        "height": arr.shape[0],
        "count": 1,
        "crs": "EPSG:32645",
        "transform": from_origin(_ORIGIN[0], _ORIGIN[1], res, res),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def _scene(lake_elev, drained_rows=0, shape=(140, 85), res=20.0):
    """Synthetic scene: rough terrain + a flat 40x40 px lake at lake_elev.

    `drained_rows` lowers the southern lake rows by 20 m to mimic a
    partially-emptied lakebed.
    """
    rng = np.random.default_rng(7)
    dem = 5400.0 + rng.normal(0, 15, shape).astype(np.float32)
    # carve a smooth bowl so the lake band is the local minimum
    dem[40:100, 20:60] = 5250.0 + rng.normal(0, 2, (60, 40)).astype(np.float32)
    lake = np.s_[45:85, 25:65]
    dem[lake] = lake_elev
    if drained_rows:
        r0, r1 = 45, 45 + drained_rows
        dem[r0:r1, 25:65] = lake_elev - 20.0
    return dem, res


def test_largest_flat_region_finds_lake():
    dem, _ = _scene(lake_elev=5197.0)
    mask = _largest_flat_region(dem, 5160.0, 5225.0)
    assert mask.any()
    ys, xs = np.where(mask)
    assert ys.min() >= 45 and ys.max() < 85
    assert xs.min() >= 25 and xs.max() < 65
    # most of the 40x40 lake survives opening (edges excluded by the
    # 9x9 std window seeing rough neighbours)
    assert mask.sum() > 900


def test_largest_flat_region_empty_when_no_flat_band():
    rng = np.random.default_rng(1)
    dem = 5500.0 + rng.normal(0, 30, (80, 80)).astype(np.float32)
    mask = _largest_flat_region(dem, 5160.0, 5225.0)
    assert not mask.any()


def test_measure_lake_synthetic(tmp_path):
    pre, res = _scene(lake_elev=5197.0)
    post, _ = _scene(lake_elev=5183.0, drained_rows=15)
    pre_path, post_path = tmp_path / "pre.tif", tmp_path / "post.tif"
    _write_dem(pre_path, pre, res)
    _write_dem(post_path, post, res)

    m = measure_lake(pre_path, post_path, box=_BOX,
                     elev_band_pre=(5160, 5225), elev_band_post=(5100, 5225))

    cell = res * res
    # flat-surface detection under-covers lake edges (9x9 std window sees
    # rough neighbours), so area is a lower bound on the true 40x40 px
    assert 0.5 * 40 * 40 * cell < m["lake_area_pre_m2"] <= 40 * 40 * cell
    assert m["surface_elev_pre_m"] == pytest.approx(5197.0, abs=1.0)
    assert m["surface_drawdown_m"] == pytest.approx(14.0, abs=1.0)
    # drained rows lowered by 20 m contribute positive (pre - post)
    assert m["visible_emptied_m3"] > 5 * 40 * 20 * cell


def test_evaluate_huggel_comparison():
    meas = {
        "lake_area_pre_m2": 0.9e6,
        "lake_area_post_m2": 0.8e6,
        "surface_drawdown_m": 14.0,
        "visible_emptied_m3": 5e6,
        "huggel_volume_m3": 0.104 * (0.9e6) ** 1.421,
    }
    out = evaluate(meas)
    assert out["lake_area_within_published_range"]
    assert "huggel_area_sweep_mcm" in out
    assert len(out["huggel_area_sweep_mcm"]["volumes_mcm"]) == 3
    # Huggel on 0.9 km2 ≈ 30 MCM < 45 MCM release midpoint → negative %
    assert out["huggel_vs_documented_release_pct"] < 0
