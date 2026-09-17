"""Tests for the optical NDWI path (detect/ndwi.py).

The critical contract: ``read_s2_band`` must return a profile describing
the AOI *window*, not the full tile. Returning the tile profile stamps a
windowed array at the tile origin — the original baseline_water_mask.tif
was misplaced ~65 km west (Rolwaling valley) by exactly this bug.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from siren.detect.ndwi import AOI_BOUNDS_UTM, baseline_water_mask, ndwi, read_s2_band, water_mask

# Fake tile: 10500x4200 px at 10 m, UTM 45N, origin west of the AOI so the
# window lands fully inside the tile.
TILE_TRANSFORM = from_origin(399960.0, 3100020.0, 10.0, 10.0)
TILE_CRS = "EPSG:32645"


def _fake_safe_zip(tmp_path: Path) -> tuple[str, str]:
    """Build a minimal S2-style zip with a GeoTIFF masquerading as .jp2.

    rasterio reads by content, not extension, so a TIFF inside
    /vsizip/ works even with a .jp2 name.
    """
    safe_name = "S2C_MSIL2A_20251122T045131_N0511_R076_T45RVL_20251122T083010"
    granule = (
        f"{safe_name}.SAFE/GRANULE/L2A_T45RVL_A006338_20251122T045408"
    )
    zip_path = tmp_path / f"{safe_name}.zip"

    band = np.zeros((4200, 10500), dtype=np.uint16)
    band[50:3700, 6500:10499] = 5000  # something in the AOI window

    for bname in ("B03", "B08"):
        tif_path = tmp_path / f"{bname}.tif"
        with rasterio.open(
            tif_path, "w", driver="GTiff", width=10500, height=4200,
            count=1, dtype="uint16", crs=TILE_CRS, transform=TILE_TRANSFORM,
        ) as dst:
            dst.write(band, 1)

        inner = (
            f"{granule}/IMG_DATA/R10m/"
            f"T45RVL_20251122T045131_{bname}_10m.jp2"
        )
        with zipfile.ZipFile(zip_path, "a") as zf:
            zf.write(tif_path, inner)

    return str(zip_path), granule


def test_ndwi_math():
    green = np.array([[6000, 4000]], dtype=np.float32)
    nir = np.array([[2000, 4000]], dtype=np.float32)
    out = ndwi(green, nir)
    assert np.isclose(out[0, 0], 0.5, atol=1e-6)
    assert np.isclose(out[0, 1], 0.0, atol=1e-6)


def test_water_mask_threshold():
    assert water_mask(np.array([[0.3]])).tolist() == [[True]]
    assert water_mask(np.array([[0.1]])).tolist() == [[False]]


def test_read_s2_band_returns_window_profile(tmp_path):
    """The returned profile must carry the WINDOW transform and dims —
    not the full-tile profile (regression for the Rolwaling mis-stamp)."""
    s2_zip, granule = _fake_safe_zip(tmp_path)
    array, profile = read_s2_band(s2_zip, granule, "B03")

    expected_win = from_bounds(*AOI_BOUNDS_UTM, transform=TILE_TRANSFORM)
    assert profile["width"] == int(round(expected_win.width))
    assert profile["height"] == int(round(expected_win.height))
    assert array.shape == (profile["height"], profile["width"])

    # Window transform origin must be inside the tile, near the AOI west
    # edge — NOT the tile origin (399960).
    t = profile["transform"]
    assert t.c > 400000.0
    assert abs(t.c - TILE_TRANSFORM.c) > 1000.0
    assert t.f < TILE_TRANSFORM.f


def test_baseline_water_mask_georeferenced(tmp_path):
    """The mask+profile must georeference to the AOI, not the tile corner."""
    s2_zip, granule = _fake_safe_zip(tmp_path)
    mask, profile = baseline_water_mask(s2_zip, granule)
    t = profile["transform"]
    left, top = t.c, t.f
    right = left + t.a * profile["width"]
    bottom = top + t.e * profile["height"]
    # Bounds must overlap the AOI bounds, not sit 65 km west
    assert left >= AOI_BOUNDS_UTM[0] - 15.0
    assert right <= AOI_BOUNDS_UTM[2] + 15.0
    assert bottom >= AOI_BOUNDS_UTM[1] - 15.0
    assert top <= AOI_BOUNDS_UTM[3] + 15.0
