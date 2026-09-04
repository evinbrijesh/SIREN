"""Generate synthetic test fixtures for SIREN.

Run once:  python backend/tests/fixtures/make_fixtures.py

Produces:
  rasters/baseline.tif        — 100x100 "normal" glacial lake (100 water px)
  rasters/expanded_water.tif  — lake expanded +28% (128 water px)
  rasters/cloudy_optical.tif  — 2-band optical scene (GREEN, NIR) with a
                                cloudy region (cloud_fraction ~0.25)
  osm/fake_assets.geojson     — 2 villages, 1 bridge, 3 wells, 1 road line

Design notes:
- The +28% expansion is exact: baseline lake is 10x10 = 100 px; expanded adds
  10 + 10 + 8 = 28 px -> 128 px = +28%.
- cloudy_optical.tif is a REAL optical scene (2 bands), not a pre-made mask.
  The quality gate must compute cloud_fraction from scene statistics (bright
  pixels in both GREEN and NIR), so the D3 test is not circular.
- Fixtures are tiny (< 1 MB total) and committed.
"""

from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.transform import from_bounds

BASE_DIR = Path(__file__).resolve().parent
RASTERS_DIR = BASE_DIR / "rasters"
OSM_DIR = BASE_DIR / "osm"
RASTERS_DIR.mkdir(parents=True, exist_ok=True)
OSM_DIR.mkdir(parents=True, exist_ok=True)

# Grid: 100x100 spanning the basin bbox (matches data/assets/dudh_koshi_aoi.geojson)
TRANSFORM = from_bounds(86.65, 27.65, 87.00, 27.98, 100, 100)
PROFILE = {
    "driver": "GTiff",
    "height": 100,
    "width": 100,
    "count": 1,
    "dtype": "float32",
    "crs": "EPSG:4326",
    "transform": TRANSFORM,
}


def write_single_band(path: Path, data: np.ndarray) -> None:
    with rasterio.open(path, "w", **PROFILE) as dst:
        dst.write(data.astype(np.float32), 1)


# ---------------------------------------------------------------------------
# 1. Baseline: 100x100 with a central 10x10 lake (100 px = normal water)
# ---------------------------------------------------------------------------
baseline = np.full((100, 100), 2000.0, dtype=np.float32)  # high backscatter = land
baseline[45:55, 45:55] = 50.0  # low backscatter = water
write_single_band(RASTERS_DIR / "baseline.tif", baseline)

# ---------------------------------------------------------------------------
# 2. Expanded water: +28% (100 -> 128 px)
# ---------------------------------------------------------------------------
expanded = baseline.copy()
expanded[44, 45:55] = 50.0  # +10
expanded[55, 45:55] = 50.0  # +10
expanded[45:53, 44] = 50.0  # +8  -> total +28 px = +28%
write_single_band(RASTERS_DIR / "expanded_water.tif", expanded)

# ---------------------------------------------------------------------------
# 3. Cloudy optical: 2-band scene (GREEN, NIR) with a cloudy region.
#    Cloud fraction ~0.25 (2500/10000 px). Clouds are bright in BOTH bands.
#    The quality gate computes cloud_fraction from scene statistics.
# ---------------------------------------------------------------------------
cloudy_profile = dict(PROFILE, count=2)
green = np.full((100, 100), 0.08, dtype=np.float32)  # dark land in GREEN
nir = np.full((100, 100), 0.30, dtype=np.float32)    # vegetation bright in NIR
# Cloudy region: top-left 50x50, bright in both bands
green[:50, :50] = 0.85
nir[:50, :50] = 0.80
with rasterio.open(RASTERS_DIR / "cloudy_optical.tif", "w", **cloudy_profile) as dst:
    dst.write(green, 1)
    dst.write(nir, 2)

# ---------------------------------------------------------------------------
# 4. Fake OSM assets: exactly 2 villages, 1 bridge, 3 wells, 1 road line
# ---------------------------------------------------------------------------
fake_osm = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"place": "village", "name": "Dingboche", "population": 450},
         "geometry": {"type": "Point", "coordinates": [86.83, 27.89]}},
        {"type": "Feature", "properties": {"place": "village", "name": "Chhukung", "population": 210},
         "geometry": {"type": "Point", "coordinates": [86.87, 27.90]}},
        {"type": "Feature", "properties": {"highway": "bridge", "name": "Imja Suspension Bridge"},
         "geometry": {"type": "Point", "coordinates": [86.82, 27.88]}},
        {"type": "Feature", "properties": {"amenity": "drinking_water", "name": "Upper Well"},
         "geometry": {"type": "Point", "coordinates": [86.84, 27.89]}},
        {"type": "Feature", "properties": {"amenity": "drinking_water", "name": "Lower Well"},
         "geometry": {"type": "Point", "coordinates": [86.82, 27.87]}},
        {"type": "Feature", "properties": {"man_made": "water_well", "name": "Community Borehole"},
         "geometry": {"type": "Point", "coordinates": [86.81, 27.86]}},
        {"type": "Feature", "properties": {"highway": "path", "name": "Main Trekking Trail"},
         "geometry": {"type": "LineString", "coordinates": [[86.81, 27.85], [86.84, 27.89], [86.87, 27.91]]}},
    ],
}
with open(OSM_DIR / "fake_assets.geojson", "w") as f:
    json.dump(fake_osm, f, indent=2)

print("✓ Created synthetic rasters (baseline, expanded +28%, cloudy 2-band) and fake OSM fixtures.")
print(f"  baseline water px: {int((baseline == 50).sum())}")
print(f"  expanded water px: {int((expanded == 50).sum())}")
print(f"  cloudy fraction:   {float((green[:50,:50] > 0.5).sum()) / 10000:.2f}")