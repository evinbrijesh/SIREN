"""D8 hydrological corridor generation (OpenCode-owned, Phase 3).

Computes the downstream flow corridor from a change polygon using SRTM D8
flow accumulation. Deterministic, rule-based (ADR-002).

CRITICAL: pysheds 0.5 uses np.in1d which was removed in numpy 2.x. The
monkeypatch below MUST run before any pysheds import. Do not "fix" by
downgrading numpy — rasterio/geopandas need numpy 2.
"""

from __future__ import annotations

import numpy as np

# --- numpy 2.x / pysheds 0.5 compatibility patch (must precede pysheds import) ---
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from pysheds.grid import Grid  # noqa: E402  (after patch)

from shapely.geometry import LineString, mapping
from shapely.ops import transform
import pyproj

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)  # D8 direction mapping
# D8 neighbor offsets for each direction value in DIRMAP (row, col deltas)
# 64=W, 128=NW, 1=N, 2=NE, 4=E, 8=SE, 16=S, 32=SW
_NEIGHBORS = {
    64: (0, -1), 128: (-1, -1), 1: (-1, 0), 2: (-1, 1),
    4: (0, 1), 8: (1, 1), 16: (1, 0), 32: (1, -1),
}


def flow_accumulation(dem_path: str) -> tuple[Grid, np.ndarray, np.ndarray]:
    """Compute D8 flow direction + accumulation from a DEM raster.

    Returns (grid, fdir, acc). grid carries the affine transform for
    cell-index -> coordinate conversion.
    """
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    filled = grid.fill_depressions(dem=dem)
    fdir = grid.flowdir(dem=filled, dirmap=DIRMAP)
    acc = grid.accumulation(fdir, dirmap=DIRMAP)
    return grid, fdir, acc


def corridor_from_change(
    dem_path: str,
    change_polygon_geojson: dict,
    threshold: float = 1000.0,
    buffer_m: float = 500.0,
) -> dict:
    """Project a downstream corridor from a change polygon.

    Steps:
      1. Compute D8 flow direction + accumulation on the DEM.
      2. Find the accumulation cell nearest the change polygon centroid.
      3. Trace the downstream flow path from that cell.
      4. Buffer the path into a corridor polygon (in UTM 45N for correct meters).

    Returns a GeoJSON FeatureCollection with the corridor polygon and the
    traced flow path.
    """
    grid, fdir, acc = flow_accumulation(dem_path)

    coords = _polygon_centroid(change_polygon_geojson)
    if coords is None:
        raise ValueError("change_polygon_geojson must be a Polygon or Feature with Polygon geometry")

    # Snap the change centroid to the nearest grid cell, then trace downstream.
    # The change source (e.g. a glacial lake) is at the headwaters where
    # accumulation is low — do NOT require a high-accumulation start cell.
    row, col = _snap_to_cell(grid, coords)
    path = _trace_downstream(fdir, row, col)

    affine = grid.affine
    path_coords = [[affine[2] + c * affine[0], affine[5] + r * affine[4]] for r, c in path]

    line = LineString(path_coords)
    proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    line_utm = transform(proj.transform, line)
    corridor_utm = line_utm.buffer(buffer_m)
    corridor = transform(lambda x, y: proj.transform(x, y, direction="INVERSE"), corridor_utm)

    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"kind": "corridor", "buffer_m": buffer_m},
             "geometry": mapping(corridor)},
            {"type": "Feature", "properties": {"kind": "flow_path"},
             "geometry": mapping(line)},
        ],
    }


def _polygon_centroid(geojson: dict) -> list[float] | None:
    """Return [lon, lat] centroid of a GeoJSON Polygon/Feature."""
    geom = geojson.get("geometry", geojson) if geojson.get("type") == "Feature" else geojson
    if geom.get("type") != "Polygon":
        return None
    ring = geom["coordinates"][0]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def _snap_to_cell(grid: Grid, coords: list[float]) -> tuple[int, int]:
    """Snap (lon, lat) to the nearest grid cell (row, col)."""
    affine = grid.affine
    col = int(round((coords[0] - affine[2]) / affine[0]))
    row = int(round((coords[1] - affine[5]) / affine[4]))
    # Clamp to grid bounds
    row = max(0, min(row, grid.shape[0] - 1))
    col = max(0, min(col, grid.shape[1] - 1))
    return row, col


def _trace_downstream(fdir: np.ndarray, row: int, col: int, max_steps: int = 5000) -> list[tuple[int, int]]:
    """Trace the D8 downstream flow path from (row, col) until it exits the grid."""
    path = [(row, col)]
    for _ in range(max_steps):
        d = fdir[row, col]
        if d not in _NEIGHBORS:
            break
        dr, dc = _NEIGHBORS[d]
        nr, nc = row + dr, col + dc
        if not (0 <= nr < fdir.shape[0] and 0 <= nc < fdir.shape[1]):
            break
        row, col = nr, nc
        path.append((row, col))
    return path