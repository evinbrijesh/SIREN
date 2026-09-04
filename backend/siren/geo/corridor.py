"""Hydrological corridor + exposure generation (OpenCode-owned, Phase 3).

Combined D8 + OSM river buffering approach (ADR-003 / Roadmap Phase 3 fallback):

  1. D8 reachability — trace the downstream flow path from the change source
     to validate which sub-basin the floodwater drains into (physical check).
  2. OSM river selection — select waterway segments reachable by the D8 path
     (the real, surveyed riverbed through inhabited valleys).
  3. Floodplain buffer — buffer the selected river segments by a nominal
     flood-plain width (default 125 m).
  4. Exposure intersection — intersect the buffered corridor against OSM
     assets using the PRD tolerance buffers (bridges ±75 m, roads ±50 m,
     settlements/wells ±100 m).

Deterministic and rule-based (ADR-002).

CRITICAL: pysheds 0.5 uses np.in1d which was removed in numpy 2.x. The
monkeypatch below MUST run before any pysheds import.
"""

from __future__ import annotations

import numpy as np

# --- numpy 2.x / pysheds 0.5 compatibility patch (must precede pysheds import) ---
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from pysheds.grid import Grid  # noqa: E402  (after patch)

import geopandas as gpd
from shapely.geometry import LineString, mapping
from shapely.ops import transform
import pyproj

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)  # D8 direction mapping
_NEIGHBORS = {
    64: (0, -1), 128: (-1, -1), 1: (-1, 0), 2: (-1, 1),
    4: (0, 1), 8: (1, 1), 16: (1, 0), 32: (1, -1),
}

# PRD §6.4 tolerance buffers (meters)
TOLERANCE_BUFFERS = {"bridge": 75.0, "road": 50.0, "settlement": 100.0, "well": 100.0}
# Default flood-plain buffer for the OSM river corridor (meters)
FLOODPLAIN_BUFFER_M = 125.0
# Radius around the D8 path used to select reachable OSM river segments (meters)
REACHABILITY_RADIUS_M = 500.0


def flow_accumulation(dem_path: str) -> tuple[Grid, np.ndarray, np.ndarray]:
    """Compute D8 flow direction + accumulation from a DEM raster."""
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    filled = grid.fill_depressions(dem=dem)
    fdir = grid.flowdir(dem=filled, dirmap=DIRMAP)
    acc = grid.accumulation(fdir, dirmap=DIRMAP)
    return grid, fdir, acc


def d8_flow_path(dem_path: str, change_polygon_geojson: dict) -> list[list[float]]:
    """Trace the D8 downstream flow path from a change polygon centroid.

    Returns a list of [lon, lat] coordinates. This is the physical-validation
    step: it confirms the change source drains into the expected sub-basin.
    """
    grid, fdir, _ = flow_accumulation(dem_path)
    coords = _polygon_centroid(change_polygon_geojson)
    if coords is None:
        raise ValueError("change_polygon_geojson must be a Polygon or Feature with Polygon geometry")
    row, col = _snap_to_cell(grid, coords)
    path = _trace_downstream(fdir, row, col)
    affine = grid.affine
    return [[affine[2] + c * affine[0], affine[5] + r * affine[4]] for r, c in path]


def exposure_corridor(
    dem_path: str,
    change_polygon_geojson: dict,
    osm_path: str,
    floodplain_buffer_m: float = FLOODPLAIN_BUFFER_M,
    reachability_radius_m: float = REACHABILITY_RADIUS_M,
) -> dict:
    """Build the exposure corridor and intersect it against OSM assets.

    Returns a GeoJSON FeatureCollection:
      - corridor polygon (buffered OSM rivers reachable from the change source)
      - d8_path (physical validation line)
      - exposures (list of affected assets with distance + buffer)

    Deterministic: same inputs -> same output.
    """
    # 1. D8 reachability (physical validation)
    d8_path = d8_flow_path(dem_path, change_polygon_geojson)
    d8_line = LineString(d8_path)

    # 2. Select OSM river segments reachable by the D8 path
    osm = gpd.read_file(osm_path)
    rivers = osm[osm["waterway"].isin(["river", "stream"])].copy()
    if rivers.empty:
        raise ValueError(f"No waterway features found in {osm_path}")

    # Reachability: river segment intersects the D8 path buffered by radius
    proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    d8_line_utm = transform(proj.transform, d8_line)
    reach_utm = d8_line_utm.buffer(reachability_radius_m)
    reach = transform(lambda x, y: proj.transform(x, y, direction="INVERSE"), reach_utm)

    reachable = rivers[rivers.intersects(reach)].copy()
    if reachable.empty:
        # Fallback: use all rivers (D8 path too short to reach any segment)
        reachable = rivers.copy()

    # 3. Buffer the reachable river segments into a flood-plain corridor
    river_union = reachable.geometry.union_all()
    river_union_utm = transform(proj.transform, river_union)
    corridor_utm = river_union_utm.buffer(floodplain_buffer_m)
    corridor = transform(lambda x, y: proj.transform(x, y, direction="INVERSE"), corridor_utm)

    # 4. Intersect corridor against assets with PRD tolerance buffers
    assets = osm.copy()
    assets["buffer_m"] = assets.apply(_asset_buffer, axis=1)
    corridor_gdf = gpd.GeoDataFrame(geometry=[corridor], crs="EPSG:4326")
    hits = gpd.sjoin(assets, corridor_gdf, how="inner", predicate="intersects")

    exposures = []
    for _, r in hits.iterrows():
        dist_m = r.geometry.distance(corridor) * 111000.0  # approx deg->m
        exposures.append({
            "asset_id": r.get("@id", f"asset-{len(exposures)}"),
            "asset_type": _asset_type(r),
            "name": r.get("name:en") or r.get("name") or "",
            "distance_m": round(dist_m, 1),
            "buffer_m": r["buffer_m"],
            "inundated": bool(r.get("amenity") == "drinking_water" or r.get("man_made") == "water_well"),
        })

    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"kind": "corridor", "buffer_m": floodplain_buffer_m},
             "geometry": mapping(corridor)},
            {"type": "Feature", "properties": {"kind": "d8_path"},
             "geometry": mapping(d8_line)},
        ],
        "exposures": exposures,
    }


def _asset_buffer(row) -> float:
    """PRD §6.4 tolerance buffer for an OSM feature."""
    if row.get("bridge") == "yes":
        return TOLERANCE_BUFFERS["bridge"]
    if row.get("highway"):
        return TOLERANCE_BUFFERS["road"]
    if row.get("place") in ("village", "hamlet", "town"):
        return TOLERANCE_BUFFERS["settlement"]
    if row.get("amenity") == "drinking_water" or row.get("man_made") == "water_well":
        return TOLERANCE_BUFFERS["well"]
    return 100.0


def _asset_type(row) -> str:
    if row.get("place") in ("village", "hamlet", "town"):
        return "settlement"
    if row.get("bridge") == "yes":
        return "bridge"
    if row.get("amenity") == "drinking_water" or row.get("man_made") == "water_well":
        return "well"
    if row.get("amenity") in ("clinic", "hospital"):
        return "health"
    if row.get("waterway"):
        return "river"
    if row.get("highway"):
        return "road"
    return "other"


def _polygon_centroid(geojson: dict) -> list[float] | None:
    geom = geojson.get("geometry", geojson) if geojson.get("type") == "Feature" else geojson
    if geom.get("type") != "Polygon":
        return None
    ring = geom["coordinates"][0]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def _snap_to_cell(grid: Grid, coords: list[float]) -> tuple[int, int]:
    affine = grid.affine
    col = int(round((coords[0] - affine[2]) / affine[0]))
    row = int(round((coords[1] - affine[5]) / affine[4]))
    row = max(0, min(row, grid.shape[0] - 1))
    col = max(0, min(col, grid.shape[1] - 1))
    return row, col


def _trace_downstream(fdir: np.ndarray, row: int, col: int, max_steps: int = 5000) -> list[tuple[int, int]]:
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