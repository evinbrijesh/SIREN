"""Hydrological corridor + exposure generation (OpenCode-owned, Phase 3).

Combined D8 + OSM river buffering approach (ADR-005):

  1. D8 reachability — trace the downstream flow path from the change source
     to validate which sub-basin the floodwater drains into (physical check).
  2. OSM river selection — select waterway segments reachable by the D8 path
     (the real, surveyed riverbed through inhabited valleys).
  3. Floodplain buffer — buffer the selected river segments by a nominal
     flood-plain width (default 125 m).
  4. Exposure intersection — buffer each asset by its PRD §6.4 tolerance
     (bridges ±75 m, roads ±50 m, settlements/wells ±100 m) and intersect
     with the corridor.

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

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)  # D8 direction mapping (ESRI encoding)
# D8 neighbor offsets — VERIFIED against pysheds _sgrid.py:19-20.
# dirmap[k] pairs with (row_offsets[k], col_offsets[k]):
#   row_offsets = [-1, -1, 0, 1, 1, 1, 0, -1]
#   col_offsets = [ 0,  1, 1, 1, 0, -1, -1, -1]
# => 64=N, 128=NE, 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW
_NEIGHBORS = {
    64: (-1, 0),   # N
    128: (-1, 1),  # NE
    1: (0, 1),     # E
    2: (1, 1),     # SE
    4: (1, 0),     # S
    8: (1, -1),    # SW
    16: (0, -1),   # W
    32: (-1, -1),  # NW
}

# PRD §6.4 tolerance buffers (meters)
TOLERANCE_BUFFERS = {"bridge": 75.0, "road": 50.0, "settlement": 100.0, "well": 100.0}
# Default flood-plain buffer for the OSM river corridor (meters)
FLOODPLAIN_BUFFER_M = 125.0
# Radius around the D8 path used to select reachable OSM river segments (meters)
REACHABILITY_RADIUS_M = 500.0
# UTM zone for the Dudh Koshi AOI (84–90°E). Assumption documented in ADR-005.
UTM_CRS = "EPSG:32645"


def flow_accumulation(dem_path: str) -> tuple[Grid, np.ndarray, np.ndarray]:
    """Compute D8 flow direction + accumulation from a DEM raster.

    Full pysheds pipeline: fill_depressions -> resolve_flats -> flowdir ->
    accumulation. resolve_flats is REQUIRED: without it, filled flats (e.g.,
    a glacial lake surface) carry fdir=-1 and every trace across them
    terminates immediately.
    """
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    filled = grid.fill_depressions(dem=dem)
    resolved = grid.resolve_flats(dem=filled)
    fdir = grid.flowdir(dem=resolved, dirmap=DIRMAP)
    acc = grid.accumulation(fdir, dirmap=DIRMAP)
    return grid, fdir, acc


def d8_flow_path(dem_path: str, change_polygon_geojson: dict, channel_threshold: float = 500.0) -> list[list[float]]:
    """Trace the D8 downstream flow path from a change polygon centroid.

    Returns a list of [lon, lat] coordinates. This is the physical-validation
    step: it confirms the change source drains into the expected sub-basin.

    Degenerate-source handling: a change source sitting in a filled depression
    (e.g., a glacial lake — pysheds turns the lake into a flat, so the trace
    terminates immediately) yields a near-empty path. In that case, snap to
    the nearest cell with flow accumulation above `channel_threshold` (the
    actual river channel receiving the overflow) and trace from there.
    """
    grid, fdir, acc = flow_accumulation(dem_path)
    coords = _polygon_centroid(change_polygon_geojson)
    if coords is None:
        raise ValueError("change_polygon_geojson must be a Polygon or Feature with Polygon geometry")

    row, col = _snap_to_cell(grid, coords)
    path = _trace_downstream(fdir, row, col)

    # Degenerate-source handling: if the trace terminates before reaching a
    # real channel (terminal accumulation below threshold — typical for a
    # glacial lake, which pysheds fills into a flat), snap to the nearest
    # significant channel cell (the river receiving the overflow) and retrace.
    end_r, end_c = path[-1]
    terminal_acc = acc[end_r, end_c]
    if terminal_acc < channel_threshold:
        row, col = _nearest_high_accumulation_cell(grid, acc, coords, channel_threshold)
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
    if len(d8_path) < 2:
        raise ValueError(
            "D8 flow path from the change source is degenerate (pit/flat/edge). "
            "Pick a change source that drains downstream."
        )
    d8_line = LineString(d8_path)

    proj = pyproj.Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    d8_line_utm = transform(proj.transform, d8_line)

    # 2. Select OSM river segments reachable by the D8 path
    osm = gpd.read_file(osm_path)
    rivers = osm[osm["waterway"].isin(["river", "stream"])].copy()
    if rivers.empty:
        raise ValueError(f"No waterway features found in {osm_path}")

    reach_utm = d8_line_utm.buffer(reachability_radius_m)
    reach = transform(lambda x, y: proj.transform(x, y, direction="INVERSE"), reach_utm)
    reachable = rivers[rivers.intersects(reach)].copy()

    if reachable.empty:
        # ADR-005 fallback: buffer the D8 path itself (NOT all rivers — that
        # would flag assets basin-wide, far from the hazard).
        corridor_utm = d8_line_utm.buffer(floodplain_buffer_m)
    else:
        river_union = reachable.geometry.union_all()
        river_union_utm = transform(proj.transform, river_union)
        corridor_utm = river_union_utm.buffer(floodplain_buffer_m)
    corridor = transform(lambda x, y: proj.transform(x, y, direction="INVERSE"), corridor_utm)

    # 3. Exposure intersection: buffer each asset by its PRD tolerance, then
    #    intersect with the corridor. Excludes waterway features (rivers are
    #    the hazard layer, not an exposure).
    assets = osm[osm["waterway"].isna()].copy()
    assets["tolerance_m"] = assets.apply(_asset_buffer, axis=1)

    # Buffer assets in UTM (correct meters), then intersect with corridor
    assets_utm = assets.to_crs(UTM_CRS)
    assets_utm["buffered"] = assets_utm.geometry.buffer(assets_utm["tolerance_m"])
    corridor_gdf_utm = gpd.GeoDataFrame(geometry=[corridor_utm], crs=UTM_CRS)

    hits = gpd.sjoin(
        assets_utm.set_geometry("buffered"),
        corridor_gdf_utm,
        how="inner",
        predicate="intersects",
    )

    exposures = []
    for idx, r in hits.iterrows():
        # Distance from the asset's true geometry to the corridor boundary,
        # computed in UTM (meters). 0.0 if the asset lies inside the corridor.
        asset_geom_utm = assets_utm.loc[idx, "geometry"]
        dist_m = asset_geom_utm.distance(corridor_utm.boundary)
        exposures.append({
            "asset_id": r.get("@id") or f"asset-{idx}",
            "asset_type": _asset_type(r),
            "name": r.get("name:en") or r.get("name") or "",
            "distance_m": round(float(dist_m), 1),
            "buffer_m": float(r["tolerance_m"]),
            "in_floodplain": True,
        })

    exposures.sort(key=lambda e: e["distance_m"])

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
    if row.get("highway"):
        return "road"
    return "other"


def _polygon_centroid(geojson: dict) -> list[float] | None:
    geom = geojson.get("geometry", geojson) if geojson.get("type") == "Feature" else geojson
    if geom.get("type") != "Polygon":
        return None
    ring = geom["coordinates"][0]
    if not ring:
        return None
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def _snap_to_cell(grid: Grid, coords: list[float]) -> tuple[int, int]:
    affine = grid.affine
    col = int(round((coords[0] - affine[2]) / affine[0]))
    row = int(round((coords[1] - affine[5]) / affine[4]))
    if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
        raise ValueError(f"Change centroid {coords} lies outside the DEM extent")
    return row, col


def _nearest_high_accumulation_cell(
    grid: Grid, acc: np.ndarray, coords: list[float], threshold: float, max_radius: int = 100
) -> tuple[int, int]:
    """Find the nearest cell to (lon, lat) with flow accumulation > threshold.

    Search outward in expanding square rings; return the first cell above
    threshold (nearest channel cell, not the global max).
    """
    affine = grid.affine
    col = int(round((coords[0] - affine[2]) / affine[0]))
    row = int(round((coords[1] - affine[5]) / affine[4]))
    for radius in range(1, max_radius):
        r0, r1 = max(0, row - radius), min(acc.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(acc.shape[1], col + radius + 1)
        window = acc[r0:r1, c0:c1]
        if window.max() > threshold:
            rr, cc = np.unravel_index(np.argmax(window), window.shape)
            return int(r0 + rr), int(c0 + cc)
    raise ValueError(f"No channel cell with accumulation > {threshold} within {max_radius} cells of {coords}")


def _trace_downstream(fdir: np.ndarray, row: int, col: int, max_steps: int = 5000) -> list[tuple[int, int]]:
    """Trace the D8 downstream flow path from (row, col) until it exits the grid.

    Stops at pits (-2), flats (-1), or nodata — pysheds encodes these outside
    the dirmap. A single-cell path means the source is a pit/flat; callers
    must handle that (see exposure_corridor's degenerate-path guard).
    """
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