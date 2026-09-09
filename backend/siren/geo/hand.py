"""Height Above Nearest Drainage (HAND) for flood exposure (V3 §3.2, Phase 3.2).

HAND is a hydrological terrain metric that computes the relative vertical
height of each pixel above the nearest drainage channel along its flow path.
Unlike a planar Euclidean buffer, HAND captures the vertical clearance
between an asset and the river — a hilltop monastery 200 m horizontally from
a canyon river but 50 m above it is NOT exposed to a 2 m flood surge.

Algorithm:
    1. Compute D8 flow direction from the DEM (pysheds).
    2. Compute flow accumulation to identify drainage channels (threshold-based).
    3. For each cell, trace downstream until reaching a channel cell.
    4. HAND = cell_elevation - channel_elevation (the vertical drop to the drain).

Exposure rule (V3 §3.2):
    An asset is exposed only if HAND(x, y) ≤ h_water_stage

where h_water_stage is the flood surge depth from the FNO surrogate (Phase 3.4)
or a policy default (0.5 m watch, 2.0 m elevated, 5.0 m critical).

This replaces the planar 125 m Euclidean buffer in corridor.py with a
physically-grounded vertical clearance check.

Deterministic and offline-safe (Hard Rules 2, 6). Uses pysheds with the
numpy 2.x compatibility monkeypatch (same as corridor.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- numpy 2.x / pysheds 0.5 compatibility patch (must precede pysheds import) ---
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from pysheds.grid import Grid  # noqa: E402  (after patch)

logger = logging.getLogger(__name__)

# D8 direction mapping (ESRI encoding, same as corridor.py)
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)

# D8 neighbor offsets matching DIRMAP (same as corridor.py)
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

# Default flow accumulation threshold for channel identification.
# Cells with accumulation above this are considered drainage channels.
DEFAULT_CHANNEL_THRESHOLD: float = 500.0

# Policy default water stage heights (metres) per severity (V3 §3.2).
# Used when the FNO surrogate is unavailable (shadow-only until validated).
DEFAULT_WATER_STAGE_M: dict[str, float] = {
    "watch": 0.5,
    "elevated": 2.0,
    "critical": 5.0,
    "informational": 0.5,
}

# Cache HAND results — the DEM is the same across all observations.
_hand_cache: dict[str, "HANDResult"] = {}


@dataclass
class HANDResult:
    """Result of a HAND computation.

    Attributes:
        hand: 2D float32 array — height above nearest drainage (metres).
            Non-channel cells have HAND ≥ 0; channel cells have HAND = 0.
        channel_mask: 2D bool array — True where flow accumulation ≥ threshold.
        dem: 2D float32 array — the input DEM (filled + resolved).
        grid: the pysheds Grid object (for coordinate transforms).
        channel_threshold: the accumulation threshold used.
    """

    hand: np.ndarray
    channel_mask: np.ndarray
    dem: np.ndarray
    grid: Grid
    channel_threshold: float


def compute_hand(
    dem_path: str | Path,
    channel_threshold: float = DEFAULT_CHANNEL_THRESHOLD,
) -> HANDResult:
    """Compute the Height Above Nearest Drainage raster from a DEM.

    Full pysheds pipeline: fill_depressions → resolve_flats → flowdir →
    accumulation → channel identification → HAND tracing.

    Results are cached per DEM path — the expensive fill_depressions step
    only runs once per process (same pattern as corridor.py).

    Args:
        dem_path: Path to the DEM GeoTIFF (e.g. SRTM 30 m or Copernicus GLO-30).
        channel_threshold: flow accumulation threshold for channel cells.
            Cells with accumulation ≥ this are drainage channels (HAND = 0).

    Returns:
        HANDResult with the HAND raster, channel mask, and grid metadata.
    """
    cache_key = f"{dem_path}:{channel_threshold}"
    if cache_key in _hand_cache:
        return _hand_cache[cache_key]

    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))
    filled = grid.fill_depressions(dem=dem)
    resolved = grid.resolve_flats(dem=filled)
    fdir = grid.flowdir(dem=resolved, dirmap=DIRMAP)
    acc = grid.accumulation(fdir, dirmap=DIRMAP)

    # Channel cells: flow accumulation above threshold
    channel_mask = acc >= channel_threshold

    # Compute HAND: for each non-channel cell, trace downstream to the
    # nearest channel cell and compute the elevation difference.
    dem_arr = np.asarray(resolved, dtype=np.float32)
    hand = _trace_hand(fdir, dem_arr, channel_mask)

    result = HANDResult(
        hand=hand.astype(np.float32),
        channel_mask=channel_mask,
        dem=dem_arr,
        grid=grid,
        channel_threshold=channel_threshold,
    )
    _hand_cache[cache_key] = result
    logger.info(
        "HAND computed: channel cells=%d (%.1f%%), HAND range=[%.1f, %.1f] m",
        int(channel_mask.sum()),
        100.0 * float(channel_mask.mean()),
        float(hand.min()),
        float(hand.max()),
    )
    return result


def _trace_hand(
    fdir: np.ndarray,
    dem: np.ndarray,
    channel_mask: np.ndarray,
) -> np.ndarray:
    """Trace downstream from each cell to the nearest channel cell.

    For each non-channel cell, follow the D8 flow direction until reaching
    a channel cell (or a boundary/pit). HAND = cell_elevation - channel_elevation.

    Channel cells have HAND = 0. Cells that drain to a boundary without
    hitting a channel have HAND = 0 (they are effectively at drainage level).

    Args:
        fdir: 2D D8 flow direction array (DIRMAP encoding).
        dem: 2D elevation array (filled + resolved).
        channel_mask: 2D bool array, True at channel cells.

    Returns:
        2D float32 HAND array.
    """
    fdir = np.asarray(fdir)
    dem = np.asarray(dem, dtype=np.float32)
    rows, cols = dem.shape
    hand = np.zeros_like(dem, dtype=np.float32)

    # Pre-compute channel cell coordinates for fast lookup
    channel_coords = set()
    ch_rows, ch_cols = np.where(channel_mask)
    for r, c in zip(ch_rows, ch_cols):
        channel_coords.add((int(r), int(c)))

    for r in range(rows):
        for c in range(cols):
            if (r, c) in channel_coords:
                hand[r, c] = 0.0
                continue

            # Trace downstream
            cell_elev = dem[r, c]
            cr, cc = r, c
            visited = set()
            found_channel = False
            channel_elev = cell_elev

            while True:
                if (cr, cc) in visited:
                    # Cycle — treat as drainage (HAND = 0)
                    break
                visited.add((cr, cc))

                if (cr, cc) in channel_coords:
                    channel_elev = dem[cr, cc]
                    found_channel = True
                    break

                # Follow D8 direction
                d = fdir[cr, cc]
                if d < 0 or d not in _NEIGHBORS:
                    # Pit or boundary — treat as drainage (HAND = 0)
                    break

                dr, dc = _NEIGHBORS[d]
                nr, nc = cr + dr, cc + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    # Off the grid — treat as drainage (HAND = 0)
                    break
                cr, cc = nr, nc

            if found_channel:
                hand[r, c] = max(0.0, cell_elev - channel_elev)
            else:
                hand[r, c] = 0.0

    return hand


def is_exposed_by_hand(
    hand: np.ndarray,
    row: int,
    col: int,
    h_water_stage: float,
) -> bool:
    """Check if a cell is exposed to flooding based on HAND.

    An asset is exposed only if its HAND value is below the water stage
    height (the flood surge depth).

    Args:
        hand: 2D HAND raster (metres above nearest drainage).
        row, col: cell coordinates of the asset.
        h_water_stage: flood surge depth in metres.

    Returns:
        True if HAND(row, col) ≤ h_water_stage (asset is exposed).
    """
    if row < 0 or row >= hand.shape[0] or col < 0 or col >= hand.shape[1]:
        return False
    return float(hand[row, col]) <= h_water_stage


def water_stage_for_severity(severity: str) -> float:
    """Policy default water stage height for a severity level (V3 §3.2).

    Used when the FNO surrogate is unavailable. The FNO provides a
    spatially-varying h_water grid; this is the scalar fallback.

    Args:
        severity: one of "informational", "watch", "elevated", "critical".

    Returns:
        Water stage height in metres.
    """
    if severity not in DEFAULT_WATER_STAGE_M:
        raise ValueError(
            f"unknown severity: {severity}. Expected one of "
            f"{list(DEFAULT_WATER_STAGE_M.keys())}"
        )
    return DEFAULT_WATER_STAGE_M[severity]


def hand_exposure_mask(
    hand: np.ndarray,
    h_water_stage: float,
) -> np.ndarray:
    """Compute a boolean exposure mask from a HAND raster + water stage.

    Args:
        hand: 2D HAND raster (metres above nearest drainage).
        h_water_stage: flood surge depth in metres.

    Returns:
        2D bool array — True where HAND ≤ h_water_stage (exposed to flooding).
    """
    return hand <= h_water_stage
