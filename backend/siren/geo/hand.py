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


# ---------------------------------------------------------------------------
# Level 2: Pure-numpy HAND for in-memory DEM arrays (no pysheds dependency)
# ---------------------------------------------------------------------------

def _d8_flow_direction_numpy(
    dem: np.ndarray,
    pixel_size_m: float = 10.0,
) -> np.ndarray:
    """Compute D8 flow direction from a DEM using pure numpy.

    D8 assigns each cell to the steepest downslope neighbor among its 8
    neighbors. The direction is encoded using the same ESRI DIRMAP as the
    pysheds pipeline: (64, 128, 1, 2, 4, 8, 16, 32) for (N, NE, E, SE, S, SW, W, NW).

    Cells with no downslope neighbor (pits/flat areas) get direction 0.

    Args:
        dem: 2D float32 elevation array.
        pixel_size_m: Pixel spacing in metres (for distance weighting of
            diagonal neighbors).

    Returns:
        2D int array with D8 direction codes (0 = pit/flat).
    """
    dem = np.asarray(dem, dtype=np.float32)
    rows, cols = dem.shape
    fdir = np.zeros((rows, cols), dtype=np.int32)

    # Neighbor offsets: (dr, dc, dircode, distance_weight)
    # Diagonal neighbors are √2 × pixel_size away
    diag_dist = np.sqrt(2) * pixel_size_m
    neighbors = [
        (-1, 0, 64, pixel_size_m),    # N
        (-1, 1, 128, diag_dist),       # NE
        (0, 1, 1, pixel_size_m),       # E
        (1, 1, 2, diag_dist),          # SE
        (1, 0, 4, pixel_size_m),       # S
        (1, -1, 8, diag_dist),         # SW
        (0, -1, 16, pixel_size_m),     # W
        (-1, -1, 32, diag_dist),       # NW
    ]

    # Compute elevation gradients for each neighbor direction
    max_slope = np.full((rows, cols), -np.inf, dtype=np.float32)
    best_dir = np.zeros((rows, cols), dtype=np.int32)

    for dr, dc, dircode, dist in neighbors:
        # Shift DEM to get neighbor elevations
        neighbor = np.full_like(dem, np.inf)
        if dr == -1 and dc == 0:  # N
            neighbor[1:, :] = dem[:-1, :]
        elif dr == -1 and dc == 1:  # NE
            neighbor[1:, :-1] = dem[:-1, 1:]
        elif dr == 0 and dc == 1:  # E
            neighbor[:, :-1] = dem[:, 1:]
        elif dr == 1 and dc == 1:  # SE
            neighbor[:-1, :-1] = dem[1:, 1:]
        elif dr == 1 and dc == 0:  # S
            neighbor[:-1, :] = dem[1:, :]
        elif dr == 1 and dc == -1:  # SW
            neighbor[:-1, 1:] = dem[1:, :-1]
        elif dr == 0 and dc == -1:  # W
            neighbor[:, 1:] = dem[:, :-1]
        elif dr == -1 and dc == -1:  # NW
            neighbor[1:, 1:] = dem[:-1, :-1]

        # Slope = (elevation - neighbor_elevation) / distance
        slope = (dem - neighbor) / dist
        # Update max slope and direction
        mask = slope > max_slope
        max_slope = np.where(mask, slope, max_slope)
        best_dir = np.where(mask, dircode, best_dir)

    # Only assign direction where there's a positive downslope
    fdir = np.where(max_slope > 0, best_dir, 0).astype(np.int32)
    return fdir


def _flow_accumulation_numpy(
    fdir: np.ndarray,
    dem: np.ndarray,
) -> np.ndarray:
    """Compute flow accumulation from D8 flow direction (pure numpy).

    Uses a priority-flood approach: process cells from highest to lowest
    elevation, accumulating flow from upstream contributors.

    Args:
        fdir: 2D D8 flow direction array (0 = pit/flat).
        dem: 2D elevation array (for sorting order).

    Returns:
        2D float32 flow accumulation array (each cell = number of upstream
        cells draining through it, including itself).
    """
    fdir = np.asarray(fdir)
    dem = np.asarray(dem, dtype=np.float32)
    rows, cols = dem.shape
    acc = np.ones((rows, cols), dtype=np.float32)

    # Reverse neighbor mapping: direction → (dr, dc)
    rev_neighbors = {v: k for k, v in _NEIGHBORS.items()}
    # Invert direction: if cell A flows to B, then B receives from A.
    # The reverse of D8 direction d is (d + 4) % 64 for the ESRI encoding,
    # but it's simpler to compute the target cell and accumulate there.

    # Sort cells by elevation (descending) — process high cells first
    flat_elev = dem.ravel()
    order = np.argsort(-flat_elev)  # descending

    # Build a reverse direction lookup: for each direction, the (dr, dc) offset
    # of the TARGET cell (where flow goes)
    target_offsets = {
        64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1),
        4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1),
    }

    for idx in order:
        r, c = divmod(idx, cols)
        d = fdir[r, c]
        if d == 0 or d not in target_offsets:
            continue  # pit or flat — no downstream flow
        dr, dc = target_offsets[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            acc[nr, nc] += acc[r, c]

    return acc


def compute_hand_from_array(
    dem: np.ndarray,
    pixel_size_m: float = 10.0,
    channel_threshold: float = 100.0,
) -> np.ndarray:
    """Compute HAND from an in-memory DEM array (pure numpy, no pysheds).

    This is the Level 2 entry point for computing HAND on Sen1Floods11 chip
    DEMs that are already reprojected to the chip grid. It avoids the
    pysheds/numba Python 3.14 incompatibility by implementing D8 flow
    direction and accumulation in pure numpy.

    Algorithm:
        1. Fill depressions (simple sink-fill via scipy.ndimage)
        2. Compute D8 flow direction (steepest downslope neighbor)
        3. Compute flow accumulation (priority-flood ordering)
        4. Identify channels (accumulation ≥ threshold)
        5. Trace downstream from each cell to nearest channel
        6. HAND = cell_elevation - channel_elevation

    Args:
        dem: 2D float32 elevation array (metres).
        pixel_size_m: Pixel spacing in metres.
        channel_threshold: Flow accumulation threshold for channel cells.
            Lower values identify more channels (finer drainage network).
            For 512×512 chips at 10m, ~100 cells ≈ 0.01 km² drainage area.

    Returns:
        2D float32 HAND array (metres above nearest drainage).
        Channel cells have HAND = 0; cells draining to boundaries have
        HAND = 0 (treated as at drainage level).
    """
    from scipy.ndimage import minimum_filter

    dem = np.asarray(dem, dtype=np.float32)
    rows, cols = dem.shape

    # Step 1: Simple sink fill — replace each cell with the minimum of its
    # 3×3 neighborhood if it's a local minimum. This is a one-pass approximation
    # of the full depression filling. For floodplain DEMs (mostly flat), this
    # is sufficient. For complex terrain, multiple passes would be needed.
    filled = dem.copy()
    # Iterative sink fill: raise sinks to the minimum of their neighbors
    for _ in range(10):  # limited iterations for performance
        local_min = minimum_filter(filled, size=3, mode="nearest")
        sinks = filled < local_min
        if not sinks.any():
            break
        filled = np.where(sinks, local_min, filled)

    # Step 2: D8 flow direction
    fdir = _d8_flow_direction_numpy(filled, pixel_size_m)

    # Step 3: Flow accumulation
    acc = _flow_accumulation_numpy(fdir, filled)

    # Step 4: Channel identification
    channel_mask = acc >= channel_threshold

    # Step 5: Trace HAND
    hand = _trace_hand(fdir, filled, channel_mask)

    return hand.astype(np.float32)


def apply_hand_filter(
    prob_mask: np.ndarray,
    hand_grid: np.ndarray,
    stage_threshold_m: float = 15.0,
) -> np.ndarray:
    """Apply deterministic HAND post-filter to a water probability mask.

    This is the Level 2 deterministic post-segmentation hydraulic filter.
    It zeros out water predictions at locations where the Height Above
    Nearest Drainage exceeds the maximum plausible flood surge stage,
    eliminating false positives at impossible elevations (ridge tops,
    plateaus, mountain shadows).

    The filter is a hard gate: any pixel where HAND(x, y) > stage_threshold_m
    is forced to 0.0, regardless of what the neural network predicted. This
    separates the vision task (backscatter → water probability) from the
    hydraulic task (elevation → plausibility).

    Args:
        prob_mask: 2D float array of water probabilities [0, 1] from the
            segmentation model (e.g. sigmoid(logits) > 0.5).
        hand_grid: 2D float32 HAND raster (metres above nearest drainage).
            Must have the same shape as prob_mask.
        stage_threshold_m: Maximum plausible flood surge stage in metres.
            Pixels with HAND above this are forced to 0.0. Typical values:
            - 5 m for shallow floodplains
            - 15 m for moderate terrain
            - 25 m for deep mountain canyons

    Returns:
        2D float array — filtered water probabilities. Same shape as input,
        with impossible-elevation predictions zeroed out.
    """
    prob_mask = np.asarray(prob_mask, dtype=np.float32)
    hand_grid = np.asarray(hand_grid, dtype=np.float32)

    if prob_mask.shape != hand_grid.shape:
        raise ValueError(
            f"prob_mask shape {prob_mask.shape} != hand_grid shape {hand_grid.shape}"
        )

    # Hard gate: zero out predictions where HAND exceeds the stage threshold
    impossible = hand_grid > stage_threshold_m
    filtered = prob_mask.copy()
    filtered[impossible] = 0.0
    return filtered
