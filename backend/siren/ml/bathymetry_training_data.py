"""Training data pipeline for neural bathymetry (ADR-013 §9.7.1).

Builds (DEM, lake_mask, bed_elevation) training samples from the Copernicus
DEM GLO30 and surveyed bathymetry points. For each of the 20 surveyed lakes:

    1. Find the DEM tile covering the lake.
    2. Clip to a window around the lake (lake extent + 100% buffer).
    3. Build a lake mask from the convex hull of surveyed points (16-lake
       dataset) or the outline polygon (4-lake dataset).
    4. Estimate the water surface elevation (z_surface) from the shoreline
       — the minimum DEM value at the lake rim (outlet elevation).
    5. Interpolate surveyed depth points to the DEM grid using linear
       interpolation → depth raster.
    6. Compute bed elevation: z_bed = z_surface - depth.
    7. Resample (DEM, lake_mask, bed_elevation) to a fixed grid size
       (default 128×128) for uniform training.

The output is a set of numpy arrays suitable for the BathymetryUNet:
    Input:  (2, H, W) — channel 0 = DEM (lake masked to 0), channel 1 = lake mask
    Target: (1, H, W) — bed elevation (normalised to [0, 1] using the DEM range)

Scientific notes:
    - The Copernicus DEM is a DSM (Digital Surface Model). Over water it
      returns the surface elevation, not the lake bed. The BathymetryUNet
      uses the surrounding moraine terrain (not the water surface) to
      predict bed elevation.
    - z_surface is estimated from the shoreline rim — the minimum DEM
      value at non-lake pixels adjacent to the lake. This is the standard
      hypsometric convention (the lake is filled to its outlet).
    - Depth interpolation uses scipy.interpolate.griddata with linear
      interpolation. Points outside the convex hull of surveyed points
      are filled with 0 depth (shoreline).
    - The bed elevation target is only defined inside the lake mask.
      Outside the lake, the target is set to z_surface (no depth).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.transform import from_origin
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point, box

from siren.ml.bathymetry_dataset import (
    LakeRecord,
    load_all_surveyed_lakes,
    ZHANG_16LAKES_DIR,
    DAS_4LAKES_DIR,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEM_DIR = REPO_ROOT / "data" / "datasets" / "copernicus_dem_glo30"

DEFAULT_GRID_SIZE = 128
BUFFER_FACTOR = 1.0  # buffer = 100% of lake extent on each side


@dataclass
class BathymetrySample:
    """A single training sample for the BathymetryUNet.

    Attributes:
        lake_id: lake identifier.
        lake_name: lake name.
        dem: (H, W) DEM array (lake region masked to 0).
        lake_mask: (H, W) binary lake mask (1 = lake, 0 = land).
        bed_elevation: (H, W) bed elevation target (z_surface - depth).
        z_surface: estimated water surface elevation (metres).
        dem_raw: (H, W) original DEM (before masking) — for debugging.
        depth_grid: (H, W) interpolated depth grid — for debugging.
        n_survey_points: number of surveyed depth points used.
        source_crs: CRS of the original DEM tile.
        bounds: (left, bottom, right, top) of the clipped window in EPSG:4326.
    """

    lake_id: str
    lake_name: str
    dem: np.ndarray
    lake_mask: np.ndarray
    bed_elevation: np.ndarray
    z_surface: float
    dem_raw: np.ndarray
    depth_grid: np.ndarray
    n_survey_points: int
    source_crs: Any
    bounds: tuple[float, float, float, float]

    def to_input_target(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to (input, target) tensors for the BathymetryUNet.

        Input:  (2, H, W) — channel 0 = normalised DEM (lake masked),
                channel 1 = lake mask
        Target: (1, H, W) — normalised bed elevation

        Returns:
            (input_array, target_array) as float32.
        """
        dem = self.dem.astype(np.float32)
        mask = self.lake_mask.astype(np.float32)
        bed = self.bed_elevation.astype(np.float32)

        # Normalise DEM to [0, 1] using the DEM range
        dem_min = float(dem[dem > 0].min()) if (dem > 0).any() else 0.0
        dem_max = float(dem.max())
        if dem_max > dem_min:
            dem_norm = (dem - dem_min) / (dem_max - dem_min)
        else:
            dem_norm = np.zeros_like(dem)
        dem_norm = np.clip(dem_norm, 0.0, 1.0)

        # Normalise bed elevation to [0, 1] using the same DEM range
        bed_norm = (bed - dem_min) / (dem_max - dem_min) if dem_max > dem_min else np.zeros_like(bed)
        bed_norm = np.clip(bed_norm, 0.0, 1.0)

        # Stack input: (2, H, W)
        x = np.stack([dem_norm, mask], axis=0).astype(np.float32)
        y = bed_norm[np.newaxis].astype(np.float32)

        return x, y


def _find_dem_tile(lon: float, lat: float) -> Path | None:
    """Find the Copernicus DEM GLO30 tile covering a point.

    Args:
        lon: longitude.
        lat: latitude.

    Returns:
        Path to the DEM tile, or None if not found.
    """
    tile_lat = int(lat)
    tile_lon = int(lon)
    tile_name = f"N{tile_lat:02d}_00_E{tile_lon:03d}_00_DEM.tif"
    tile_path = DEM_DIR / tile_name
    if tile_path.exists():
        return tile_path
    return None


def _build_lake_mask(
    record: LakeRecord,
    dem_bounds: tuple[float, float, float, float],
    dem_transform: Any,
    dem_width: int,
    dem_height: int,
) -> np.ndarray:
    """Build a binary lake mask on the DEM grid.

    For the 4-lake dataset: rasterise the outline polygon.
    For the 16-lake dataset: rasterise the convex hull of surveyed points.

    Args:
        record: LakeRecord with points and optional outline.
        dem_bounds: (left, bottom, right, top) of the DEM window.
        dem_transform: affine transform of the DEM window.
        dem_width: width of the DEM window in pixels.
        dem_height: height of the DEM window in pixels.

    Returns:
        (H, W) binary array — 1 = lake, 0 = land.
    """
    from rasterio.features import rasterize

    # Determine the lake geometry
    if record.outline is not None:
        # Use the outline polygon (4-lake dataset)
        outline_gdf = gpd.GeoDataFrame(
            {"geometry": [record.outline]},
            crs=record.crs,
        )
        outline_wgs84 = outline_gdf.to_crs("EPSG:4326")
        lake_geom = outline_wgs84.geometry.iloc[0]
    else:
        # Use convex hull of surveyed points (16-lake dataset)
        points_wgs84 = record.points.to_crs("EPSG:4326")
        lake_geom = points_wgs84.geometry.union_all().convex_hull

    # Rasterise to the DEM grid
    mask = rasterize(
        [(lake_geom, 1)],
        out_shape=(dem_height, dem_width),
        transform=dem_transform,
        fill=0,
        dtype=np.float32,
    )

    return mask


def _estimate_z_surface(dem: np.ndarray, lake_mask: np.ndarray) -> float:
    """Estimate water surface elevation from the lake shoreline.

    The water surface elevation equals the minimum DEM value at the lake
    rim — the set of non-lake pixels immediately adjacent to the lake.
    This is the standard hypsometric convention: the lake is filled to
    its outlet (the lowest point on the rim).

    Args:
        dem: (H, W) DEM array.
        lake_mask: (H, W) binary lake mask.

    Returns:
        Water surface elevation in metres.

    Raises:
        ValueError: if the lake mask is empty or fills the entire grid.
    """
    w = lake_mask > 0.5
    if not w.any():
        raise ValueError("lake mask is empty")

    # Dilate the lake mask by one pixel (4-connectivity)
    padded = np.pad(w, 1, mode="constant", constant_values=False)
    dilated = (
        padded[:-2, 1:-1] | padded[2:, 1:-1]
        | padded[1:-1, :-2] | padded[1:-1, 2:]
    )
    rim = dilated & ~w
    if not rim.any():
        raise ValueError("lake mask has no rim pixels (fills entire grid)")

    rim_elevations = dem[rim]
    # Filter out nodata values
    valid = rim_elevations[rim_elevations > -1000]
    if len(valid) == 0:
        raise ValueError("all rim pixels are nodata")

    return float(valid.min())


def _interpolate_depths(
    record: LakeRecord,
    dem_bounds: tuple[float, float, float, float],
    dem_transform: Any,
    dem_width: int,
    dem_height: int,
) -> tuple[np.ndarray, int]:
    """Interpolate surveyed depth points to the DEM grid.

    Uses scipy.interpolate.griddata with linear interpolation. Points
    outside the convex hull of surveyed points are filled with 0 depth
    (shoreline).

    Args:
        record: LakeRecord with surveyed depth points.
        dem_bounds: (left, bottom, right, top) of the DEM window.
        dem_transform: affine transform of the DEM window.
        dem_width: width of the DEM window in pixels.
        dem_height: height of the DEM window in pixels.

    Returns:
        (depth_grid, n_points) — (H, W) depth array in metres and the
        number of surveyed points used.
    """
    # Get surveyed points in WGS84
    points_wgs84 = record.points.to_crs("EPSG:4326")
    lons = points_wgs84.geometry.x.values
    lats = points_wgs84.geometry.y.values
    depths = points_wgs84["depth_m"].values

    # Build target grid coordinates (pixel centres)
    left, bottom, right, top = dem_bounds
    dx = (right - left) / dem_width
    dy = (top - bottom) / dem_height
    x_grid = left + dx * (np.arange(dem_width) + 0.5)
    y_grid = bottom + dy * (np.arange(dem_height) + 0.5)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Interpolate
    points = np.column_stack([lons, lats])
    depth_grid = griddata(
        points, depths, (xx, yy), method="linear", fill_value=0.0
    ).astype(np.float32)

    return depth_grid, len(depths)


def _resample_array(
    arr: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """Resample a 2D array to a target size using bilinear interpolation.

    Args:
        arr: (H, W) input array.
        target_size: target dimension (output is target_size × target_size).

    Returns:
        (target_size, target_size) resampled array.
    """
    from scipy.ndimage import zoom

    h, w = arr.shape
    if h == target_size and w == target_size:
        return arr

    zoom_factors = (target_size / h, target_size / w)
    return zoom(arr, zoom_factors, order=1).astype(np.float32)


def build_sample(
    record: LakeRecord,
    grid_size: int = DEFAULT_GRID_SIZE,
    buffer_factor: float = BUFFER_FACTOR,
) -> BathymetrySample | None:
    """Build a training sample for a single lake.

    Args:
        record: LakeRecord with surveyed depth points.
        grid_size: target grid size for resampling (default 128).
        buffer_factor: buffer as a fraction of lake extent (default 1.0 = 100%).

    Returns:
        BathymetrySample, or None if the lake cannot be processed.
    """
    # Get lake centroid in WGS84
    points_wgs84 = record.points.to_crs("EPSG:4326")
    lon = float(points_wgs84.geometry.x.mean())
    lat = float(points_wgs84.geometry.y.mean())

    # Find the DEM tile
    tile_path = _find_dem_tile(lon, lat)
    if tile_path is None:
        logger.warning("No DEM tile found for %s at (%.4f, %.4f)", record.lake_name, lon, lat)
        return None

    # Compute lake extent
    if record.outline is not None:
        outline_wgs84 = gpd.GeoDataFrame(
            {"geometry": [record.outline]}, crs=record.crs
        ).to_crs("EPSG:4326")
        lake_geom = outline_wgs84.geometry.iloc[0]
    else:
        lake_geom = points_wgs84.geometry.union_all().convex_hull

    minx, miny, maxx, maxy = lake_geom.bounds
    lake_width = maxx - minx
    lake_height = maxy - miny

    # Add buffer
    buffer_x = lake_width * buffer_factor
    buffer_y = lake_height * buffer_factor
    window_left = minx - buffer_x
    window_right = maxx + buffer_x
    window_bottom = miny - buffer_y
    window_top = maxy + buffer_y

    with rasterio.open(tile_path) as ds:
        # Clip to the window
        win = from_bounds(
            window_left, window_bottom, window_right, window_top,
            ds.transform,
        )
        win = win.round_offsets(op="floor").round_lengths(op="ceil")

        dem_raw = ds.read(1, window=win, fill_value=-9999.0)
        win_transform = ds.window_transform(win)
        win_bounds = rasterio.windows.bounds(win, ds.transform)
        win_crs = ds.crs

    dem_h, dem_w = dem_raw.shape
    logger.debug(
        "%s: DEM window %dx%d, bounds=%s",
        record.lake_name, dem_w, dem_h, win_bounds,
    )

    # Build lake mask on the DEM grid
    lake_mask = _build_lake_mask(
        record, win_bounds, win_transform, dem_w, dem_h
    )

    # Estimate water surface elevation
    try:
        z_surface = _estimate_z_surface(dem_raw, lake_mask)
    except ValueError as e:
        logger.warning("Cannot estimate z_surface for %s: %s", record.lake_name, e)
        return None

    # Interpolate depths to the DEM grid
    depth_grid, n_points = _interpolate_depths(
        record, win_bounds, win_transform, dem_w, dem_h
    )

    # Compute bed elevation: z_bed = z_surface - depth
    bed_elevation = z_surface - depth_grid

    # Mask DEM: set lake region to 0 (as the model expects)
    dem_masked = dem_raw.copy().astype(np.float32)
    dem_masked[lake_mask > 0.5] = 0.0
    # Set nodata to 0 as well
    dem_masked[dem_masked < -1000] = 0.0

    # Resample to target grid size
    dem_resampled = _resample_array(dem_masked, grid_size)
    mask_resampled = _resample_array(lake_mask, grid_size)
    bed_resampled = _resample_array(bed_elevation, grid_size)
    dem_raw_resampled = _resample_array(dem_raw.astype(np.float32), grid_size)
    depth_resampled = _resample_array(depth_grid, grid_size)

    # Binarise the mask after resampling
    mask_resampled = (mask_resampled > 0.5).astype(np.float32)

    # Re-apply the mask to the DEM after resampling (resampling can cause
    # edge misalignment between the mask and DEM)
    dem_resampled[mask_resampled > 0.5] = 0.0

    return BathymetrySample(
        lake_id=record.lake_id,
        lake_name=record.lake_name,
        dem=dem_resampled,
        lake_mask=mask_resampled,
        bed_elevation=bed_resampled,
        z_surface=z_surface,
        dem_raw=dem_raw_resampled,
        depth_grid=depth_resampled,
        n_survey_points=n_points,
        source_crs=win_crs,
        bounds=win_bounds,
    )


def build_all_samples(
    records: list[LakeRecord] | None = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    buffer_factor: float = BUFFER_FACTOR,
) -> list[BathymetrySample]:
    """Build training samples for all surveyed lakes.

    Args:
        records: list of LakeRecord objects. Loaded if None.
        grid_size: target grid size for resampling.
        buffer_factor: buffer as a fraction of lake extent.

    Returns:
        List of BathymetrySample objects (one per successfully processed lake).
    """
    if records is None:
        records = load_all_surveyed_lakes()

    samples: list[BathymetrySample] = []
    for record in records:
        sample = build_sample(record, grid_size=grid_size, buffer_factor=buffer_factor)
        if sample is not None:
            samples.append(sample)
            logger.debug(
                "Built sample for %s: %d points, z_surface=%.1f, max_depth=%.1f",
                record.lake_name,
                sample.n_survey_points,
                sample.z_surface,
                sample.depth_grid.max(),
            )
        else:
            logger.warning("Skipped %s — could not build sample", record.lake_name)

    logger.info("Built %d/%d bathymetry training samples", len(samples), len(records))
    return samples


def compute_sample_volume(
    sample: BathymetrySample,
    pixel_area_m2: float | None = None,
) -> float:
    """Compute the lake volume from a bathymetry sample.

    V = Σ max(0, z_surface - z_bed) * pixel_area

    This uses the bed elevation target (from surveyed depths) to compute
    the "ground truth" volume for the sample.

    Args:
        sample: BathymetrySample with bed_elevation and z_surface.
        pixel_area_m2: area of one pixel in m². If None, estimated from
            the sample bounds and grid size.

    Returns:
        Volume in m³.
    """
    if pixel_area_m2 is None:
        # Estimate pixel area from bounds and grid size
        left, bottom, right, top = sample.bounds
        width_m = (right - left) * 111320 * np.cos(np.radians((top + bottom) / 2))
        height_m = (top - bottom) * 111320
        pixel_area_m2 = (width_m * height_m) / (sample.dem.shape[0] * sample.dem.shape[1])

    lake_pixels = sample.lake_mask > 0.5
    depths = np.maximum(0.0, sample.z_surface - sample.bed_elevation)
    return float(depths[lake_pixels].sum() * pixel_area_m2)
