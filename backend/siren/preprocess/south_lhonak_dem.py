"""South Lhonak Pléiades DEM differencing — 2023 GLOF event terrain change.

The 2023-10-03 South Lhonak GLOF was triggered by the collapse of a frozen
lateral moraine into the lake, producing a multi-hazard cascade through
the Teesta valley (Sattar et al. 2025). This module computes the pre/post-
event DEM difference to quantify:

    - Moraine collapse volume (material lost from the source area)
    - Lakebed changes (sediment redistribution)
    - Downstream deposition (debris accumulation along the flood path)

The two DEMs have different footprints and must be clipped to their common
area before differencing. Both are 1 m Pléiades DEMs (height above WGS84
ellipsoid, UTM 45N / EPSG:32645), coregistered to the GLO30 Copernicus DEM
by the data provider (Gascoin & Cook 2024).

Usage:
    from siren.preprocess.south_lhonak_dem import compute_dem_difference

    result = compute_dem_difference()
    print(f"Moraine loss: {result.moraine_loss_m3:.0f} m³")
    print(f"Deposition: {result.deposition_m3:.0f} m³")

Scientific notes:
    - The pre-event DEM (2022-10-18) was computed from a PHR1A triplet.
    - The post-event DEM (2023-10-29) was computed from a PHR1A pair, 26
      days after the GLOF.
    - DEM differencing is subject to vertical bias and noise. The data
      provider coregistered to Copernicus GLO30, but residual co-registration
      errors may produce artefacts in steep terrain.
    - This is retrospective event evidence, not generalization proof.
    - NoData = -9999.0 in both DEMs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEM_DIR = REPO_ROOT / "data" / "datasets" / "south_lhonak_pleiades_dem"

PRE_EVENT_DEM = DEM_DIR / "20221018.tif"
POST_EVENT_DEM = DEM_DIR / "20231029.tif"
PROFILES_CSV = DEM_DIR / "profilesDEMsLake.csv"

NODATA = -9999.0


@dataclass
class DEMDifferenceResult:
    """Result of pre/post-event DEM differencing.

    Attributes:
        diff: (H, W) elevation difference (post - pre) in metres.
            Negative = elevation loss (erosion/collapse).
            Positive = elevation gain (deposition).
        common_bounds: bounds of the overlapping area in UTM 45N.
        common_shape: (height, width) of the overlapping area.
        pixel_area_m2: area of one pixel in m² (1.0 for 1 m DEM).
        loss_m3: total volume of elevation loss (negative diff * pixel_area).
        gain_m3: total volume of elevation gain (positive diff * pixel_area).
        net_change_m3: net volume change (gain - |loss|).
        n_loss_pixels: number of pixels with significant elevation loss.
        n_gain_pixels: number of pixels with significant elevation gain.
        loss_threshold_m: threshold for "significant" elevation loss.
        gain_threshold_m: threshold for "significant" elevation gain.
        pre_valid_fraction: fraction of valid pixels in the pre-event DEM.
        post_valid_fraction: fraction of valid pixels in the post-event DEM.
        pre_crs: CRS of the pre-event DEM.
        post_crs: CRS of the post-event DEM.
    """

    diff: np.ndarray
    common_bounds: tuple[float, float, float, float]
    common_shape: tuple[int, int]
    pixel_area_m2: float
    loss_m3: float
    gain_m3: float
    net_change_m3: float
    n_loss_pixels: int
    n_gain_pixels: int
    loss_threshold_m: float
    gain_threshold_m: float
    pre_valid_fraction: float
    post_valid_fraction: float
    pre_crs: Any
    post_crs: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "common_bounds": list(self.common_bounds),
            "common_shape": list(self.common_shape),
            "pixel_area_m2": self.pixel_area_m2,
            "loss_m3": round(self.loss_m3, 1),
            "gain_m3": round(self.gain_m3, 1),
            "net_change_m3": round(self.net_change_m3, 1),
            "n_loss_pixels": self.n_loss_pixels,
            "n_gain_pixels": self.n_gain_pixels,
            "loss_threshold_m": self.loss_threshold_m,
            "gain_threshold_m": self.gain_threshold_m,
            "pre_valid_fraction": round(self.pre_valid_fraction, 4),
            "post_valid_fraction": round(self.post_valid_fraction, 4),
            "pre_crs": str(self.pre_crs),
            "post_crs": str(self.post_crs),
        }


def _compute_common_bounds(
    pre_bounds: tuple[float, float, float, float],
    post_bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Compute the intersection of two bounding boxes.

    Args:
        pre_bounds: (left, bottom, right, top) of the pre-event DEM.
        post_bounds: (left, bottom, right, top) of the post-event DEM.

    Returns:
        (left, bottom, right, top) of the overlapping area.
    """
    left = max(pre_bounds[0], post_bounds[0])
    bottom = max(pre_bounds[1], post_bounds[1])
    right = min(pre_bounds[2], post_bounds[2])
    top = min(pre_bounds[3], post_bounds[3])
    return (left, bottom, right, top)


def compute_dem_difference(
    pre_path: Path | None = None,
    post_path: Path | None = None,
    loss_threshold_m: float = 2.0,
    gain_threshold_m: float = 2.0,
    output_path: Path | None = None,
) -> DEMDifferenceResult:
    """Compute the pre/post-event DEM difference for South Lhonak.

    Clips both DEMs to their common footprint, computes the elevation
    difference (post - pre), and quantifies erosion and deposition volumes.

    Args:
        pre_path: path to the pre-event DEM. Defaults to the standard location.
        post_path: path to the post-event DEM. Defaults to the standard location.
        loss_threshold_m: threshold for "significant" elevation loss (metres).
            Differences below this (in absolute value) are considered noise.
        gain_threshold_m: threshold for "significant" elevation gain (metres).
        output_path: if given, write the difference raster as a GeoTIFF.

    Returns:
        DEMDifferenceResult with the difference raster and volume statistics.

    Raises:
        FileNotFoundError: if either DEM file does not exist.
        ValueError: if the DEMs have no overlapping area.
    """
    if pre_path is None:
        pre_path = PRE_EVENT_DEM
    if post_path is None:
        post_path = POST_EVENT_DEM

    if not pre_path.exists():
        raise FileNotFoundError(f"Pre-event DEM not found: {pre_path}")
    if not post_path.exists():
        raise FileNotFoundError(f"Post-event DEM not found: {post_path}")

    logger.info("Computing DEM difference: pre=%s, post=%s", pre_path.name, post_path.name)

    with rasterio.open(pre_path) as pre_ds, rasterio.open(post_path) as post_ds:
        # Verify CRS compatibility
        if pre_ds.crs != post_ds.crs:
            raise ValueError(
                f"CRS mismatch: pre={pre_ds.crs}, post={post_ds.crs}"
            )

        # Compute common bounds
        common = _compute_common_bounds(pre_ds.bounds, post_ds.bounds)
        if common[0] >= common[2] or common[1] >= common[3]:
            raise ValueError(
                f"No overlapping area: pre={pre_ds.bounds}, post={post_ds.bounds}"
            )

        logger.info("Common bounds: %s", common)

        # Read the overlapping window from each DEM
        # Both are 1 m resolution, so windows should align
        pre_window = from_bounds(*common, pre_ds.transform)
        post_window = from_bounds(*common, post_ds.transform)

        # Round to integer pixel boundaries
        pre_window = pre_window.round_offsets(op="floor").round_lengths(op="ceil")
        post_window = post_window.round_offsets(op="floor").round_lengths(op="ceil")

        # Read data
        pre_data = pre_ds.read(1, window=pre_window, fill_value=NODATA)
        post_data = post_ds.read(1, window=post_window, fill_value=NODATA)

        # The windows may have slightly different shapes due to rounding.
        # Crop to the minimum common shape.
        min_h = min(pre_data.shape[0], post_data.shape[0])
        min_w = min(pre_data.shape[1], post_data.shape[1])
        pre_data = pre_data[:min_h, :min_w]
        post_data = post_data[:min_h, :min_w]

        logger.info("Common shape: (%d, %d)", min_h, min_w)

        # Compute valid pixel fractions
        pre_valid = pre_data != NODATA
        post_valid = post_data != NODATA
        pre_valid_frac = float(pre_valid.sum()) / pre_data.size
        post_valid_frac = float(post_valid.sum()) / post_data.size

        # Compute difference only where both DEMs have valid data
        both_valid = pre_valid & post_valid
        diff = np.full_like(pre_data, NODATA, dtype=np.float32)
        diff[both_valid] = post_data[both_valid].astype(np.float32) - pre_data[both_valid].astype(np.float32)

        # Compute volume statistics
        pixel_area = float(pre_ds.res[0] * pre_ds.res[1])  # 1.0 for 1 m DEM

        # Significant loss: diff < -loss_threshold
        loss_mask = (diff < -loss_threshold_m) & both_valid
        loss_m3 = float(np.abs(diff[loss_mask]).sum() * pixel_area)

        # Significant gain: diff > gain_threshold
        gain_mask = (diff > gain_threshold_m) & both_valid
        gain_m3 = float(diff[gain_mask].sum() * pixel_area)

        net_change = gain_m3 - loss_m3

        # Write output raster if requested
        if output_path is not None:
            # Compute the transform for the common window
            # Use the pre-event DEM's transform (both are 1 m, same CRS)
            win_transform = pre_ds.window_transform(pre_window)
            profile = {
                "driver": "GTiff",
                "height": min_h,
                "width": min_w,
                "count": 1,
                "dtype": "float32",
                "crs": pre_ds.crs,
                "transform": win_transform,
                "nodata": NODATA,
                "compress": "lzw",
                "tiled": True,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(diff, 1)
            logger.info("Difference raster written: %s", output_path)

        logger.info(
            "DEM difference complete: loss=%.0f m³ (%d pixels), gain=%.0f m³ (%d pixels), "
            "net=%.0f m³, pre_valid=%.1f%%, post_valid=%.1f%%",
            loss_m3,
            loss_mask.sum(),
            gain_m3,
            gain_mask.sum(),
            net_change,
            pre_valid_frac * 100,
            post_valid_frac * 100,
        )

        return DEMDifferenceResult(
            diff=diff,
            common_bounds=common,
            common_shape=(min_h, min_w),
            pixel_area_m2=pixel_area,
            loss_m3=loss_m3,
            gain_m3=gain_m3,
            net_change_m3=net_change,
            n_loss_pixels=int(loss_mask.sum()),
            n_gain_pixels=int(gain_mask.sum()),
            loss_threshold_m=loss_threshold_m,
            gain_threshold_m=gain_threshold_m,
            pre_valid_fraction=pre_valid_frac,
            post_valid_fraction=post_valid_frac,
            pre_crs=pre_ds.crs,
            post_crs=post_ds.crs,
        )
