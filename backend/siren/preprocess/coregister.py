"""Co-register a moving raster onto a reference raster's grid (PRD §6.2).

Pure function: resamples the moving raster so its CRS, transform and
dimensions match the reference. Returns an alignment-error RMSE in reference
pixels (0.0 when the two already share an identical grid).
"""

from __future__ import annotations

import tempfile

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject as _warp_reproject
from rasterio.warp import transform as _warp_transform_coords


def co_register(
    reference_path: str,
    moving_path: str,
    output_path: str | None = None,
) -> tuple[str, float]:
    """Co-register a moving raster to a reference raster's grid.

    Resamples the moving raster to match the reference's CRS, transform and
    dimensions.

    Args:
        reference_path: path to the reference (baseline) GeoTIFF.
        moving_path: path to the moving (to-be-aligned) GeoTIFF.
        output_path: where to write the aligned raster. If None, a temporary
            file is created.

    Returns:
        (path to aligned raster, alignment_error) where alignment_error is the
        RMSE in reference pixels between the moving grid's corners and the
        reference grid's corners (0.0 if they already share the same grid).
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            prefix="siren_coreg_", suffix=".tif", delete=False
        )
        output_path = tmp.name
        tmp.close()

    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_height = ref.height
        ref_width = ref.width
        ref_profile = ref.profile.copy()

    with rasterio.open(moving_path) as mov:
        mov_crs = mov.crs
        mov_transform = mov.transform
        mov_height = mov.height
        mov_width = mov.width
        mov_count = mov.count
        mov_nodata = mov.nodata
        mov_data = mov.read()

        same_grid = (
            mov_crs == ref_crs
            and mov_transform == ref_transform
            and mov_height == ref_height
            and mov_width == ref_width
        )

        if same_grid:
            alignment_error = 0.0
        else:
            # Corner offsets of the moving grid expressed in reference pixels,
            # compared against the reference grid corners (0,0)..(W,H).
            mov_corners_px = [
                (0, 0),
                (mov_width, 0),
                (mov_width, mov_height),
                (0, mov_height),
            ]
            mov_corners_world = [
                mov_transform * (x, y) for (x, y) in mov_corners_px
            ]
            xs = [c[0] for c in mov_corners_world]
            ys = [c[1] for c in mov_corners_world]
            if mov_crs != ref_crs:
                xs, ys = _warp_transform_coords(
                    str(mov_crs), str(ref_crs), xs, ys
                )
            mov_in_ref_px = [
                ~ref_transform * (x, y) for (x, y) in zip(xs, ys)
            ]
            ref_corners_px = [
                (0, 0),
                (ref_width, 0),
                (ref_width, ref_height),
                (0, ref_height),
            ]
            sq = 0.0
            for (mx, my), (rx, ry) in zip(mov_in_ref_px, ref_corners_px):
                sq += (mx - rx) ** 2 + (my - ry) ** 2
            alignment_error = float((sq / len(ref_corners_px)) ** 0.5)

    # Write the aligned raster on the reference grid.
    out_profile = ref_profile.copy()
    out_profile.update(count=mov_count, dtype=ref_profile.get("dtype", "float32"))
    with rasterio.open(output_path, "w", **out_profile) as dst:
        for b in range(mov_count):
            dst_buf = np.empty((ref_height, ref_width), dtype=mov_data.dtype)
            _warp_reproject(
                source=mov_data[b],
                destination=dst_buf,
                src_transform=mov_transform,
                src_crs=mov_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
                src_nodata=mov_nodata,
                dst_nodata=ref_profile.get("nodata"),
            )
            dst.write(dst_buf, b + 1)

    return str(output_path), alignment_error
