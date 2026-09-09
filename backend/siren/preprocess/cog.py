"""Cloud-Optimized GeoTIFF / windowed AOI reading (Sprint 1 Step 5).

The Production Roadmap §2.2 calls for reading only the AOI bounding box
from cloud-hosted scenes instead of downloading entire 1.7 GB Sentinel-1
SAFE archives. This module provides the windowed-read primitive that
makes that possible, plus the CDSE-specific helpers to address the
cloud object.

Two read paths share the same windowed core:

  * **Local SAFE** (``/vsizip/``) — the offline-demo path. Reading only
    the AOI window from the measurement TIFF cuts memory and I/O roughly
    to the AOI's share of the scene footprint (~5 % for the Dudh Koshi
    basin), even from a local archive.
  * **Cloud COG** (``/vsis3/``) — the production path. CDSE HTTPS asset
    links do *not* support spatial subsetting (full-file download only);
    partial reads require the S3 endpoint ``s3://eodata/...`` with AWS
    credentials, accessed via GDAL's ``/vsis3/`` virtual filesystem.

The window computation (:func:`aoi_window`) and the S3 URI transform
(:func:`cdse_s3_uri_from_href`) are pure and unit-tested against
synthetic fixtures. The actual cloud read is an integration step
verified manually against CDSE (it needs network + credentials and so
cannot run in the offline test suite — Hard Rule 2).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

logger = logging.getLogger(__name__)

# CDSE S3 endpoint for EO data (Production Roadmap §2.2 / CDSE S3 docs).
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
CDSE_S3_BUCKET = "eodata"


def aoi_window(
    src: rasterio.io.DatasetReader,
    bbox: tuple[float, float, float, float],
    bbox_crs: str = "EPSG:4326",
) -> Window:
    """Compute the rasterio Window covering ``bbox`` within an open dataset.

    The bbox is reprojected from ``bbox_crs`` into the dataset's CRS
    before windowing, so this works regardless of the source projection.

    For Sentinel-1 GRD SAFE measurement TIFFs — which carry Ground
    Control Points instead of an affine CRS — this falls back to
    :func:`gcp_window` (GCPs are in EPSG:4326).

    Args:
        src: An open rasterio dataset.
        bbox: ``(lon_min, lat_min, lon_max, lat_max)`` in ``bbox_crs``.
        bbox_crs: CRS of the bbox (default EPSG:4326).

    Returns:
        A rasterio :class:`~rasterio.windows.Window` (integer, floored)
        covering the bbox, clipped to the dataset bounds.
    """
    if src.crs is None:
        gcps, gcp_crs = src.gcps
        if gcps and gcp_crs is not None:
            return gcp_window(gcps, gcp_crs, src, bbox, bbox_crs)
        raise ValueError("source dataset has no CRS — cannot transform bbox")
    if bbox_crs != src.crs.to_string():
        from rasterio.warp import transform_bounds

        left, bottom, right, top = transform_bounds(
            bbox_crs, src.crs, *bbox, densify_pts=21
        )
    else:
        left, bottom, right, top = bbox

    # Clamp to the dataset's own bounds so we never request a window
    # outside the raster (which would raise or return nodata padding).
    ds_left, ds_bottom, ds_right, ds_top = src.bounds
    left = max(left, ds_left)
    right = min(right, ds_right)
    bottom = max(bottom, ds_bottom)
    top = min(top, ds_top)
    if right <= left or top <= bottom:
        raise ValueError(
            f"bbox {bbox} does not intersect dataset bounds {tuple(src.bounds)}"
        )

    win = from_bounds(left, bottom, right, top, transform=src.transform)
    return win.round_offsets().round_lengths()


def gcp_window(
    gcps,
    gcp_crs,
    src: rasterio.io.DatasetReader,
    bbox: tuple[float, float, float, float],
    bbox_crs: str = "EPSG:4326",
) -> Window:
    """Compute a pixel Window from Ground Control Points (S1 GRD SAFE).

    Sentinel-1 GRD measurement TIFFs are georeferenced by a grid of GCPs
    (in EPSG:4326) rather than an affine transform. The GCP grid is
    typically far coarser than a small AOI bbox, so filtering GCPs
    *inside* the bbox is unreliable. Instead this fits an affine
    ``(lon, lat) → (row, col)`` transform over **all** GCPs by least
    squares, maps the four bbox corners into pixel space, and takes the
    bounding window — expanded by one GCP cell on each side and clamped
    to the raster.

    The affine fit is exact for a nominally-geolocated S1 GRD scene and
    accurate to within a few pixels over the swath; the one-cell margin
    absorbs the residual. The result is an *over-read* (superset of the
    AOI pixels) — callers needing exact clipping should reproject/clip
    afterwards.
    """
    from rasterio.warp import transform_bounds

    if bbox_crs != str(gcp_crs):
        left, bottom, right, top = transform_bounds(
            bbox_crs, str(gcp_crs), *bbox, densify_pts=21
        )
    else:
        left, bottom, right, top = bbox

    rows = np.array([g.row for g in gcps], dtype=np.float64)
    cols = np.array([g.col for g in gcps], dtype=np.float64)
    xs = np.array([g.x for g in gcps], dtype=np.float64)
    ys = np.array([g.y for g in gcps], dtype=np.float64)

    # S1 GRD geolocation is non-linear across the wide swath, so a global
    # affine (lon,lat)→(row,col) fit is inaccurate (residuals of hundreds
    # of pixels). Fit a *local* affine using only GCPs near the AOI.
    # Expand the bbox by a margin large enough to capture a few GCP cells
    # on each side; grow until we have enough points for a stable fit.
    margin = 0.5
    sel = np.zeros_like(xs, dtype=bool)
    for _ in range(6):
        sel = (
            (xs >= left - margin) & (xs <= right + margin)
            & (ys >= bottom - margin) & (ys <= top + margin)
        )
        if int(sel.sum()) >= 6:
            break
        margin *= 2.0
    if int(sel.sum()) < 3:
        raise ValueError(
            f"bbox {bbox} has too few nearby GCPs ({int(sel.sum())}) for a fit "
            f"(gcp extent x=[{xs.min():.4f},{xs.max():.4f}] "
            f"y=[{ys.min():.4f},{ys.max():.4f}])"
        )

    A = np.column_stack([np.ones_like(xs[sel]), xs[sel], ys[sel]])
    a_coef, *_ = np.linalg.lstsq(A, rows[sel], rcond=None)
    b_coef, *_ = np.linalg.lstsq(A, cols[sel], rcond=None)

    corners = np.array([
        [left, bottom],
        [left, top],
        [right, bottom],
        [right, top],
    ])
    Ac = np.column_stack([np.ones(4), corners[:, 0], corners[:, 1]])
    r_corners = Ac @ a_coef
    c_corners = Ac @ b_coef

    r0, r1 = float(r_corners.min()), float(r_corners.max())
    c0, c1 = float(c_corners.min()), float(c_corners.max())

    # Sanity: if the fit maps the bbox entirely outside the raster, raise.
    if r1 < 0 or r0 > src.height or c1 < 0 or c0 > src.width:
        raise ValueError(
            f"bbox {bbox} maps outside the raster via GCP affine fit "
            f"(rows=[{r0:.0f},{r1:.0f}] cols=[{c0:.0f},{c1:.0f}], "
            f"raster {src.height}x{src.width})"
        )

    # Margin: one GCP cell (median gap) on each side, clamped to the raster.
    uniq_rows = np.unique(rows)
    uniq_cols = np.unique(cols)
    row_gap = float(np.median(np.diff(uniq_rows))) if len(uniq_rows) > 1 else 0.0
    col_gap = float(np.median(np.diff(uniq_cols))) if len(uniq_cols) > 1 else 0.0

    r0 = max(0.0, r0 - row_gap)
    c0 = max(0.0, c0 - col_gap)
    r1 = min(float(src.height), r1 + row_gap)
    c1 = min(float(src.width), c1 + col_gap)

    if c1 <= c0 or r1 <= r0:
        raise ValueError(
            f"bbox {bbox} yields an empty window after GCP fit + clamp"
        )

    return Window(
        col_off=int(np.floor(c0)),
        row_off=int(np.floor(r0)),
        width=int(np.ceil(c1 - c0)),
        height=int(np.ceil(r1 - r0)),
    )


def read_aoi_window(
    path_or_url: str,
    bbox: tuple[float, float, float, float],
    bbox_crs: str = "EPSG:4326",
    band: int = 1,
    decimation: int = 1,
    env_options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Read only the AOI bbox window from any rasterio-openable raster.

    Works for local GeoTIFFs, ``/vsizip/`` SAFE archives, and ``/vsis3/``
    cloud COGs (pass the appropriate ``env_options`` for auth).

    Args:
        path_or_url: rasterio-openable path or virtual filesystem URL.
        bbox: AOI ``(lon_min, lat_min, lon_max, lat_max)`` in ``bbox_crs``.
        bbox_crs: CRS of the bbox.
        band: 1-based band index to read.
        decimation: Integer read decimation (``out_shape`` scaling).
        env_options: Extra ``rasterio.Env`` options (e.g. AWS credentials,
            ``GDAL_HTTP_HEADER_FILE``) for authenticated cloud reads.

    Returns:
        ``(data, transform, crs)`` where ``data`` is a 2D float32 array
        covering only the AOI window, ``transform`` is the window's
        geotransform (scaled for decimation), and ``crs`` is the source CRS.
    """
    with rasterio.Env(**(env_options or {})):
        with rasterio.open(path_or_url) as src:
            win = aoi_window(src, bbox, bbox_crs=bbox_crs)
            out_h = max(1, int(win.height) // decimation)
            out_w = max(1, int(win.width) // decimation)
            data = src.read(
                band, window=win, out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)
            # Window transform, then scale for decimation.
            win_transform = src.window_transform(win)
            scale_x = win.width / out_w
            scale_y = win.height / out_h
            transform = rasterio.transform.Affine(
                win_transform.a * scale_x, win_transform.b, win_transform.c,
                win_transform.d, win_transform.e * scale_y, win_transform.f,
            )
            crs = src.crs
    return data, transform, crs


def cdse_s3_uri_from_href(href: str) -> str:
    """Convert a CDSE HTTPS eodata asset URL to an ``s3://eodata/...`` URI.

    CDSE STAC assets are served over HTTPS as full-file download links
    (no spatial subsetting). The same objects are reachable via the S3
    endpoint ``s3://eodata/<key>`` which *does* support byte-range reads
    for COGs. This helper rewrites the host to the S3 URI form.

    Examples:
        ``https://eodata.dataspace.copernicus.eu/Sentinel-1/GRD/.../x.tif``
        → ``s3://eodata/Sentinel-1/GRD/.../x.tif``

    For hrefs not on the eodata host, the original string is returned
    unchanged (caller should fall back to full download).
    """
    if not isinstance(href, str) or not href:
        return href  # type: ignore[return-value]
    # Match the eodata host (with or without a region segment).
    m = re.match(
        r"^https?://eodata(?:\.[a-z]+)?\.dataspace\.copernicus\.eu/(.*)$",
        href,
        re.IGNORECASE,
    )
    if not m:
        return href
    return f"s3://{CDSE_S3_BUCKET}/{m.group(1)}"


def cdse_s3_env(
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
) -> dict[str, Any]:
    """Build rasterio.Env options for authenticated /vsis3/ reads from CDSE.

    Credentials default to the ``AWS_ACCESS_KEY_ID`` /
    ``AWS_SECRET_ACCESS_KEY`` environment variables (the standard AWS
    vars that GDAL reads). CDSE issues these from the Data Space account
    "Add Credentials" page — they are *not* the CDSE login password.

    Returns a dict suitable for passing as ``env_options`` to
    :func:`read_aoi_window`. CDSE's S3 uses path-style addressing, so
    ``AWS_VIRTUAL_HOSTING=False`` is set (required per CDSE docs).
    """
    ak = access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    sk = secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    opts: dict[str, Any] = {
        "AWS_VIRTUAL_HOSTING": False,
        "AWS_S3_ENDPOINT": CDSE_S3_ENDPOINT,
        "AWS_HTTPS": "YES",
    }
    if ak:
        opts["AWS_ACCESS_KEY_ID"] = ak
    if sk:
        opts["AWS_SECRET_ACCESS_KEY"] = sk
    return opts


def calibrate_s1_window_to_db(
    safe_zip: str,
    pol: str,
    bbox: tuple[float, float, float, float],
    bbox_crs: str = "EPSG:4326",
    decimation: int = 10,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Read + calibrate only the AOI window of a Sentinel-1 SAFE σ⁰ band.

    This is the windowed equivalent of
    :func:`siren.preprocess.sar_calibrate.calibrate_s1_to_db` — it reads
    only the AOI window from the measurement TIFF (via ``/vsizip/``),
    then applies the ESA calibration LUT interpolated to that window.

    The calibration LUT is parsed from the full archive (it is tiny) and
    interpolated to the windowed, decimated grid. This keeps the
    calibration math identical to the full-scene path while reading
    only the AOI pixels from the multi-GB measurement TIFF.

    Args:
        safe_zip: Path to the Sentinel-1 SAFE ZIP archive.
        pol: Polarisation (``"vv"`` or ``"vh"``).
        bbox: AOI bbox in ``bbox_crs``.
        bbox_crs: CRS of the bbox.
        decimation: Read decimation factor.

    Returns:
        ``(sigma0_db, transform, crs)`` for the AOI window only.
    """
    from siren.preprocess.sar_calibrate import (
        _find_calibration_xml,
        _find_measurement_tiff,
        _interpolate_calibration,
        _parse_calibration_lut,
    )

    inner_tiff = _find_measurement_tiff(safe_zip, pol)
    vsi = f"/vsizip/{safe_zip}/{inner_tiff}"

    with rasterio.open(vsi) as src:
        win = aoi_window(src, bbox, bbox_crs=bbox_crs)
        out_h = max(1, int(win.height) // decimation)
        out_w = max(1, int(win.width) // decimation)
        # Nearest-neighbour resampling, matching calibrate_s1_to_db: the
        # σ⁰ calibration formula squares the DN, so averaging DNs before
        # squaring (bilinear) would be physically wrong.
        dn = src.read(
            1, window=win, out_shape=(out_h, out_w),
            resampling=rasterio.enums.Resampling.nearest,
        ).astype(np.float32)
        win_transform = src.window_transform(win)
        crs = src.crs

    # Interpolate the calibration LUT to the windowed, decimated grid.
    # The LUT line/pixel indices are in full-scene coordinates, so map
    # the decimated window grid back to full-scene pixel coordinates.
    lines, pixels, sigma_nought = _parse_calibration_lut(safe_zip, pol)
    # Full-scene coordinates of the window's decimated grid.
    row0 = int(win.row_off)
    col0 = int(win.col_off)
    dec_lines = row0 + np.arange(out_h) * decimation
    dec_pixels = col0 + np.arange(out_w) * decimation
    grid_lines, grid_pixels = np.meshgrid(dec_lines, dec_pixels, indexing="ij")
    points = np.stack([grid_lines.ravel(), grid_pixels.ravel()], axis=-1)

    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (lines, pixels), sigma_nought,
        method="linear", bounds_error=False,
        fill_value=float(sigma_nought[0, 0]),
    )
    calib = interp(points).reshape(out_h, out_w).astype(np.float32)

    calib_safe = np.where(calib > 0, calib, 1.0)
    sigma0 = (dn ** 2) / (calib_safe ** 2)
    sigma0 = np.where(sigma0 > 0, sigma0, 1e-10)
    sigma0_db = (10.0 * np.log10(sigma0)).astype(np.float32)

    scale_x = win.width / out_w
    scale_y = win.height / out_h
    # Output pixels span `decimation` source pixels each; scale_x ≈ decimation.
    transform = rasterio.transform.Affine(
        win_transform.a * scale_x, win_transform.b, win_transform.c,
        win_transform.d, win_transform.e * scale_y, win_transform.f,
    )
    logger.info(
        "Windowed σ⁰ %s dB: shape=%s, range=[%.1f, %.1f]",
        pol, sigma0_db.shape, float(sigma0_db.min()), float(sigma0_db.max()),
    )
    return sigma0_db, transform, crs
