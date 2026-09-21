"""SAR backscatter change detection (Phase 2).

Detects water expansion between two Sentinel-1 GRD scenes via log-ratio of
backscatter (DN²). This is the SAR path — primary during monsoon cloud cover
(ADR-003).

Pipeline (all real, validated on the Dudh Koshi scene pair):
  1. Read VV bands from both SAFE archives (decimated for memory).
  2. Multi-look (5x5 spatial averaging) to suppress speckle.
  3. Log-ratio: log10((DN2²+ε)/(DN1²+ε)). Same-orbit repeat-pass means radar
     shadow is dark in BOTH scenes (ratio ≈ 0), while new water is
     bright→dark (strong negative ratio).
  4. Threshold the ratio to flag water expansion.
  5. Geolocate change pixels via the annotation geolocation grid (GCPs).

KNOWN DATA CONSTRAINT (documented 2026-09-05): the available ascending-orbit
scene pair covers only the WESTERN AOI — the eastern swath edge at lat 27.9
is ~86.69°E, so the Imja lake (86.925°E) is OUTSIDE the swath. The demo
scenario mask near Imja is therefore PREPARED (Roadmap fallback), clearly
labeled; the pipeline itself is demonstrated on the real covered region.
A descending-pass S1 pair would provide real Imja coverage (V2).
"""

from __future__ import annotations

import re
import zipfile

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator

# Decimation factor for reading the 25762x16732 scenes (memory bound)
DECIMATION = 10
# Multi-look window (speckle suppression)
MULTILOOK_SIZE = 5
# Log-ratio threshold for "strong decrease" (candidate new water)
CHANGE_THRESHOLD = -0.8
# Land floor in scene 1: pixels below this are shadow/dark, not land
LAND_FLOOR_DN = 300.0


def read_s1_vv(s1_zip: str, inner_tiff: str, decimation: int = DECIMATION) -> np.ndarray:
    """Read the VV measurement band, decimated by the given factor."""
    with rasterio.open(f"/vsizip/{s1_zip}/{inner_tiff}") as src:
        out_shape = (src.height // decimation, src.width // decimation)
        return src.read(1, out_shape=out_shape)


def multilook(data: np.ndarray, size: int = MULTILOOK_SIZE) -> np.ndarray:
    """Spatial averaging to suppress speckle."""
    return uniform_filter(data.astype(np.float32), size=size)


def log_ratio(s1_dn: np.ndarray, s2_dn: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Log-ratio of backscatter (DN²) between two dates."""
    s1 = s1_dn.astype(np.float32)
    s2 = s2_dn.astype(np.float32)
    return np.log10((s2**2 + eps) / (s1**2 + eps))


def detect_expansion(
    s1_dn: np.ndarray, s2_dn: np.ndarray, threshold: float = CHANGE_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    """Detect water expansion between two scenes.

    Returns (expansion_mask, log_ratio). Expansion = strong negative
    log-ratio on multi-looked data (was bright land, became dark water).
    """
    s1f = multilook(s1_dn)
    s2f = multilook(s2_dn)
    lr = log_ratio(s1f, s2f)
    return lr < threshold, lr


def extract_gcps(s1_zip: str) -> list[tuple[int, int, float, float]]:
    """Extract the geolocation grid points (line, pixel, lat, lon) from the
    scene annotation XML."""
    with zipfile.ZipFile(s1_zip) as z:
        name = next(
            n for n in z.namelist()
            if "annotation/s1d-iw-grd-vv" in n
            and "calibration" not in n and "noise" not in n and "rfi" not in n
        )
        xml = z.read(name).decode("utf-8", errors="ignore")
    pattern = re.compile(
        r"<geolocationGridPoint>.*?<line>(\d+)</line>.*?<pixel>(\d+)</pixel>"
        r".*?<latitude>([\d.eE+-]+)</latitude>"
        r".*?<longitude>([\d.eE+-]+)</longitude>.*?</geolocationGridPoint>",
        re.DOTALL,
    )
    return [
        (int(l), int(p), float(la), float(lo))
        for l, p, la, lo in pattern.findall(xml)
    ]


def geolocate(
    s1_zip: str, rows: np.ndarray, cols: np.ndarray, decimation: int = DECIMATION
) -> tuple[np.ndarray, np.ndarray]:
    """Map decimated radar indices to (lat, lon) via the GCP grid."""
    gcps = extract_gcps(s1_zip)
    lines = sorted({g[0] for g in gcps})
    pixels = sorted({g[1] for g in gcps})
    gcp_map = {(g[0], g[1]): (g[2], g[3]) for g in gcps}
    lat_grid = np.zeros((len(lines), len(pixels)))
    lon_grid = np.zeros((len(lines), len(pixels)))
    for i, l in enumerate(lines):
        for j, p in enumerate(pixels):
            la, lo = gcp_map[(l, p)]
            lat_grid[i, j] = la
            lon_grid[i, j] = lo
    interp_lat = RegularGridInterpolator((lines, pixels), lat_grid)
    interp_lon = RegularGridInterpolator((lines, pixels), lon_grid)
    pts = np.column_stack([rows * decimation, cols * decimation])
    return interp_lat(pts), interp_lon(pts)


def swath_coverage(s1_zip: str) -> dict[float, float]:
    """Return the eastern swath edge (max lon) per latitude band.

    Used to verify which parts of the AOI the scenes actually cover.
    """
    gcps = extract_gcps(s1_zip)
    by_lat: dict[float, list[float]] = {}
    for _, _, la, lo in gcps:
        if 27.6 <= la <= 28.0:
            by_lat.setdefault(round(la, 1), []).append(lo)
    return {lat: max(lons) for lat, lons in sorted(by_lat.items())}


# ---------------------------------------------------------------------------
# Fix for finding #2 (shadow vs water): slope masking.
# Water is FLAT (slope < ~5°); radar shadow occurs on STEEP slopes. Combining
# the log-ratio change signal with a DEM slope mask removes shadow-induced
# false positives that survive multi-looking.
# ---------------------------------------------------------------------------

def dem_slope(dem_path: str) -> tuple[np.ndarray, "rasterio.DatasetReader"]:
    """Compute terrain slope (degrees) from the DEM.

    Handles lat/lon scaling: converts the lat/lon gradient to meters using
    the local latitude (1° lon ≈ 111.32·cos(lat) km, 1° lat ≈ 110.57 km).
    Returns (slope_degrees_array, dataset) — keep the dataset open while
    using the array (for windowed sampling).
    """
    ds = rasterio.open(dem_path)
    dem = ds.read(1).astype(np.float32)
    nodata = ds.nodata
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    # Per-pixel metre conversion: np.gradient returns dz per pixel, so the
    # divisor must be metres per PIXEL (metres per degree × degrees per
    # pixel), not metres per degree. (Dividing by m/degree shrinks the
    # slope by ~10^5 — the gate would never fire.)
    lats = np.array([ds.xy(r, 0)[1] for r in range(ds.height)])
    m_per_px_lon = 111_320.0 * np.cos(np.deg2rad(lats)) * abs(ds.transform.a)
    m_per_px_lat = 110_540.0 * abs(ds.transform.e)

    dy, dx = np.gradient(dem)
    gx = dx / m_per_px_lon[:, np.newaxis]   # dz/dx in metres/metre
    gz = dy / m_per_px_lat                  # dz/dy in metres/metre
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gz**2))).astype(np.float32)
    return slope, ds


def sample_slope(slope: np.ndarray, ds: "rasterio.DatasetReader", lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Sample the slope raster at geographic points (nearest cell)."""
    inv = ~ds.transform
    cols, rows = inv * (lons, lats)
    rows = np.clip(rows.astype(int), 0, slope.shape[0] - 1)
    cols = np.clip(cols.astype(int), 0, slope.shape[1] - 1)
    return slope[rows, cols]


def filter_change_by_slope(
    change_mask: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    dem_path: str,
    max_slope_deg: float = 5.0,
) -> np.ndarray:
    """Keep only change pixels on flat terrain (plausible water).

    Shadow-induced false positives sit on steep slopes; water cannot.
    Returns the filtered boolean mask (same length as lats).
    """
    slope, ds = dem_slope(dem_path)
    try:
        s = sample_slope(slope, ds, lats, lons)
    finally:
        ds.close()
    keep = s < max_slope_deg
    return change_mask & keep


def sar_grid_dem_slope(
    sar_path: str,
    dem_path: str,
    stride: int = 8,
) -> np.ndarray | None:
    """Sample DEM slope onto a calibrated SAR cache grid via its GCPs.

    Calibrated SAR caches (``extract_and_cache_vv_vh_db``) carry the
    Sentinel-1 GCP geolocation grid (EPSG:4326) rather than an affine —
    GRD geolocation is not affine over mountain terrain (a fitted affine
    has 1–4 km residuals). Each SAR pixel is mapped to (lon, lat) through
    GDAL's GCP polynomial transformer, then the DEM-derived slope raster
    is nearest-sampled at those coordinates.

    Args:
        sar_path: calibrated VV/VH cache GeoTIFF (must carry GCPs).
        dem_path: DEM GeoTIFF (EPSG:4326) to derive slope from.
        stride: geolocation is smooth — transform every ``stride``-th
            pixel and bilinearly upsample (much faster than transforming
            all ~4M pixels individually).

    Returns:
        (H, W) float32 slope in degrees aligned with the SAR grid, NaN
        where the pixel falls outside the DEM extent — or None when the
        SAR cache carries no GCPs (old ungeoreferenced caches).
    """
    ll = sar_grid_lonlat(sar_path, stride=stride)
    if ll is None:
        return None
    lon, lat = ll
    h, w = lon.shape

    slope, ds = dem_slope(dem_path)
    try:
        dem_inv = ~ds.transform
        # (lon, lat) -> DEM pixel coordinates
        dem_cols = dem_inv.a * lon + dem_inv.b * lat + dem_inv.c
        dem_rows = dem_inv.d * lon + dem_inv.e * lat + dem_inv.f
        dc = np.rint(dem_cols).astype(np.int64)
        dr = np.rint(dem_rows).astype(np.int64)

        valid = (
            (dr >= 0) & (dr < slope.shape[0])
            & (dc >= 0) & (dc < slope.shape[1])
        )
        out = np.full((h, w), np.nan, dtype=np.float32)
        out[valid] = slope[dr[valid], dc[valid]]
        return out
    finally:
        ds.close()


def sar_grid_lonlat(
    sar_path: str,
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-pixel (lon, lat) geolocation for a calibrated SAR cache grid.

    Reads the S1 GCP grid persisted by ``extract_and_cache_vv_vh_db`` and
    maps pixel coordinates to EPSG:4326 through GDAL's GCP polynomial
    transformer (a fitted affine is inaccurate by 1–4 km over mountain
    terrain — the GCPs are the honest geolocation). Every ``stride``-th
    pixel is transformed and the smooth field bilinearly upsampled.

    Returns:
        (lon, lat) float64 arrays shaped (H, W) — or None when the cache
        carries no GCPs (old ungeoreferenced caches).
    """
    import rasterio
    from rasterio.transform import GCPTransformer
    from scipy.ndimage import zoom

    with rasterio.open(sar_path) as src:
        gcps, _ = src.gcps
        h, w = src.height, src.width
    if not gcps:
        return None

    transformer = GCPTransformer(gcps)
    rows_s = np.arange(0, h, stride, dtype=np.float64)
    cols_s = np.arange(0, w, stride, dtype=np.float64)
    rr, cc = np.meshgrid(rows_s, cols_s, indexing="ij")
    xs, ys = transformer.xy(rr.ravel().tolist(), cc.ravel().tolist())
    lon_s = np.asarray(xs).reshape(rr.shape)
    lat_s = np.asarray(ys).reshape(rr.shape)

    # Upsample geolocation to the full grid — the field is smooth, so
    # bilinear interpolation introduces sub-pixel error only.
    zh, zw = h / lon_s.shape[0], w / lon_s.shape[1]
    lon = zoom(lon_s, (zh, zw), order=1)
    lat = zoom(lat_s, (zh, zw), order=1)
    lon = lon[:h, :w]
    lat = lat[:h, :w]
    if lon.shape != (h, w):
        pad_h, pad_w = h - lon.shape[0], w - lon.shape[1]
        lon = np.pad(lon, ((0, pad_h), (0, pad_w)), mode="edge")
        lat = np.pad(lat, ((0, pad_h), (0, pad_w)), mode="edge")
    return lon, lat


def sar_grid_polygon_mask(
    vector_path: str,
    lon: np.ndarray,
    lat: np.ndarray,
    cell_deg: float = 0.001,
) -> np.ndarray:
    """Rasterize vector polygons onto a geolocated SAR grid.

    The polygons are burned onto a regular EPSG:4326 grid (``cell_deg``
    resolution ≈ 110 m, matched to the ~90 m SAR pixel pitch) covering the
    lon/lat extent, then nearest-sampled at each SAR pixel's geolocation.

    Args:
        vector_path: polygon source readable by geopandas (shapefile,
            GeoJSON, or a GDAL ``/vsizip/`` path).
        lon, lat: per-pixel geolocation from ``sar_grid_lonlat``.
        cell_deg: rasterisation cell size in degrees.

    Returns:
        (H, W) bool array — True where the SAR pixel falls inside a polygon.
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    h, w = lon.shape
    west, east = float(np.nanmin(lon)), float(np.nanmax(lon))
    south, north = float(np.nanmin(lat)), float(np.nanmax(lat))

    try:
        gdf = gpd.read_file(vector_path, bbox=(west, south, east, north))
    except TypeError:
        gdf = gpd.read_file(vector_path)
        gdf = gdf[gdf.intersects(
            gpd.GeoSeries.from_xy([west, east, east, west],
                                [south, south, north, north]).union_all()
        )]
    out = np.zeros((h, w), dtype=bool)
    if gdf.empty:
        return out

    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    grid_w = int(np.ceil((east - west) / cell_deg)) + 2
    grid_h = int(np.ceil((north - south) / cell_deg)) + 2
    transform = from_origin(west - cell_deg, north + cell_deg, cell_deg, cell_deg)
    burned = rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None],
        out_shape=(grid_h, grid_w),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    inv = ~transform
    cols = np.rint(inv.a * lon + inv.b * lat + inv.c).astype(np.int64)
    rows = np.rint(inv.d * lon + inv.e * lat + inv.f).astype(np.int64)
    valid = (rows >= 0) & (rows < grid_h) & (cols >= 0) & (cols < grid_w)
    out[valid] = burned[rows[valid], cols[valid]] > 0
    return out


def sar_grid_sample(
    raster_path: str,
    lon: np.ndarray,
    lat: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    """Nearest-sample any georeferenced raster onto a geolocated SAR grid.

    Reprojects the SAR pixel (lon, lat) into the source raster's CRS, then
    nearest-samples band 1. Pixels outside the source extent get ``fill``.

    This is the geographically correct way to compare the full-scene SAR
    shadow mask against the small AOI masks — index-based resizing
    (``consensus._resize_mask``) assumes identical extents and produces
    meaningless overlap counts when the scenes differ.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(raster_path) as ds:
        arr = ds.read(1)
        if ds.crs is not None and ds.crs.to_epsg() != 4326:
            xs, ys = warp_transform(
                "EPSG:4326", ds.crs,
                lon.ravel().tolist(), lat.ravel().tolist(),
            )
        else:
            xs, ys = lon.ravel().tolist(), lat.ravel().tolist()
        inv = ~ds.transform

    bx = np.asarray(xs).reshape(lon.shape)
    by = np.asarray(ys).reshape(lat.shape)
    cols = np.rint(inv.a * bx + inv.b * by + inv.c).astype(np.int64)
    rows = np.rint(inv.d * bx + inv.e * by + inv.f).astype(np.int64)
    valid = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    out = np.full(lon.shape, fill, dtype=np.float32)
    out[valid] = arr[rows[valid], cols[valid]].astype(np.float32)
    return out