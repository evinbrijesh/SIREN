"""SAR backscatter change detection (OpenCode-owned, Phase 2).

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