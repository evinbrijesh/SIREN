"""Himalayan lake chip dataset builder — high-altitude SAR training data.

Rasterizes the verified glacial lake inventory
(``data/datasets/glacial_lake_2022-2024/Glacial_Lake_2022.shp``, ~31.7k
polygons across High Mountain Asia) onto the GCP-geolocated Sentinel-1 GRD
grid and extracts lake-centred chips in the 6-channel Kuro Siwo contract
(``VV_post, VH_post, VV_pre, VH_pre, dVV, dVH``) — see
``ml/contract.py::build_kuro_siwo_tensor``.

Rationale (domain-shift fix): the gate-passed Kuro Siwo model was trained
on lowland floods and is badly out-of-distribution on high-Himalaya
terrain — glacier ice reads as water (~61k false-positive px on the Imja
scene). A single descending swath covers ~1.4k verified lake polygons,
which is the missing high-altitude label source for a decoder fine-tune
(frozen encoder, ADR-013 shadow track).

Label semantics (honest): the inventory polygons are *median-outlined*
lake footprints over 2022–2024, not per-pixel SAR water boundaries on the
scene date. Chips therefore carry WEAK positive labels inside the polygon
and negatives elsewhere — surrounding moraine/glacier/terrain becomes the
domain-specific negative context the model lacks. Each chip records its
provenance (lake ID, elevation, area, scenes) so a later eval can filter
by confidence.

Geolocation: polygon vertices and centroids are mapped through the
scene's GCP polynomial (``GCPTransformer.rowcol``), the same geolocation
the runtime shadow layer uses — no affine approximation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INVENTORY_PATH = (
    _REPO_ROOT
    / "data"
    / "datasets"
    / "glacial_lake_2022-2024"
    / "Glacial_Lake_2022.shp"
)

CHIP = 96           # ~8.6 km context window at ~90 m decimated GRD pitch
STRIDE_BG = 512     # background-chip sampling stride (grid px)


@dataclass
class LakeChip:
    """One extracted chip + provenance."""

    chip_id: str
    lake_id: str
    lake_elev_m: float
    lake_area_km2: float
    lon: float
    lat: float
    row: int
    col: int
    pos_px: int
    kind: str = "lake"          # "lake" | "background"
    extra: dict = field(default_factory=dict)


def load_lake_inventory(
    path: Path | str | None = None,
    min_elev_m: float = 0.0,
    min_area_km2: float = 0.0,
):
    """Load the glacial lake inventory, filtered by elevation/area.

    Args:
        path: shapefile path (defaults to the 2022 inventory).
        min_elev_m: minimum lake elevation (metres).
        min_area_km2: minimum lake area (km²).

    Returns:
        GeoDataFrame with ID, Latitude, Longitude, Lake_Elev, Area, geometry.
    """
    import geopandas as gpd

    p = Path(path) if path else DEFAULT_INVENTORY_PATH
    gdf = gpd.read_file(p)
    # The inventory's Area column is square metres (Imja = 1,738,224 →
    # 1.74 km²). Normalise to km² once so filters/manifests are honest.
    gdf["area_km2"] = gdf["Area"].astype(float) / 1e6
    if min_elev_m:
        gdf = gdf[gdf["Lake_Elev"].astype(float) >= min_elev_m]
    if min_area_km2:
        gdf = gdf[gdf["area_km2"] >= min_area_km2]
    return gdf.reset_index(drop=True)


def lake_grid_positions(lakes, sar_cache_path: str):
    """Map lake centroids to SAR-grid (row, col) via the GCP polynomial.

    Args:
        lakes: GeoDataFrame with Latitude/Longitude columns.
        sar_cache_path: calibrated VV/VH dB cache carrying GCPs.

    Returns:
        (positions, lon_grid, lat_grid):
        positions is a list of (idx, row, col) for centroids inside the
        grid; lon_grid/lat_grid are the full (H, W) geolocation arrays.
    """
    import rasterio
    from rasterio.transform import GCPTransformer

    from siren.detect.sar import sar_grid_lonlat

    with rasterio.open(sar_cache_path) as src:
        gcps, _ = src.gcps
        h, w = src.height, src.width
    if not gcps:
        raise ValueError(f"no GCPs in {sar_cache_path} — cannot geolocate lakes")

    ll = sar_grid_lonlat(sar_cache_path)
    if ll is None:
        raise ValueError(f"sar_grid_lonlat returned None for {sar_cache_path}")
    lon_g, lat_g = ll

    transformer = GCPTransformer(gcps)
    rows, cols = transformer.rowcol(
        lakes["Longitude"].astype(float).tolist(),
        lakes["Latitude"].astype(float).tolist(),
        op=float,
    )
    positions = []
    for idx, (r, c) in enumerate(zip(rows, cols)):
        if 0 <= r < h and 0 <= c < w:
            positions.append((int(idx), int(round(r)), int(round(c))))
    return positions, lon_g, lat_g


def rasterize_lake_labels(lakes, positions, sar_cache_path: str) -> np.ndarray:
    """Rasterize lake polygons onto the SAR grid via GCP vertex mapping.

    Each polygon's exterior/interior vertices are mapped lon/lat →
    (row, col) through the GCP polynomial, then burned into a full-grid
    binary label with ``rasterio.features.rasterize`` (identity transform
    — coords are already pixel space).

    Args:
        lakes: GeoDataFrame of lake polygons (EPSG:4326).
        positions: (idx, row, col) list from ``lake_grid_positions`` —
            only lakes whose centroid falls in the grid are burned.
        sar_cache_path: calibrated cache (for GCPs + grid shape).

    Returns:
        uint8 label array (H, W): 1 = inside a lake polygon.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import GCPTransformer
    from shapely.geometry import MultiPolygon, Polygon

    with rasterio.open(sar_cache_path) as src:
        gcps, _ = src.gcps
        h, w = src.height, src.width
    transformer = GCPTransformer(gcps)

    shapes = []
    for idx, _r, _c in positions:
        geom = lakes.geometry.iloc[idx]
        if geom is None or geom.is_empty:
            continue
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            if not isinstance(poly, Polygon):
                continue
            rings = []
            for ring in [poly.exterior, *poly.interiors]:
                xs, ys = ring.xy
                rr, cc = transformer.rowcol(list(xs), list(ys), op=float)
                rings.append(list(zip(cc.astype(float), rr.astype(float))))
            if len(rings[0]) >= 3:
                shapes.append((Polygon(rings[0], rings[1:]), 1))

    label = rasterize(shapes, out_shape=(h, w), fill=0, dtype="uint8")
    return label


def extract_lake_chips(
    tensor6: np.ndarray,
    label: np.ndarray,
    lakes,
    positions,
    chip: int = CHIP,
    background_stride: int = STRIDE_BG,
    background_per: int = 25,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[LakeChip]]:
    """Extract lake-centred 6-ch chips plus sparse background negatives.

    Args:
        tensor6: (6, H, W) Kuro Siwo tensor.
        label: (H, W) lake-polygon mask from ``rasterize_lake_labels``.
        lakes: GeoDataFrame (for ID/elev/area columns).
        positions: (idx, row, col) centroids inside the grid.
        chip: square chip side (grid px).
        background_stride: grid stride for background chip candidates.
        background_per: max number of no-lake chips (stratified sample).
        rng_seed: seed for background sampling (Hard Rule 6).

    Returns:
        (x, y, chips): float32 (N, 6, chip, chip), uint8 (N, chip, chip),
        and the LakeChip manifest list.
    """
    h, w = label.shape
    half = chip // 2
    x_l, y_l, chips = [], [], []

    def _emit(r0: int, c0: int, meta: LakeChip) -> None:
        x_l.append(tensor6[:, r0:r0 + chip, c0:c0 + chip])
        y_l.append(label[r0:r0 + chip, c0:c0 + chip])
        chips.append(meta)

    for n, (idx, r, c) in enumerate(positions):
        r0, c0 = r - half, c - half
        if r0 < 0 or c0 < 0 or r0 + chip > h or c0 + chip > w:
            continue  # edge-clipped chips skipped — full context required
        lake = lakes.iloc[idx]
        pos = int(label[r0:r0 + chip, c0:c0 + chip].sum())
        _emit(r0, c0, LakeChip(
            chip_id=f"lake_{n:05d}",
            lake_id=str(lake.get("ID", f"idx{idx}")),
            lake_elev_m=float(lake.get("Lake_Elev", 0)),
            lake_area_km2=float(lake.get("area_km2", lake.get("Area", 0.0))),
            lon=float(lake["Longitude"]),
            lat=float(lake["Latitude"]),
            row=r, col=c, pos_px=pos,
        ))

    # Background negatives: lake-free chips on a strided grid, sampled so
    # terrain variety (glacier/rock/slope) is preserved.
    rng = np.random.RandomState(rng_seed)
    bg_candidates = []
    for r0 in range(0, h - chip + 1, background_stride):
        for c0 in range(0, w - chip + 1, background_stride):
            if label[r0:r0 + chip, c0:c0 + chip].sum() == 0:
                bg_candidates.append((r0, c0))
    rng.shuffle(bg_candidates)
    for n, (r0, c0) in enumerate(bg_candidates[:background_per]):
        _emit(r0, c0, LakeChip(
            chip_id=f"bg_{n:04d}",
            lake_id="", lake_elev_m=0.0, lake_area_km2=0.0,
            lon=0.0, lat=0.0, row=r0 + half, col=c0 + half,
            pos_px=0, kind="background",
        ))

    if not x_l:
        return (
            np.empty((0, 6, chip, chip), np.float32),
            np.empty((0, chip, chip), np.uint8),
            [],
        )
    return np.stack(x_l).astype(np.float32), np.stack(y_l).astype(np.uint8), chips


def build_chip_dataset(
    pre_cache: Path | str,
    post_cache: Path | str,
    inventory_path: Path | str | None = None,
    min_elev_m: float = 4000.0,
    min_area_km2: float = 0.02,
    chip: int = CHIP,
    background_per: int = 25,
    out_dir: Path | str | None = None,
) -> dict:
    """Build the Himalayan lake chip dataset end-to-end.

    Args:
        pre_cache: calibrated t0 VV/VH dB raster (with GCPs).
        post_cache: calibrated t1 VV/VH dB raster (with GCPs).
        inventory_path: lake inventory shapefile (default: 2022 inventory).
        min_elev_m / min_area_km2: lake filters.
        chip: chip side in grid px.
        background_per: background chips to include.
        out_dir: where to write chips.npz + manifest + report
            (default: ``data/datasets/himalayan_lake_chips/``).

    Returns:
        Stats/report dict (also written to ``report.json``).
    """
    import rasterio

    from siren.ml.contract import build_kuro_siwo_tensor

    out = Path(out_dir) if out_dir else (
        _REPO_ROOT / "data" / "datasets" / "himalayan_lake_chips"
    )
    out.mkdir(parents=True, exist_ok=True)

    lakes = load_lake_inventory(
        inventory_path, min_elev_m=min_elev_m, min_area_km2=min_area_km2
    )
    logger.info(f"Lake inventory: {len(lakes)} lakes after filters")

    positions, lon_g, lat_g = lake_grid_positions(lakes, str(post_cache))
    logger.info(f"{len(positions)} lake centroids inside the SAR grid")

    label = rasterize_lake_labels(lakes, positions, str(post_cache))

    with rasterio.open(pre_cache) as src:
        pre_db = src.read().astype(np.float32)
    with rasterio.open(post_cache) as src:
        post_db = src.read().astype(np.float32)
    if pre_db.shape != post_db.shape:
        raise ValueError(
            f"pre/post grid mismatch: {pre_db.shape} vs {post_db.shape} — "
            "same-orbit aligned pair required"
        )
    tensor6 = build_kuro_siwo_tensor(pre_db[:2], post_db[:2])

    x, y, chips = extract_lake_chips(
        tensor6, label, lakes, positions, chip=chip,
        background_per=background_per,
    )

    manifest = [
        {
            "chip_id": c.chip_id, "lake_id": c.lake_id, "kind": c.kind,
            "lake_elev_m": c.lake_elev_m, "lake_area_km2": c.lake_area_km2,
            "lon": c.lon, "lat": c.lat, "row": c.row, "col": c.col,
            "pos_px": c.pos_px,
        }
        for c in chips
    ]
    np.savez_compressed(out / "chips.npz", x=x, y=y)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    lake_chips = [c for c in chips if c.kind == "lake"]
    pos_counts = [c.pos_px for c in lake_chips]
    elevs = [c.lake_elev_m for c in lake_chips]
    areas = [c.lake_area_km2 for c in lake_chips]
    report = {
        "inventory": str(inventory_path or DEFAULT_INVENTORY_PATH),
        "scenes": {
            "t0": Path(pre_cache).name,
            "t1": Path(post_cache).name,
            "contract": "kuro_siwo_6ch",
        },
        "label_semantics": (
            "weak_positive: median-outlined inventory polygon "
            "(2022–2024 composite), not per-pixel scene-date water"
        ),
        "filters": {"min_elev_m": min_elev_m, "min_area_km2": min_area_km2},
        "lakes_after_filters": len(lakes),
        "lake_centroids_in_grid": len(positions),
        "chips_total": len(chips),
        "lake_chips": len(lake_chips),
        "background_chips": len(chips) - len(lake_chips),
        "chip_px": chip,
        "pos_px_median": float(np.median(pos_counts)) if pos_counts else 0.0,
        "pos_px_max": int(max(pos_counts)) if pos_counts else 0,
        "chips_with_pos_gt_10px": int(sum(p > 10 for p in pos_counts)),
        "elev_m": {
            "min": float(min(elevs)) if elevs else None,
            "median": float(np.median(elevs)) if elevs else None,
            "max": float(max(elevs)) if elevs else None,
        },
        "area_km2": {
            "min": float(min(areas)) if areas else None,
            "median": float(np.median(areas)) if areas else None,
            "max": float(max(areas)) if areas else None,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    logger.info(
        f"Wrote {len(chips)} chips ({len(lake_chips)} lake + "
        f"{len(chips) - len(lake_chips)} bg) to {out}"
    )
    return report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build Himalayan lake SAR chip dataset")
    ap.add_argument("--pre", required=True, help="t0 calibrated VV/VH dB cache")
    ap.add_argument("--post", required=True, help="t1 calibrated VV/VH dB cache")
    ap.add_argument("--inventory", default=None)
    ap.add_argument("--min-elev", type=float, default=4000.0)
    ap.add_argument("--min-area", type=float, default=0.02)
    ap.add_argument("--chip", type=int, default=CHIP)
    ap.add_argument("--background", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = build_chip_dataset(
        args.pre, args.post,
        inventory_path=args.inventory,
        min_elev_m=args.min_elev, min_area_km2=args.min_area,
        chip=args.chip, background_per=args.background, out_dir=args.out,
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
