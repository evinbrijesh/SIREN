"""Manual Imja shoreline labelling — independent truth for the §17.2 gate.

The segmentation gate's precision leg is contested by weak labels:
inventory polygons are 2022–2024 *median* outlines and SCL carries its
own shoreline bias. A hand-verified label on a near-contemporaneous S2
scene separates model error from label bias.

Workflow:

  1. ``extract_roi`` pulls a ~2.5 km B02/B03/B04/B08/SCL crop around the
     Imja centroid at native 10 m from an S2 L2A SAFE zip and computes
     NDWI at 10 m.
  2. ``render_panel`` writes a side-by-side PNG (true colour | NDWI |
     SCL water | candidate label over RGB) with the inventory polygon
     outline — the image an analyst annotates against.
  3. ``candidate_label`` thresholds NDWI on SCL-clear pixels; the
     analyst then applies include/exclude polygons in ``apply_edits``
     until the boundary matches the visible shoreline.
  4. ``write_label`` persists the final (H, W) uint8 GeoTIFF
     (1 = water, 0 = land, 255 = unlabelled/cloud) — sampled onto the
     SAR grid by ``heldout_eval._optical_labels``-style machinery.

Label provenance is recorded in a JSON sidecar (scene, analyst, NDWI
threshold, edit count) — PRD §17.1 requires label provenance.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, from_bounds

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
IMJA_LON, IMJA_LAT = 86.9282, 27.8983
INVENTORY_SHP = (
    _REPO_ROOT
    / "data"
    / "datasets"
    / "glacial_lake_2022-2024"
    / "Glacial_Lake_2022.shp"
)
OUT_DIR = _REPO_ROOT / "data" / "processed"

ROI_HALF_M = 1250.0   # ~2.5 km box — lake is ~1.1 km long


def _roi_window_utm(band_meta: dict, half_m: float = ROI_HALF_M) -> Window:
    """Pixel window for a half_m box around the Imja centroid in the
    band's native UTM grid."""
    xs, ys = warp_transform(
        "EPSG:4326", band_meta["crs"], [IMJA_LON], [IMJA_LAT]
    )
    cx, cy = xs[0], ys[0]
    win = from_bounds(
        cx - half_m, cy - half_m, cx + half_m, cy + half_m,
        band_meta["transform"],
    )
    return win.round_offsets().round_lengths()


def extract_roi(s2_zip_path: Path | str, half_m: float = ROI_HALF_M) -> dict:
    """Read B02/B03/B04 (10 m) + SCL (20 m, NN-resampled) over the Imja
    ROI and compute 10 m NDWI.

    Returns dict: rgb (3,H,W uint8), ndwi (H,W float32), scl (H,W uint8
    on the 10 m grid), transform, crs.
    """
    from rasterio.warp import reproject
    from rasterio.enums import Resampling

    s2_zip_path = Path(s2_zip_path)
    with zipfile.ZipFile(s2_zip_path) as z:
        def _band(name: str, res: str):
            p = next(
                n for n in z.namelist()
                if "IMG_DATA" in n and n.endswith(f"_{name}_{res}.jp2")
            )
            with z.open(p) as f:
                ds = rasterio.open(f)
                return ds, ds.read(1, window=_roi_window_utm(
                    {"crs": ds.crs, "transform": ds.transform}, half_m
                )), _roi_window_utm(
                    {"crs": ds.crs, "transform": ds.transform}, half_m
                )

        # 10 m bands share the grid — one window for all
        ref_ds = None
        bands = {}
        for b in ("B02", "B03", "B04", "B08"):
            p = next(
                n for n in z.namelist()
                if "IMG_DATA" in n and n.endswith(f"_{b}_10m.jp2")
            )
            with z.open(p) as f:
                ds = rasterio.open(f)
                if ref_ds is None:
                    win = _roi_window_utm(
                        {"crs": ds.crs, "transform": ds.transform}, half_m
                    )
                    ref_ds = ds
                    ref_transform = ds.window_transform(win)
                    ref_crs = ds.crs
                bands[b] = ds.read(1, window=win).astype(np.float32)

        scl_p = next(
            n for n in z.namelist()
            if "IMG_DATA" in n and n.endswith("_SCL_20m.jp2")
        )
        with z.open(scl_p) as f:
            ds = rasterio.open(f)
            scl = np.zeros(bands["B03"].shape, dtype=np.uint8)
            reproject(
                source=rasterio.band(ds, 1),
                destination=scl,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )

    b02, b03, b04, b08 = bands["B02"], bands["B03"], bands["B04"], bands["B08"]
    ndwi = (b03 - b08) / (b03 + b08 + 1e-6)

    def _norm(x):
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        return np.clip((x - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

    rgb = np.stack([_norm(b04), _norm(b03), _norm(b02)])
    return {
        "rgb": rgb,
        "ndwi": ndwi.astype(np.float32),
        "scl": scl,
        "b11": None,
        "transform": ref_transform,
        "crs": ref_crs,
    }


def candidate_label(
    roi: dict, ndwi_threshold: float = 0.15, clear_scl: set | None = None
) -> np.ndarray:
    """uint8 label: 1=water, 0=land, 255=unlabelled (cloud/shadow)."""
    clear = clear_scl or {4, 5, 6, 7, 11}
    valid = np.isin(roi["scl"], list(clear))
    out = np.where(valid, 0, 255).astype(np.uint8)
    out[valid & (roi["ndwi"] > ndwi_threshold)] = 1
    return out


def _imja_inventory_poly(transform):
    """Imja inventory polygon in pixel coords of the ROI grid."""
    import geopandas as gpd

    gdf = gpd.read_file(INVENTORY_SHP)
    gdf = gdf.to_crs("EPSG:4326")
    d = (gdf["Longitude"] - IMJA_LON) ** 2 + (gdf["Latitude"] - IMJA_LAT) ** 2
    geom = gdf.geometry.iloc[int(d.idxmin() if hasattr(d, "idxmin") else d.argmin())]
    geom = gdf.geometry.iloc[int(d.values.argmin())]
    from shapely.geometry import MultiPolygon
    poly = geom.geoms[0] if isinstance(geom, MultiPolygon) else geom
    xs, ys = poly.exterior.xy
    inv = ~transform
    return [tuple(inv * (x, y)) for x, y in zip(xs, ys)]


def render_panel(
    roi: dict, label: np.ndarray, out_png: Path, title: str = ""
) -> Path:
    """4-up PNG: RGB | NDWI | SCL-water | label overlay on RGB, with the
    inventory polygon outline drawn on each tile."""
    from PIL import Image, ImageDraw

    h, w = label.shape
    tiles = []

    def _rgb_base() -> Image.Image:
        return Image.fromarray(np.moveaxis(roi["rgb"], 0, -1))

    # tile 1: true colour
    tiles.append(_rgb_base())

    # tile 2: NDWI grayscale (blue-high)
    nd = roi["ndwi"]
    nd_img = np.clip((nd + 1) / 2 * 255, 0, 255).astype(np.uint8)
    nd_rgb = np.stack([255 - nd_img, 255 - nd_img // 2, 255 - nd_img], -1)
    tiles.append(Image.fromarray(nd_rgb.astype(np.uint8)))

    # tile 3: SCL water (blue) / other-clear (grey) / cloud (black)
    scl_rgb = np.zeros((h, w, 3), np.uint8) + 20
    scl_rgb[np.isin(roi["scl"], [4, 5, 7, 11])] = [110, 110, 110]
    scl_rgb[roi["scl"] == 6] = [30, 120, 255]
    tiles.append(Image.fromarray(scl_rgb))

    # tile 4: label boundary over RGB
    t4 = np.moveaxis(roi["rgb"].copy(), 0, -1)
    water = label == 1
    edge = water & ~np.roll(water, 1, 0) | water & ~np.roll(water, 1, 1)
    t4[edge] = [255, 40, 40]
    t4[label == 255] = (t4[label == 255] * 0.4).astype(np.uint8)
    tiles.append(Image.fromarray(t4))

    panel = Image.new("RGB", (w * 2 + 8, h * 2 + 8), (15, 15, 15))
    for i, t in enumerate(tiles):
        panel.paste(t, ((i % 2) * (w + 8), (i // 2) * (h + 8)))

    # inventory polygon on each tile
    try:
        poly = _imja_inventory_poly(roi["transform"])
        dr = ImageDraw.Draw(panel)
        for i in range(4):
            ox, oy = (i % 2) * (w + 8), (i // 2) * (h + 8)
            pts = [(ox + c, oy + r) for c, r in poly]
            dr.line(pts + [pts[0]], fill=(255, 230, 0), width=1)
    except Exception as exc:
        logger.warning("inventory overlay failed: %s", exc)

    out_png = Path(out_png)
    panel.save(out_png)
    logger.info("wrote %s %s", out_png, title)
    return out_png


def apply_edits(
    label: np.ndarray, transform, include: list | None = None,
    exclude: list | None = None,
) -> np.ndarray:
    """Apply analyst include/exclude polygons (lon/lat vertex lists) to
    the candidate label."""
    from rasterio.features import rasterize
    from shapely.geometry import Polygon

    out = label.copy()
    for polys, val in ((include, 1), (exclude, 0)):
        for ring in polys or []:
            mask = rasterize(
                [(Polygon(ring), 1)], out_shape=out.shape,
                transform=transform, fill=0, dtype="uint8",
            ).astype(bool)
            out[mask] = val
    return out


def write_label(
    label: np.ndarray, roi: dict, out_tif: Path, provenance: dict
) -> Path:
    """Persist label GeoTIFF + JSON provenance sidecar."""
    out_tif = Path(out_tif)
    profile = {
        "driver": "GTiff", "height": label.shape[0], "width": label.shape[1],
        "count": 1, "dtype": "uint8", "crs": roi["crs"],
        "transform": roi["transform"], "nodata": 255,
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(label, 1)
    out_tif.with_suffix(out_tif.suffix + ".json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    logger.info("wrote %s", out_tif)
    return out_tif
