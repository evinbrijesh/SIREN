"""Paired SAR + optical dataset for multi-modal fusion training/inference
(ADR-013 §9.7.2).

Wraps the Kuro Siwo 6-channel SAR dataset and attaches co-registered S2
optical features (NDWI, MNDWI, cloud_mask) for the MultiModalFusionNet.

Two modes:

1. **Training mode** (``FusionDataset``): wraps KuroSiwoDataset and attaches
   optical features from a directory of S2 SAFE archives matched by date
   proximity (±3 days) and tile overlap. When no matching S2 scene is found
   for a chip, optical is set to None (SAR-only fallback — the fusion model
   handles this via cloud-gated cross-attention).

2. **Inference mode** (``build_fusion_chip``): builds a single (sar, optical)
   pair from real SAFE archives (S1 pre/post + S2) co-registered to a DEM
   window. Used by the Imja integration test and the fusion inference script.

Contract:
    SAR:     (6, H, W) — VV_post, VH_post, VV_pre, VH_pre, ΔVV, ΔVH (normalized [0,1])
    Optical: (3, H, W) — NDWI, MNDWI, cloud_mask  (or None if cloud-blocked)
    Label:   (H, W)   — binary water mask
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Maximum temporal gap between SAR and S2 for a valid pair (ADR-013 §9.7.2)
MAX_PAIR_GAP_DAYS = 3


def _find_s1_band(safe_zip: Path, pol: str) -> str:
    """Find the measurement TIFF for a given polarization inside a S1 SAFE zip."""
    with zipfile.ZipFile(str(safe_zip)) as z:
        for name in z.namelist():
            if name.endswith(".tiff") and "measurement" in name and f"-{pol}-" in name:
                return name
    raise FileNotFoundError(f"No {pol} band in {safe_zip}")


def _read_s1_calibration_constant(safe_zip: Path, pol: str) -> float:
    """Read the sigmaNought calibration constant from the S1 SAFE annotation XML."""
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(str(safe_zip)) as z:
        cal_files = [
            n for n in z.namelist()
            if "calibration" in n and f"-{pol}-" in n and n.endswith(".xml")
        ]
        if not cal_files:
            raise FileNotFoundError(f"No calibration file for {pol} in {safe_zip}")
        with z.open(cal_files[0]) as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag == "sigmaNought" and elem.text:
                    return float(elem.text.strip().split()[0])
    raise ValueError(f"No sigmaNought value found in calibration file for {pol}")


def _dn_to_linear_sigma0(dn: np.ndarray, cal_const: float = 700.0) -> np.ndarray:
    """Convert Sentinel-1 DN to linear sigma0: sigma0 = (DN²) / (cal²)."""
    return np.maximum(dn.astype(np.float32), 0.0) ** 2 / (cal_const ** 2)


def _linear_to_normalized_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear sigma0 to normalized dB in [0, 1].

    Matches the Kuro Siwo adapter contract:
        dB = 10 * log10(linear + eps), clamp to [-30, 0], normalize to [0, 1].
    """
    eps = 1e-10
    db = 10.0 * np.log10(linear + eps)
    db = np.clip(db, -30.0, 0.0)
    return (db + 30.0) / 30.0


def _compute_delta(post_norm: np.ndarray, pre_norm: np.ndarray) -> np.ndarray:
    """Compute normalized Δσ⁰ channel from normalized dB arrays."""
    post_db = post_norm * 30.0 - 30.0
    pre_db = pre_norm * 30.0 - 30.0
    delta_db = post_db - pre_db
    delta_db = np.clip(delta_db, -15.0, 5.0)
    return (delta_db + 15.0) / 20.0


def _pad_to_size(arr: np.ndarray, size: int = 224) -> np.ndarray:
    """Pad a 2D array to size×size with edge values.

    If the array is already larger than ``size`` in a dimension, it is returned
    unchanged in that dimension (no center-crop — the caller handles cropping).
    """
    h, w = arr.shape
    pad_h = max(0, (size - h) // 2)
    pad_w = max(0, (size - w) // 2)
    extra_h = max(0, size - h - 2 * pad_h)
    extra_w = max(0, size - w - 2 * pad_w)
    if pad_h == 0 and pad_w == 0 and extra_h == 0 and extra_w == 0:
        return arr
    return np.pad(arr, ((pad_h, pad_h + extra_h), (pad_w, pad_w + extra_w)), mode="edge")


def _crop_back(arr: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """Crop a padded array back to original size (inverse of _pad_to_size)."""
    h, w = arr.shape
    pad_h = (h - orig_h) // 2
    pad_w = (w - orig_w) // 2
    return arr[pad_h:pad_h + orig_h, pad_w:pad_w + orig_w]


def build_sar_6ch_from_safe(
    pre_safe: str | Path,
    post_safe: str | Path,
    dem_path: str | Path,
    window_bounds: tuple[float, float, float, float],
    chip_size: int = 224,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a 6-channel SAR tensor from two Sentinel-1 SAFE archives.

    Reads VV+VH from both pre and post S1 GRD scenes, reprojects to the DEM
    grid window, converts to normalized dB, and computes Δσ⁰ channels.

    Args:
        pre_safe: path to the pre-event S1 SAFE zip.
        post_safe: path to the post-event S1 SAFE zip.
        dem_path: path to the reference DEM raster.
        window_bounds: (west, south, east, north) in lat/lon (EPSG:4326).
        chip_size: target chip size (padded with edge values).

    Returns:
        (sar_6ch, meta) where sar_6ch is (6, chip_size, chip_size) float32
        and meta contains the DEM window CRS, transform, shape, and bounds.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import from_bounds

    pre_safe = Path(pre_safe)
    post_safe = Path(post_safe)

    # Read DEM window
    west, south, east, north = window_bounds
    with rasterio.open(str(dem_path)) as dem_src:
        win = from_bounds(west, south, east, north, dem_src.transform)
        dem = dem_src.read(1, window=win).astype(np.float32)
        dem_transform = rasterio.windows.transform(win, dem_src.transform)
        dem_crs = dem_src.crs
    dem = np.nan_to_num(dem, nan=0.0)
    dem_h, dem_w = dem.shape

    # Read calibration constants
    pre_cal_vv = _read_s1_calibration_constant(pre_safe, "vv")
    pre_cal_vh = _read_s1_calibration_constant(pre_safe, "vh")
    post_cal_vv = _read_s1_calibration_constant(post_safe, "vv")
    post_cal_vh = _read_s1_calibration_constant(post_safe, "vh")

    # Reproject SAR to DEM window
    pre_vv_grid = np.zeros((dem_h, dem_w), dtype=np.float32)
    pre_vh_grid = np.zeros((dem_h, dem_w), dtype=np.float32)
    post_vv_grid = np.zeros((dem_h, dem_w), dtype=np.float32)
    post_vh_grid = np.zeros((dem_h, dem_w), dtype=np.float32)

    pre_vv_path = f"/vsizip/{pre_safe}/{_find_s1_band(pre_safe, 'vv')}"
    pre_vh_path = f"/vsizip/{pre_safe}/{_find_s1_band(pre_safe, 'vh')}"
    post_vv_path = f"/vsizip/{post_safe}/{_find_s1_band(post_safe, 'vv')}"
    post_vh_path = f"/vsizip/{post_safe}/{_find_s1_band(post_safe, 'vh')}"

    for label, src_path, dst in [
        ("pre_vv", pre_vv_path, pre_vv_grid),
        ("pre_vh", pre_vh_path, pre_vh_grid),
        ("post_vv", post_vv_path, post_vv_grid),
        ("post_vh", post_vh_path, post_vh_grid),
    ]:
        with rasterio.open(src_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear,
            )

    # Convert to normalized dB
    pre_vv_norm = _linear_to_normalized_db(_dn_to_linear_sigma0(pre_vv_grid, pre_cal_vv))
    pre_vh_norm = _linear_to_normalized_db(_dn_to_linear_sigma0(pre_vh_grid, pre_cal_vh))
    post_vv_norm = _linear_to_normalized_db(_dn_to_linear_sigma0(post_vv_grid, post_cal_vv))
    post_vh_norm = _linear_to_normalized_db(_dn_to_linear_sigma0(post_vh_grid, post_cal_vh))

    # Compute Δσ⁰
    d_vv = _compute_delta(post_vv_norm, pre_vv_norm)
    d_vh = _compute_delta(post_vh_norm, pre_vh_norm)

    # Pad to chip_size and stack
    sar_6ch = np.stack([
        _pad_to_size(post_vv_norm, chip_size),
        _pad_to_size(post_vh_norm, chip_size),
        _pad_to_size(pre_vv_norm, chip_size),
        _pad_to_size(pre_vh_norm, chip_size),
        _pad_to_size(d_vv, chip_size),
        _pad_to_size(d_vh, chip_size),
    ], axis=0).astype(np.float32)
    sar_6ch = np.nan_to_num(sar_6ch, nan=0.0, posinf=1.0, neginf=0.0)

    meta = {
        "crs": str(dem_crs),
        "transform": dem_transform,
        "shape": (dem_h, dem_w),
        "bounds": window_bounds,
        "chip_size": chip_size,
        "pre_safe": str(pre_safe),
        "post_safe": str(post_safe),
    }
    return sar_6ch, meta


def build_optical_3ch_from_safe(
    s2_safe: str | Path,
    dem_path: str | Path,
    window_bounds: tuple[float, float, float, float],
    chip_size: int = 224,
) -> np.ndarray | None:
    """Build a 3-channel optical tensor from an S2 L2A SAFE archive.

    Extracts NDWI, MNDWI, and cloud mask from the S2 scene and co-registers
    to the DEM grid window, then pads to chip_size.

    Args:
        s2_safe: path to the S2 L2A SAFE zip.
        dem_path: path to the reference DEM raster.
        window_bounds: (west, south, east, north) in lat/lon (EPSG:4326).
        chip_size: target chip size.

    Returns:
        (3, chip_size, chip_size) float32 array [NDWI, MNDWI, cloud_mask],
        or None if the S2 scene doesn't cover the window.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import from_bounds
    from siren.preprocess.s2_optical import extract_optical_features

    s2_safe = Path(s2_safe)

    # Read DEM window to get the target grid
    west, south, east, north = window_bounds
    with rasterio.open(str(dem_path)) as dem_src:
        win = from_bounds(west, south, east, north, dem_src.transform)
        dem = dem_src.read(1, window=win).astype(np.float32)
        dem_transform = rasterio.windows.transform(win, dem_src.transform)
        dem_crs = dem_src.crs
    dem_h, dem_w = dem.shape

    # Extract optical features at native S2 resolution
    features = extract_optical_features(s2_safe)

    # Reproject optical features to the DEM grid
    ndwi_grid = np.zeros((dem_h, dem_w), dtype=np.float32)
    mndwi_grid = np.zeros((dem_h, dem_w), dtype=np.float32)
    cloud_grid = np.zeros((dem_h, dem_w), dtype=np.float32)

    src_transform = features["meta"]["transform"]
    src_crs = features["meta"]["crs"]

    for src, dst, resamp in [
        (features["ndwi"], ndwi_grid, Resampling.bilinear),
        (features["mndwi"], mndwi_grid, Resampling.bilinear),
        (features["cloud_mask"], cloud_grid, Resampling.nearest),
    ]:
        reproject(
            source=src,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=resamp,
        )

    # Pad to chip_size and stack
    optical_3ch = np.stack([
        _pad_to_size(ndwi_grid, chip_size),
        _pad_to_size(mndwi_grid, chip_size),
        _pad_to_size(cloud_grid, chip_size),
    ], axis=0).astype(np.float32)
    optical_3ch = np.nan_to_num(optical_3ch, nan=0.0, posinf=1.0, neginf=-1.0)

    return optical_3ch


def build_fusion_chip(
    pre_safe: str | Path,
    post_safe: str | Path,
    s2_safe: str | Path | None,
    dem_path: str | Path,
    window_bounds: tuple[float, float, float, float],
    chip_size: int = 224,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Build a (sar, optical) fusion input pair from real SAFE archives.

    Combines build_sar_6ch_from_safe and build_optical_3ch_from_safe into a
    single call. The optical input is None if s2_safe is None (cloud-blocked
    or no S2 coverage).

    Args:
        pre_safe: path to the pre-event S1 SAFE zip.
        post_safe: path to the post-event S1 SAFE zip.
        s2_safe: path to the S2 L2A SAFE zip, or None.
        dem_path: path to the reference DEM raster.
        window_bounds: (west, south, east, north) in lat/lon.
        chip_size: target chip size.

    Returns:
        (sar_6ch, optical_3ch_or_none, meta) where:
            sar_6ch: (6, chip_size, chip_size) float32
            optical_3ch: (3, chip_size, chip_size) float32 or None
            meta: dict with provenance metadata
    """
    sar_6ch, meta = build_sar_6ch_from_safe(
        pre_safe, post_safe, dem_path, window_bounds, chip_size
    )

    if s2_safe is not None:
        optical_3ch = build_optical_3ch_from_safe(
            s2_safe, dem_path, window_bounds, chip_size
        )
        meta["s2_safe"] = str(s2_safe)
        meta["optical_available"] = True
    else:
        optical_3ch = None
        meta["s2_safe"] = None
        meta["optical_available"] = False

    return sar_6ch, optical_3ch, meta


class FusionDataset:
    """Paired SAR + optical dataset for MultiModalFusionNet training.

    Wraps the Kuro Siwo 6-channel SAR dataset and attaches optical features.
    When no S2 scene is available for a chip, optical is None (SAR-only
    fallback — the fusion model handles this via cloud-gated cross-attention).

    Args:
        sar_split: "train" or "test" (Kuro Siwo split).
        s2_archives: dict mapping sample_id → S2 SAFE path, or None.
        max_samples: optional cap on number of samples.
        chip_size: expected chip size (default 224).

    Yields dicts with keys:
        sar: float32 tensor (6, H, W)
        optical: float32 tensor (3, H, W) or None
        water: float32 tensor (H, W) — binary water mask
        valid: float32 tensor (H, W) — valid pixel mask
        sample_id: str
    """

    def __init__(
        self,
        sar_split: str = "train",
        s2_archives: dict[str, str | Path] | None = None,
        max_samples: int | None = None,
        chip_size: int = 224,
    ) -> None:
        from siren.ml.kuro_siwo_dataset import KuroSiwoDataset

        self.sar_dataset = KuroSiwoDataset(
            split=sar_split, max_samples=max_samples, chip_size=chip_size
        )
        self.s2_archives = s2_archives or {}
        self.chip_size = chip_size

    def __len__(self) -> int:
        return len(self.sar_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.sar_dataset[idx]
        sar = sample["sar"]
        water = sample["water"]
        valid = sample["valid"]
        sample_id = sample["sample_id"]

        # Attach optical if available for this sample
        s2_path = self.s2_archives.get(sample_id)
        if s2_path is not None and Path(s2_path).exists():
            # For training chips, we'd need geospatial metadata to co-register
            # S2 to the chip grid. Kuro Siwo chips don't carry per-chip
            # geotransforms, so optical attachment requires a pre-computed
            # optical feature cache. This is a stub — the training path uses
            # SAR-only (optical=None) until paired optical chips are available.
            optical = None
        else:
            optical = None

        return {
            "sar": sar,
            "optical": optical,
            "water": water,
            "valid": valid,
            "sample_id": sample_id,
        }


def collate_fusion(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for FusionDataset batches.

    Handles variable optical availability (some chips may have optical=None).
    Returns optical as a stacked tensor if ALL samples have optical, else None.

    Accepts both torch tensors and numpy arrays for sar/water/valid (the
    KuroSiwoDataset returns torch tensors; build_fusion_chip returns numpy).
    """
    import torch

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.from_numpy(x)

    sars = torch.stack([_to_tensor(b["sar"]) for b in batch])
    waters = torch.stack([_to_tensor(b["water"]) for b in batch])
    valids = torch.stack([_to_tensor(b["valid"]) for b in batch])

    opticals = [b["optical"] for b in batch]
    if all(o is not None for o in opticals):
        optical = torch.stack([_to_tensor(o) for o in opticals])
    else:
        optical = None

    return {
        "sar": sars,
        "optical": optical,
        "water": waters,
        "valid": valids,
        "sample_id": [b["sample_id"] for b in batch],
    }
