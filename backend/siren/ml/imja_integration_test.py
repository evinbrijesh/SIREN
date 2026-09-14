"""End-to-end integration test: Imja Lake Sentinel-1 pair through the
6-channel Kuro Siwo WaterResUNet → breach volume estimation.

This script validates the full pipeline on real data:
  1. Read VV+VH from both Sentinel-1 SAFE zips (pre: 2026-07-02, post: 2026-07-14)
  2. Crop to a ~5×5 km window around Imja Tsho (~27.701N, 86.928E)
  3. Reproject SAR to the DEM grid at ~30m resolution
  4. Pad to 224×224 (model's expected chip size)
  5. Run the v1 model (τ=0.30) on the single chip
  6. Feed pre/post water masks to breach_volume.estimate_breach_volume()

Usage:
    python -m siren.ml.imja_integration_test
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

# --- Paths ---
PRE_SAFE = REPO_ROOT / "data/raw/S1D_IW_GRDH_1SDV_20260702T001034_20260702T001059_003487_0062AB_5083.SAFE.zip"
POST_SAFE = REPO_ROOT / "data/raw/S1D_IW_GRDH_1SDV_20260714T001035_20260714T001100_003662_00689F_377F.SAFE.zip"
DEM_PATH = REPO_ROOT / "data/raw/srtm_30m.tif"
CKPT_PATH = REPO_ROOT / "models/checkpoints/water_resunet_kuro_siwo_full/water_resunet_6ch_kuro_siwo_v1.pt"

CHIP_SIZE = 224
THRESHOLD = 0.30  # ADR-011.1 calibrated threshold

# Imja Tsho coordinates (ICIMOD literature: 27°54'N, 86°55'E ≈ 27.90°N, 86.92°E)
# Lake surface elevation ~5004-5010 m a.s.l.
IMJA_LAT = 27.90
IMJA_LON = 86.92
HALF_SIZE = 0.020  # ~2.2 km half-width = 4.4 km window (tight enough to exclude valley rivers)

# Elevation sanity bounds at the lake centroid (no-silent-fallbacks policy)
IMJA_ELEV_MIN = 4950.0
IMJA_ELEV_MAX = 5050.0


def _find_band(safe_zip: Path, pol: str) -> str:
    """Find the measurement TIFF for a given polarization inside a SAFE zip."""
    with zipfile.ZipFile(safe_zip) as z:
        for name in z.namelist():
            if name.endswith(".tiff") and "measurement" in name and f"-{pol}-" in name:
                return name
    raise FileNotFoundError(f"No {pol} band in {safe_zip}")


def _read_calibration_constant(safe_zip: Path, pol: str) -> float:
    """Read the sigmaNought calibration constant from the SAFE annotation XML.

    For Sentinel-1 GRD: sigma0 = (DN²) / (sigmaNought²)
    """
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(safe_zip) as z:
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
    raise ValueError(f"No sigmaNought value found in calibration file")


def _dn_to_linear_sigma0(dn: np.ndarray, cal_const: float = 700.0) -> np.ndarray:
    """Convert Sentinel-1 DN to linear sigma0.

    For GRD products: sigma0 = (DN²) / (calibration_constant²)
    """
    return np.maximum(dn.astype(np.float32), 0.0) ** 2 / (cal_const ** 2)


def _linear_to_normalized_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear sigma0 to normalized dB in [0, 1].

    Matches the Kuro Siwo adapter:
        dB = 10 * log10(linear + eps)
        clamp to [-30, 0]
        normalize to [0, 1]
    """
    eps = 1e-10
    db = 10.0 * np.log10(linear + eps)
    db = np.clip(db, -30.0, 0.0)
    return (db + 30.0) / 30.0


def _compute_delta(post_norm: np.ndarray, pre_norm: np.ndarray) -> np.ndarray:
    """Compute normalized Δσ⁰ channel.

    Matches the Kuro Siwo adapter:
        delta_db = post_db - pre_db
        clamp to [-15, 5]
        normalize to [0, 1]
    """
    post_db = post_norm * 30.0 - 30.0
    pre_db = pre_norm * 30.0 - 30.0
    delta_db = post_db - pre_db
    delta_db = np.clip(delta_db, -15.0, 5.0)
    return (delta_db + 15.0) / 20.0


def _pad_to_224(arr: np.ndarray) -> np.ndarray:
    """Pad a 2D array to 224×224 with edge values."""
    h, w = arr.shape
    pad_h = (CHIP_SIZE - h) // 2
    pad_w = (CHIP_SIZE - w) // 2
    extra_h = CHIP_SIZE - h - 2 * pad_h
    extra_w = CHIP_SIZE - w - 2 * pad_w
    return np.pad(arr, ((pad_h, pad_h + extra_h), (pad_w, pad_w + extra_w)), mode="edge")


def _crop_back(arr: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """Crop a 224×224 array back to original size (inverse of _pad_to_224)."""
    h, w = arr.shape
    pad_h = (h - orig_h) // 2
    pad_w = (w - orig_w) // 2
    return arr[pad_h:pad_h + orig_h, pad_w:pad_w + orig_w]


def run_imja_integration() -> dict:
    """Run the full Imja integration test."""
    import torch
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import from_bounds
    from siren.ml.model import WaterResUNet
    from siren.risk.breach_volume import estimate_breach_volume

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- 1. Load model ---
    logger.info("Loading 6-channel WaterResUNet v1 (ADR-011.1 gate-passed)...")
    model = WaterResUNet(in_channels=6, base_channels=32).to(device)
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    logger.info("Model loaded: %d params", sum(p.numel() for p in model.parameters()))

    # --- 2. Read DEM window around Imja ---
    logger.info("Reading DEM window around Imja Tsho (%.4f, %.4f)...", IMJA_LAT, IMJA_LON)
    with rasterio.open(DEM_PATH) as dem_src:
        win = from_bounds(
            IMJA_LON - HALF_SIZE, IMJA_LAT - HALF_SIZE,
            IMJA_LON + HALF_SIZE, IMJA_LAT + HALF_SIZE,
            dem_src.transform,
        )
        dem = dem_src.read(1, window=win).astype(np.float32)
        dem_transform = rasterio.windows.transform(win, dem_src.transform)
        dem_crs = dem_src.crs
        dem_res = dem_src.res
    dem = np.nan_to_num(dem, nan=0.0)
    dem_h, dem_w = dem.shape
    logger.info("DEM window: %dx%d, range [%.1f, %.1f] m", dem_w, dem_h, dem.min(), dem.max())

    # --- 2b. Elevation sanity check at centroid (no-silent-fallbacks policy) ---
    cy, cx = dem_h // 2, dem_w // 2
    centroid_elev = float(dem[cy, cx])
    logger.info("DEM elevation at centroid (pixel %d,%d): %.1f m", cy, cx, centroid_elev)
    if not (IMJA_ELEV_MIN <= centroid_elev <= IMJA_ELEV_MAX):
        raise ValueError(
            f"DEM elevation at centroid ({centroid_elev:.1f} m) is outside the "
            f"expected range for Imja Tsho [{IMJA_ELEV_MIN}, {IMJA_ELEV_MAX}] m. "
            f"The extraction window is misregistered — check IMJA_LAT/IMJA_LON. "
            f"(Current: {IMJA_LAT}, {IMJA_LON})"
        )

    # --- 3. Read calibration constants ---
    pre_cal_vv = _read_calibration_constant(PRE_SAFE, "vv")
    pre_cal_vh = _read_calibration_constant(PRE_SAFE, "vh")
    post_cal_vv = _read_calibration_constant(POST_SAFE, "vv")
    post_cal_vh = _read_calibration_constant(POST_SAFE, "vh")
    logger.info("Calibration: pre_vv=%.2f pre_vh=%.2f post_vv=%.2f post_vh=%.2f",
                pre_cal_vv, pre_cal_vh, post_cal_vv, post_cal_vh)

    # --- 4. Reproject SAR to DEM window ---
    logger.info("Reprojecting SAR to DEM window...")
    pre_vv_grid = np.zeros((dem_h, dem_w), dtype=np.float32)
    pre_vh_grid = np.zeros((dem_h, dem_w), dtype=np.float32)
    post_vv_grid = np.zeros((dem_h, dem_w), dtype=np.float32)
    post_vh_grid = np.zeros((dem_h, dem_w), dtype=np.float32)

    pre_vv_path = f"/vsizip/{PRE_SAFE}/{_find_band(PRE_SAFE, 'vv')}"
    pre_vh_path = f"/vsizip/{PRE_SAFE}/{_find_band(PRE_SAFE, 'vh')}"
    post_vv_path = f"/vsizip/{POST_SAFE}/{_find_band(POST_SAFE, 'vv')}"
    post_vh_path = f"/vsizip/{POST_SAFE}/{_find_band(POST_SAFE, 'vh')}"

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
    logger.info("Reprojected SAR grids: %s", pre_vv_grid.shape)

    # --- 5. Convert to normalized dB ---
    logger.info("Converting to normalized dB (with sigma0 calibration)...")
    pre_vv_norm = _linear_to_normalized_db(_dn_to_linear_sigma0(pre_vv_grid, pre_cal_vv))
    pre_vh_norm = _linear_to_normalized_db(_dn_to_linear_sigma0(pre_vh_grid, pre_cal_vh))
    post_vv_norm = _linear_to_normalized_db(_dn_to_linear_sigma0(post_vv_grid, post_cal_vv))
    post_vh_norm = _linear_to_normalized_db(_dn_to_linear_sigma0(post_vh_grid, post_cal_vh))
    logger.info("  VV_post range: [%.4f, %.4f]", post_vv_norm.min(), post_vv_norm.max())
    logger.info("  VH_post range: [%.4f, %.4f]", post_vh_norm.min(), post_vh_norm.max())

    # --- 6. Compute Δσ⁰ channels ---
    logger.info("Computing Δσ⁰ channels...")
    d_vv = _compute_delta(post_vv_norm, pre_vv_norm)
    d_vh = _compute_delta(post_vh_norm, pre_vh_norm)

    # --- 7. Pad to 224×224 and run inference ---
    logger.info("Padding to %dx%d and running inference (τ=%.2f)...", CHIP_SIZE, CHIP_SIZE, THRESHOLD)

    # Post-event water mask: (VV_post, VH_post, VV_pre, VH_pre, ΔVV, ΔVH)
    post_chip = np.stack([
        _pad_to_224(post_vv_norm),
        _pad_to_224(post_vh_norm),
        _pad_to_224(pre_vv_norm),
        _pad_to_224(pre_vh_norm),
        _pad_to_224(d_vv),
        _pad_to_224(d_vh),
    ], axis=0)
    post_chip = np.nan_to_num(post_chip, nan=0.0, posinf=1.0, neginf=0.0)

    # Pre-event water mask: swap roles (pre becomes "post")
    d_vv_rev = _compute_delta(pre_vv_norm, post_vv_norm)
    d_vh_rev = _compute_delta(pre_vh_norm, post_vh_norm)
    pre_chip = np.stack([
        _pad_to_224(pre_vv_norm),
        _pad_to_224(pre_vh_norm),
        _pad_to_224(post_vv_norm),
        _pad_to_224(post_vh_norm),
        _pad_to_224(d_vv_rev),
        _pad_to_224(d_vh_rev),
    ], axis=0)
    pre_chip = np.nan_to_num(pre_chip, nan=0.0, posinf=1.0, neginf=0.0)

    with torch.no_grad():
        x_post = torch.from_numpy(post_chip).float().unsqueeze(0).to(device)
        logits_post = model(x_post)
        probs_post = torch.sigmoid(logits_post).cpu().numpy()[0, 0]

        x_pre = torch.from_numpy(pre_chip).float().unsqueeze(0).to(device)
        logits_pre = model(x_pre)
        probs_pre = torch.sigmoid(logits_pre).cpu().numpy()[0, 0]

    # Crop back to original DEM window size
    post_water_raw = _crop_back(probs_post > THRESHOLD, dem_h, dem_w)
    pre_water_raw = _crop_back(probs_pre > THRESHOLD, dem_h, dem_w)

    pixel_area_m2 = float(dem_res[0] * dem_res[1] * 111000 * 111000)  # deg → m²
    logger.info("Post-event water pixels (raw): %d (%.3f km²)", post_water_raw.sum(), post_water_raw.sum() * pixel_area_m2 / 1e6)
    logger.info("Pre-event water pixels (raw): %d (%.3f km²)", pre_water_raw.sum(), pre_water_raw.sum() * pixel_area_m2 / 1e6)

    # --- 7b. Connected-component filtering: isolate the lake containing the centroid ---
    # Without this, fragmented water bodies (rivers, tarns, valley shadows) are
    # treated as a single lake basin, producing nonsensical hypsometric volumes.
    from scipy.ndimage import label as cc_label

    cy, cx = dem_h // 2, dem_w // 2  # centroid pixel = Imja Tsho center

    def _isolate_target_lake(water_mask: np.ndarray, target_y: int, target_x: int, label: str) -> np.ndarray:
        """Keep only the connected component containing the target pixel."""
        if not water_mask[target_y, target_x]:
            # The target pixel isn't classified as water — find the nearest
            # connected component to the centroid and use that, with a warning.
            labeled, n_features = cc_label(water_mask)
            if n_features == 0:
                logger.warning("%s: no water pixels found at all — returning empty mask", label)
                return water_mask
            # Find the nearest non-zero component to the centroid
            ys, xs = np.nonzero(labeled)
            if len(ys) == 0:
                return water_mask
            dists = (ys - target_y) ** 2 + (xs - target_x) ** 2
            nearest = labeled[ys[dists.argmin()], xs[dists.argmin()]]
            logger.warning(
                "%s: target pixel (%d,%d) is not water; using nearest component (label %d, "
                "distance %d px)",
                label, target_y, target_x, nearest, int(np.sqrt(dists.min())),
            )
            return labeled == nearest
        labeled, n_features = cc_label(water_mask)
        target_label = labeled[target_y, target_x]
        isolated = labeled == target_label
        logger.info(
            "%s: isolated target lake (component %d of %d): %d → %d pixels (%.3f → %.3f km²)",
            label, target_label, n_features,
            water_mask.sum(), isolated.sum(),
            water_mask.sum() * pixel_area_m2 / 1e6,
            isolated.sum() * pixel_area_m2 / 1e6,
        )
        return isolated

    post_water = _isolate_target_lake(post_water_raw, cy, cx, "post")
    pre_water = _isolate_target_lake(pre_water_raw, cy, cx, "pre")
    logger.info("Post-event lake pixels (filtered): %d (%.3f km²)", post_water.sum(), post_water.sum() * pixel_area_m2 / 1e6)
    logger.info("Pre-event lake pixels (filtered): %d (%.3f km²)", pre_water.sum(), pre_water.sum() * pixel_area_m2 / 1e6)

    # --- 8. Estimate breach volume ---
    logger.info("Estimating breach volume (method='auto': hypsometric → Huggel fallback)...")
    try:
        result = estimate_breach_volume(
            pre_water_mask=pre_water,
            post_water_mask=post_water,
            dem=dem,
            pixel_area_m2=pixel_area_m2,
            method="auto",
        )
        logger.info("=== BREACH VOLUME RESULT ===")
        logger.info("  V_breach: %.1f m³", result.v_breach_m3)
        logger.info("  Method: %s (bathymetry_resolved=%s)", result.method, result.bathymetry_resolved)
        logger.info("  z_pre: %.2f m", result.z_pre_m)
        logger.info("  z_post: %s", f"{result.z_post_m:.2f} m" if result.z_post_m is not None else "N/A")
        logger.info("  Lake area pre: %.1f m² (%.3f km²)", result.lake_area_pre_m2, result.lake_area_pre_m2 / 1e6)
        logger.info("  Lake area post: %.1f m² (%.3f km²)", result.lake_area_post_m2, result.lake_area_post_m2 / 1e6)
        logger.info("  Delta area: %.1f m²", result.delta_area_m2)
        logger.info("  Lake volume pre: %.1f m³", result.lake_volume_pre_m3)
        logger.info("  Mode: %s", result.mode)
        return result.to_dict()
    except ValueError as e:
        logger.error("Breach volume estimation failed: %s", e)
        return {"error": str(e)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    result = run_imja_integration()
    print("\n=== RESULT ===")
    import json
    print(json.dumps(result, indent=2))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
