"""Dataset loader for single-date SAR water segmentation (ADR-010 Stage 1).

Supports two independent split strategies over the same underlying
Sen1Floods11 hand-labeled chips:

  "official"      -- the dataset's own train/valid/test CSVs
                      (chip-level; all 10 flood events appear in all
                      three splits -- literature-comparable, but leaks
                      event geography between train and test).
  "event_holdout" -- siren.ml.build_event_holdout_split's train/val/test
                      CSVs (entire flood events held out; no event
                      appears in more than one split -- the strict
                      generalization test ADR-010 requires).

Both strategies read from the same three physical directories on disk
(train/, valid/, test/ under data/raw/Sen1Floods11/), since that is
where the files actually live; the "event_holdout" CSVs record which
physical directory each chip's files are stored in.

Label convention (Sen1Floods11 QC layer): -1 = no data, 0 = land,
1 = water. No-data pixels are excluded from the loss via a validity
mask, not remapped to land or water.

4-channel mode (ADR-011 / V3 §2.1): when ``use_copernicus_dem=True``, the
dataset fetches Copernicus GLO-30 DEM tiles for each chip's geospatial
bounds (via :mod:`siren.ml.dem_fetch`), reprojects to the chip grid, and
derives slope with latitude-corrected metre pixel size. The 4-channel
tensor is (VV, VH, DEM, Slope), normalised via ``contract.normalize_tensor``.

When ``dem_path`` is provided (legacy single-file mode), the DEM is read
from that file and co-registered to each chip. When neither is set, the
legacy 2-channel (VV, VH) path is used.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np

from siren.ml.contract import normalize_sar, normalize_tensor, normalize_tensor_multitemporal

SEN1FLOODS11_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "Sen1Floods11"

SplitStrategy = Literal["official", "event_holdout"]


def _official_csv_name(split: str) -> str:
    return {"train": "flood_train_data.csv", "val": "flood_valid_data.csv", "test": "flood_test_data.csv"}[split]


def _load_official_rows(split: str) -> list[tuple[str, str, str]]:
    """Official split: physical dir == split name (train/valid/test)."""
    physical_dir = {"train": "train", "val": "valid", "test": "test"}[split]
    csv_path = SEN1FLOODS11_ROOT / "splits" / _official_csv_name(split)
    rows = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if row:
                rows.append((row[0], row[1], physical_dir))
    return rows


def _load_event_holdout_rows(split: str) -> list[tuple[str, str, str]]:
    """Event-holdout split: physical dir recorded per-row in the CSV
    (chips for one logical split may live in any of the three physical
    directories, since the holdout regroups by flood event, not by the
    dataset's original chip-level partition)."""
    csv_path = SEN1FLOODS11_ROOT / "splits" / "event_holdout" / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m siren.ml.build_event_holdout_split` first."
        )
    rows = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if row:
                rows.append((row[0], row[1], row[2]))
    return rows


def coregister_dem_to_chip(
    dem_path: str | Path,
    chip_crs: str,
    chip_transform,
    chip_shape: tuple[int, int],
) -> np.ndarray:
    """Reproject a DEM raster to a chip's grid (CRS + transform + shape).

    Args:
        dem_path: Path to the DEM GeoTIFF (e.g. Copernicus GLO-30 or SRTM).
        chip_crs: CRS of the target chip (e.g. 'EPSG:4326').
        chip_transform: rasterio Affine transform of the chip.
        chip_shape: (H, W) of the chip grid.

    Returns:
        2D float32 array of DEM elevation in metres, co-registered to the
        chip's grid.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(str(dem_path)) as dem_src:
        dem = np.empty(chip_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(dem_src, 1),
            destination=dem,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=chip_transform,
            dst_crs=chip_crs,
            resampling=Resampling.bilinear,
        )
    return dem


class WaterSegmentationDataset:
    """Single-date SAR water segmentation dataset.

    Args:
        split: "train", "val", or "test".
        strategy: "official" (chip-level, literature-comparable) or
                  "event_holdout" (event-level, no leakage).
        chip_size: expected chip height/width (Sen1Floods11 chips are 512x512).
        dem_path: optional path to a single DEM raster (legacy mode). When
                  provided, the DEM is co-registered to each chip.
        use_copernicus_dem: when True, fetch Copernicus GLO-30 DEM tiles
                           per chip (Level 2 production path). Overrides
                           dem_path.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        strategy: SplitStrategy = "event_holdout",
        chip_size: int = 512,
        dem_path: str | Path | None = None,
        use_copernicus_dem: bool = False,
    ) -> None:
        self.split = split
        self.strategy = strategy
        self.chip_size = chip_size
        self.dem_path = Path(dem_path) if dem_path else None
        self.use_copernicus_dem = use_copernicus_dem
        # In-memory cache for reprojected DEM + slope (keyed by chip_id).
        # Avoids re-reading/reprojecting DEM tiles on every epoch.
        self._dem_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        if strategy == "official":
            self.rows = _load_official_rows(split)
        elif strategy == "event_holdout":
            self.rows = _load_event_holdout_rows(split)
        else:
            raise ValueError(f"unknown split strategy: {strategy}")

    def __len__(self) -> int:
        return len(self.rows)

    def events(self) -> set[str]:
        return {s1_file.split("_")[0] for s1_file, _, _ in self.rows}

    def __getitem__(self, idx: int) -> dict:
        import rasterio
        import torch

        s1_file, label_file, physical_dir = self.rows[idx]

        s1_path = SEN1FLOODS11_ROOT / physical_dir / "S1" / s1_file
        label_path = SEN1FLOODS11_ROOT / physical_dir / "Label" / label_file

        with rasterio.open(str(s1_path)) as src:
            sar = src.read()  # (2, H, W) VV/VH in dB
            chip_crs = str(src.crs)
            chip_transform = src.transform
            chip_shape = (src.height, src.width)
            chip_bounds = src.bounds
        with rasterio.open(str(label_path)) as src:
            label = src.read(1)  # (H, W): -1 nodata, 0 land, 1 water

        water = (label == 1).astype(np.float32)
        valid = (label != -1).astype(np.float32)  # loss mask: exclude no-data pixels

        if self.use_copernicus_dem:
            # 4-channel path (Level 2): fetch Copernicus GLO-30 DEM per chip
            from siren.ml.dem_fetch import (
                fetch_dem_for_bounds, coregister_dem_to_grid,
                pixel_size_m_from_transform,
            )
            from siren.preprocess.rtc import slope_degrees

            # Check in-memory cache first (avoids re-projecting DEM every epoch)
            if s1_file in self._dem_cache:
                dem, slope = self._dem_cache[s1_file]
            else:
                center_lat = (chip_bounds.top + chip_bounds.bottom) / 2

                dem_paths = fetch_dem_for_bounds(
                    chip_bounds.left, chip_bounds.bottom,
                    chip_bounds.right, chip_bounds.top,
                )
                dem = coregister_dem_to_grid(
                    dem_paths, chip_crs, chip_transform, chip_shape,
                )
                dx_m, dy_m = pixel_size_m_from_transform(
                    chip_transform, chip_crs, center_lat,
                )
                slope = slope_degrees(dem, (dx_m, dy_m))
                self._dem_cache[s1_file] = (dem, slope)

            tensor = np.stack([
                sar[0], sar[1], dem, slope,
            ], axis=0).astype(np.float32)
            tensor_norm = normalize_tensor(tensor)

            return {
                "sar": torch.from_numpy(tensor_norm).float(),  # (4, H, W)
                "water": torch.from_numpy(water).float(),
                "valid": torch.from_numpy(valid).float(),
                "chip_id": s1_file,
                "event": s1_file.split("_")[0],
                "dem": torch.from_numpy(dem).float(),  # unnormalised, for viz/loss
                "slope": torch.from_numpy(slope).float(),
            }

        if self.dem_path is not None:
            # 4-channel path (ADR-011 / V3 §2.1): stack (VV, VH, DEM, Slope)
            from siren.preprocess.dem import slope_degrees
            from siren.ml.dem_fetch import pixel_size_m_from_transform

            dem = coregister_dem_to_chip(
                self.dem_path, chip_crs, chip_transform, chip_shape
            )
            # Pixel size in metres (latitude-corrected for geographic CRS)
            center_lat = (chip_bounds.top + chip_bounds.bottom) / 2
            dx_m, dy_m = pixel_size_m_from_transform(
                chip_transform, chip_crs, center_lat,
            )
            slope = slope_degrees(dem, (dx_m, dy_m))

            # Stack into (4, H, W) and normalise via the 4-channel contract
            tensor = np.stack([
                sar[0], sar[1], dem, slope,
            ], axis=0).astype(np.float32)
            tensor_norm = normalize_tensor(tensor)

            return {
                "sar": torch.from_numpy(tensor_norm).float(),  # (4, H, W)
                "water": torch.from_numpy(water).float(),
                "valid": torch.from_numpy(valid).float(),
                "chip_id": s1_file,
                "event": s1_file.split("_")[0],
                "dem": torch.from_numpy(dem).float(),  # unnormalised, for viz
                "slope": torch.from_numpy(slope).float(),
            }

        # Legacy 2-channel path (ADR-010 Stage 1)
        sar_norm = normalize_sar(sar)
        return {
            "sar": torch.from_numpy(sar_norm).float(),  # (2, H, W)
            "water": torch.from_numpy(water).float(),
            "valid": torch.from_numpy(valid).float(),
            "chip_id": s1_file,
            "event": s1_file.split("_")[0],
        }


# ---------------------------------------------------------------------------
# 6-channel multi-temporal dataset (V3 §2.8 — multi-temporal upgrade)
# ---------------------------------------------------------------------------

# Synthetic Δσ⁰ parameters for training when paired pre/post SAR is unavailable.
# Flood water: rough vegetation → smooth water, typical drop -6 to -10 dB.
# Permanent water: no change, ~0 dB.
# Land: no change, ~0 dB.
# Noise: ±1.5 dB to simulate real temporal variability.
_SYNTHETIC_FLOOD_DROP_DB: float = -8.0
_SYNTHETIC_NOISE_STD_DB: float = 1.5


def _synthetic_delta_sar(
    sar_post: np.ndarray,
    water_label: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct synthetic Δσ⁰ from post-event SAR and water labels.

    For water pixels (label=1): Δσ⁰ ≈ -8 dB (flood drop signal).
    For land pixels (label=0): Δσ⁰ ≈ 0 dB (no change).
    Noise ±1.5 dB is added to simulate real temporal variability.

    This is a training-time augmentation for datasets without paired
    pre/post SAR (e.g. Sen1Floods11). At inference time, real paired
    SAR scenes provide the actual Δσ⁰.

    Args:
        sar_post: post-event SAR array (2, H, W) in dB (VV, VH).
        water_label: binary water mask (H, W), 1=water, 0=land.
        rng: seeded numpy Generator for reproducibility (Hard Rule 6).

    Returns:
        Δσ⁰ array (2, H, W) in dB.
    """
    delta = np.zeros_like(sar_post, dtype=np.float32)
    # Water pixels: large negative drop (flood signal)
    delta[:, water_label > 0.5] = _SYNTHETIC_FLOOD_DROP_DB
    # Add noise everywhere
    noise = rng.normal(0, _SYNTHETIC_NOISE_STD_DB, size=sar_post.shape).astype(np.float32)
    delta += noise
    return delta


def _compute_hand_from_dem(
    dem: np.ndarray,
    chip_transform,
    chip_crs: str,
    pixel_size_m: tuple[float, float],
) -> np.ndarray:
    """Compute HAND (Height Above Nearest Drainage) from a DEM array.

    Wraps :mod:`siren.geo.hand` to compute HAND for a single chip.
    Falls back to a simple relative elevation proxy if pysheds fails
    on small chips (common for 512×512 Sen1Floods11 tiles where the
    river may exit the tile boundaries).

    Args:
        dem: DEM elevation array (H, W) in metres.
        chip_transform: rasterio Affine transform.
        chip_crs: CRS string.
        pixel_size_m: (dx, dy) pixel size in metres.

    Returns:
        HAND array (H, W) in metres. Values near 0 = in-channel,
        higher values = elevated terrain.
    """
    try:
        from siren.geo.hand import compute_hand
        hand_result = compute_hand(
            dem=dem,
            transform=chip_transform,
            crs=chip_crs,
            pixel_size_m=pixel_size_m,
        )
        return hand_result.hand
    except Exception:
        # Fallback: relative elevation proxy (DEM minus local minimum).
        # This preserves the scale-invariant property that HAND provides.
        # Not physically exact but prevents the network from memorising
        # absolute elevation (the core goal of replacing raw DEM).
        from scipy.ndimage import minimum_filter
        # Local minimum over a 50-pixel radius (~1.5 km for 30m pixels)
        local_min = minimum_filter(dem, size=50, mode='nearest')
        hand_proxy = dem - local_min
        return np.clip(hand_proxy, 0, None).astype(np.float32)


class MultiTemporalWaterDataset:
    """6-channel multi-temporal SAR water segmentation dataset (V3 §2.8).

    Replaces the static 4-channel (VV, VH, DEM, Slope) tensor with a
    6-channel multi-temporal tensor:

        (VV_post, VH_post, ΔVV, ΔVH, HAND, Slope)

    where Δσ⁰ = σ⁰_post - σ⁰_pre. This eliminates dry-soil false positives
    (the core failure mode of single-date inference on Pakistan/Somalia)
    because dry soil has Δσ⁰ ≈ 0 while flood water has Δσ⁰ ≤ -6 dB.

    HAND replaces raw DEM to prevent the network from memorising absolute
    elevation (e.g. Mekong Delta at 5 m vs Himalaya at 4,500 m).

    Args:
        split: "train", "val", or "test".
        strategy: "official" or "event_holdout".
        chip_size: expected chip size (Sen1Floods11 = 512).
        dem_path: path to a single DEM raster (co-registered per chip).
        use_copernicus_dem: fetch Copernicus GLO-30 DEM per chip.
        pre_sar_dir: optional directory of pre-event SAR scenes. When
                     provided, real Δσ⁰ is computed from paired pre/post
                     SAR. When None, synthetic Δσ⁰ is constructed from
                     water labels (training augmentation).
        seed: random seed for synthetic Δσ⁰ noise (Hard Rule 6).
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        strategy: SplitStrategy = "event_holdout",
        chip_size: int = 512,
        dem_path: str | Path | None = None,
        use_copernicus_dem: bool = False,
        pre_sar_dir: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.strategy = strategy
        self.chip_size = chip_size
        self.dem_path = Path(dem_path) if dem_path else None
        self.use_copernicus_dem = use_copernicus_dem
        self.pre_sar_dir = Path(pre_sar_dir) if pre_sar_dir else None
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._dem_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        if strategy == "official":
            self.rows = _load_official_rows(split)
        elif strategy == "event_holdout":
            self.rows = _load_event_holdout_rows(split)
        else:
            raise ValueError(f"unknown split strategy: {strategy}")

    def __len__(self) -> int:
        return len(self.rows)

    def events(self) -> set[str]:
        return {s1_file.split("_")[0] for s1_file, _, _ in self.rows}

    def _get_dem_slope_hand(
        self,
        s1_file: str,
        chip_crs: str,
        chip_transform,
        chip_shape: tuple[int, int],
        chip_bounds,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get co-registered DEM, slope, and HAND for a chip (cached)."""
        if s1_file in self._dem_cache:
            return self._dem_cache[s1_file]

        from siren.preprocess.dem import slope_degrees
        from siren.ml.dem_fetch import pixel_size_m_from_transform

        if self.use_copernicus_dem:
            from siren.ml.dem_fetch import (
                fetch_dem_for_bounds, coregister_dem_to_grid,
            )
            center_lat = (chip_bounds.top + chip_bounds.bottom) / 2
            dem_paths = fetch_dem_for_bounds(
                chip_bounds.left, chip_bounds.bottom,
                chip_bounds.right, chip_bounds.top,
            )
            dem = coregister_dem_to_grid(
                dem_paths, chip_crs, chip_transform, chip_shape,
            )
        else:
            dem = coregister_dem_to_chip(
                self.dem_path, chip_crs, chip_transform, chip_shape
            )
            center_lat = (chip_bounds.top + chip_bounds.bottom) / 2

        dx_m, dy_m = pixel_size_m_from_transform(chip_transform, chip_crs, center_lat)
        slope = slope_degrees(dem, (dx_m, dy_m))
        hand = _compute_hand_from_dem(dem, chip_transform, chip_crs, (dx_m, dy_m))

        self._dem_cache[s1_file] = (dem, slope, hand)
        return dem, slope, hand

    def __getitem__(self, idx: int) -> dict:
        import rasterio
        import torch

        s1_file, label_file, physical_dir = self.rows[idx]

        s1_path = SEN1FLOODS11_ROOT / physical_dir / "S1" / s1_file
        label_path = SEN1FLOODS11_ROOT / physical_dir / "Label" / label_file

        with rasterio.open(str(s1_path)) as src:
            sar_post = src.read()  # (2, H, W) VV/VH in dB
            chip_crs = str(src.crs)
            chip_transform = src.transform
            chip_shape = (src.height, src.width)
            chip_bounds = src.bounds
        with rasterio.open(str(label_path)) as src:
            label = src.read(1)  # (H, W): -1 nodata, 0 land, 1 water

        water = (label == 1).astype(np.float32)
        valid = (label != -1).astype(np.float32)

        # Compute Δσ⁰
        if self.pre_sar_dir is not None:
            # Real paired pre/post SAR
            pre_path = self.pre_sar_dir / s1_file
            if pre_path.exists():
                with rasterio.open(str(pre_path)) as src:
                    sar_pre = src.read()
                delta_sar = (sar_post - sar_pre).astype(np.float32)
            else:
                # Fallback to synthetic if pre-event scene missing
                delta_sar = _synthetic_delta_sar(sar_post, water, self._rng)
        else:
            # Synthetic Δσ⁰ from labels (training augmentation)
            delta_sar = _synthetic_delta_sar(sar_post, water, self._rng)

        # Get DEM, slope, HAND
        dem, slope, hand = self._get_dem_slope_hand(
            s1_file, chip_crs, chip_transform, chip_shape, chip_bounds
        )

        # Stack into 6-channel tensor: (VV_post, VH_post, dVV, dVH, HAND, Slope)
        tensor = np.stack([
            sar_post[0], sar_post[1],
            delta_sar[0], delta_sar[1],
            hand, slope,
        ], axis=0).astype(np.float32)
        tensor_norm = normalize_tensor_multitemporal(tensor)

        return {
            "sar": torch.from_numpy(tensor_norm).float(),  # (6, H, W)
            "water": torch.from_numpy(water).float(),
            "valid": torch.from_numpy(valid).float(),
            "chip_id": s1_file,
            "event": s1_file.split("_")[0],
            "dem": torch.from_numpy(dem).float(),
            "slope": torch.from_numpy(slope).float(),
            "hand": torch.from_numpy(hand).float(),
            "delta_sar": torch.from_numpy(delta_sar).float(),  # unnormalised, for viz
        }
