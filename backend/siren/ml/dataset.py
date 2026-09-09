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

from siren.ml.contract import normalize_sar, normalize_tensor

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
