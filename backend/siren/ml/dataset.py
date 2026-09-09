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

4-channel mode (ADR-011 / V3 §2.1): when ``dem_path`` is provided, the
dataset stacks (VV, VH, DEM, Slope) per chip. The DEM is reprojected to
each chip's grid and slope is derived via ``preprocess.dem.slope_degrees``.
The 4-channel tensor is normalised via ``contract.normalize_tensor``.
When ``dem_path`` is None, the legacy 2-channel path is used.
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
        dem_path: optional path to a DEM raster. When provided, the dataset
                  returns 4-channel (VV, VH, DEM, Slope) tensors (ADR-011 /
                  V3 §2.1). When None, returns legacy 2-channel (VV, VH).
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        strategy: SplitStrategy = "event_holdout",
        chip_size: int = 512,
        dem_path: str | Path | None = None,
    ) -> None:
        self.split = split
        self.strategy = strategy
        self.chip_size = chip_size
        self.dem_path = Path(dem_path) if dem_path else None

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
        with rasterio.open(str(label_path)) as src:
            label = src.read(1)  # (H, W): -1 nodata, 0 land, 1 water

        water = (label == 1).astype(np.float32)
        valid = (label != -1).astype(np.float32)  # loss mask: exclude no-data pixels

        if self.dem_path is not None:
            # 4-channel path (ADR-011 / V3 §2.1): stack (VV, VH, DEM, Slope)
            from siren.preprocess.dem import slope_degrees, DEFAULT_PIXEL_SIZE_M

            dem = coregister_dem_to_chip(
                self.dem_path, chip_crs, chip_transform, chip_shape
            )
            # Pixel size from the chip transform (assumes square pixels)
            dx = abs(chip_transform.a)
            dy = abs(chip_transform.e)
            px = float((dx + dy) / 2.0) if dx > 0 and dy > 0 else DEFAULT_PIXEL_SIZE_M
            slope = slope_degrees(dem, px)

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
