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
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np

from siren.ml.contract import normalize_sar

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


class WaterSegmentationDataset:
    """Single-date SAR water segmentation dataset.

    Args:
        split: "train", "val", or "test".
        strategy: "official" (chip-level, literature-comparable) or
                  "event_holdout" (event-level, no leakage).
        chip_size: expected chip height/width (Sen1Floods11 chips are 512x512).
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        strategy: SplitStrategy = "event_holdout",
        chip_size: int = 512,
    ) -> None:
        self.split = split
        self.strategy = strategy
        self.chip_size = chip_size

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
        with rasterio.open(str(label_path)) as src:
            label = src.read(1)  # (H, W): -1 nodata, 0 land, 1 water

        # Apply the frozen input contract -- identical to inference-time normalization.
        sar_norm = normalize_sar(sar)

        water = (label == 1).astype(np.float32)
        valid = (label != -1).astype(np.float32)  # loss mask: exclude no-data pixels

        return {
            "sar": torch.from_numpy(sar_norm).float(),
            "water": torch.from_numpy(water).float(),
            "valid": torch.from_numpy(valid).float(),
            "chip_id": s1_file,
            "event": s1_file.split("_")[0],
        }
