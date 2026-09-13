"""Kuro Siwo GRD dataset adapter — real paired pre/post SAR (PRD v4.7 §17.3).

Reads the Kuro Siwo WebDataset .tar shards directly from disk using Python's
standard ``tarfile`` module (zero external dependencies). Each sample provides
real paired pre/post Sentinel-1 GRD SAR, eliminating the synthetic Δσ⁰ label
leakage that disqualified ``water_resunet_6ch_v1`` (PRD v4.7 §17.3).

Channel contract (6-channel):
    (VV_post, VH_post, VV_pre, VH_pre, ΔVV, ΔVH)

where Δσ⁰ = σ⁰_post − σ⁰_pre in dB. SAR values in the shards are in linear
scale and are converted to dB before differencing.

Label convention (Kuro Siwo):
    0 = no water, 1 = permanent water, 2 = flood

For binary water segmentation the target is ``water = (mask >= 1)`` — both
permanent water and flood are positive. The valid mask excludes invalid pixels
from the loss.

Shard layout (``data/datasets/Kuro Siwo/``):
    train_GRD/shard-00000.tar ... shard-00004.tar
    test_GRD/shard-00004.tar, shard-00005.tar, shard-00011.tar

Each .tar is a WebDataset archive: samples are stored as consecutive members
``{sample_id}.{field}.npy`` and ``{sample_id}.info.json``. Tars may be
truncated (incomplete downloads); the adapter handles this by building an index
via forward iteration rather than relying on ``tarfile.getmembers()``.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Root directory for Kuro Siwo GRD shards.
KURO_SIWO_ROOT = (
    Path(__file__).resolve().parents[3] / "data" / "datasets" / "Kuro Siwo"
)

# Per-sample file suffixes in the tar archive.
_SAMPLE_FIELDS = (
    "flood_vv", "flood_vh",
    "sec1_vv", "sec1_vh",
    "sec2_vv", "sec2_vh",
    "dem",
    "mask", "valid_mask",
)

# SAR conversion constants.
_LINEAR_EPS = 1e-10       # avoid log(0)
_DB_MIN = -30.0           # clamp floor (dB)
_DB_MAX = 0.0             # clamp ceiling (dB)
_DELTA_MIN = -15.0        # Δσ⁰ clamp floor (dB)
_DELTA_MAX = 5.0          # Δσ⁰ clamp ceiling (dB)


def _linear_to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear SAR backscatter to dB: 10 * log10(x + eps)."""
    return 10.0 * np.log10(np.maximum(linear, 0.0) + _LINEAR_EPS)


def _normalize_db(db: np.ndarray) -> np.ndarray:
    """Clamp dB to [-30, 0] and rescale to [0, 1]."""
    return np.clip(db, _DB_MIN, _DB_MAX)
    # Normalization to [0,1] is done in _build_tensor for efficiency.


def _normalize_delta(delta_db: np.ndarray) -> np.ndarray:
    """Clamp Δσ⁰ (dB) to [-15, 5] and rescale to [0, 1]."""
    return np.clip(delta_db, _DELTA_MIN, _DELTA_MAX)


def _build_tensor(
    flood_vv: np.ndarray,
    flood_vh: np.ndarray,
    pre_vv: np.ndarray,
    pre_vh: np.ndarray,
) -> np.ndarray:
    """Stack the 6-channel tensor from raw linear-scale SAR arrays.

    Channel order: (VV_post, VH_post, VV_pre, VH_pre, ΔVV, ΔVH)

    All channels are normalized to [0, 1]:
        - VV/VH: dB clamped to [-30, 0] → (dB + 30) / 30
        - Δσ⁰: dB clamped to [-15, 5] → (Δ + 15) / 20

    Args:
        flood_vv, flood_vh: post-flood SAR (linear scale, shape (1, H, W)).
        pre_vv, pre_vh: pre-flood SAR (linear scale, same shape).

    Returns:
        float32 array of shape (6, H, W) in [0, 1].
    """
    # Squeeze channel dim if present: (1, H, W) → (H, W)
    vv_post = np.squeeze(flood_vv, axis=0) if flood_vv.ndim == 3 else flood_vv
    vh_post = np.squeeze(flood_vh, axis=0) if flood_vh.ndim == 3 else flood_vh
    vv_pre = np.squeeze(pre_vv, axis=0) if pre_vv.ndim == 3 else pre_vv
    vh_pre = np.squeeze(pre_vh, axis=0) if pre_vh.ndim == 3 else pre_vh

    # Convert to dB
    vv_post_db = _linear_to_db(vv_post)
    vh_post_db = _linear_to_db(vh_post)
    vv_pre_db = _linear_to_db(vv_pre)
    vh_pre_db = _linear_to_db(vh_pre)

    # Normalize SAR channels to [0, 1]: (clamp(dB, -30, 0) + 30) / 30
    vv_post_norm = (np.clip(vv_post_db, _DB_MIN, _DB_MAX) + 30.0) / 30.0
    vh_post_norm = (np.clip(vh_post_db, _DB_MIN, _DB_MAX) + 30.0) / 30.0
    vv_pre_norm = (np.clip(vv_pre_db, _DB_MIN, _DB_MAX) + 30.0) / 30.0
    vh_pre_norm = (np.clip(vh_pre_db, _DB_MIN, _DB_MAX) + 30.0) / 30.0

    # Compute Δσ⁰ in dB and normalize to [0, 1]: (clamp(Δ, -15, 5) + 15) / 20
    d_vv = _normalize_delta(vv_post_db - vv_pre_db)
    d_vh = _normalize_delta(vh_post_db - vh_pre_db)
    d_vv_norm = (d_vv + 15.0) / 20.0
    d_vh_norm = (d_vh + 15.0) / 20.0

    tensor = np.stack([
        vv_post_norm, vh_post_norm,
        vv_pre_norm, vh_pre_norm,
        d_vv_norm, d_vh_norm,
    ], axis=0).astype(np.float32)

    return tensor


def _index_shard(shard_path: Path) -> list[dict[str, int]]:
    """Build a sample index for a single .tar shard via forward iteration.

    Handles truncated tars (incomplete downloads) by iterating forward
    rather than calling ``getmembers()``.

    Returns:
        List of dicts, one per sample, mapping field names to tar member
        offsets. Each dict has keys: ``sample_id``, ``flood_vv``, ``flood_vh``,
        ``sec1_vv``, ``sec1_vh``, ``sec2_vv``, ``sec2_vh``, ``dem``, ``mask``,
        ``valid_mask``, ``info``. Values are tar member offsets (int).
    """
    samples: dict[str, dict[str, int]] = {}
    try:
        with tarfile.open(str(shard_path), "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name
                # Parse: {sample_id}.{field}.npy or {sample_id}.info.json
                if name.endswith(".info.json"):
                    sample_id = name[:-len(".info.json")]
                    samples.setdefault(sample_id, {})["info"] = member.offset
                elif name.endswith(".npy"):
                    # Split from the right: sample_id.field.npy
                    parts = name.rsplit(".", 2)
                    if len(parts) == 3:
                        sample_id, field, _ = parts
                        samples.setdefault(sample_id, {})[field] = member.offset
    except tarfile.ReadError:
        # Truncated tar (incomplete download) — keep whatever samples we
        # indexed before hitting the truncation point.
        logger.warning(
            "Shard %s is truncated (ReadError) — indexed %d samples before truncation",
            shard_path.name, len(samples),
        )

    # Only keep complete samples (have all required fields)
    required = {"flood_vv", "flood_vh", "sec1_vv", "sec1_vh", "mask", "valid_mask"}
    complete = []
    for sample_id, fields in samples.items():
        if required.issubset(fields.keys()):
            complete.append({"sample_id": sample_id, **fields})

    return complete


def _read_member(tar: tarfile.TarFile, offset: int) -> bytes:
    """Read a tar member at a known offset (avoids getmembers() on truncated tars)."""
    tar.offset = offset
    member = tar.next()
    if member is None:
        raise IOError(f"Could not read tar member at offset {offset}")
    f = tar.extractfile(member)
    if f is None:
        raise IOError(f"Could not extract tar member at offset {offset}")
    return f.read()


def _load_npy(tar: tarfile.TarFile, offset: int) -> np.ndarray:
    """Load a .npy array from a tar member at a known offset."""
    return np.load(io.BytesIO(_read_member(tar, offset)))


class KuroSiwoDataset:
    """PyTorch-compatible dataset reading Kuro Siwo GRD .tar shards from disk.

    Provides real paired pre/post Sentinel-1 SAR for 6-channel multi-temporal
    water segmentation, replacing the disqualified synthetic Δσ⁰ path
    (PRD v4.7 §17.3).

    Args:
        split: "train" or "test" (maps to train_GRD/ or test_GRD/).
        root: path to the Kuro Siwo dataset root (default: data/datasets/Kuro Siwo).
        pre_event: which pre-flood image to use: "sec1" (first, default) or "sec2".
        chip_size: expected chip size (Kuro Siwo = 224).
        max_samples: optional cap on number of samples (for debugging/testing).

    Yields dicts with keys:
        sar: float32 tensor (6, H, W) — (VV_post, VH_post, VV_pre, VH_pre, ΔVV, ΔVH)
        water: float32 tensor (H, W) — binary water mask (mask >= 1)
        valid: float32 tensor (H, W) — valid pixel mask
        sample_id: str
        flood_date: str (from info.json)
        dem: float32 tensor (H, W) — unnormalized DEM (for viz/optional terrain)
    """

    def __init__(
        self,
        split: Literal["train", "test"] = "train",
        root: str | Path | None = None,
        pre_event: Literal["sec1", "sec2"] = "sec1",
        chip_size: int = 224,
        max_samples: int | None = None,
    ) -> None:
        self.split = split
        self.root = Path(root) if root else KURO_SIWO_ROOT
        self.pre_event = pre_event
        self.chip_size = chip_size

        shard_dir = self.root / f"{split}_GRD"
        if not shard_dir.exists():
            raise FileNotFoundError(
                f"Kuro Siwo {split} shard directory not found: {shard_dir}. "
                f"Ensure data/datasets/Kuro Siwo/{split}_GRD/ is present."
            )

        # Index all shards
        shard_paths = sorted(shard_dir.glob("shard-*.tar"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No .tar shards found in {shard_dir}"
            )

        self._index: list[tuple[Path, dict[str, int]]] = []
        for shard_path in shard_paths:
            shard_samples = _index_shard(shard_path)
            logger.info("Indexed %s: %d complete samples", shard_path.name, len(shard_samples))
            for s in shard_samples:
                self._index.append((shard_path, s))

        if max_samples is not None:
            self._index = self._index[:max_samples]

        logger.info(
            "KuroSiwoDataset(%s): %d total samples across %d shards",
            split, len(self._index), len(shard_paths),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        import torch

        shard_path, sample_info = self._index[idx]
        sample_id = sample_info["sample_id"]

        with tarfile.open(str(shard_path), "r") as tar:
            # Load arrays
            flood_vv = _load_npy(tar, sample_info["flood_vv"])
            flood_vh = _load_npy(tar, sample_info["flood_vh"])
            pre_vv = _load_npy(tar, sample_info[f"{self.pre_event}_vv"])
            pre_vh = _load_npy(tar, sample_info[f"{self.pre_event}_vh"])
            mask = _load_npy(tar, sample_info["mask"])
            valid_mask = _load_npy(tar, sample_info["valid_mask"])

            # Optional: DEM and info
            dem = None
            if "dem" in sample_info:
                dem = _load_npy(tar, sample_info["dem"])

            flood_date = ""
            if "info" in sample_info:
                try:
                    info = json.loads(_read_member(tar, sample_info["info"]))
                    flood_date = info.get("flood_date", "")
                except Exception:
                    pass

        # Squeeze (1, H, W) → (H, W) for mask/valid
        mask = np.squeeze(mask, axis=0) if mask.ndim == 3 else mask
        valid_mask = np.squeeze(valid_mask, axis=0) if valid_mask.ndim == 3 else valid_mask

        # Build 6-channel tensor
        tensor = _build_tensor(flood_vv, flood_vh, pre_vv, pre_vh)

        # Binary water: mask >= 1 (permanent water + flood)
        water = (mask >= 1).astype(np.float32)
        valid = (valid_mask >= 1).astype(np.float32)

        # Clean NaN/Inf from SAR tensor (some Kuro Siwo chips have NaN backscatter)
        tensor = np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

        result = {
            "sar": torch.from_numpy(tensor).float(),           # (6, H, W)
            "water": torch.from_numpy(water).float(),          # (H, W)
            "valid": torch.from_numpy(valid).float(),            # (H, W)
            "sample_id": sample_id,
            "flood_date": flood_date,
        }
        if dem is not None:
            dem = np.squeeze(dem, axis=0) if dem.ndim == 3 else dem
            # Replace NaN DEM with 0 (some Kuro Siwo chips have missing DEM).
            # The gravity penalty uses DEM elevation; NaN would propagate to
            # the loss. Zero elevation is neutral for the gravity variance
            # computation (it's the reference plane).
            dem = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)
            result["dem"] = torch.from_numpy(dem).float()       # (H, W), unnormalized

        return result

    def events(self) -> set[str]:
        """Return unique flood event IDs (sample_id prefix before first underscore)."""
        return {
            info["sample_id"].split("_")[0]
            for _, info in self._index
        }
