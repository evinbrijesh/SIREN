"""Water Segmentation Inference Engine — wraps WaterUNet for pipeline use.

Loads trained weights if available; falls back to the deterministic mask
when torch is missing or no weights are found (ADR-002, Hard Rule 1).

Architecture change (ADR-010): the previous engine wrapped a bi-temporal
SiameseUNet that took (t0, t1) and directly predicted a change mask. That
model was disqualified (label-leaked training, runtime/training input
mismatch). The replacement is a single-date water segmenter (WaterUNet):
it segments water from one SAR image, and change detection is done by
deterministic bi-temporal differencing of two independently-segmented
per-date water masks. This keeps the ML model's job narrow (per-date
water segmentation, which is what Sen1Floods11 actually labels) and
keeps the change decision deterministic (Hard Rule 1).

Usage in the pipeline:
    from siren.ml.engine import ChangeDetectionEngine

    engine = ChangeDetectionEngine(weights_path=weights_path)
    if engine.is_ready:
        # Single-date water mask (the ML model's actual job):
        water_mask = engine.predict_water_mask(sar_raster_db)

        # Bi-temporal change mask (deterministic differencing of two
        # per-date ML water masks — NOT a learned change detector):
        change_mask = engine.predict_change_mask(t0_sar, t1_sar)
        consensus = compute_consensus_mask(change_mask, rule_based_mask, dem_slope)
    else:
        # Fall back to deterministic NDWI/backscatter mask
        consensus = rule_based_mask
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from siren.ml.contract import normalize_sar

logger = logging.getLogger(__name__)

# Default weights path — trained weights saved here by train.py
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "water_unet_weights.pt"


class ChangeDetectionEngine:
    """Inference engine for single-date SAR water segmentation (WaterUNet).

    Despite the class name (kept for pipeline interface compatibility), this
    engine now wraps WaterUNet, a single-date water segmenter. The
    ``predict_change_mask`` method segments both dates independently and
    differences the results — it is NOT a learned change detector.

    Gracefully degrades:
      - If torch is not installed -> is_ready = False, falls back to deterministic
      - If no weights file exists  -> is_ready = False, falls back to deterministic
      - If weights load successfully -> is_ready = True, produces ML predictions
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        device: str = "cpu",
        in_channels: int = 2,
    ) -> None:
        self.device = device
        self.in_channels = in_channels
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        self.is_ready = False
        self.model = None
        self.checkpoint_metadata: dict | None = None

        try:
            import torch  # noqa: F401
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            logger.info("torch not installed — ML engine disabled, using deterministic fallback")
            return

        self._load_model()

    def _load_model(self) -> None:
        """Load WaterUNet with trained weights if available."""
        import torch
        from siren.ml.model import WaterUNet

        if self.weights_path.exists():
            try:
                checkpoint = torch.load(
                    str(self.weights_path), map_location=self.device, weights_only=True
                )
                # Auto-detect in_channels from checkpoint metadata
                if isinstance(checkpoint, dict) and "in_channels" in checkpoint:
                    self.in_channels = checkpoint["in_channels"]
                state_dict = (
                    checkpoint["state_dict"]
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                    else checkpoint
                )
                self.model = WaterUNet(in_channels=self.in_channels).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_ready = True
                if isinstance(checkpoint, dict):
                    self.checkpoint_metadata = {
                        k: v for k, v in checkpoint.items() if k != "state_dict"
                    }
                logger.info(
                    f"ML engine loaded WaterUNet weights from {self.weights_path} "
                    f"(in_channels={self.in_channels})"
                )
            except Exception as exc:
                logger.warning(f"Failed to load ML weights: {exc} — using deterministic fallback")
                self.model = WaterUNet(in_channels=self.in_channels).to(self.device)
                self.model.eval()
                self.is_ready = False
        else:
            self.model = WaterUNet(in_channels=self.in_channels).to(self.device)
            self.model.eval()
            logger.info(
                f"No trained weights at {self.weights_path} — "
                "ML engine in scaffold mode (deterministic fallback active). "
                "Run `python -m siren.ml.train` to produce weights."
            )
            self.is_ready = False

    def predict_water_mask(
        self,
        sar_raster: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Segment surface water from a single-date SAR raster.

        Args:
            sar_raster: SAR sigma0 in dB, shape (2, H, W) or (C, H, W).
                        VV/VH channel order expected (per ml/contract.py).
                        Values may be raw dB or already normalized — the
                        contract normalization is applied idempotently.
            threshold: water probability threshold for binary mask.

        Returns:
            Binary water mask (H, W) as uint8 (1 = water, 0 = land).
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                "ML engine not ready — no trained weights loaded. "
                "Use the deterministic mask instead."
            )

        import torch

        # Apply the frozen input contract (idempotent on already-normalized input).
        sar_norm = normalize_sar(sar_raster)

        # Ensure channel count matches model expectations
        if sar_norm.shape[0] != self.in_channels:
            sar_norm = self._adjust_channels(sar_norm, self.in_channels)

        # Pad to nearest multiple of 16 (WaterUNet has 4 downsampling stages)
        h, w = sar_norm.shape[1], sar_norm.shape[2]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h or pad_w:
            sar_norm = np.pad(sar_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        with torch.no_grad():
            x = torch.from_numpy(sar_norm).float().unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Crop back to original dimensions
        if pad_h or pad_w:
            probs = probs[:h, :w]

        return (probs >= threshold).astype(np.uint8)

    def predict_water_probability(
        self,
        sar_raster: np.ndarray,
    ) -> np.ndarray:
        """Segment water and return the raw probability map (H, W) in [0, 1].

        Useful for heatmap visualization and consensus masking.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                "ML engine not ready — no trained weights loaded."
            )

        import torch

        sar_norm = normalize_sar(sar_raster)
        if sar_norm.shape[0] != self.in_channels:
            sar_norm = self._adjust_channels(sar_norm, self.in_channels)

        h, w = sar_norm.shape[1], sar_norm.shape[2]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h or pad_w:
            sar_norm = np.pad(sar_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        with torch.no_grad():
            x = torch.from_numpy(sar_norm).float().unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        if pad_h or pad_w:
            probs = probs[:h, :w]

        return probs

    def predict_change_mask(
        self,
        t0_raster: np.ndarray,
        t1_raster: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Bi-temporal change mask via deterministic differencing of two
        independently-segmented per-date water masks.

        This is NOT a learned change detector. The ML model segments water
        on each date independently; the change decision is the deterministic
        set difference (water at t1 but not at t0). This preserves Hard Rule 1:
        the change decision remains deterministic even when the ML water
        segmenter is available.

        Args:
            t0_raster: Baseline SAR image (C, H, W) in dB or normalized.
            t1_raster: Current SAR image (C, H, W) in dB or normalized.
            threshold: Water probability threshold per date.

        Returns:
            Binary change mask (H, W) as uint8 (1 = new water at t1).
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                "ML engine not ready — no trained weights loaded. "
                "Use the deterministic mask instead."
            )

        water_t0 = self.predict_water_mask(t0_raster, threshold=threshold)
        water_t1 = self.predict_water_mask(t1_raster, threshold=threshold)

        # Deterministic change: water at t1 that was not water at t0.
        # (Expansion only — contraction is not a flood hazard signal.)
        change = (water_t1 == 1) & (water_t0 == 0)
        return change.astype(np.uint8)

    def predict_change_probability(
        self,
        t0_raster: np.ndarray,
        t1_raster: np.ndarray,
    ) -> np.ndarray:
        """Continuous change signal in [0, 1] from per-date water probabilities.

        Returns the positive part of (p_t1 - p_t0), clipped to [0, 1]:
        pixels where water probability increased. Useful for heatmap
        visualization. The binary change mask (predict_change_mask) is
        the thresholded version of this.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                "ML engine not ready — no trained weights loaded."
            )

        p_t0 = self.predict_water_probability(t0_raster)
        p_t1 = self.predict_water_probability(t1_raster)

        # Positive change only (expansion), clipped to [0, 1]
        delta = p_t1 - p_t0
        return np.clip(delta, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _adjust_channels(arr: np.ndarray, target: int) -> np.ndarray:
        """Adjust channel count: truncate or replicate to match target."""
        c = arr.shape[0]
        if c == target:
            return arr
        if c > target:
            return arr[:target]
        # Replicate channels to reach target count
        if c == 1:
            return np.repeat(arr, target, axis=0)
        # General case: tile then truncate
        repeats = (target + c - 1) // c  # ceil division
        return np.concatenate([arr] * repeats, axis=0)[:target]
