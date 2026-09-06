"""Change Detection Inference Engine — wraps the Siamese U-Net for pipeline use.

Loads trained weights if available; falls back to the deterministic mask
when torch is missing or no weights are found (ADR-002, Hard Rule 1).

Usage in the pipeline:
    from siren.ml.engine import ChangeDetectionEngine

    engine = ChangeDetectionEngine(weights_path=weights_path)
    if engine.is_ready:
        ml_mask = engine.predict_change_mask(t0_array, t1_array)
        consensus = compute_consensus_mask(ml_mask, rule_based_mask, dem_slope)
    else:
        # Fall back to deterministic NDWI/backscatter mask
        consensus = rule_based_mask
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default weights path — trained weights would be saved here by train.py
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "siamese_unet_weights.pt"


class ChangeDetectionEngine:
    """Inference engine for the Siamese U-Net change detection model.

    Gracefully degrades:
      - If torch is not installed → is_ready = False, falls back to deterministic
      - If no weights file exists → is_ready = False, falls back to deterministic
      - If weights load successfully → is_ready = True, produces ML predictions
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        device: str = "cpu",
        in_channels: int = 3,
    ) -> None:
        self.device = device
        self.in_channels = in_channels
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        self.is_ready = False
        self.model = None

        try:
            import torch  # noqa: F401
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            logger.info("torch not installed — ML engine disabled, using deterministic fallback")
            return

        self._load_model()

    def _load_model(self) -> None:
        """Load the Siamese U-Net with trained weights if available."""
        import torch
        from siren.ml.model import SiameseUNet

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
                self.model = SiameseUNet(in_channels=self.in_channels).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_ready = True
                logger.info(f"ML engine loaded trained weights from {self.weights_path} (in_channels={self.in_channels})")
            except Exception as exc:
                logger.warning(f"Failed to load ML weights: {exc} — using deterministic fallback")
                self.model = SiameseUNet(in_channels=self.in_channels).to(self.device)
                self.model.eval()
                self.is_ready = False
        else:
            self.model = SiameseUNet(in_channels=self.in_channels).to(self.device)
            self.model.eval()
            logger.info(
                f"No trained weights at {self.weights_path} — "
                "ML engine in scaffold mode (deterministic fallback active). "
                "Run train.py to produce weights."
            )
            self.is_ready = False

    def predict_change_mask(
        self,
        t0_raster: np.ndarray,
        t1_raster: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Run tiled inference over aligned raster arrays.

        Args:
            t0_raster: Baseline image (C, H, W) normalized to [0, 1]
            t1_raster: Current image (C, H, W) normalized to [0, 1]
            threshold: Change probability threshold for binary mask

        Returns:
            Binary change mask (H, W) as uint8 array (1 = changed)

        Raises:
            RuntimeError: if the engine is not ready (no torch or no weights)
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                "ML engine not ready — no trained weights loaded. "
                "Use the deterministic mask instead."
            )

        import torch

        # Ensure channel count matches model expectations
        n_ch = self.in_channels
        if t0_raster.shape[0] != n_ch:
            t0_raster = self._adjust_channels(t0_raster, n_ch)
        if t1_raster.shape[0] != n_ch:
            t1_raster = self._adjust_channels(t1_raster, n_ch)

        with torch.no_grad():
            t0_t = torch.from_numpy(t0_raster).float().unsqueeze(0).to(self.device)
            t1_t = torch.from_numpy(t1_raster).float().unsqueeze(0).to(self.device)

            logits = self.model(t0_t, t1_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        return (probs >= threshold).astype(np.uint8)

    def predict_change_probability(
        self,
        t0_raster: np.ndarray,
        t1_raster: np.ndarray,
    ) -> np.ndarray:
        """Run inference and return the raw probability map (H, W) in [0, 1].

        Unlike predict_change_mask, this returns the continuous probability
        values, useful for heatmap visualization and consensus masking.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                "ML engine not ready — no trained weights loaded."
            )

        import torch

        with torch.no_grad():
            t0_t = torch.from_numpy(t0_raster).float().unsqueeze(0).to(self.device)
            t1_t = torch.from_numpy(t1_raster).float().unsqueeze(0).to(self.device)

            logits = self.model(t0_t, t1_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        return probs

    @staticmethod
    def _adjust_channels(arr: np.ndarray, target: int) -> np.ndarray:
        """Adjust channel count: truncate or replicate to match target."""
        c = arr.shape[0]
        if c == target:
            return arr
        if c > target:
            return arr[:target]
        # Replicate first channel
        reps = [target // c] + [1] * (target - (target // c) * c)
        return np.concatenate(
            [np.repeat(arr, reps[0], axis=0)] + [arr[:1]] * reps[1], axis=0
        )
