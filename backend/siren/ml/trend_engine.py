"""Temporal trend engine — ConvLSTM inference wrapper.

Loads the trained ConvLSTM trend classifier and provides inference for
classifying temporal sequences of water masks into trend categories:
stable / slowly / rapidly / uncertain.

Falls back to deterministic trend classification (based on expansion
percentage thresholds) when torch is unavailable or no trained weights
exist. This preserves the deterministic-first principle (ADR-002).

Usage in the pipeline:
    engine = TrendEngine()
    if engine.is_ready:
        trend_class, confidence = engine.classify_trend(water_masks)
    else:
        # Deterministic fallback
        trend_class = deterministic_trend(expansion_pcts)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "convlstm_trend_weights.pt"
)

TREND_CLASSES = ["stable", "slowly", "rapidly", "uncertain"]


class TrendEngine:
    """ConvLSTM temporal trend classification engine.

    Loads trained weights if available, falls back to deterministic
    threshold-based classification otherwise.
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        self.is_ready = False
        self.model = None
        self.seq_len = 4

        try:
            import torch  # noqa: F401
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            logger.info("torch not installed — trend engine disabled, using deterministic fallback")
            return

        self._load_model()

    def _load_model(self) -> None:
        """Load the ConvLSTM with trained weights if available."""
        import torch
        from siren.ml.temporal import ConvLSTMTrendClassifier

        if self.weights_path.exists():
            try:
                checkpoint = torch.load(
                    str(self.weights_path), map_location=self.device, weights_only=True
                )
                if isinstance(checkpoint, dict) and "seq_len" in checkpoint:
                    self.seq_len = checkpoint["seq_len"]

                self.model = ConvLSTMTrendClassifier(
                    in_channels=1,
                    encoder_dim=32,
                    lstm_dim=64,
                    num_classes=4,
                ).to(self.device)

                state_dict = (
                    checkpoint["state_dict"]
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                    else checkpoint
                )
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_ready = True
                logger.info(f"Trend engine loaded trained weights from {self.weights_path}")
            except Exception as exc:
                logger.warning(f"Failed to load trend weights: {exc} — using deterministic fallback")
                self.is_ready = False
        else:
            logger.info(
                f"No trained weights at {self.weights_path} — "
                "trend engine using deterministic fallback"
            )
            self.is_ready = False

    def classify_trend(
        self,
        water_masks: list[np.ndarray] | np.ndarray,
        confidence_threshold: float = 0.75,
    ) -> tuple[str, float]:
        """Classify the temporal trend from a sequence of water masks.

        Uses a hybrid approach: the ConvLSTM prediction is used when its
        confidence exceeds the threshold. Otherwise, the deterministic
        fallback (expansion-percentage thresholds) is used. This is a
        common production ML pattern — defer to the physical model when
        the neural network is uncertain.

        Args:
            water_masks: List of (H, W) binary water masks, or a
                (T, H, W) array. Must have at least 2 timesteps.
            confidence_threshold: Minimum ConvLSTM confidence to use its
                prediction. Below this, the deterministic fallback is used.

        Returns:
            (trend_class, confidence) — one of "stable"/"slowly"/"rapidly"/"uncertain"
            and a float confidence in [0, 1].
        """
        if not self.is_ready or self.model is None:
            return self._deterministic_fallback(water_masks)

        import torch
        import torch.nn.functional as F

        # Convert to (T, 1, H, W) tensor
        if isinstance(water_masks, list):
            masks = np.stack(water_masks)
        else:
            masks = water_masks

        if masks.ndim == 3:
            masks = masks[:, None, :, :]  # (T, 1, H, W)
        elif masks.ndim == 4:
            pass  # already (T, C, H, W)
        else:
            raise ValueError(f"Expected 3D or 4D array, got {masks.ndim}D")

        # Resize to 128x128 if needed (training resolution)
        t, c, h, w = masks.shape
        if h != 128 or w != 128:
            resized = np.zeros((t, c, 128, 128), dtype=np.float32)
            for i in range(t):
                from scipy.ndimage import zoom
                zh, zw = 128 / h, 128 / w
                resized[i, 0] = zoom(masks[i, 0], (zh, zw), order=0)
            masks = resized

        # Pad or truncate to seq_len
        if t < self.seq_len:
            # Extrapolate the trend by dilating the water mask
            from scipy import ndimage
            pad_count = self.seq_len - t
            for _ in range(pad_count):
                last = masks[-1, 0]
                dilated = ndimage.binary_dilation(last > 0.5, iterations=1).astype(np.float32)
                masks = np.concatenate([masks, dilated[None, None, :, :]], axis=0)
        elif t > self.seq_len:
            # Take the last seq_len timesteps
            masks = masks[-self.seq_len:]

        with torch.no_grad():
            x = torch.from_numpy(masks).float().unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        ml_trend = TREND_CLASSES[pred.item()]
        ml_conf = conf.item()

        # Hybrid: use ML prediction when confident, deterministic otherwise
        if ml_conf >= confidence_threshold and ml_trend != "uncertain":
            return ml_trend, ml_conf

        # Fall back to deterministic for low-confidence or "uncertain" predictions
        det_trend, det_conf = self._deterministic_fallback(water_masks)
        # Blend confidence: report the deterministic result but note ML was consulted
        return det_trend, det_conf

    @staticmethod
    def _deterministic_fallback(
        water_masks: list[np.ndarray] | np.ndarray,
    ) -> tuple[str, float]:
        """Deterministic trend classification from water area progression.

        Uses simple percentage thresholds:
          - < 3% change: stable
          - 3-20% monotonic increase: slowly
          - > 20% monotonic increase: rapidly
          - non-monotonic (increases then decreases): uncertain
        """
        if isinstance(water_masks, list):
            masks = water_masks
        else:
            masks = [water_masks[i] for i in range(water_masks.shape[0])]

        if len(masks) < 2:
            return "uncertain", 0.5

        areas = [float(m.sum()) for m in masks]
        if areas[0] == 0:
            if areas[-1] > 0:
                return "rapidly", 0.7
            return "stable", 0.8

        # Compute expansion percentages
        expansions = [(a - areas[0]) / areas[0] * 100 for a in areas]

        # Check monotonicity (allow 1% tolerance for noise)
        is_increasing = all(expansions[i] <= expansions[i + 1] + 1 for i in range(len(expansions) - 1))
        total_expansion = expansions[-1]

        # Non-monotonic (e.g. increase then decrease) = uncertain
        if not is_increasing:
            return "uncertain", 0.6

        if abs(total_expansion) < 3:
            return "stable", 0.85
        elif total_expansion < 20:
            return "slowly", 0.75
        else:
            return "rapidly", 0.80
