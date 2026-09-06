"""SegFormer inference engine — Stage 2 land-cover classification.

Loads the trained SegFormer head and classifies changed-pixel crops into:
  0: Water, 1: Debris, 2: Snowmelt, 3: Shadow, 4: Bare rock

Used after Stage 1 (Siamese U-Net) to filter false alarms. Crops flagged
as "shadow" or "snowmelt" are excluded from the final change mask.

Falls back to a deterministic heuristic (backscatter thresholding) when
the model is unavailable.
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
    / "segformer_classifier_weights.pt"
)

CLASS_NAMES = ["water", "debris", "snowmelt", "shadow", "bare_rock"]
# Classes that represent false alarms (filtered out)
FALSE_ALARM_CLASSES = {"shadow", "snowmelt"}


class SegFormerEngine:
    """SegFormer land-cover classification engine.

    Classifies changed-pixel crops into functional categories and
    filters false alarms (shadow, snowmelt).
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
        self.class_names = CLASS_NAMES

        try:
            import torch  # noqa: F401
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            logger.info("torch not installed — SegFormer engine disabled")
            return

        self._load_model()

    def _load_model(self) -> None:
        import torch
        from siren.ml.model import SegFormerHead

        if self.weights_path.exists():
            try:
                checkpoint = torch.load(
                    str(self.weights_path), map_location=self.device, weights_only=True
                )
                if isinstance(checkpoint, dict) and "in_channels" in checkpoint:
                    self.in_channels = checkpoint["in_channels"]
                if isinstance(checkpoint, dict) and "class_names" in checkpoint:
                    self.class_names = checkpoint["class_names"]

                self.model = SegFormerHead(
                    in_channels=self.in_channels, num_classes=len(self.class_names)
                ).to(self.device)
                state_dict = (
                    checkpoint["state_dict"]
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                    else checkpoint
                )
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_ready = True
                logger.info(f"SegFormer engine loaded weights from {self.weights_path}")
            except Exception as exc:
                logger.warning(f"Failed to load SegFormer weights: {exc}")
                self.is_ready = False
        else:
            logger.info(f"No SegFormer weights at {self.weights_path}")
            self.is_ready = False

    def classify_crop(self, crop: np.ndarray) -> tuple[str, float]:
        """Classify a single crop of a changed region.

        Args:
            crop: (C, H, W) array — satellite imagery crop

        Returns:
            (class_name, confidence) — predicted class and softmax confidence
        """
        if not self.is_ready or self.model is None:
            return self._deterministic_classify(crop)

        import torch
        import torch.nn.functional as F

        # Ensure correct channels
        if crop.shape[0] != self.in_channels:
            if crop.shape[0] > self.in_channels:
                crop = crop[:self.in_channels]
            else:
                crop = np.concatenate([crop, crop[:1]] * (self.in_channels - crop.shape[0]), axis=0)

        # Ensure 64x64
        c, h, w = crop.shape
        if h != 64 or w != 64:
            from scipy.ndimage import zoom
            resized = np.zeros((c, 64, 64), dtype=np.float32)
            for i in range(c):
                resized[i] = zoom(crop[i], (64 / h, 64 / w), order=0)
            crop = resized

        with torch.no_grad():
            x = torch.from_numpy(crop).float().unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        class_name = self.class_names[pred.item()]
        return class_name, conf.item()

    def classify_change_crops(
        self,
        image: np.ndarray,
        change_mask: np.ndarray,
        crop_size: int = 64,
    ) -> dict:
        """Classify all changed regions in an image.

        Extracts crops from changed regions, classifies each, and returns
        a summary with per-class counts and false-alarm filtering.

        Args:
            image: (C, H, W) satellite imagery
            change_mask: (H, W) binary change mask from Stage 1
            crop_size: Size of crops to extract

        Returns:
            Dict with:
              - classifications: list of (class_name, confidence) per crop
              - class_distribution: dict of class_name → count
              - false_alarm_count: number of crops classified as false alarms
              - filtered_mask: change mask with false-alarm regions removed
              - source: "segformer" | "deterministic_fallback"
        """
        from scipy import ndimage

        h, w = change_mask.shape
        source = "segformer" if self.is_ready else "deterministic_fallback"

        # Find changed regions
        labeled, n_regions = ndimage.label(change_mask > 0)
        if n_regions == 0:
            return {
                "classifications": [],
                "class_distribution": {},
                "false_alarm_count": 0,
                "filtered_mask": change_mask.copy(),
                "source": source,
            }

        classifications = []
        class_counts = {name: 0 for name in self.class_names}
        false_alarm_labels = set()

        for region_id in range(1, n_regions + 1):
            region_mask = (labeled == region_id)
            cy, cx = ndimage.center_of_mass(region_mask)

            # Extract crop around the region center
            y0 = max(0, int(cy) - crop_size // 2)
            x0 = max(0, int(cx) - crop_size // 2)
            y1 = min(h, y0 + crop_size)
            x1 = min(w, x0 + crop_size)

            crop = image[:, y0:y1, x0:x1]
            if crop.shape[1] < crop_size or crop.shape[2] < crop_size:
                pad_h = crop_size - crop.shape[1]
                pad_w = crop_size - crop.shape[2]
                crop = np.pad(crop, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

            class_name, conf = self.classify_crop(crop)
            classifications.append((class_name, conf))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            if class_name in FALSE_ALARM_CLASSES:
                false_alarm_labels.add(region_id)

        # Build filtered mask (remove false alarm regions)
        filtered_mask = change_mask.copy()
        for region_id in false_alarm_labels:
            filtered_mask[labeled == region_id] = 0

        return {
            "classifications": classifications,
            "class_distribution": class_counts,
            "false_alarm_count": len(false_alarm_labels),
            "filtered_mask": filtered_mask,
            "source": source,
        }

    @staticmethod
    def _deterministic_classify(crop: np.ndarray) -> tuple[str, float]:
        """Deterministic fallback using backscatter statistics."""
        vv = crop[0] if crop.shape[0] >= 1 else crop.mean()
        vh = crop[1] if crop.shape[0] >= 2 else vv

        # Denormalize if needed (training normalizes [-50, 20] dB → [0, 1])
        if float(np.abs(vv).max()) <= 1.0:
            vv_db = vv * 70 - 50
            vh_db = vh * 70 - 50
        else:
            vv_db, vh_db = vv, vh

        vv_mean = float(np.mean(vv_db))

        # Order matters: check shadow (darkest) first, then water, then rock
        if vv_mean < -25:
            return "shadow", 0.6
        elif vv_mean < -22:
            return "water", 0.7
        elif vv_mean > -8:
            return "bare_rock", 0.65
        elif -15 < vv_mean <= -8:
            return "debris", 0.55
        else:
            return "snowmelt", 0.5
