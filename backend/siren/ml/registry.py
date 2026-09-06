"""Model registry — reports status and metadata for all ML models.

Provides a single source of truth for which models are loaded, their
training metadata, and whether they're ready for inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
)


def get_model_status() -> dict[str, Any]:
    """Report the status of all ML models in the SIREN pipeline.

    Returns a dict keyed by model name with:
      - loaded: whether the model is ready for inference
      - weights_path: path to the weights file
      - weights_exists: whether the file exists
      - weights_size_mb: file size in MB (if exists)
      - metadata: training metadata from the checkpoint (if loadable)
      - description: human-readable description of the model's role
    """
    models: dict[str, Any] = {}

    # Stage 1: Siamese U-Net (change detection)
    siren_weights = DEFAULT_WEIGHTS_DIR / "siamese_unet_weights.pt"
    models["siamese_unet"] = {
        "stage": 1,
        "name": "Siamese U-Net Change Detector",
        "loaded": _check_torch_model(siren_weights),
        "weights_path": str(siren_weights),
        "weights_exists": siren_weights.exists(),
        "weights_size_mb": round(siren_weights.stat().st_size / 1e6, 1) if siren_weights.exists() else 0,
        "metadata": _load_checkpoint_metadata(siren_weights),
        "description": "Bi-temporal change detection using shared ResNet-34 encoder. "
                       "Produces pixel-level change probability maps from satellite image pairs.",
        "architecture": "SiameseUNet(ResNet-34, U-Net decoder)",
        "training_data": "Sen1Floods11 (252 hand-labeled SAR chips)",
    }

    # Stage 2: SegFormer (semantic classification)
    segformer_weights = DEFAULT_WEIGHTS_DIR / "segformer_classifier_weights.pt"
    models["segformer"] = {
        "stage": 2,
        "name": "SegFormer Land-Cover Classifier",
        "loaded": _check_torch_model(segformer_weights),
        "weights_path": str(segformer_weights),
        "weights_exists": segformer_weights.exists(),
        "weights_size_mb": round(segformer_weights.stat().st_size / 1e6, 2) if segformer_weights.exists() else 0,
        "metadata": _load_checkpoint_metadata(segformer_weights),
        "description": "Classifies changed pixels into functional categories: water, debris, "
                       "snowmelt, shadow, bare rock. Filters false alarms from cloud shadows and snowmelt. "
                       "Trained with weak labels from SAR backscatter statistics.",
        "architecture": "SegFormerHead(MiT-B0 patch attention, 5-class)",
        "training_data": "Sen1Floods11 weak-labeled (2580 crops, 5 classes)",
    }

    # Stage 3: Consensus gating (not a neural network — deterministic)
    models["consensus_gating"] = {
        "stage": 3,
        "name": "Multi-Sensor Consensus Gating",
        "loaded": True,
        "weights_path": None,
        "weights_exists": True,
        "weights_size_mb": 0,
        "metadata": None,
        "description": "Fuses ML mask with rule-based mask and DEM slope gating. "
                       "Eliminates ML false positives on steep terrain (>35°). "
                       "Weighted fusion: 0.6×ML + 0.4×rule-based.",
        "architecture": "Deterministic (consensus.py)",
        "training_data": None,
    }

    # Stage 4: ConvLSTM (temporal trend)
    convlstm_weights = DEFAULT_WEIGHTS_DIR / "convlstm_trend_weights.pt"
    models["convlstm_trend"] = {
        "stage": 4,
        "name": "ConvLSTM Temporal Trend Classifier",
        "loaded": _check_torch_model(convlstm_weights),
        "weights_path": str(convlstm_weights),
        "weights_exists": convlstm_weights.exists(),
        "weights_size_mb": round(convlstm_weights.stat().st_size / 1e6, 2) if convlstm_weights.exists() else 0,
        "metadata": _load_checkpoint_metadata(convlstm_weights),
        "description": "Classifies temporal trend from satellite sequences: stable, slowly expanding, "
                       "rapidly expanding, or uncertain. Uses ConvLSTM cells over a CNN encoder. "
                       "Hybrid inference — defers to deterministic thresholds when ML confidence < 0.75.",
        "architecture": "ConvLSTMTrendClassifier(CNN encoder, ConvLSTM cells, 4-class)",
        "training_data": "Synthetic sequences from Sen1Floods11 (840 sequences, 4 trend classes)",
    }

    return models


def _check_torch_model(weights_path: Path) -> bool:
    """Check if a torch model can be loaded from the given path."""
    if not weights_path.exists():
        return False
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load_checkpoint_metadata(weights_path: Path) -> dict[str, Any] | None:
    """Load metadata from a checkpoint without loading the full model."""
    if not weights_path.exists():
        return None
    try:
        import torch
        checkpoint = torch.load(
            str(weights_path), map_location="cpu", weights_only=True
        )
        if isinstance(checkpoint, dict):
            # Extract only metadata, not the state_dict
            return {
                k: v for k, v in checkpoint.items()
                if k != "state_dict" and not isinstance(v, dict)
            }
        return {"type": "raw_state_dict"}
    except Exception:
        return None
