"""Model registry — reports status and metadata for all ML models.

Provides a single source of truth for which models are loaded, their
training metadata, and whether they're ready for inference.

Per ADR-010, the only qualified model is the single-date SAR water
segmentation U-Net (``water_unet``). The three previous checkpoints
(Siamese U-Net, SegFormer crop classifier, ConvLSTM trend) were
disqualified by the 2026-09-07 DL audit and archived to
``data/archived_disqualified/``. They are reported here as ``archived``
with their disqualification reason, not as ``loaded``.

PRD v4.7 §17.3 adds four additional disqualified checkpoints that remain
on disk for audit history but must never be promoted to inference:
``water_resunet_6ch_v1`` (label leakage), ``xgboost_susceptibility_v1``
and ``xgboost_susceptibility_v2_real`` (generated/label-dependent
features), and ``fno_hydro_surrogate_v1`` (synthetic terrain, no real
multi-basin validation). Their status is reported as ``disqualified``
regardless of whether the weight files deserialize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
)

ARCHIVED_WEIGHTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "archived_disqualified"
)

CHECKPOINTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "checkpoints"
)


# Disqualified checkpoints that remain on disk for audit history (PRD v4.7 §17.3).
# Status is determined by provenance, not by whether the file deserializes.
DISQUALIFIED_CHECKPOINTS = [
    {
        "name": "water_resunet_6ch_v1",
        "display": "WaterResUNet 6-channel (multi-temporal, disqualified)",
        "reason": (
            "Label leakage: the Δσ⁰ channel is synthesized from water labels "
            "(flood drop = -8 dB + noise) during training, encoding the answer "
            "in the input. Reported test IoU 0.9999 is invalid. Retrain with "
            "real paired pre/post SAR to obtain an honest evaluation."
        ),
    },
    {
        "name": "xgboost_susceptibility_v1",
        "display": "XGBoost breach susceptibility v1 (disqualified)",
        "reason": (
            "Generated/label-dependent features: the training dataset fills "
            "missing lake area, dam geometry, expansion rate, and rainfall "
            "anomaly with label-conditioned random distributions. The reported "
            "CV Brier 0.0253 and ROC-AUC 0.9948 are not valid real-data evidence."
        ),
    },
    {
        "name": "xgboost_susceptibility_v2_real",
        "display": "XGBoost breach susceptibility v2 'real' (disqualified)",
        "reason": (
            "Same generated-feature contamination as v1. The 'real' label was "
            "applied before the label-dependent feature generation path was "
            "identified. The Brier gate pass is withdrawn pending a genuine "
            "real-data evaluation."
        ),
    },
    {
        "name": "fno_hydro_surrogate_v1",
        "display": "FNO hydrodynamic surrogate v1 (disqualified)",
        "reason": (
            "Synthetic training data only (800 simulated runs). No real "
            "multi-basin hydraulic validation. The runtime shadow path "
            "previously fabricated a Teesta-shaped DEM when real terrain was "
            "absent. Inference is blocked until a valid checkpoint with real "
            "terrain validation is produced."
        ),
    },
]


def get_model_status() -> dict[str, Any]:
    """Report the status of all ML models in the SIREN pipeline.

    Returns a dict keyed by model name with:
      - loaded: whether the model is ready for inference
      - weights_path: path to the weights file
      - weights_exists: whether the file exists
      - weights_size_mb: file size in MB (if exists)
      - metadata: training metadata from the checkpoint (if loadable)
      - description: human-readable description of the model's role
      - status: qualification status (active / shadow / unavailable /
        disqualified / archived_disqualified)
      - inference_allowed: whether inference may run on this checkpoint
      - evaluation_valid: whether the reported metrics are valid evidence
    """
    models: dict[str, Any] = {}

    # --- Active model (ADR-010 Stage 1) ---
    # Single-date SAR water segmentation U-Net.
    # Trained on Sen1Floods11 hand-labeled chips with official event-level splits.
    # Input contract: 2-ch VV/VH sigma0 dB, normalized via ml/contract.py.
    water_weights = DEFAULT_WEIGHTS_DIR / "water_unet_weights.pt"
    water_loaded = _check_torch_model(water_weights)
    models["water_unet"] = {
        "stage": 1,
        "name": "Single-Date SAR Water Segmentation U-Net",
        "loaded": water_loaded,
        "weights_path": str(water_weights),
        "weights_exists": water_weights.exists(),
        "weights_size_mb": round(water_weights.stat().st_size / 1e6, 1) if water_weights.exists() else 0,
        "metadata": _load_checkpoint_metadata(water_weights),
        "description": "Per-date surface water segmentation from Sentinel-1 VV/VH sigma0. "
                       "Change detection is deterministic bi-temporal differencing of per-date masks. "
                       "ML evidence only — never the sole source of a hazard score (ADR-010).",
        "architecture": "WaterUNet(2-ch VV/VH, U-Net decoder, <=10M params)",
        "training_data": "Sen1Floods11 hand-labeled, official event-level splits",
        "input_contract": "ml/contract.py: normalize_sar(), [-30, 0] dB → [0, 1]",
        "status": "active" if water_loaded else "unavailable",
        "inference_allowed": water_loaded,
        "evaluation_valid": water_loaded,
    }

    # --- 6-channel real-data SAR model (ADR-011.1) ---
    # Trained on real paired pre/post Sentinel-1 GRD from Kuro Siwo.
    # First honest 6-channel baseline (no synthetic Δσ⁰ leakage).
    # Passes ADR-011.1 calibrated gate (IoU > 0.60 AND Precision >= 0.84).
    ks_dir = CHECKPOINTS_DIR / "water_resunet_kuro_siwo_full"
    ks_weights = ks_dir / "water_resunet_6ch_kuro_siwo_v1.pt"
    ks_meta = ks_dir / "water_resunet_6ch_kuro_siwo_v1.meta.json"
    ks_loaded = _check_torch_model(ks_weights)
    ks_metadata = _load_sidecar_metadata(ks_meta)
    ks_gate_passed = False
    if ks_metadata and ks_metadata.get("gate", {}).get("adr_passed"):
        ks_gate_passed = True
    models["water_resunet_6ch_kuro_siwo"] = {
        "stage": 2,
        "name": "WaterResUNet 6-channel (Kuro Siwo real paired SAR)",
        "loaded": ks_loaded,
        "weights_path": str(ks_weights),
        "weights_exists": ks_weights.exists(),
        "weights_size_mb": round(ks_weights.stat().st_size / 1e6, 1) if ks_weights.exists() else 0,
        "metadata": ks_metadata,
        "description": (
            "6-channel multi-temporal water segmentation from real paired pre/post "
            "Sentinel-1 GRD (VV_post, VH_post, VV_pre, VH_pre, ΔVV, ΔVH). "
            "First honest 6-channel baseline — no synthetic Δσ⁰ leakage (PRD v4.7 §17.3). "
            "Passes ADR-011.1 calibrated gate (IoU > 0.60 AND P >= 0.84) at τ=0.30. "
            "Pending shadow-mode observation cycle before load-bearing promotion."
        ),
        "architecture": "WaterResUNet(6-ch, U-Net decoder, 7.94M params)",
        "training_data": "Kuro Siwo GRD, real paired pre/post Sentinel-1, ~6,775 samples (truncated shards)",
        "input_contract": "6-ch: VV_post, VH_post, VV_pre, VH_pre, ΔVV, ΔVH (linear→dB→normalized)",
        "status": "shadow_pending_promotion" if ks_gate_passed else "shadow_only",
        "inference_allowed": ks_gate_passed,
        "evaluation_valid": True,
        "gate": ks_metadata.get("gate", {}) if ks_metadata else None,
    }

    # --- Promoted component: Himalayan adapter expansion evidence ---
    # ADR-014-am1 (2026-09-19, owner-directed): the Δp expansion product
    # of the label-refined v2 adapter is promoted to primary expansion
    # evidence under the union policy — deterministic change evidence
    # stays live as the labeled cross-check. Per-date extent masks of
    # the same checkpoint remain unqualified shadow evidence.
    try:
        from siren.ml.promotion import (
            is_promoted,
            promotion_record,
        )
        exp_rec = (
            promotion_record("sar_segmentation_expansion")
            if is_promoted("sar_segmentation_expansion") else None
        )
    except ImportError:
        exp_rec = None
    if exp_rec is not None:
        ckpt = CHECKPOINTS_DIR / exp_rec["checkpoint"]
        models["himalayan_adapter_multidate"] = {
            "stage": 3,
            "name": "Himalayan adapter (multi-date) — Δp expansion evidence",
            "loaded": ckpt.exists(),
            "weights_path": str(ckpt),
            "weights_exists": ckpt.exists(),
            "weights_size_mb": round(ckpt.stat().st_size / 1e6, 1) if ckpt.exists() else 0,
            "description": (
                "Promoted expansion evidence (ADR-014-am1): "
                "(p1>=0.5)&(p1-p0>=0.2) within monitorable-lake vicinity "
                "on ro-121 desc unfrozen pairs. Multi-date fine-tune "
                "(3 SAR pairs) — 50% verified-change recall on the "
                "held-out gold pair vs 28% for the single-date adapter. "
                "Union policy: deterministic change stays live as "
                "labeled cross-check. Extent masks remain shadow. "
                "Caveat: thin verified-change truth (~30-50px/eval) — "
                "see ml/promotion.py."
            ),
            "status": "promoted_component",
            "promoted_component": "sar_segmentation_expansion",
            "promotion": exp_rec,
            "inference_allowed": True,
            "evaluation_valid": True,
        }

    # --- Disqualified checkpoints (PRD v4.7 §17.3) ---
    # These remain on disk for audit history but must not be promoted.
    for entry in DISQUALIFIED_CHECKPOINTS:
        name = entry["name"]
        weights_file = CHECKPOINTS_DIR / f"{name}.pt"
        meta_file = CHECKPOINTS_DIR / f"{name}.meta.json"
        # Fallback to .json for XGBoost checkpoints (no .pt file).
        if not weights_file.exists():
            alt = CHECKPOINTS_DIR / f"{name}.json"
            if alt.exists():
                weights_file = alt
        models[name] = {
            "stage": "disqualified",
            "name": entry["display"],
            "loaded": False,
            "weights_path": str(weights_file),
            "weights_exists": weights_file.exists(),
            "weights_size_mb": round(weights_file.stat().st_size / 1e6, 2) if weights_file.exists() else 0,
            "metadata": _load_sidecar_metadata(meta_file),
            "description": f"DISQUALIFIED — {entry['reason']}",
            "status": "disqualified",
            "inference_allowed": False,
            "evaluation_valid": False,
            "disqualification_reason": entry["reason"],
        }

    # --- Archived / disqualified models (ADR-010) ---
    # These checkpoints are NOT loaded. They are reported for transparency.
    archived_models = [
        ("siamese_unet", "Siamese U-Net (bi-temporal, disqualified)",
         "Label leakage: training synthesizes 'before' images from labels; "
         "runtime feeds binary masks instead of sigma0. ADR-010 §1 verdict: replace."),
        ("segformer_classifier", "SegFormer crop classifier (disqualified)",
         "Not the SegFormer architecture; threshold-generated weak labels; "
         "unreachable 'shadow' class; can delete rule-detected evidence. ADR-010 §1 verdict: drop."),
        ("convlstm_trend", "ConvLSTM trend classifier (disqualified)",
         "Trained on synthetic mask progressions; no elapsed-time input; "
         "inference fabricates missing timesteps by dilation. ADR-010 §1 verdict: replace."),
    ]

    for name, display_name, reason in archived_models:
        weights_file = ARCHIVED_WEIGHTS_DIR / f"{name}_weights.pt"
        models[name] = {
            "stage": "archived",
            "name": display_name,
            "loaded": False,
            "weights_path": str(weights_file),
            "weights_exists": weights_file.exists(),
            "weights_size_mb": round(weights_file.stat().st_size / 1e6, 2) if weights_file.exists() else 0,
            "metadata": _load_checkpoint_metadata(weights_file),
            "description": f"ARCHIVED — disqualified by 2026-09-07 DL audit. {reason}",
            "status": "archived_disqualified",
            "inference_allowed": False,
            "evaluation_valid": False,
            "disqualification_reason": reason,
        }

    # --- Deterministic consensus (not a neural network) ---
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
                       "Weighted fusion: 0.6xML + 0.4xrule-based.",
        "architecture": "Deterministic (consensus.py)",
        "training_data": None,
        "status": "active",
        "inference_allowed": True,
        "evaluation_valid": True,
    }

    return models


def _check_torch_model(weights_path: Path) -> bool:
    """Check if a torch model can be loaded from the given path.

    Per ADR-010 audit finding §5.8, this checks actual loadability
    (file exists AND torch is installed AND the checkpoint can be
    deserialized), not just file existence.

    Handles both wrapped checkpoints (dict with "state_dict" key) and
    raw state_dicts (dict of param_name → tensor).
    """
    if not weights_path.exists():
        return False
    try:
        import torch  # noqa: F401
        checkpoint = torch.load(
            str(weights_path), map_location="cpu", weights_only=True
        )
        # Wrapped checkpoint: {"state_dict": {...}, "metadata": {...}}
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return True
        # Raw state_dict: {"enc1.conv1.weight": tensor, ...}
        # Check that it's a non-empty dict with tensor values
        if isinstance(checkpoint, dict) and len(checkpoint) > 0:
            first_val = next(iter(checkpoint.values()))
            if hasattr(first_val, "shape"):
                return True
        return False
    except Exception:
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


def _load_sidecar_metadata(meta_path: Path) -> dict[str, Any] | None:
    """Load a JSON metadata sidecar without loading model weights."""
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None
