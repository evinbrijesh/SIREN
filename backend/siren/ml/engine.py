"""Water Segmentation Inference Engine — wraps WaterUNet / WaterResUNet.

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

Checkpoint resolution (2026-09-14): the engine auto-detects the checkpoint
architecture and input contract from the state_dict, so it can serve both

  * ``WaterUNet``      — 2-channel single-date (VV, VH)          [ADR-010]
  * ``WaterResUNet``   — 4-channel terrain-aware (V3 §2.1)
  * ``WaterResUNet``   — 6-channel multi-temporal Kuro Siwo      [ADR-011.1]

The 6-channel Kuro Siwo variant is the gate-passed model
(IoU 0.62 > 0.60, Precision 0.87 >= 0.84 at tau=0.30). Its contract is
``(VV_post, VH_post, VV_pre, VH_pre, dVV, dVH)`` — see
``ml/contract.py::build_kuro_siwo_tensor``. For that variant the engine
builds the multi-temporal tensor from the pre/post date pair.

Usage in the pipeline:
    from siren.ml.engine import ChangeDetectionEngine

    engine = ChangeDetectionEngine()
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

from siren.ml.contract import (
    KURO_SIWO_CHANNELS,
    build_kuro_siwo_tensor,
    normalize_sar,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Default weights path — trained weights saved here by train.py
DEFAULT_WEIGHTS_PATH = _REPO_ROOT / "data" / "processed" / "water_unet_weights.pt"

# Gate-passed 6-channel multi-temporal checkpoint (ADR-011.1).
# This is the only trained, gate-passed neural checkpoint in the repo.
KURO_SIWO_WEIGHTS_PATH = (
    _REPO_ROOT
    / "models"
    / "checkpoints"
    / "water_resunet_kuro_siwo_full"
    / "water_resunet_6ch_kuro_siwo_v1.pt"
)

# Calibrated operating threshold for the Kuro Siwo checkpoint (ADR-011.1).
# The threshold sweep picked tau=0.30 (IoU 0.6197, P 0.8457, R 0.6987);
# the default 0.50 gives IoU 0.6147 / P 0.8710 / R 0.6762. Both clear the
# ADR-011.1 gate; 0.30 is the documented operating point.
KURO_SIWO_CALIBRATED_THRESHOLD: float = 0.30


def _detect_architecture(state_dict: dict) -> tuple[str, int, int]:
    """Detect (architecture, in_channels, base_channels) from a state_dict.

    Distinguishes the two model families by their first encoder block:

      * ``WaterResUNet`` uses ``ResidualBlock`` -> keys ``enc1.conv1.weight``
        and ``enc1.bn1.*``
      * ``WaterUNet`` uses ``DoubleConv`` (nn.Sequential) -> keys
        ``enc1.conv.0.weight`` and ``enc1.conv.1.*``

    Returns:
        (architecture_name, in_channels, base_channels)

    Raises:
        ValueError: if no recognised encoder weight is found.
    """
    key_res = "enc1.conv1.weight"          # ResidualBlock (WaterResUNet)
    key_unet = "enc1.conv.0.weight"        # DoubleConv (WaterUNet)

    if key_res in state_dict:
        weight = state_dict[key_res]
        return "WaterResUNet", int(weight.shape[1]), int(weight.shape[0])
    if key_unet in state_dict:
        weight = state_dict[key_unet]
        return "WaterUNet", int(weight.shape[1]), int(weight.shape[0])

    raise ValueError(
        "unrecognised checkpoint: neither 'enc1.conv1.weight' (WaterResUNet) "
        "nor 'enc1.conv.0.weight' (WaterUNet) found in state_dict"
    )


class ChangeDetectionEngine:
    """Inference engine for SAR water segmentation.

    Despite the class name (kept for pipeline interface compatibility), this
    engine wraps a single-date water segmenter (WaterUNet) or the 6-channel
    multi-temporal WaterResUNet. The ``predict_change_mask`` method segments
    both dates and differences the results — it is NOT a learned change
    detector (Hard Rule 1: the change decision stays deterministic).

    Checkpoint resolution order (first existing wins):
      1. explicit ``weights_path`` argument
      2. ``models/checkpoints/water_resunet_kuro_siwo_full/
         water_resunet_6ch_kuro_siwo_v1.pt``       (ADR-011.1, gate-passed)
      3. ``data/processed/water_unet_weights.pt``   (ADR-010 Stage 1)

    The gate-passed Kuro Siwo 6-channel checkpoint is preferred: it clears
    the ADR-011.1 calibrated gate (IoU 0.62 > 0.60, Precision 0.87 >= 0.84
    at tau=0.30), whereas the 2-channel Sen1Floods11 model has an
    event-holdout IoU of 0.24 (below the load-bearing gate). Both remain
    shadow-only evidence (ADR-010) — neither enters the hazard score.

    Gracefully degrades:
      - If torch is not installed -> is_ready = False, falls back to deterministic
      - If no weights file exists  -> is_ready = False, falls back to deterministic
      - If weights load successfully -> is_ready = True, produces ML predictions
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        device: str = "cpu",
        in_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        self.device = device
        self.weights_path: Path | None = (
            Path(weights_path) if weights_path else None
        )
        self.is_ready = False
        self.model = None
        self.checkpoint_metadata: dict | None = None
        self.architecture: str | None = None
        # in_channels/base_channels are detected from the checkpoint; the
        # explicit argument only acts as a fallback for scaffold mode.
        self.in_channels = in_channels
        self.base_channels = 32
        # Dropout rate for MC Dropout uncertainty (E1, ADR-013 §9.7.4).
        # Dropout2d is parameter-free, so instantiating WaterResUNet with
        # dropout > 0 is checkpoint-compatible — the same state_dict loads
        # unchanged. In eval mode dropout is identity, so the deterministic
        # path is bit-identical to dropout=0; the rate only matters when
        # predict_change_uncertainty() re-enables dropout for MC passes.
        self.dropout_rate = float(dropout)
        # Conformal calibration sidecar (E1): populated by _load_model from
        # conformal_calibration.json next to the checkpoint when present.
        self.conformal_quantile: float | None = None
        self.conformal_gate_passed = False

        try:
            import torch  # noqa: F401
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            logger.info("torch not installed — ML engine disabled, using deterministic fallback")
            return

        self._load_model()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _candidate_paths(self) -> list[Path]:
        """Ordered checkpoint candidates (first existing wins).

        When ``sar_segmentation_expansion`` is promoted, its recorded
        checkpoint is the runtime model — the promotion record must
        describe what actually runs, not just what was evaluated.
        Otherwise the gate-passed Kuro Siwo 6-channel checkpoint leads
        and the ADR-010 Stage 1 2-channel checkpoint is the fallback.
        """
        if self.weights_path is not None:
            return [self.weights_path]
        candidates: list[Path] = []
        try:
            from siren.ml.promotion import is_promoted, promotion_record

            if is_promoted("sar_segmentation_expansion"):
                rec = promotion_record("sar_segmentation_expansion")
                promoted = (
                    _REPO_ROOT / "models" / "checkpoints" / rec["checkpoint"]
                )
                if promoted.exists():
                    candidates.append(promoted)
        except ImportError:
            pass
        candidates += [KURO_SIWO_WEIGHTS_PATH, DEFAULT_WEIGHTS_PATH]
        return candidates

    def _resolve_weights_path(self) -> Path | None:
        for candidate in self._candidate_paths():
            if candidate.exists():
                return candidate
        return None

    def _load_model(self) -> None:
        """Load the first available checkpoint, auto-detecting architecture."""
        import torch
        from siren.ml.model import WaterResUNet, WaterUNet

        resolved = self._resolve_weights_path()

        if resolved is None:
            # Scaffold mode: no weights anywhere. Build a 2-channel WaterUNet
            # so the object is usable, but report is_ready=False.
            self.weights_path = DEFAULT_WEIGHTS_PATH
            fallback_channels = self.in_channels or 2
            self.model = WaterUNet(in_channels=fallback_channels).to(self.device)
            self.model.eval()
            self.architecture = "WaterUNet"
            self.in_channels = fallback_channels
            logger.info(
                f"No trained weights found (looked in {self._candidate_paths()}) — "
                "ML engine in scaffold mode (deterministic fallback active)."
            )
            return

        self.weights_path = resolved
        try:
            checkpoint = torch.load(
                str(resolved), map_location=self.device, weights_only=True
            )
            state_dict = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )

            architecture, detected_in, detected_base = _detect_architecture(state_dict)
            self.architecture = architecture
            self.in_channels = detected_in
            self.base_channels = detected_base

            if architecture == "WaterResUNet":
                self.model = WaterResUNet(
                    in_channels=detected_in,
                    base_channels=detected_base,
                    dropout=self.dropout_rate,
                ).to(self.device)
            else:
                self.model = WaterUNet(
                    in_channels=detected_in, base_channels=detected_base
                ).to(self.device)

            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_ready = True

            if isinstance(checkpoint, dict) and "state_dict" not in checkpoint:
                self.checkpoint_metadata = None
            elif isinstance(checkpoint, dict):
                self.checkpoint_metadata = {
                    k: v for k, v in checkpoint.items() if k != "state_dict"
                }

            # Sidecar metadata (the Kuro Siwo checkpoint ships a .meta.json)
            meta_path = resolved.with_suffix(".meta.json")
            if meta_path.exists():
                try:
                    import json

                    self.checkpoint_metadata = {
                        **(self.checkpoint_metadata or {}),
                        **json.loads(meta_path.read_text()),
                    }
                except Exception as exc:  # noqa: BLE001 — metadata is advisory
                    logger.warning(f"Could not read checkpoint sidecar: {exc}")

            # Conformal calibration sidecar (E1): when the MC Dropout
            # quantile has been calibrated on the held-out Kuro Siwo split
            # (ml/calibrate_uncertainty.py), predict_change_uncertainty
            # uses it instead of nominal quantiles. The gate flag records
            # whether coverage met PRD §17.2 (±5% of nominal 90%).
            cal_path = resolved.parent / "conformal_calibration.json"
            if cal_path.exists():
                try:
                    cal = json.loads(cal_path.read_text())
                    self.conformal_quantile = cal.get("conformal_quantile")
                    self.conformal_gate_passed = bool(cal.get("gate_passed", False))
                    logger.info(
                        f"Conformal calibration loaded: q*={self.conformal_quantile} "
                        f"(gate_passed={self.conformal_gate_passed})"
                    )
                except Exception as exc:  # noqa: BLE001 — advisory
                    logger.warning(f"Could not read conformal sidecar: {exc}")

            logger.info(
                f"ML engine loaded {architecture} weights from {resolved} "
                f"(in_channels={detected_in}, base_channels={detected_base})"
            )
        except Exception as exc:
            logger.warning(
                f"Failed to load ML weights from {resolved}: {exc} — "
                "using deterministic fallback"
            )
            self.model = WaterUNet(in_channels=self.in_channels or 2).to(self.device)
            self.model.eval()
            self.architecture = "WaterUNet"
            self.in_channels = self.in_channels or 2
            self.is_ready = False

    @property
    def is_multitemporal(self) -> bool:
        """True when the loaded checkpoint uses the 6-channel Kuro Siwo contract."""
        return self.in_channels == KURO_SIWO_CHANNELS

    @property
    def default_threshold(self) -> float:
        """Operating threshold for the loaded checkpoint.

        The Kuro Siwo 6-channel model is calibrated at tau=0.30 (ADR-011.1);
        all other checkpoints use the conventional 0.50.
        """
        if self.is_multitemporal:
            return KURO_SIWO_CALIBRATED_THRESHOLD
        return 0.5

    # ------------------------------------------------------------------ #
    # Inference helpers
    # ------------------------------------------------------------------ #
    def _pad_to_multiple(self, arr: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, int, int]:
        """Reflect-pad (C, H, W) to a multiple of ``multiple``. Returns (arr, pad_h, pad_w)."""
        h, w = arr.shape[1], arr.shape[2]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        return arr, pad_h, pad_w

    def _forward_logits(self, tensor_chw: np.ndarray) -> np.ndarray:
        """Run the model on a (C, H, W) tensor, returning (H, W) probabilities."""
        import torch

        tensor, pad_h, pad_w = self._pad_to_multiple(tensor_chw.astype(np.float32))
        with torch.no_grad():
            x = torch.from_numpy(tensor).float().unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        if pad_h or pad_w:
            probs = probs[: tensor.shape[1] - pad_h, : tensor.shape[2] - pad_w]
        return probs.astype(np.float32)

    def _build_input(self, pre_db: np.ndarray, post_db: np.ndarray) -> np.ndarray:
        """Build the model input tensor for the loaded contract.

        Multi-temporal (6-ch): full (VV_post, VH_post, VV_pre, VH_pre, dVV, dVH)
        contract via ``build_kuro_siwo_tensor``.

        Legacy (2-ch/4-ch): single-date channels from the post image, passed
        through the frozen ``normalize_sar`` contract (idempotent).
        """
        if self.is_multitemporal:
            return build_kuro_siwo_tensor(pre_db, post_db)

        post = np.asarray(post_db, dtype=np.float32)
        if post.shape[0] != self.in_channels:
            post = self._adjust_channels(post, self.in_channels)
        return normalize_sar(post)

    # ------------------------------------------------------------------ #
    # Public inference API
    # ------------------------------------------------------------------ #
    def predict_water_probability(
        self,
        sar_raster: np.ndarray,
    ) -> np.ndarray:
        """Segment water and return the raw probability map (H, W) in [0, 1].

        For the multi-temporal contract there is no pre-event image available
        in this call, so the tensor is built with pre == post (Δσ⁰ = 0) — the
        model then behaves as a single-date segmenter. Use
        ``predict_change_probability`` when both dates are available.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                "ML engine not ready — no trained weights loaded."
            )

        sar = np.asarray(sar_raster, dtype=np.float32)
        if sar.ndim != 3 or sar.shape[0] < 2:
            raise ValueError(
                f"expected (2, H, W) VV/VH dB input, got shape {sar.shape}"
            )
        pair = sar[:2]
        tensor = self._build_input(pair, pair)
        return self._forward_logits(tensor)

    def predict_water_mask(
        self,
        sar_raster: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Segment surface water from a single-date SAR raster.

        Args:
            sar_raster: SAR sigma0 in dB, shape (2, H, W) or (C, H, W).
                        VV/VH channel order expected (per ml/contract.py).
                        Values may be raw dB or already normalized — the
                        contract normalization is applied idempotently.
            threshold: water probability threshold for binary mask.
                       Defaults to ``self.default_threshold`` (0.30 for the
                       calibrated Kuro Siwo checkpoint, 0.50 otherwise).

        Returns:
            Binary water mask (H, W) as uint8 (1 = water, 0 = land).
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                "ML engine not ready — no trained weights loaded. "
                "Use the deterministic mask instead."
            )
        tau = self.default_threshold if threshold is None else threshold
        return (self.predict_water_probability(sar_raster) >= tau).astype(np.uint8)

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

        t0 = np.asarray(t0_raster, dtype=np.float32)[:2]
        t1 = np.asarray(t1_raster, dtype=np.float32)[:2]
        if t0.shape != t1.shape:
            raise ValueError(
                f"pre/post spatial mismatch: {t0.shape} vs {t1.shape}"
            )

        # Water probability at each date under the loaded contract.
        # Multi-temporal: post=t1 with the real pre-event context (in-distribution),
        # and post=t0 with itself (Δ=0) for the baseline date.
        p_t1 = self._forward_logits(self._build_input(t0, t1))
        p_t0 = self._forward_logits(self._build_input(t0, t0))

        # Positive change only (expansion), clipped to [0, 1]
        delta = p_t1 - p_t0
        return np.clip(delta, 0.0, 1.0).astype(np.float32)

    def predict_change_mask(
        self,
        t0_raster: np.ndarray,
        t1_raster: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Bi-temporal change mask via deterministic differencing of two
        independently-segmented per-date water masks.

        This is NOT a learned change detector. The ML model segments water
        on each date; the change decision is the deterministic set difference
        (water at t1 but not at t0). This preserves Hard Rule 1: the change
        decision remains deterministic even when the ML water segmenter is
        available.

        Args:
            t0_raster: Baseline SAR image (2, H, W) in dB (VV, VH).
            t1_raster: Current SAR image (2, H, W) in dB (VV, VH).
            threshold: Water probability threshold per date. Defaults to
                       ``self.default_threshold`` (0.30 for the calibrated
                       Kuro Siwo checkpoint, 0.50 otherwise).

        Returns:
            Binary change mask (H, W) as uint8 (1 = new water at t1).
        """
        return self.predict_state_and_change(t0_raster, t1_raster, threshold)[
            "expansion"
        ]

    def predict_state_and_change(
        self,
        t0_raster: np.ndarray,
        t1_raster: np.ndarray,
        threshold: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-date water state plus directional change masks.

        Same two forward passes as ``predict_change_mask``, but returns the
        full state/change decomposition instead of expansion alone:

          * ``water_t0``   — water extent at the baseline date
          * ``water_t1``   — water extent at the current date
          * ``expansion``  — new water: ``water_t1 & ~water_t0``
          * ``expansion_dp`` — Δp expansion (ADR-014-am1 contract):
            ``(p_t1 >= 0.5) & (p_t1 - p_t0 >= 0.2)``
          * ``drainage``   — receded water: ``water_t0 & ~water_t1``

        The state-vs-change split matters for persistent lakes: a correct
        segmenter detects Imja at BOTH dates, so ``expansion`` is ~empty
        even though the lake is clearly present — a change-only display
        makes a working model look broken (verified in the 2026-09-17
        weak-label domain-adaptation experiment). Displaying ``water_t1``
        alongside ``expansion`` keeps persistent water visible.

        Args:
            t0_raster: Baseline SAR image (2, H, W) in dB (VV, VH).
            t1_raster: Current SAR image (2, H, W) in dB (VV, VH).
            threshold: Water probability threshold per date. Defaults to
                       ``self.default_threshold`` (0.30 for the calibrated
                       Kuro Siwo checkpoint, 0.50 otherwise).

        Returns:
            Dict of binary masks (H, W) uint8: water_t0, water_t1,
            expansion, expansion_dp, drainage.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                "ML engine not ready — no trained weights loaded. "
                "Use the deterministic mask instead."
            )

        tau = self.default_threshold if threshold is None else threshold
        t0 = np.asarray(t0_raster, dtype=np.float32)[:2]
        t1 = np.asarray(t1_raster, dtype=np.float32)[:2]
        if t0.shape != t1.shape:
            raise ValueError(
                f"pre/post spatial mismatch: {t0.shape} vs {t1.shape}"
            )

        # Per-date water masks. For the multi-temporal contract the t1
        # prediction uses the real pre-event context (the model's trained
        # task); t0 uses itself with Δσ⁰ = 0.
        p_t1 = self._forward_logits(self._build_input(t0, t1))
        p_t0 = self._forward_logits(self._build_input(t0, t0))
        water_t0 = p_t0 >= tau
        water_t1 = p_t1 >= tau

        # Deterministic change decomposition (Hard Rule 1 — set differences,
        # not a learned change detector). ``expansion_dp`` adds the Δp
        # variant evaluated under ADR-014-am1: confident post-prob AND a
        # meaningful probability rise — catches sub-pixel footprint growth
        # the binary extent difference structurally misses.
        dp = p_t1 - p_t0
        return {
            "water_t0": water_t0.astype(np.uint8),
            "water_t1": water_t1.astype(np.uint8),
            "expansion": (water_t1 & ~water_t0).astype(np.uint8),
            "expansion_dp": ((p_t1 >= 0.5) & (dp >= 0.2)).astype(np.uint8),
            "drainage": (water_t0 & ~water_t1).astype(np.uint8),
        }

    # ------------------------------------------------------------------ #
    # MC Dropout uncertainty (E1, ADR-013 §9.7.4)
    # ------------------------------------------------------------------ #
    def has_dropout_layers(self) -> bool:
        """True when the loaded model contains Dropout modules.

        The gate-passed checkpoints were trained with ``dropout=0.0`` (the
        ResidualBlock uses ``nn.Identity``), so an engine constructed with
        the default ``dropout=0.0`` has no stochastic layers and MC
        sampling would produce degenerate zero-variance maps. Construct the
        engine with ``dropout > 0`` to enable MC Dropout.
        """
        if self.model is None:
            return False
        import torch.nn as nn

        return any(
            isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
            for m in self.model.modules()
        )

    def _mc_forward(self, tensor_chw: np.ndarray, **mc_kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Run MC Dropout on a (C, H, W) tensor; returns (mean, variance) (H, W)."""
        import torch

        from siren.ml.uncertainty import mc_dropout_inference

        tensor, pad_h, pad_w = self._pad_to_multiple(tensor_chw.astype(np.float32))
        x = torch.from_numpy(tensor).float().unsqueeze(0).to(self.device)
        result = mc_dropout_inference(self.model, x, apply_sigmoid=True, **mc_kwargs)
        mean = result.mean[0, 0]
        var = result.variance[0, 0]
        if pad_h or pad_w:
            mean = mean[: tensor.shape[1] - pad_h, : tensor.shape[2] - pad_w]
            var = var[: tensor.shape[1] - pad_h, : tensor.shape[2] - pad_w]
        return mean.astype(np.float32), var.astype(np.float32)

    def predict_change_uncertainty(
        self,
        t0_raster: np.ndarray,
        t1_raster: np.ndarray,
        n_samples: int = 20,
        confidence_level: float = 0.90,
        conformal_quantile: float | None = None,
        threshold: float | None = None,
        seed: int = 42,
    ) -> dict:
        """Bi-temporal change mask + spatially-resolved epistemic uncertainty.

        Runs T stochastic MC Dropout forward passes (Gal & Ghahramani 2016)
        on the post-date tensor and on the self-paired pre-date tensor —
        the same two tensors ``predict_change_mask`` uses — and returns the
        per-pixel mean probability, variance σ²(x, y), and the change mask
        derived from the MC mean.

        The variance of the change signal is var(p_t1) + var(p_t0): the two
        MC runs use independent dropout draws, so their errors are
        independent and variances add.

        Informational only until the conformal gate passes (PRD §17.2:
        empirical coverage within ±5% of the nominal 90% level on a
        held-out calibration set). Never enters the hazard score (ADR-010).

        Args:
            t0_raster: Baseline SAR image (2, H, W) in dB (VV, VH).
            t1_raster: Current SAR image (2, H, W) in dB (VV, VH).
            n_samples: number of MC forward passes T per date.
            confidence_level: nominal coverage (0.90 = 90% CI).
            conformal_quantile: calibrated quantile from
                ``uncertainty.calibrate_conformal``; None = nominal quantiles.
            threshold: water probability threshold; defaults to
                ``self.default_threshold``.
            seed: RNG seed for the dropout draws (Hard Rule 6 — identical
                inputs + seed → identical uncertainty map on the same
                device/backend).

        Returns:
            Dict with per-date mean/variance maps, the change mask, the
            combined change variance, and the uncertainty method tag.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                "ML engine not ready — no trained weights loaded. "
                "Use the deterministic mask instead."
            )
        import torch

        tau = self.default_threshold if threshold is None else threshold
        t0 = np.asarray(t0_raster, dtype=np.float32)[:2]
        t1 = np.asarray(t1_raster, dtype=np.float32)[:2]
        if t0.shape != t1.shape:
            raise ValueError(
                f"pre/post spatial mismatch: {t0.shape} vs {t1.shape}"
            )

        # Prefer the calibrated conformal quantile when a sidecar exists.
        if conformal_quantile is None:
            conformal_quantile = self.conformal_quantile

        has_dropout = self.has_dropout_layers()
        if not has_dropout:
            logger.warning(
                "Loaded checkpoint has no dropout layers (engine dropout=%.2f) "
                "— MC variance is degenerate (all passes identical). "
                "Construct the engine with dropout>0 for real uncertainty.",
                self.dropout_rate,
            )

        mc_kwargs = {
            "n_samples": n_samples,
            "confidence_level": confidence_level,
            "conformal_quantile": conformal_quantile,
        }

        # Seed before the stochastic passes (Hard Rule 6): identical inputs
        # + seed → identical dropout draws on the same device/backend.
        torch.manual_seed(seed)
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        mean_t1, var_t1 = self._mc_forward(self._build_input(t0, t1), **mc_kwargs)
        mean_t0, var_t0 = self._mc_forward(self._build_input(t0, t0), **mc_kwargs)

        # Restore strict eval mode (mc_dropout_inference enables dropout).
        self.model.eval()

        change = ((mean_t1 >= tau) & ~(mean_t0 >= tau)).astype(np.uint8)
        change_var = (var_t0 + var_t1).astype(np.float32)

        return {
            "change_mask": change,
            "water_prob_t1": mean_t1,
            "variance_t1": var_t1,
            "water_prob_t0": mean_t0,
            "variance_t0": var_t0,
            "change_variance": change_var,
            "n_samples": n_samples,
            "confidence_level": confidence_level,
            "conformal_quantile": conformal_quantile,
            "conformal_gate_passed": self.conformal_gate_passed,
            "method": f"mc_dropout_t{n_samples}",
            "has_dropout": has_dropout,
        }

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
