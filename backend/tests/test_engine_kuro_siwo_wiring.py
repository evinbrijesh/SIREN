"""Tests for the Kuro Siwo 6-channel checkpoint wiring (ADR-011.1).

Covers:
  * ``build_kuro_siwo_tensor`` matches the training-side contract exactly
  * architecture auto-detection from a state_dict
  * ``ChangeDetectionEngine`` prefers the gate-passed 6-channel checkpoint
  * 6-channel inference preserves the deterministic-differencing property
    (Hard Rule 1: identical inputs -> zero change)
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- #
# Contract: build_kuro_siwo_tensor
# --------------------------------------------------------------------------- #

def test_build_kuro_siwo_tensor_matches_training_contract() -> None:
    """The inference-side tensor builder must match training exactly.

    Training builds the 6-channel tensor from linear-scale SAR via
    ``kuro_siwo_dataset._build_tensor``; the pipeline carries already-calibrated
    dB. Both paths must produce identical tensors.
    """
    from siren.ml.contract import build_kuro_siwo_tensor
    from siren.ml.kuro_siwo_dataset import _build_tensor, _linear_to_db

    rng = np.random.RandomState(0)
    flood_vv = rng.uniform(0.001, 0.3, (1, 64, 64)).astype(np.float32)
    flood_vh = rng.uniform(0.0005, 0.15, (1, 64, 64)).astype(np.float32)
    pre_vv = rng.uniform(0.002, 0.4, (1, 64, 64)).astype(np.float32)
    pre_vh = rng.uniform(0.001, 0.2, (1, 64, 64)).astype(np.float32)

    train_tensor = _build_tensor(flood_vv, flood_vh, pre_vv, pre_vh)

    pre_db = np.stack([_linear_to_db(pre_vv[0]), _linear_to_db(pre_vh[0])], axis=0)
    post_db = np.stack([_linear_to_db(flood_vv[0]), _linear_to_db(flood_vh[0])], axis=0)
    infer_tensor = build_kuro_siwo_tensor(pre_db, post_db)

    assert infer_tensor.shape == (6, 64, 64)
    assert infer_tensor.dtype == np.float32
    assert float(infer_tensor.min()) >= 0.0
    assert float(infer_tensor.max()) <= 1.0
    np.testing.assert_allclose(infer_tensor, train_tensor, atol=1e-5)


def test_build_kuro_siwo_tensor_channel_semantics() -> None:
    """Channel order must be (VV_post, VH_post, VV_pre, VH_pre, dVV, dVH)."""
    from siren.ml.contract import build_kuro_siwo_tensor

    # Distinct constant levels so each channel is identifiable
    pre = np.full((2, 8, 8), -20.0, dtype=np.float32)
    post = np.full((2, 8, 8), -10.0, dtype=np.float32)
    post[1] = -24.0  # VH: dVH = -24 - (-20) = -4 dB

    tensor = build_kuro_siwo_tensor(pre, post)

    # (clamp(dB, -30, 0) + 30) / 30
    assert tensor[0, 0, 0] == pytest.approx(20.0 / 30.0, abs=1e-5)   # VV_post
    assert tensor[1, 0, 0] == pytest.approx(6.0 / 30.0, abs=1e-5)    # VH_post
    assert tensor[2, 0, 0] == pytest.approx(10.0 / 30.0, abs=1e-5)   # VV_pre
    assert tensor[3, 0, 0] == pytest.approx(10.0 / 30.0, abs=1e-5)   # VH_pre
    # dVV = +10 dB -> (clamp(10, -15, 5) + 15) / 20 = 20/20 = 1.0
    assert tensor[4, 0, 0] == pytest.approx(1.0, abs=1e-5)
    # dVH = -4 dB -> (clamp(-4, -15, 5) + 15) / 20 = 11/20 = 0.55
    assert tensor[5, 0, 0] == pytest.approx(11.0 / 20.0, abs=1e-5)


def test_build_kuro_siwo_tensor_rejects_wrong_channels() -> None:
    from siren.ml.contract import build_kuro_siwo_tensor

    with pytest.raises(ValueError, match="2-channel"):
        build_kuro_siwo_tensor(
            np.zeros((3, 8, 8), dtype=np.float32),
            np.zeros((2, 8, 8), dtype=np.float32),
        )


def test_build_kuro_siwo_tensor_rejects_shape_mismatch() -> None:
    from siren.ml.contract import build_kuro_siwo_tensor

    with pytest.raises(ValueError, match="shape mismatch"):
        build_kuro_siwo_tensor(
            np.zeros((2, 8, 8), dtype=np.float32),
            np.zeros((2, 16, 16), dtype=np.float32),
        )


# --------------------------------------------------------------------------- #
# Architecture detection
# --------------------------------------------------------------------------- #

def test_detect_architecture_resunet() -> None:
    from siren.ml.engine import _detect_architecture
    from siren.ml.model import WaterResUNet

    model = WaterResUNet(in_channels=6, base_channels=16)
    arch, in_ch, base = _detect_architecture(model.state_dict())
    assert arch == "WaterResUNet"
    assert in_ch == 6
    assert base == 16


def test_detect_architecture_unet() -> None:
    from siren.ml.engine import _detect_architecture
    from siren.ml.model import WaterUNet

    model = WaterUNet(in_channels=2, base_channels=16)
    arch, in_ch, base = _detect_architecture(model.state_dict())
    assert arch == "WaterUNet"
    assert in_ch == 2
    assert base == 16


def test_detect_architecture_rejects_unknown() -> None:
    from siren.ml.engine import _detect_architecture

    with pytest.raises(ValueError, match="unrecognised checkpoint"):
        _detect_architecture({"some.other.weight": torch.zeros(1)})


# --------------------------------------------------------------------------- #
# Engine: checkpoint resolution
# --------------------------------------------------------------------------- #

def test_engine_prefers_gate_passed_kuro_siwo_checkpoint() -> None:
    """The default engine must load the gate-passed 6-channel checkpoint."""
    from siren.ml.engine import KURO_SIWO_WEIGHTS_PATH, ChangeDetectionEngine

    if not KURO_SIWO_WEIGHTS_PATH.exists():
        pytest.skip("Kuro Siwo checkpoint not present")

    engine = ChangeDetectionEngine()
    assert engine.is_ready is True
    assert engine.architecture == "WaterResUNet"
    assert engine.in_channels == 6
    assert engine.is_multitemporal is True
    assert engine.weights_path == KURO_SIWO_WEIGHTS_PATH


def test_engine_explicit_path_overrides_default(tmp_path) -> None:
    """An explicit weights_path must bypass the candidate search."""
    from siren.ml.engine import ChangeDetectionEngine
    from siren.ml.model import WaterUNet

    wpath = tmp_path / "explicit.pt"
    model = WaterUNet(in_channels=2, base_channels=8)
    torch.save(model.state_dict(), str(wpath))

    engine = ChangeDetectionEngine(weights_path=wpath)
    assert engine.is_ready is True
    assert engine.weights_path == wpath
    assert engine.in_channels == 2
    assert engine.is_multitemporal is False


def test_engine_missing_path_is_not_ready(tmp_path) -> None:
    from siren.ml.engine import ChangeDetectionEngine

    engine = ChangeDetectionEngine(weights_path=tmp_path / "nope.pt")
    assert engine.is_ready is False


# --------------------------------------------------------------------------- #
# Engine: 6-channel inference
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def kuro_engine():
    from siren.ml.engine import KURO_SIWO_WEIGHTS_PATH, ChangeDetectionEngine

    if not KURO_SIWO_WEIGHTS_PATH.exists():
        pytest.skip("Kuro Siwo checkpoint not present")
    return ChangeDetectionEngine()


def test_engine_6ch_inference_shapes(kuro_engine) -> None:
    """6-channel inference returns the documented shapes and dtypes."""
    rng = np.random.RandomState(42)
    t0 = rng.uniform(-22, -6, (2, 224, 224)).astype(np.float32)
    t1 = t0.copy()
    t1[:, 100:150, 100:150] -= 8.0  # synthetic flood drop

    change = kuro_engine.predict_change_mask(t0, t1)
    assert change.shape == (224, 224)
    assert change.dtype == np.uint8

    water = kuro_engine.predict_water_mask(t1)
    assert water.shape == (224, 224)
    assert water.dtype == np.uint8

    prob = kuro_engine.predict_change_probability(t0, t1)
    assert prob.shape == (224, 224)
    assert prob.dtype == np.float32
    assert float(prob.min()) >= 0.0
    assert float(prob.max()) <= 1.0


def test_engine_6ch_identical_inputs_zero_change(kuro_engine) -> None:
    """Hard Rule 1: identical inputs must produce zero change pixels."""
    rng = np.random.RandomState(7)
    sar = rng.uniform(-22, -6, (2, 128, 128)).astype(np.float32)

    change = kuro_engine.predict_change_mask(sar, sar)
    assert change.sum() == 0, "Identical inputs should produce no change"


def test_engine_6ch_non_multiple_of_16(kuro_engine) -> None:
    """Padding logic must handle non-multiple-of-16 spatial dims."""
    rng = np.random.RandomState(11)
    t0 = rng.uniform(-22, -6, (2, 100, 130)).astype(np.float32)
    t1 = t0.copy()

    change = kuro_engine.predict_change_mask(t0, t1)
    assert change.shape == (100, 130)
    assert change.sum() == 0


def test_engine_6ch_rejects_shape_mismatch(kuro_engine) -> None:
    t0 = np.zeros((2, 64, 64), dtype=np.float32)
    t1 = np.zeros((2, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="spatial mismatch"):
        kuro_engine.predict_change_mask(t0, t1)


# --------------------------------------------------------------------------- #
# Calibrated operating threshold (ADR-011.1)
# --------------------------------------------------------------------------- #

def test_kuro_engine_uses_calibrated_threshold(kuro_engine) -> None:
    """The 6-channel checkpoint must default to its calibrated tau=0.30."""
    from siren.ml.engine import KURO_SIWO_CALIBRATED_THRESHOLD

    assert KURO_SIWO_CALIBRATED_THRESHOLD == 0.30
    assert kuro_engine.default_threshold == 0.30


def test_non_multitemporal_engine_uses_half_threshold(tmp_path) -> None:
    """Non-Kuro checkpoints keep the conventional 0.50 threshold."""
    from siren.ml.engine import ChangeDetectionEngine
    from siren.ml.model import WaterUNet

    wpath = tmp_path / "unet2ch.pt"
    model = WaterUNet(in_channels=2, base_channels=8)
    torch.save(model.state_dict(), str(wpath))

    engine = ChangeDetectionEngine(weights_path=wpath)
    assert engine.is_multitemporal is False
    assert engine.default_threshold == 0.5


def test_explicit_threshold_overrides_default(kuro_engine) -> None:
    """An explicit threshold argument must override the calibrated default."""
    rng = np.random.RandomState(3)
    sar = rng.uniform(-22, -6, (2, 64, 64)).astype(np.float32)

    strict = kuro_engine.predict_water_mask(sar, threshold=0.99)
    lenient = kuro_engine.predict_water_mask(sar, threshold=0.01)
    assert lenient.sum() >= strict.sum()
