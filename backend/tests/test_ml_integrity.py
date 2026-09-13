from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from siren.ml import dataset, registry
from siren.risk import shadow_evidence
from siren.risk.susceptibility import DEFAULT_CHECKPOINT_PATH, SusceptibilityScorer


CHECKPOINTS = Path(__file__).resolve().parents[2] / "models" / "checkpoints"


def test_default_susceptibility_checkpoint_is_rejected():
    scorer = SusceptibilityScorer()
    assert scorer.load_checkpoint() is False
    assert scorer.is_trained is False
    assert scorer.brier_score is None
    assert scorer.passes_acceptance_gate() is False


def test_renamed_contaminated_checkpoint_is_rejected(tmp_path):
    renamed = tmp_path / "renamed.json"
    renamed.write_bytes(DEFAULT_CHECKPOINT_PATH.read_bytes())
    renamed.with_suffix(".meta.json").write_text(json.dumps({
        "inference_allowed": True, "evaluation_valid": True,
        "brier_score_cv": 0.001, "calibration_q": 0.01,
    }))
    scorer = SusceptibilityScorer()
    assert scorer.load_checkpoint(renamed) is False
    assert scorer.is_calibrated is False


def test_failed_reload_clears_all_model_and_calibration_state(tmp_path):
    scorer = SusceptibilityScorer()
    scorer._model = Mock()
    scorer._calibrator = Mock()
    scorer._is_trained = True
    scorer._is_calibrated = True
    scorer._is_isotonic_calibrated = True
    scorer._calibration_q = 0.01
    scorer._brier_score = 0.01
    scorer._brier_score_raw = 0.02
    assert scorer.load_checkpoint(tmp_path / "missing.json") is False
    assert scorer._model is None
    assert scorer._calibrator is None
    assert scorer._calibration_q is None
    assert scorer._brier_score_raw is None
    assert scorer.is_trained is False
    assert scorer.is_calibrated is False
    assert scorer._is_isotonic_calibrated is False
    assert scorer.passes_acceptance_gate() is False
    with pytest.raises(RuntimeError):
        scorer.predict(np.zeros((1, 6)))


def test_shadow_does_not_train_or_invent_probability(monkeypatch):
    monkeypatch.setattr(SusceptibilityScorer, "load_checkpoint", lambda *a, **k: False)
    train = Mock(side_effect=AssertionError("runtime training is forbidden"))
    monkeypatch.setattr(SusceptibilityScorer, "train", train)
    stats = {"hazard_score": 0.75}
    result = shadow_evidence.attach_shadow_evidence(stats, {}, 0.0, 0.0)
    train.assert_not_called()
    assert stats["hazard_score"] == 0.75
    assert result["susceptibility"]["is_available"] is False
    assert "p_breach" not in result["susceptibility"]
    assert result["hydro_surrogate"]["is_triggered"] is False
    assert "0.000" not in result["hydro_surrogate"]["reason"]
    assert "t_arrival_by_sector" not in result["hydro_surrogate"]


def test_fno_runtime_returns_unavailable_without_generating_terrain():
    result = shadow_evidence._compute_shadow_hydro({}, 0.85, None)
    assert result["is_available"] is False
    assert result["status"] == "disqualified"
    assert "h_water_max_m" not in result
    assert "t_arrival_by_sector" not in result


@pytest.mark.parametrize("name", [
    "water_resunet_6ch_v1", "xgboost_susceptibility_v1",
    "xgboost_susceptibility_v2_real", "fno_hydro_surrogate_v1",
])
def test_registry_discloses_disqualified_models(name):
    status = registry.get_model_status()[name]
    assert status["status"] == "disqualified"
    assert status["loaded"] is False
    assert status["inference_allowed"] is False
    assert status["evaluation_valid"] is False
    assert status["disqualification_reason"]


@pytest.mark.parametrize("name", [
    "water_resunet_6ch_v1", "xgboost_susceptibility_v1",
    "xgboost_susceptibility_v2_real", "fno_hydro_surrogate_v1",
])
def test_sidecars_cannot_advertise_gate_pass(name):
    meta = json.loads((CHECKPOINTS / f"{name}.meta.json").read_text())
    assert meta["evaluation_valid"] is False
    assert meta["inference_allowed"] is False
    assert meta["status"] == "disqualified"
    assert meta.get("gate_passed", False) is False
    assert meta.get("passes_brier_gate", False) is False
    if isinstance(meta.get("gate"), dict):
        assert meta["gate"]["passed"] is False


def test_multitemporal_dataset_requires_real_pairs(monkeypatch):
    monkeypatch.setattr(dataset, "_load_event_holdout_rows", lambda split: [])
    with pytest.raises(ValueError, match="pre_sar_dir"):
        dataset.MultiTemporalWaterDataset("train")


def test_multitemporal_dataset_rejects_missing_pair_before_training(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "_load_event_holdout_rows", lambda split: [
        ("Event_1.tif", "Event_1_label.tif", "train"),
    ])
    with pytest.raises(ValueError, match="Event_1.tif"):
        dataset.MultiTemporalWaterDataset("train", pre_sar_dir=tmp_path,
                                          dem_path=tmp_path / "dem.tif")


@pytest.mark.parametrize("real", [False, True])
def test_susceptibility_training_is_blocked_before_dataset_loading(monkeypatch, real):
    from siren.ml import train_susceptibility
    load = Mock(side_effect=AssertionError("must reject before reading data"))
    monkeypatch.setattr(train_susceptibility, "load_dataset", load)
    with pytest.raises(ValueError, match="disqualified"):
        train_susceptibility.train_susceptibility_model(save=False, use_real_data=real)
    load.assert_not_called()


def test_real_glof_loader_rejects_generated_feature_path(monkeypatch):
    from siren.ml import real_glof_dataset
    load = Mock(side_effect=AssertionError("must reject before reading data"))
    monkeypatch.setattr(real_glof_dataset, "_load_hmaglofdb", load)
    with pytest.raises(ValueError, match="disqualified"):
        real_glof_dataset.load_real_dataset()
    load.assert_not_called()
