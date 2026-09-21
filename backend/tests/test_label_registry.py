"""Tests for the label registry and operational gate evaluation harness.

These tests verify the bookkeeping and gating logic without needing a GPU
or full SAR/label rasters on disk. Heavy inference tests belong in the
integration suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_list_labels_reports_existing_gold_labels() -> None:
    from siren.ml.label_registry import list_labels

    labels = list_labels()
    assert "20260819" in labels
    assert "20260912" in labels
    for d in ("20260819", "20260912"):
        assert labels[d]["tier"] == "gold"
        assert labels[d]["exists"] is True
        assert labels[d]["path"].endswith(".tif")


def test_best_label_prefers_gold_over_auto_and_scl() -> None:
    from siren.ml.label_registry import best_label_for_date

    # 20260726 has both auto and scl_merged; best is auto (higher tier
    # than scl_merged, lower than gold which does not exist for this date).
    rec = best_label_for_date("20260726")
    assert rec is not None
    assert rec["tier"] == "auto"


def test_eval_eligible_pairs_excludes_training_pair() -> None:
    from siren.ml.label_registry import eval_eligible_pairs

    pairs = eval_eligible_pairs("eval")
    names = {p["pair"] for p in pairs}
    assert "early_desc" not in names  # train-only pair
    # unfrozen_desc has gold labels on disk, so it should appear
    assert "unfrozen_desc" in names


def test_operational_gate_status_reports_sufficient_gold() -> None:
    """With the newly downloaded shoulder + winter labels the registry now
    sees three gold-complete pairs and can attempt the gate."""
    from siren.ml.label_registry import operational_gate_status

    status = operational_gate_status()
    assert status["can_attempt_gate"] is True
    assert "unfrozen_desc" in status["gold_complete_pairs"]
    assert "shoulder" in status["gold_complete_pairs"]
    assert "winter" in status["gold_complete_pairs"]


def test_operational_gate_status_reports_insufficient_gold_when_empty() -> None:
    """If gold labels are empty, the gate cannot be attempted."""
    from unittest.mock import patch

    from siren.ml import label_registry as lr
    from siren.ml.label_registry import operational_gate_status

    with patch.object(lr, "GOLD_LABELS", {}):
        status = operational_gate_status()
    assert status["can_attempt_gate"] is False
    assert "acquire" in status["next_action"].lower()


def test_registry_report_can_be_serialized(tmp_path: Path) -> None:
    from siren.ml.label_registry import registry_report

    report = registry_report()
    out = tmp_path / "report.json"
    out.write_text(json.dumps(report, default=str))
    data = json.loads(out.read_text())
    assert "operational_gate" in data
    assert data["operational_gate"]["can_attempt_gate"] is True


def test_operational_gate_eval_reports_insufficient_labels_when_empty() -> None:
    """When fewer than two gold-complete pairs exist, the harness refuses
    to evaluate and returns a clear status."""
    from unittest.mock import patch

    from siren.ml import label_registry as lr
    from siren.ml.operational_gate_eval import evaluate_operational_gate

    with patch.object(lr, "GOLD_LABELS", {}):
        report = evaluate_operational_gate()
    assert report["status"] == "insufficient_labels"
    assert report["verdict"] == "not_attempted"
    assert report["gate"]["iou"] == 0.60
    assert report["gate"]["precision"] == 0.84
    assert report["gate"]["glacier_fp_frac"] == 0.05


def test_operational_gate_eval_fails_gracefully_without_promotion() -> None:
    from siren.ml import operational_gate_eval as oge

    with patch.object(oge, "_promoted_checkpoint_path", return_value=None):
        report = oge.evaluate_operational_gate()
    assert report["status"] == "cannot_evaluate"
    assert "no promoted" in report["reason"].lower()
