"""Tests for the neural/deterministic cross-check harness (PRD §9.8.3)."""

import numpy as np

from siren.ml.cross_check import DISAGREEMENT_BOUND, evaluate_overlap


def _mask(shape=(10, 10), box=None):
    m = np.zeros(shape, dtype=np.uint8)
    if box is not None:
        r0, r1, c0, c1 = box
        m[r0:r1, c0:c1] = 1
    return m


def test_full_overlap_no_flag():
    ml = _mask(box=(2, 5, 2, 5))
    rule = _mask(box=(2, 6, 2, 6))
    v = evaluate_overlap(ml, rule)
    assert v["material_disagreement"] is False
    assert v["rule_recall"] == 9 / 16
    assert v["ml_precision"] == 1.0
    assert "reason" not in v


def test_both_empty_no_flag():
    v = evaluate_overlap(_mask(), _mask())
    assert v["material_disagreement"] is False
    assert v["rule_recall"] is None
    assert v["ml_precision"] is None


def test_one_empty_flags_material():
    v = evaluate_overlap(_mask(), _mask(box=(2, 6, 2, 6)))
    assert v["material_disagreement"] is True
    assert "cross-check" in v["reason"]


def test_low_overlap_flags_with_reason():
    ml = _mask(box=(7, 9, 7, 9))
    rule = _mask(box=(2, 6, 2, 6))
    v = evaluate_overlap(ml, rule)
    assert v["material_disagreement"] is True
    assert v["rule_recall"] == 0.0
    assert "fallback mask remains authoritative" in v["reason"]


def test_partial_overlap_below_bound_flags():
    # rule 16px, ml overlaps only 4 (recall 0.25 < 0.5)
    ml = _mask(box=(2, 4, 2, 4))
    rule = _mask(box=(2, 6, 2, 6))
    v = evaluate_overlap(ml, rule)
    assert v["rule_recall"] == 0.25
    assert v["material_disagreement"] is True


def test_precision_direction_flags():
    # ml fully inside rule but rule is 4x bigger → recall low; and a
    # large disjoint ml region → precision low too.
    ml = _mask(box=(2, 4, 2, 4)) | _mask(box=(7, 10, 7, 10))
    rule = _mask(box=(2, 4, 2, 4))
    v = evaluate_overlap(ml, rule)
    assert v["ml_precision"] == round(4 / 13, 4)
    assert v["material_disagreement"] is True


def test_verdict_schema_keys():
    v = evaluate_overlap(_mask(), _mask(box=(0, 2, 0, 2)))
    for key in (
        "metric", "bound", "ml_px", "rule_px", "overlap_px",
        "rule_recall", "ml_precision", "material_disagreement",
    ):
        assert key in v
    assert v["bound"] == DISAGREEMENT_BOUND
