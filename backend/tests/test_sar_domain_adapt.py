"""Tests for the weak-label domain-adaptation helpers."""

from __future__ import annotations

import numpy as np

from siren.ml.sar_domain_adapt import extract_chips


def test_extract_chips_respects_labelled_fraction():
    rng = np.random.default_rng(0)
    tensor = rng.random((6, 200, 200), dtype=np.float32)
    label = np.full((200, 200), -1, dtype=np.int8)
    aoi = np.zeros((200, 200), dtype=bool)
    aoi[20:180, 20:180] = True
    # fully labelled block in the centre of the AOI
    label[60:140, 60:140] = 0
    label[90:110, 90:110] = 1

    x, y, v = extract_chips(tensor, label, aoi, chip=96, stride=48,
                            min_labelled=0.10)
    assert x.shape[1:] == (6, 96, 96)
    assert len(x) > 0
    # every returned chip has >=10% labelled pixels
    assert (v.mean(axis=(1, 2)) >= 0.10).all()
    # at least one chip contains positives
    assert (y * v).sum() > 0
    # labels are clipped to {0,1}
    assert set(np.unique(y)) <= {0.0, 1.0}


def test_extract_chips_empty_when_nothing_labelled():
    tensor = np.zeros((6, 150, 150), dtype=np.float32)
    label = np.full((150, 150), -1, dtype=np.int8)
    aoi = np.ones((150, 150), dtype=bool)
    x, y, v = extract_chips(tensor, label, aoi, chip=96, stride=48)
    assert len(x) == 0
