"""Tests for the gold-label panel generator.

These tests only exercise the bookkeeping and scene-selection helpers.
Heavy ROI/panel rendering is tested by running the CLI on a real date.
"""

from __future__ import annotations

import pytest


def test_missing_gold_dates_includes_eval_pairs_without_gold() -> None:
    from siren.ml.generate_gold_label_panels import _missing_gold_dates

    missing = _missing_gold_dates()
    # The existing gold pair should not appear.
    for d in ("20260819", "20260912"):
        assert d not in missing
    # unfrozen_desc2 is the top-priority missing pair.
    assert missing.get("20260726") == "unfrozen_desc2"
    assert missing.get("20260807") == "unfrozen_desc2"


def test_scene_candidates_for_known_dates() -> None:
    from siren.ml.generate_gold_label_panels import _scene_candidates_for_sar_date

    # 20251121 has a curated clear scene from 20251122.
    scenes = _scene_candidates_for_sar_date("20251121")
    assert scenes
    assert "20251122" in scenes[0].name

    # 20260916 now has a downloaded +2d scene (20260918, ~35% cloud).
    scenes = _scene_candidates_for_sar_date("20260916")
    assert scenes
    assert "20260918" in scenes[0].name


@pytest.mark.skip(reason="requires full S2 ROI extraction + Pillow")
def test_generate_panel_produces_files() -> None:
    """Integration smoke test: run manually with a real S2 scene."""
    pass


def test_generate_all_returns_expected_shape() -> None:
    from siren.ml.generate_gold_label_panels import generate_all

    # Restrict to a single missing date whose scene is known to exist.
    report = generate_all(pair_names=["unfrozen_desc2"], out_dir="/tmp/siren_test_panels")
    assert report["status"] in ("generated", "failed")
    assert "generated" in report
    assert "failed" in report
