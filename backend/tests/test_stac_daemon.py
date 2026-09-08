"""Tests for the STAC polling daemon (Sprint 1 Step 3).

All tests are fully mocked — no real network calls. Network is simulated by
monkeypatching the CDSE search function.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from siren.ingest.stac_daemon import (
    DEFAULT_BBOX,
    DEFAULT_LOOKBACK_DAYS,
    _extract_scene_metadata,
    _date_range,
    dispatch_calibration,
    poll_stac,
    register_new_scenes,
    run_poll_cycle,
)
from siren.db.repo import Repository


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def repo():
    return Repository(":memory:")


def _fake_stac_item(
    scene_id: str = "S1A_IW_GRDH_1SDV_20260811T120000_20260811T120030_055000_066000_ABCD",
    acquired_at: str = "2026-08-11T12:00:00Z",
    orbit: int = 85,
) -> dict:
    """Build a minimal STAC item dict matching CDSE response shape."""
    return {
        "id": scene_id,
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[86.6, 27.6], [87.0, 27.6], [87.0, 28.0], [86.6, 28.0], [86.6, 27.6]]]},
        "properties": {
            "datetime": acquired_at,
            "sat:relative_orbit": orbit,
            "sar:product_type": "IW_GRDH_1S",
            "sar:polarizations": "VV+VH",
        },
        "assets": {
            "data": {"href": "https://example.com/download/" + scene_id + ".zip"},
        },
    }


# --------------------------------------------------------------------------- #
# _extract_scene_metadata
# --------------------------------------------------------------------------- #
def test_extract_scene_metadata_basic():
    item = _fake_stac_item()
    meta = _extract_scene_metadata(item)
    assert meta["scene_id"] == "S1A_IW_GRDH_1SDV_20260811T120000_20260811T120030_055000_066000_ABCD"
    assert meta["acquired_at"] == "2026-08-11T12:00:00Z"
    assert meta["orbit"] == 85
    assert meta["product_type"] == "IW_GRDH_1S"
    assert meta["polarization"] == "VV+VH"
    assert meta["download_url"] is not None
    assert "ABCD" in meta["download_url"]


def test_extract_scene_metadata_missing_assets():
    item = _fake_stac_item()
    del item["assets"]
    meta = _extract_scene_metadata(item)
    assert meta["download_url"] is None


def test_extract_scene_metadata_missing_orbit():
    item = _fake_stac_item()
    del item["properties"]["sat:relative_orbit"]
    meta = _extract_scene_metadata(item)
    assert meta["orbit"] is None


# --------------------------------------------------------------------------- #
# _date_range
# --------------------------------------------------------------------------- #
def test_date_range_returns_two_dates():
    start, end = _date_range(30)
    assert len(start) == 10  # YYYY-MM-DD
    assert len(end) == 10
    assert start < end


# --------------------------------------------------------------------------- #
# register_new_scenes
# --------------------------------------------------------------------------- #
def test_register_new_scenes_inserts_into_db(repo):
    scenes = [
        _extract_scene_metadata(_fake_stac_item("scene-A")),
        _extract_scene_metadata(_fake_stac_item("scene-B")),
    ]
    new = register_new_scenes(scenes, repo=repo)
    assert len(new) == 2
    jobs = repo.list_acquisition_jobs()
    assert len(jobs) >= 2
    # Verify the jobs are there
    job_a = repo.find_acquisition_job("cdse-s1", "scene-A")
    assert job_a is not None
    assert job_a["status"] == "pending"
    assert job_a["download_url"] is not None


def test_register_new_scenes_idempotent(repo):
    """Registering the same scene twice does NOT create a duplicate."""
    scenes = [_extract_scene_metadata(_fake_stac_item("scene-A"))]
    # First registration
    new1 = register_new_scenes(scenes, repo=repo)
    assert len(new1) == 1
    # Second registration — should skip
    new2 = register_new_scenes(scenes, repo=repo)
    assert len(new2) == 0
    # Only one job in the DB
    jobs = [j for j in repo.list_acquisition_jobs() if j["source"] == "cdse-s1"]
    assert len(jobs) == 1


def test_register_new_scenes_mixed_new_and_existing(repo):
    """A mix of new and already-registered scenes."""
    # Pre-register scene-A
    register_new_scenes([_extract_scene_metadata(_fake_stac_item("scene-A"))], repo=repo)
    # Now register scene-A + scene-B
    scenes = [
        _extract_scene_metadata(_fake_stac_item("scene-A")),
        _extract_scene_metadata(_fake_stac_item("scene-B")),
    ]
    new = register_new_scenes(scenes, repo=repo)
    assert len(new) == 1  # only scene-B is new
    assert new[0]["scene_id"] == "scene-B"


def test_register_new_scenes_empty_list(repo):
    new = register_new_scenes([], repo=repo)
    assert new == []


def test_register_new_scenes_skips_no_id(repo):
    scenes = [_extract_scene_metadata(_fake_stac_item(""))]
    new = register_new_scenes(scenes, repo=repo)
    assert new == []


# --------------------------------------------------------------------------- #
# dispatch_calibration
# --------------------------------------------------------------------------- #
def test_dispatch_calibration_returns_scene_ids():
    scenes = [_extract_scene_metadata(_fake_stac_item("scene-A"))]
    dispatched = dispatch_calibration(scenes)
    assert "scene-A" in dispatched


def test_dispatch_calibration_empty():
    dispatched = dispatch_calibration([])
    assert dispatched == []


def test_dispatch_calibration_skips_no_id():
    scenes = [_extract_scene_metadata(_fake_stac_item(""))]
    dispatched = dispatch_calibration(scenes)
    assert dispatched == []


# --------------------------------------------------------------------------- #
# poll_stac (mocked)
# --------------------------------------------------------------------------- #
def test_poll_stac_returns_metadata(monkeypatch):
    """poll_stac calls search_all_scenes and extracts metadata."""
    fake_features = [_fake_stac_item("scene-A"), _fake_stac_item("scene-B")]
    monkeypatch.setattr(
        "siren.ingest.stac_daemon.search_all_scenes",
        lambda **kwargs: fake_features,
    )
    monkeypatch.setattr("siren.ingest.stac_daemon._auth_token", lambda: None)
    scenes = poll_stac()
    assert len(scenes) == 2
    assert scenes[0]["scene_id"] == "scene-A"
    assert scenes[1]["scene_id"] == "scene-B"


def test_poll_stac_empty_results(monkeypatch):
    monkeypatch.setattr(
        "siren.ingest.stac_daemon.search_all_scenes",
        lambda **kwargs: [],
    )
    monkeypatch.setattr("siren.ingest.stac_daemon._auth_token", lambda: None)
    scenes = poll_stac()
    assert scenes == []


def test_poll_stac_network_error(monkeypatch):
    """Network errors propagate (the caller handles them)."""
    def _raise(**kwargs):
        raise urllib.error.URLError("Network unavailable")
    monkeypatch.setattr(
        "siren.ingest.stac_daemon.search_all_scenes",
        _raise,
    )
    monkeypatch.setattr("siren.ingest.stac_daemon._auth_token", lambda: None)
    with pytest.raises(urllib.error.URLError):
        poll_stac()


# --------------------------------------------------------------------------- #
# run_poll_cycle (end-to-end with mocks)
# --------------------------------------------------------------------------- #
def test_run_poll_cycle_finds_and_registers_new_scene(monkeypatch, repo):
    """Full poll cycle: search → register → dispatch."""
    fake_features = [_fake_stac_item("scene-new-001")]
    monkeypatch.setattr(
        "siren.ingest.stac_daemon.search_all_scenes",
        lambda **kwargs: fake_features,
    )
    monkeypatch.setattr("siren.ingest.stac_daemon._auth_token", lambda: None)

    summary = run_poll_cycle(repo=repo)
    assert summary["found"] == 1
    assert summary["registered"] == 1
    assert summary["dispatched"] == 1
    assert summary["already_known"] == 0
    assert "error" not in summary


def test_run_poll_cycle_skips_already_known(monkeypatch, repo):
    """Second poll cycle finds the same scene — skips it."""
    fake_features = [_fake_stac_item("scene-A")]
    monkeypatch.setattr(
        "siren.ingest.stac_daemon.search_all_scenes",
        lambda **kwargs: fake_features,
    )
    monkeypatch.setattr("siren.ingest.stac_daemon._auth_token", lambda: None)

    # First cycle: registers the scene
    s1 = run_poll_cycle(repo=repo)
    assert s1["registered"] == 1
    # Second cycle: same scene, should skip
    s2 = run_poll_cycle(repo=repo)
    assert s2["found"] == 1
    assert s2["registered"] == 0
    assert s2["already_known"] == 1


def test_run_poll_cycle_network_error(monkeypatch, repo):
    """Network error returns error summary, does not crash."""
    def _raise(**kwargs):
        raise urllib.error.URLError("Network unavailable")
    monkeypatch.setattr(
        "siren.ingest.stac_daemon.search_all_scenes",
        _raise,
    )
    monkeypatch.setattr("siren.ingest.stac_daemon._auth_token", lambda: None)

    summary = run_poll_cycle(repo=repo)
    assert "error" in summary
    assert summary["found"] == 0
    assert summary["registered"] == 0


def test_run_poll_cycle_multiple_scenes(monkeypatch, repo):
    """Multiple scenes: some new, some already known."""
    # Pre-register scene-A
    register_new_scenes([_extract_scene_metadata(_fake_stac_item("scene-A"))], repo=repo)

    fake_features = [
        _fake_stac_item("scene-A"),  # already known
        _fake_stac_item("scene-B"),  # new
        _fake_stac_item("scene-C"),  # new
    ]
    monkeypatch.setattr(
        "siren.ingest.stac_daemon.search_all_scenes",
        lambda **kwargs: fake_features,
    )
    monkeypatch.setattr("siren.ingest.stac_daemon._auth_token", lambda: None)

    summary = run_poll_cycle(repo=repo)
    assert summary["found"] == 3
    assert summary["registered"] == 2
    assert summary["already_known"] == 1
