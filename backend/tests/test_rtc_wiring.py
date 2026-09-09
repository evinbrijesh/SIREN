"""Tests for Sprint 1 Step 6 RTC wiring helpers.

Covers:
  * :func:`siren.preprocess.sar_calibrate.find_safe_for_scene_id`
  * :func:`siren.preprocess.sar_calibrate.extract_incidence_and_look`
  * the ``calibrate_scene_task`` pending / not_found paths (no real SAFE
    archives required — uses a temp dir and a fake repo).

The full RTC-on-real-SAFE path (``calibrate_scene_with_rtc``) is verified
manually against the real ESA archives in ``data/raw/``; the pure RTC
math is covered by ``tests/test_rtc.py``.
"""

from __future__ import annotations

import sqlite3
import tempfile
import zipfile
from pathlib import Path

import pytest

from siren.preprocess.sar_calibrate import (
    extract_incidence_and_look,
    find_safe_for_scene_id,
)


# --------------------------------------------------------------------------- #
# find_safe_for_scene_id
# --------------------------------------------------------------------------- #
def test_find_safe_for_scene_id_exact_match(tmp_path):
    scene = "S1D_IW_GRDH_1SDV_20260723T122115_20260723T122140_003801_006D5D_D3B2"
    safe = tmp_path / f"{scene}.SAFE.zip"
    safe.write_bytes(b"")
    assert find_safe_for_scene_id(scene, tmp_path) == str(safe)


def test_find_safe_for_scene_id_token_fallback(tmp_path):
    """Exact file absent → fall back to the acquisition-time token substring."""
    scene = "S1D_IW_GRDH_1SDV_20260723T122115_20260723T122140_003801_006D5D_D3B2"
    # SAFE on disk with a slightly different name but same time token.
    safe = tmp_path / "S1D_IW_GRDH_1SDV_20260723T122115_20260723T122140_003801_006D5D_D3B2.SAFE.zip"
    safe.write_bytes(b"")
    found = find_safe_for_scene_id(scene, tmp_path)
    assert found is not None
    assert Path(found).name == safe.name


def test_find_safe_for_scene_id_not_found(tmp_path):
    scene = "S1D_IW_GRDH_1SDV_20260723T122115_20260723T122140_003801_006D5D_D3B2"
    assert find_safe_for_scene_id(scene, tmp_path) is None


def test_find_safe_for_scene_id_no_token_returns_none(tmp_path):
    """A scene ID without an acquisition-time token can't substring-match."""
    assert find_safe_for_scene_id("not-a-real-scene", tmp_path) is None


# --------------------------------------------------------------------------- #
# extract_incidence_and_look — synthetic SAFE zip
# --------------------------------------------------------------------------- #
_SYNTH_ANNOTATION = """\
<?xml version="1.0" encoding="UTF-8"?>
<product>
  <productInformation>
    <pass>Ascending</pass>
    <timelinessCategory>Fast-24h</timelinessCategory>
  </productInformation>
  <geolocationGrid>
    <tiePoint>
      <platformHeading>-1.249844831044351e+01</platformHeading>
      <incidenceAngleMidSwath>3.916447039946686e+01</incidenceAngleMidSwath>
    </tiePoint>
  </geolocationGrid>
</product>
"""


def _make_synthetic_safe(tmp_path: Path, pol: str = "vv") -> str:
    """Build a minimal SAFE zip with one annotation XML."""
    scene = "S1D_IW_GRDH_1SDV_20260723T122115_20260723T122140_003801_006D5D_D3B2"
    safe = tmp_path / f"{scene}.SAFE.zip"
    ann_name = f"{scene}/annotation/s1d-iw-grd-{pol}-20260723t122115-20260723t122140-003801-006d5d-001.xml"
    with zipfile.ZipFile(safe, "w") as z:
        z.writestr(ann_name, _SYNTH_ANNOTATION)
    return str(safe)


def test_extract_incidence_and_look_ascending(tmp_path):
    safe = _make_synthetic_safe(tmp_path)
    geom = extract_incidence_and_look(safe)
    assert geom["incidence_deg"] == pytest.approx(39.16447, abs=1e-4)
    assert geom["platform_heading_deg"] == pytest.approx(-12.49845, abs=1e-4)
    assert geom["pass"] == "Ascending"
    # look_azimuth = heading + 90, wrapped to [0, 360)
    expected_az = (-12.49845 + 90.0) % 360.0
    assert geom["look_azimuth_deg"] == pytest.approx(expected_az, abs=1e-4)


def test_extract_incidence_and_look_descending_wraps(tmp_path):
    """A descending heading near -167° → look azimuth wraps past 360 → ~283°."""
    ann = _SYNTH_ANNOTATION.replace(
        "<pass>Ascending</pass>", "<pass>Descending</pass>"
    ).replace(
        "-1.249844831044351e+01", "-1.671234567e+01"
    )
    scene = "S1D_IW_GRDH_1SDV_20260714T001035_20260714T001100_003662_00689F_377F"
    safe = tmp_path / f"{scene}.SAFE.zip"
    ann_name = f"{scene}/annotation/s1d-iw-grd-vv-20260714t001035-20260714t001100-003662-00689f-001.xml"
    with zipfile.ZipFile(safe, "w") as z:
        z.writestr(ann_name, ann)
    geom = extract_incidence_and_look(str(safe))
    assert geom["pass"] == "Descending"
    expected_az = (-16.71234567 + 90.0) % 360.0
    assert geom["look_azimuth_deg"] == pytest.approx(expected_az, abs=1e-4)


def test_extract_incidence_and_look_missing_fields_raises(tmp_path):
    safe = tmp_path / "S1D_IW_GRDH_1SDV_20260723T122115_20260723T122140_003801_006D5D_XXXX_XXXX.SAFE.zip"
    ann_name = "S1D_IW_GRDH_1SDV_20260723T122115_20260723T122140_003801_006D5D_XXXX_XXXX.SAFE/annotation/s1d-iw-grd-vv-20260723t122115-20260723t122140-003801-006d5d-001.xml"
    with zipfile.ZipFile(safe, "w") as z:
        z.writestr(ann_name, "<product><geolocationGrid><tiePoint></tiePoint></geolocationGrid></product>")
    with pytest.raises(ValueError):
        extract_incidence_and_look(str(safe))


# --------------------------------------------------------------------------- #
# calibrate_scene_task — pending / not_found paths (no real SAFE required)
# --------------------------------------------------------------------------- #
class _FakeRepo:
    """Minimal repo stub matching the acquisition_job interface used by the task."""

    def __init__(self, jobs: dict[str, dict] | None = None):
        self.jobs = jobs or {}
        self.updates: list[tuple[str, str, str | None]] = []

    def find_acquisition_job(self, source: str, provider_product_id: str):
        return self.jobs.get(provider_product_id)

    def update_acquisition_job(self, job_id, status, local_path=None, last_error=None):
        self.updates.append((job_id, status, local_path))


def _run_calibrate_task(scene_id, repo, raw_dir, dem_exists=False):
    """Invoke the real calibrate_scene_task closure with a controlled env.

    We rebuild the closure via _register_celery_tasks on a fake app so we
    don't need celery installed; the task body is plain Python.
    """
    import os
    import types

    # Fake celery module with a no-op app.task decorator.
    class _FakeApp:
        def task(self, *a, **k):
            def deco(fn):
                return fn
            return deco
        def conf(self):  # pragma: no cover
            pass

    import siren.ingest.stac_daemon as mod
    real_celery = sys_modules_get("celery", None)
    fake_celery = types.ModuleType("celery")
    fake_celery.Celery = lambda *a, **k: _FakeApp()
    sys_modules_set("celery", fake_celery)
    try:
        # _register_celery_tasks imports celery at call time.
        _, calibrate = mod._register_celery_tasks(_FakeApp())
    finally:
        if real_celery is not None:
            sys_modules_set("celery", real_celery)
        else:
            sys_modules_del("celery")

    # Patch get_repository to return our fake repo, and env vars for paths.
    import siren.db.repo as repo_mod
    orig_get_repo = repo_mod.get_repository
    repo_mod.get_repository = lambda: repo
    orig_env = dict(os.environ)
    os.environ["SIREN_RAW_DIR"] = str(raw_dir)
    os.environ["SIREN_DEM_PATH"] = str(raw_dir / "srtm_30m.tif") if not dem_exists else str(raw_dir / "srtm_30m.tif")
    try:
        return calibrate(scene_id)
    finally:
        repo_mod.get_repository = orig_get_repo
        os.environ.clear()
        os.environ.update(orig_env)


def _sys_modules():
    import sys
    return sys.modules


def sys_modules_get(name, default=None):
    return _sys_modules().get(name, default)


def sys_modules_set(name, mod):
    _sys_modules()[name] = mod


def sys_modules_del(name):
    _sys_modules().pop(name, None)


def test_calibrate_task_not_found(tmp_path):
    repo = _FakeRepo(jobs={})
    result = _run_calibrate_task("S1_missing", repo, tmp_path)
    assert result["status"] == "not_found"
    assert result["scene_id"] == "S1_missing"


def test_calibrate_task_pending_when_safe_missing(tmp_path):
    """Job exists but no SAFE on disk → status pending, rtc_applied False."""
    scene = "S1D_IW_GRDH_1SDV_20260723T122115_20260723T122140_003801_006D5D_D3B2"
    repo = _FakeRepo(jobs={scene: {"job_id": "acq-12345", "status": "pending"}})
    result = _run_calibrate_task(scene, repo, tmp_path)
    assert result["status"] == "pending"
    assert result["rtc_applied"] is False
    assert repo.updates == [("acq-12345", "pending", None)]


def test_calibrate_task_pending_when_dem_missing(tmp_path):
    """SAFE present but DEM absent → still pending."""
    scene = "S1D_IW_GRDH_1SDV_20260723T122115_20260723T122140_003801_006D5D_D3B2"
    safe = tmp_path / f"{scene}.SAFE.zip"
    safe.write_bytes(b"")
    repo = _FakeRepo(jobs={scene: {"job_id": "acq-12345", "status": "pending"}})
    # DEM does not exist in tmp_path.
    result = _run_calibrate_task(scene, repo, tmp_path, dem_exists=False)
    assert result["status"] == "pending"
    assert result["rtc_applied"] is False
