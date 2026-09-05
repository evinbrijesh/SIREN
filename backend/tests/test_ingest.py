"""Tests for the ingest toolkit CLIs (PRD §11).

All tests are fully mocked — no real network calls. Network is simulated by
monkeypatching urllib.request.urlopen.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
from pathlib import Path

import pytest

from siren.ingest import cdse, imerg, overpass, srtm

MODULES = (cdse, srtm, imerg, overpass)


# --------------------------------------------------------------------------- #
# fake HTTP response context manager
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------- #
# parse_bbox
# --------------------------------------------------------------------------- #
def test_parse_bbox_valid():
    assert cdse.parse_bbox("86.65,27.65,87.00,27.98") == (86.65, 27.65, 87.00, 27.98)


@pytest.mark.parametrize("bad", ["1,2,3", "a,b,c,d", "87,27,86,28", "1,2,3,4,5"])
def test_parse_bbox_invalid(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        cdse.parse_bbox(bad)


def test_parse_date_range_valid():
    assert cdse.parse_date_range("2026-07-01:2026-08-31") == ("2026-07-01", "2026-08-31")


def test_parse_date_range_invalid():
    with pytest.raises(argparse.ArgumentTypeError):
        cdse.parse_date_range("2026-07-01")


@pytest.mark.parametrize("mod", MODULES)
def test_all_modules_parse_bbox(mod):
    assert mod.parse_bbox("86.65,27.65,87.00,27.98") == (86.65, 27.65, 87.00, 27.98)


# --------------------------------------------------------------------------- #
# --help works without network
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", MODULES)
def test_help_exits_zero(mod, capsys):
    with pytest.raises(SystemExit) as ei:
        mod.main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "--bbox" in out


# --------------------------------------------------------------------------- #
# provenance sidecar writing (mocked download)
# --------------------------------------------------------------------------- #
def test_cdse_provenance_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("CDSE_TOKEN", "fake-token")
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    stac_resp = json.dumps(
        {
            "features": [
                {
                    "id": "S1_SCENE_001",
                    "properties": {"datetime": "2026-07-23T12:00:00Z"},
                    "assets": {"data": {"href": "https://example.com/scene.zip"}},
                }
            ]
        }
    ).encode()
    download_bytes = b"FAKE-ZIP-CONTENT"

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        if req.get_method() == "POST":
            return _FakeResp(stac_resp)
        return _FakeResp(download_bytes)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    code = cdse.main(
        ["--bbox", "86.65,27.65,87.00,27.98", "--sensor", "s1",
         "--date", "2026-07-01:2026-08-31", "--out", str(tmp_path)]
    )
    assert code == 0

    zips = [f for f in tmp_path.iterdir() if f.suffix == ".zip"]
    assert len(zips) == 1
    assert zips[0].read_bytes() == download_bytes

    sidecars = [f for f in tmp_path.iterdir() if f.suffix == ".json"]
    assert len(sidecars) == 1
    prov = json.loads(sidecars[0].read_text())
    assert prov["source"] == "copernicus-cdse-stac"
    assert prov["scene_id"] == "S1_SCENE_001"
    assert prov["download_url"] == "https://example.com/scene.zip"
    assert prov["bbox"] == [86.65, 27.65, 87.00, 27.98]
    assert prov["acquired_at"] == "2026-07-23T12:00:00Z"
    assert prov["retries"] == 0


def test_srtm_provenance_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "u")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "p")
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    tile_bytes = b"FAKE-HGT-ZIP"

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        return _FakeResp(tile_bytes)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    code = srtm.main(["--bbox", "86.65,27.65,87.00,27.98", "--out", str(tmp_path)])
    assert code == 0

    # bbox spans lon 86..87, lat 27 -> tiles N27E086, N27E087
    zips = sorted(f.name for f in tmp_path.iterdir() if f.suffix == ".zip")
    assert zips == ["N27E086.SRTMGL1.hgt.zip", "N27E087.SRTMGL1.hgt.zip"]
    sidecars = [f for f in tmp_path.iterdir() if f.suffix == ".json"]
    assert len(sidecars) == 2
    prov = json.loads((tmp_path / "N27E086.SRTMGL1.hgt.zip.json").read_text())
    assert prov["source"] == "nasa-earthdata-srtm"
    assert prov["scene_id"] == "N27E086.SRTMGL1.hgt.zip"
    assert prov["acquired_at"] == "2000-02-11"
    assert prov["retries"] == 0
    assert "SRTMGL1.003" in prov["download_url"]


def test_imerg_provenance_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "u")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "p")
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        return _FakeResp(b"FAKE-IMERG-NC4")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    code = imerg.main(
        ["--bbox", "86.65,27.65,87.00,27.98",
         "--date", "2026-07-01:2026-07-02", "--out", str(tmp_path)]
    )
    assert code == 0

    nc4 = sorted(f.name for f in tmp_path.iterdir() if f.suffix == ".nc4")
    assert nc4 == ["3IMERDL.20260701.nc4", "3IMERDL.20260702.nc4"]
    prov = json.loads((tmp_path / "3IMERDL.20260701.nc4.json").read_text())
    assert prov["source"] == "nasa-gesdisc-imerg-late-daily"
    assert prov["scene_id"] == "3IMERDL.20260701.nc4"
    assert prov["acquired_at"] == "2026-07-01"
    assert prov["retries"] == 0


def test_overpass_provenance_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    overpass_resp = json.dumps(
        {
            "elements": [
                {"type": "node", "id": 1, "lon": 86.9, "lat": 27.8,
                 "tags": {"place": "village", "name": "Chukhung"}},
                {"type": "node", "id": 2, "lon": 86.85, "lat": 27.7,
                 "tags": {"amenity": "drinking_water"}},
                {"type": "way", "id": 3,
                 "geometry": [{"lon": 86.8, "lat": 27.7}, {"lon": 86.9, "lat": 27.8}],
                 "tags": {"highway": "residential"}},
            ]
        }
    ).encode()

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        return _FakeResp(overpass_resp)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out = tmp_path / "osm_infrastructure.geojson"
    code = overpass.main(["--bbox", "86.65,27.65,87.00,27.98", "--out", str(out)])
    assert code == 0

    fc = json.loads(out.read_text())
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 3
    assert fc["features"][0]["geometry"]["type"] == "Point"
    assert fc["features"][2]["geometry"]["type"] == "LineString"

    sidecar = tmp_path / "osm_infrastructure.geojson.json"
    assert sidecar.exists()
    prov = json.loads(sidecar.read_text())
    assert prov["source"] == "osm-overpass"
    assert prov["scene_id"] == "overpass-extract"
    assert prov["bbox"] == [86.65, 27.65, 87.00, 27.98]
    assert prov["retries"] == 0


# --------------------------------------------------------------------------- #
# offline-safe: network unavailable -> exit 0, clear message, no crash
# --------------------------------------------------------------------------- #
def _offline_argv(mod, out):
    if mod is cdse:
        return ["--bbox", "86.65,27.65,87.00,27.98", "--sensor", "s1",
                "--date", "2026-07-01:2026-08-31", "--out", out]
    if mod is srtm:
        return ["--bbox", "86.65,27.65,87.00,27.98", "--out", out]
    if mod is imerg:
        return ["--bbox", "86.65,27.65,87.00,27.98",
                "--date", "2026-07-01:2026-08-31", "--out", out]
    if mod is overpass:
        return ["--bbox", "86.65,27.65,87.00,27.98", "--out", out]
    raise AssertionError(mod)


@pytest.mark.parametrize("mod", MODULES)
def test_offline_safe_exit_zero(mod, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    def fake_urlopen(*a, **k):  # noqa: ARG001
        raise urllib.error.URLError("offline: no network")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out = str(tmp_path / "out")
    code = mod.main(_offline_argv(mod, out))
    assert code == 0
    err = capsys.readouterr().err
    assert "Network unavailable" in err


# --------------------------------------------------------------------------- #
# retry logic: transient failure then success
# --------------------------------------------------------------------------- #
def test_retry_succeeds_after_transient(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("transient")
        return "ok"

    result, retries = cdse.retry(flaky)
    assert result == "ok"
    assert retries == 2
    assert calls["n"] == 3


def test_retry_exhausts_then_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    def always_fail():
        raise urllib.error.URLError("down")

    with pytest.raises(urllib.error.URLError):
        cdse.retry(always_fail, max_retries=2)
