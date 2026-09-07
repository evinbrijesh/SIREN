"""Tests for the ingest toolkit CLIs (PRD §11).

All tests are fully mocked — no real network calls. Network is simulated by
monkeypatching urllib.request.urlopen.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from siren.ingest import cdse, imerg, overpass, srtm

MODULES = (cdse, srtm, imerg, overpass)


# --------------------------------------------------------------------------- #
# fake HTTP response context manager
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, size: int | None = None) -> bytes:
        if size is None:
            return self._data
        # Streaming mode: return chunks until exhausted
        if not self._data:
            return b""
        chunk = self._data[:size]
        self._data = self._data[size:]
        return chunk

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
            ],
            "links": [],
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
    assert prov["source"] == "nasa-earthdata-cloud-srtm"
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
                {"type": "way", "id": 4,
                 "geometry": [{"lon": 86.7, "lat": 27.7}, {"lon": 86.9, "lat": 27.9}],
                 "tags": {"waterway": "river", "name": "Dudh Koshi"}},
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
    assert len(fc["features"]) == 4
    assert fc["features"][0]["geometry"]["type"] == "Point"
    assert fc["features"][2]["geometry"]["type"] == "LineString"

    # Verify flat properties: tags merged into top-level properties
    village = fc["features"][0]
    assert village["properties"]["place"] == "village"
    assert village["properties"]["name"] == "Chukhung"
    assert village["properties"]["id"] == 1
    assert village["properties"]["osm_type"] == "node"
    # Original tags also preserved as nested key
    assert "tags" in village["properties"]

    # Verify river feature has flat waterway property
    river = fc["features"][3]
    assert river["properties"]["waterway"] == "river"
    assert river["properties"]["name"] == "Dudh Koshi"

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


# --------------------------------------------------------------------------- #
# overpass: empty response does not overwrite existing file
# --------------------------------------------------------------------------- #
def test_overpass_empty_response_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    # Write a pre-existing file
    out = tmp_path / "osm_infrastructure.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": [{"existing": True}]}))

    empty_resp = json.dumps({"elements": []}).encode()

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        return _FakeResp(empty_resp)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    code = overpass.main(["--bbox", "86.65,27.65,87.00,27.98", "--out", str(out)])
    # Non-strict: exit 0, but file preserved
    assert code == 0
    # The existing file must NOT have been overwritten
    fc = json.loads(out.read_text())
    assert "existing" in fc.get("features", [{}])[0]


def test_overpass_empty_response_strict_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    empty_resp = json.dumps({"elements": []}).encode()

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        return _FakeResp(empty_resp)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out = tmp_path / "osm_infrastructure.geojson"
    code = overpass.main(["--bbox", "86.65,27.65,87.00,27.98", "--out", str(out), "--strict"])
    assert code == 1


# --------------------------------------------------------------------------- #
# overpass: query includes waterway=river
# --------------------------------------------------------------------------- #
def test_overpass_query_includes_rivers():
    bbox = (86.65, 27.65, 87.00, 27.98)
    q = overpass.build_query(bbox)
    assert 'waterway"="river"' in q
    assert 'waterway"="stream"' in q


# --------------------------------------------------------------------------- #
# cdse: --strict flag returns non-zero on network failure
# --------------------------------------------------------------------------- #
def test_cdse_strict_nonzero_on_network_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    def fake_urlopen(*a, **k):  # noqa: ARG001
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    code = cdse.main([
        "--bbox", "86.65,27.65,87.00,27.98", "--sensor", "s1",
        "--date", "2026-07-01:2026-08-31", "--out", str(tmp_path), "--strict",
    ])
    assert code == 1


# --------------------------------------------------------------------------- #
# openmeteo: date window walks backward (bug fix)
# --------------------------------------------------------------------------- #
def test_openmeteo_date_window_backward():
    """Verify _days_before walks backward (positive n = past)."""
    from siren.ingest import openmeteo
    # 7 days before 2026-08-04 should be 2026-07-28
    assert openmeteo._days_before("2026-08-04", 7) == "2026-07-28"
    # 0 days before = same date
    assert openmeteo._days_before("2026-08-04", 0) == "2026-08-04"
    # 6 days before 2026-08-04 should be 2026-07-29
    assert openmeteo._days_before("2026-08-04", 6) == "2026-07-29"


def test_openmeteo_cli_accepts_args(tmp_path, monkeypatch):
    """Verify openmeteo has a proper CLI with --lat, --lon, --date, --out."""
    from siren.ingest import openmeteo

    # Mock the API response
    api_resp = json.dumps({
        "daily": {
            "time": ["2026-07-17", "2026-07-18", "2026-07-23"],
            "precipitation_sum": [0.0, 5.0, 18.2],
            "temperature_2m_mean": [10.0, 12.0, 15.0],
        }
    }).encode()

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        return _FakeResp(api_resp)

    # openmeteo imports urlopen directly, so patch at the module level
    monkeypatch.setattr("siren.ingest.openmeteo.urlopen", fake_urlopen)

    out = tmp_path / "weather.json"
    code = openmeteo.main([
        "--lat", "27.815", "--lon", "86.825",
        "--date", "2026-07-23", "--out", str(out),
    ])
    assert code == 0
    data = json.loads(out.read_text())
    assert data["source"] == "open-meteo-archive"
    assert len(data["series"]) == 1
    s = data["series"][0]
    assert s["date"] == "2026-07-23"
    assert s["rainfall_24h_mm"] == 18.2
    # 7-day window: [2026-07-17 .. 2026-07-23] = 0.0 + 5.0 + 18.2 = 23.2
    # (only 3 days in the mock data, 4 days missing)
    assert s["rainfall_7d_days_missing"] == 4
    assert s["rainfall_7d_complete"] is False


# --------------------------------------------------------------------------- #
# acquisition_jobs table + repo methods
# --------------------------------------------------------------------------- #
def test_acquisition_job_create_and_update():
    from siren.db.repo import Repository
    repo = Repository(":memory:")

    # Create a job
    job = repo.create_acquisition_job(
        source="cdse-s1",
        provider_product_id="S1_SCENE_001",
        download_url="https://example.com/scene.zip",
        acquired_at="2026-07-23T12:00:00Z",
    )
    assert job["source"] == "cdse-s1"
    assert job["provider_product_id"] == "S1_SCENE_001"
    assert job["status"] == "pending"

    # Idempotency: creating the same job again doesn't duplicate
    job2 = repo.create_acquisition_job(
        source="cdse-s1",
        provider_product_id="S1_SCENE_001",
    )
    assert job2["job_id"] == job["job_id"]

    # Update status
    repo.update_acquisition_job(job["job_id"], "verified", local_path="data/raw/scene.zip")
    updated = repo.get_acquisition_job(job["job_id"])
    assert updated["status"] == "verified"
    assert updated["local_path"] == "data/raw/scene.zip"
    assert updated["attempts"] == 1

    # Find by source + product_id
    found = repo.find_acquisition_job("cdse-s1", "S1_SCENE_001")
    assert found is not None
    assert found["job_id"] == job["job_id"]

    # List jobs
    all_jobs = repo.list_acquisition_jobs()
    assert len(all_jobs) == 1
    failed_jobs = repo.list_acquisition_jobs(status="failed")
    assert len(failed_jobs) == 0


# --------------------------------------------------------------------------- #
# pipeline: live observation unblock
# --------------------------------------------------------------------------- #
def test_pipeline_rejects_truly_unknown_observation():
    """run_pipeline() should still reject IDs not in demo config or DB."""
    from siren.pipeline import run_pipeline
    from siren.db.repo import Repository
    repo = Repository(":memory:")
    with pytest.raises(ValueError, match="Unknown observation"):
        run_pipeline("nonexistent-obs", repo)


def test_pipeline_accepts_live_observation(tmp_path):
    """run_pipeline() should accept an observation registered in the DB.

    This is the Live Phase 4 unblock test. The observation must be registered
    with a raster_uri pointing to a real change mask file.
    """
    from siren.pipeline import run_pipeline
    from siren.db.repo import Repository
    from siren.detect.scenario import scenario_expansion_mask
    import rasterio

    repo = Repository(":memory:")

    # Create a change mask file for the live observation
    mask_dir = tmp_path / "processed"
    mask_dir.mkdir()
    mask_path = mask_dir / "live-001_expansion_mask.tif"
    mask, meta = scenario_expansion_mask(0.15, seed=99)
    with rasterio.open(
        str(mask_path), "w", driver="GTiff",
        height=mask.shape[0], width=mask.shape[1],
        count=1, dtype="uint8", crs="EPSG:4326",
        transform=meta["transform"],
    ) as dst:
        dst.write(mask.astype(np.uint8), 1)

    # Register a live observation pointing to this mask
    repo.register_observation(
        observation_id="live-001",
        basin_id="dudh-koshi-demo-01",
        acquired_at="2026-09-01T12:00:00Z",
        source="sentinel-1-grd-nrt",
        raster_uri=str(mask_path),
        cloud_fraction=0.0,
        optical_cloud_fraction=0.80,
        water_area_km2=3.5,
        water_area_change_percent=15.0,
        rainfall_24h_mm=30.0,
        rainfall_7d_mm=90.0,
    )

    # The pipeline should now accept this observation
    result = run_pipeline("live-001", repo)
    assert result["observation_id"] == "live-001"
    assert result["status"] == "processed"
    assert result["score"] is not None
    assert result["score"]["severity"] in ("informational", "watch", "elevated", "critical")
