"""Tests for ICIMOD and HMA glacial lake inventory ingest (V3 §3.5).

All tests are fully mocked — no real network calls. Network is simulated by
monkeypatching urllib.request.urlopen.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from siren.ingest import icimod, hma


class _FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, size: int | None = None) -> bytes:
        if size is None:
            return self._data
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
# Shared helpers (icimod)
# --------------------------------------------------------------------------- #

def test_icimod_parse_bbox_valid():
    assert icimod.parse_bbox("86.0,27.0,87.5,28.5") == (86.0, 27.0, 87.5, 28.5)


def test_icimod_parse_bbox_invalid_count():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        icimod.parse_bbox("1,2,3")


def test_icimod_parse_bbox_invalid_order():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        icimod.parse_bbox("87.0,27.0,86.0,28.5")  # west > east


def test_icimod_provenance_path_for(tmp_path):
    p = tmp_path / "lakes.geojson"
    sidecar = icimod.provenance_path_for(p)
    assert sidecar.name == "lakes.geojson.json"


def test_icimod_write_provenance_creates_sidecar(tmp_path):
    sidecar = tmp_path / "test.geojson.json"
    icimod.write_provenance(
        sidecar,
        source=icimod.ICIMOD_SOURCE,
        url="https://example.com/data.geojson",
        bbox=(86.0, 27.0, 87.5, 28.5),
        downloaded_at="2026-09-08T12:00:00Z",
        record_count=42,
    )
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["source"] == icimod.ICIMOD_SOURCE
    assert payload["bbox"] == [86.0, 27.0, 87.5, 28.5]
    assert payload["record_count"] == 42
    assert payload["license"] == icimod.ICIMOD_LICENSE


def test_icimod_filter_geojson_by_bbox():
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [86.5, 27.5]}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [90.0, 30.0]}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [86.8, 28.0]}},
        ],
    }
    filtered = icimod.filter_geojson_by_bbox(geojson, (86.0, 27.0, 87.5, 28.5))
    assert len(filtered["features"]) == 2
    assert filtered["features"][0]["geometry"]["coordinates"] == [86.5, 27.5]


def test_icimod_filter_geojson_polygon():
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[86.5, 27.5], [86.6, 27.5], [86.6, 27.6], [86.5, 27.6], [86.5, 27.5]]],
                },
            },
        ],
    }
    filtered = icimod.filter_geojson_by_bbox(geojson, (86.0, 27.0, 87.5, 28.5))
    assert len(filtered["features"]) == 1


def test_icimod_download_file_success(tmp_path, monkeypatch):
    dest = tmp_path / "data.geojson"
    monkeypatch.setattr(
        "siren.ingest.icimod.urllib.request.urlopen",
        lambda req, timeout: _FakeResp(b'{"type":"FeatureCollection","features":[]}'),
    )
    assert icimod.download_file("https://example.com/data.geojson", dest)
    assert dest.exists()
    assert dest.read_bytes() == b'{"type":"FeatureCollection","features":[]}'


def test_icimod_download_file_network_error(tmp_path, monkeypatch):
    dest = tmp_path / "data.geojson"
    def raise_urlerror(*a, **kw):
        raise urllib.error.URLError("network down")
    monkeypatch.setattr("siren.ingest.icimod.urllib.request.urlopen", raise_urlerror)
    assert not icimod.download_file("https://example.com/data.geojson", dest)
    assert not dest.exists()


def test_icimod_main_downloads_and_filters(tmp_path, monkeypatch):
    """Full main() flow with mocked network: downloads, filters, writes provenance."""
    lake_geojson = json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [86.5, 27.5]}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [90.0, 30.0]}},
        ],
    }).encode()

    call_count = [0]
    def fake_urlopen(req, timeout):
        call_count[0] += 1
        if "geojson" in req.full_url:
            return _FakeResp(lake_geojson)
        return _FakeResp(b"date,lake_id,area_km2\n2020-01-01,L001,0.5\n")

    monkeypatch.setattr("siren.ingest.icimod.urllib.request.urlopen", fake_urlopen)

    out_dir = tmp_path / "icimod_out"
    rc = icimod.main(["--bbox", "86.0,27.0,87.5,28.5", "--out", str(out_dir)])
    assert rc == 0

    lake_path = out_dir / "icimod_lakes_2020.geojson"
    assert lake_path.exists()
    filtered = json.loads(lake_path.read_text())
    assert len(filtered["features"]) == 1  # only the in-bbox lake

    sidecar = out_dir / "icimod_lakes_2020.geojson.json"
    assert sidecar.exists()
    provenance = json.loads(sidecar.read_text())
    assert provenance["source"] == icimod.ICIMOD_SOURCE
    assert provenance["record_count"] == 1

    glof_path = out_dir / "icimod_glof_events.csv"
    assert glof_path.exists()


def test_icimod_main_offline_safe(tmp_path, monkeypatch):
    """When network is down, main() exits 0 and no files are written."""
    def raise_urlerror(*a, **kw):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr("siren.ingest.icimod.urllib.request.urlopen", raise_urlerror)

    out_dir = tmp_path / "icimod_out"
    rc = icimod.main(["--bbox", "86.0,27.0,87.5,28.5", "--out", str(out_dir)])
    assert rc == 0
    assert not (out_dir / "icimod_lakes_2020.geojson").exists()


# --------------------------------------------------------------------------- #
# HMA tests
# --------------------------------------------------------------------------- #

def test_hma_uses_icimod_helpers():
    """HMA reuses the provenance + download helpers from icimod."""
    assert hma.provenance_path_for is icimod.provenance_path_for
    assert hma.download_file is icimod.download_file
    assert hma.filter_geojson_by_bbox is icimod.filter_geojson_by_bbox
    assert hma.parse_bbox is icimod.parse_bbox


def test_hma_constants():
    """HMA has its own source, license, and URLs."""
    assert hma.HMA_SOURCE == "HMA Glacial Lake Inventory (Chen et al., 2021)"
    assert hma.HMA_LICENSE == "CC-BY 4.0"
    assert "zenodo.org" in hma.HMA_LAKE_INVENTORY_URL


def test_hma_write_provenance(tmp_path):
    sidecar = tmp_path / "hma.geojson.json"
    hma.write_hma_provenance(
        sidecar,
        url=hma.HMA_LAKE_INVENTORY_URL,
        bbox=(86.0, 27.0, 87.5, 28.5),
        downloaded_at="2026-09-08T12:00:00Z",
        record_count=10,
    )
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["source"] == hma.HMA_SOURCE
    assert payload["license"] == hma.HMA_LICENSE
    assert payload["record_count"] == 10


def test_hma_main_downloads_and_filters(tmp_path, monkeypatch):
    """HMA main() flow with mocked network."""
    lake_geojson = json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [86.5, 27.5]}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [90.0, 30.0]}},
        ],
    }).encode()

    def fake_urlopen(req, timeout):
        if "geojson" in req.full_url:
            return _FakeResp(lake_geojson)
        return _FakeResp(b"lake_id,area_2000,area_2020\nL001,0.3,0.5\n")

    monkeypatch.setattr("siren.ingest.hma.urllib.request.urlopen", fake_urlopen)

    out_dir = tmp_path / "hma_out"
    rc = hma.main(["--bbox", "86.0,27.0,87.5,28.5", "--out", str(out_dir)])
    assert rc == 0

    lake_path = out_dir / "hma_lakes_2020.geojson"
    assert lake_path.exists()
    filtered = json.loads(lake_path.read_text())
    assert len(filtered["features"]) == 1

    sidecar = out_dir / "hma_lakes_2020.geojson.json"
    assert sidecar.exists()
    provenance = json.loads(sidecar.read_text())
    assert provenance["source"] == hma.HMA_SOURCE

    changes_path = out_dir / "hma_lake_area_changes.csv"
    assert changes_path.exists()


def test_hma_main_offline_safe(tmp_path, monkeypatch):
    """When network is down, HMA main() exits 0 and no files are written."""
    def raise_urlerror(*a, **kw):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr("siren.ingest.hma.urllib.request.urlopen", raise_urlerror)

    out_dir = tmp_path / "hma_out"
    rc = hma.main(["--bbox", "86.0,27.0,87.5,28.5", "--out", str(out_dir)])
    assert rc == 0
    assert not (out_dir / "hma_lakes_2020.geojson").exists()
