"""Deterministic lake thermal-state gate tests."""

import json
from datetime import date, timedelta

import pytest

from siren.detect.thermal_state import (
    LakeThermalState,
    estimate_thermal_state,
)


def _write_series(tmp_path, days: dict) -> str:
    path = tmp_path / "series.json"
    path.write_text(json.dumps({
        "provenance": "test",
        "station_elev_m": 5085.0,
        "days": days,
    }))
    return str(path)


def _series_around(center: str, temps: list[float]) -> dict:
    d0 = date.fromisoformat(center)
    return {
        (d0 - timedelta(days=i)).isoformat(): {"mean_c": t, "min_c": t - 5}
        for i, t in enumerate(temps)
    }


def test_frozen_when_sustained_cold(tmp_path):
    p = _write_series(tmp_path, _series_around("2026-01-15", [-15.0] * 7))
    r = estimate_thermal_state("2026-01-15", series_path=p)
    assert r["state"] == LakeThermalState.FROZEN_SURFACE.value
    # station at 5085 m vs Imja 5010 m: correction = 6.5 * (5010-5085)/1000
    assert r["lake_temp_c"] == pytest.approx(-15.0 - (-0.4875), abs=0.05)


def test_liquid_when_warm(tmp_path):
    p = _write_series(tmp_path, _series_around("2026-07-15", [3.0] * 7))
    r = estimate_thermal_state("2026-07-15", series_path=p)
    assert r["state"] == LakeThermalState.LIQUID.value


def test_transition_band(tmp_path):
    p = _write_series(tmp_path, _series_around("2026-10-15", [-0.5] * 7))
    r = estimate_thermal_state("2026-10-15", series_path=p)
    assert r["state"] == LakeThermalState.FREEZE_TRANSITION.value


def test_insufficient_coverage_is_unknown(tmp_path):
    p = _write_series(tmp_path, _series_around("2026-01-15", [-15.0] * 2))
    r = estimate_thermal_state("2026-01-15", series_path=p)
    assert r["state"] == LakeThermalState.UNKNOWN.value
    assert r["coverage"] < 0.5


def test_missing_file_is_unknown(tmp_path):
    r = estimate_thermal_state(
        "2026-01-15", series_path=tmp_path / "nonexistent.json"
    )
    assert r["state"] == LakeThermalState.UNKNOWN.value


def test_lapse_correction_applied(tmp_path):
    """A station BELOW the lake should shift lake temp downward."""
    path = tmp_path / "low_station.json"
    path.write_text(json.dumps({
        "station_elev_m": 3000.0,
        "days": _series_around("2026-06-01", [10.0] * 7),
    }))
    r = estimate_thermal_state(
        "2026-06-01", series_path=str(path), lake_elev_m=5010.0
    )
    # 10.0 - 6.5*(5010-3000)/1000 = 10.0 - 13.065 = -3.065 → rounds to -3.1
    assert r["lake_temp_c"] == pytest.approx(-3.1, abs=0.01)
    assert r["state"] == LakeThermalState.FROZEN_SURFACE.value


def test_real_series_classifies_eval_and_demo_dates():
    """The committed series must classify the known dates correctly —
    Jan eval pair frozen, July demo observations liquid (DoD intact)."""
    for d in ("2026-01-08", "2026-01-20"):
        r = estimate_thermal_state(d)
        assert r["state"] == LakeThermalState.FROZEN_SURFACE.value, d
    for d in ("2026-07-23", "2026-08-04", "2026-08-12"):
        r = estimate_thermal_state(d)
        assert r["state"] == LakeThermalState.LIQUID.value, d


def test_frozen_gate_suppresses_hydro_trigger(monkeypatch):
    """FROZEN_SURFACE must bypass the P_breach≥0.70 hydro/volume trigger
    even when susceptibility would fire."""
    from siren.risk import shadow_evidence

    monkeypatch.setattr(
        shadow_evidence,
        "_compute_shadow_susceptibility",
        lambda *a, **k: {"is_available": True, "p_breach": 0.95},
    )
    def _must_not_run(*a, **k):  # pragma: no cover
        raise AssertionError("hydro surrogate ran on a frozen lake")
    monkeypatch.setattr(shadow_evidence, "_compute_shadow_hydro", _must_not_run)

    change_stats = {"lake_thermal_state": "frozen_surface"}
    shadow = shadow_evidence.attach_shadow_evidence(
        change_stats=change_stats,
        obs_config={},
        rainfall_24h=0.0,
        rainfall_7d=0.0,
        dem_path=None,
    )
    hydro = shadow["hydro_surrogate"]
    assert hydro["is_triggered"] is False
    assert "frozen" in hydro["reason"].lower()


def test_liquid_state_leaves_gate_untouched(monkeypatch):
    """Without a frozen state, the normal P_breach logic applies."""
    from siren.risk import shadow_evidence

    monkeypatch.setattr(
        shadow_evidence,
        "_compute_shadow_susceptibility",
        lambda *a, **k: {"is_available": True, "p_breach": 0.3},
    )
    shadow = shadow_evidence.attach_shadow_evidence(
        change_stats={"lake_thermal_state": "liquid"},
        obs_config={},
        rainfall_24h=0.0,
        rainfall_7d=0.0,
        dem_path=None,
    )
    assert "0.70" in shadow["hydro_surrogate"]["reason"]
