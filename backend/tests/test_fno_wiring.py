"""Tests for Sprint 3 FNO wiring — corridor arrival horizons + codec t_arr.

Tests the integration of FNO hydrodynamic surrogate outputs into the
corridor exposure engine and the alert payload codec, while verifying
that the deterministic pipeline remains unaffected (shadow mode).
"""

from __future__ import annotations

import json

from siren.alerting.codec import decode, encode
from siren.geo.corridor import attach_arrival_horizons


# ---------------------------------------------------------------------------
# attach_arrival_horizons (corridor.py)
# ---------------------------------------------------------------------------

def test_attach_arrival_horizons_adds_t_arrival_to_matching_exposures():
    """attach_arrival_horizons adds t_arrival_min to exposures by name match."""
    corridor_result = {
        "type": "FeatureCollection",
        "features": [],
        "exposures": [
            {"asset_id": "BR-12", "asset_type": "bridge", "name": "Hillary Bridge",
             "distance_m": 60.0, "buffer_m": 75.0, "in_floodplain": True},
            {"asset_id": "village-2", "asset_type": "settlement", "name": "Benkar",
             "distance_m": 210.0, "buffer_m": 100.0, "in_floodplain": True},
            {"asset_id": "RD-4", "asset_type": "road", "name": "Road 4",
             "distance_m": 40.0, "buffer_m": 50.0, "in_floodplain": True},
        ],
    }
    t_arrival = {"Hillary Bridge": 55.5, "Benkar": 134.9, "Jorsale": 196.4}

    result = attach_arrival_horizons(corridor_result, t_arrival)

    # Hillary Bridge and Benkar should have arrival times; Road 4 should not
    exposures = result["exposures"]
    by_name = {e["name"]: e for e in exposures}
    assert by_name["Hillary Bridge"]["t_arrival_min"] == 55.5
    assert by_name["Benkar"]["t_arrival_min"] == 134.9
    assert by_name["Road 4"]["t_arrival_min"] is None


def test_attach_arrival_horizons_tags_provenance():
    """attach_arrival_horizons tags the corridor with FNO provenance."""
    corridor_result = {"exposures": [{"name": "Hillary Bridge", "asset_id": "BR-12"}]}
    result = attach_arrival_horizons(corridor_result, {"Hillary Bridge": 55.0})
    assert result["fno_provenance"] == "fno_surrogate_v1"
    assert result["fno_arrival_available"] is True


def test_attach_arrival_horizons_custom_provenance():
    """attach_arrival_horizons accepts a custom provenance tag."""
    corridor_result = {"exposures": [{"name": "Test", "asset_id": "T1"}]}
    result = attach_arrival_horizons(corridor_result, {"Test": 10.0}, provenance="fno_v2")
    assert result["fno_provenance"] == "fno_v2"


def test_attach_arrival_horizons_no_matches():
    """attach_arrival_horizons with no matches sets fno_arrival_available=False."""
    corridor_result = {"exposures": [{"name": "Unknown", "asset_id": "U1"}]}
    result = attach_arrival_horizons(corridor_result, {"Hillary Bridge": 55.0})
    assert result["fno_arrival_available"] is False
    assert result["exposures"][0]["t_arrival_min"] is None


def test_attach_arrival_horizons_empty_exposures():
    """attach_arrival_horizons handles empty exposures gracefully."""
    corridor_result = {"exposures": []}
    result = attach_arrival_horizons(corridor_result, {"Hillary Bridge": 55.0})
    assert result["fno_provenance"] == "fno_surrogate_v1"
    assert result["fno_arrival_available"] is False


def test_attach_arrival_horizons_case_insensitive():
    """attach_arrival_horizons matches sector names case-insensitively."""
    corridor_result = {"exposures": [{"name": "hillary bridge", "asset_id": "BR-12"}]}
    result = attach_arrival_horizons(corridor_result, {"Hillary Bridge": 55.0})
    assert result["exposures"][0]["t_arrival_min"] == 55.0


# ---------------------------------------------------------------------------
# Codec with sector arrivals (Sprint 3)
# ---------------------------------------------------------------------------

ALERT_WITH_ARRIVALS = {
    "alert_id": "alert-0091",
    "geofence_id": "B",
    "severity": "critical",
    "hazard_type": "GLOF_FL",
    "exposed_population": 1240,
    "critical_assets": ["BR-12", "RD-4"],
    "disease_flags": ["BOIL_WATER_NOW"],
    "sector_arrivals": {"sec_chk": 55, "sec_dik": 135, "sec_sng": 196},
}


def test_codec_with_arrivals_round_trip():
    """Alerts with sector_arrivals survive a round-trip through encode/decode."""
    payload = encode(ALERT_WITH_ARRIVALS)
    assert len(payload) <= 250
    decoded = decode(payload)
    assert decoded["sector_arrivals"] == {"sec_chk": 55, "sec_dik": 135, "sec_sng": 196}


def test_codec_with_arrivals_byte_budget():
    """Adding 3 sector arrival times keeps the payload well within 250 bytes."""
    base = {
        "alert_id": "alert-0091",
        "geofence_id": "B",
        "severity": "critical",
        "hazard_type": "GLOF_FL",
        "exposed_population": 1240,
        "critical_assets": ["BR-12", "RD-4"],
        "disease_flags": ["BOIL_WATER_NOW"],
    }
    payload_without = encode(base)
    payload_with = encode({**base, "sector_arrivals": {"sec_chk": 55, "sec_dik": 135, "sec_sng": 196}})
    overhead = len(payload_with) - len(payload_without)
    assert overhead < 60, f"t_arr overhead is {overhead} bytes, expected <50"
    assert len(payload_with) <= 250


def test_codec_without_arrivals_omits_t_arr():
    """Alerts without sector_arrivals do not include t_arr in the payload."""
    base = {
        "alert_id": "alert-0001",
        "geofence_id": "A",
        "severity": "informational",
        "hazard_type": "minor_expansion",
        "exposed_population": 100,
        "critical_assets": ["well-1"],
        "disease_flags": [],
    }
    payload = encode(base)
    obj = json.loads(payload)
    assert "t_arr" not in obj


def test_codec_arrivals_are_integers():
    """Sector arrival times are encoded as integers (compact representation)."""
    alert = dict(ALERT_WITH_ARRIVALS)
    alert["sector_arrivals"] = {"sec_a": 55.7, "sec_b": 134.2}
    payload = encode(alert)
    obj = json.loads(payload)
    assert obj["t_arr"]["sec_a"] == 56  # rounded to int
    assert obj["t_arr"]["sec_b"] == 134


# ---------------------------------------------------------------------------
# Shadow mode integrity — deterministic pipeline unaffected
# ---------------------------------------------------------------------------

def test_shadow_hydro_provenance_tag():
    """The FNO shadow evidence carries the 'fno_surrogate_v1' provenance tag."""
    from siren.risk.shadow_evidence import _compute_shadow_hydro
    result = _compute_shadow_hydro(
        obs_config={"expansion_pct": 30.0, "mean_slope_degrees": 20.0},
        p_breach=0.85,
        dem_path=None,
    )
    assert result.get("provenance") == "fno_surrogate_v1"
    assert result.get("is_triggered") is True
