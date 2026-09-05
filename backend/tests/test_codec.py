"""Tests for the alert payload codec (PRD §10.4, Track 7.ii, Hard Rule 4)."""

from __future__ import annotations

import json

import pytest

from siren.alerting.codec import decode, encode
from siren.alerting.validate import PayloadTooLargeError, validate_size

# ---------------------------------------------------------------------------
# Representative test alerts (PRD §10.3 / §10.4).
# ---------------------------------------------------------------------------

# Verbose alert per PRD §10.3 — long strings push it over 250 bytes with the
# full field set, so it serves the oversize test.
ALERT_VERBOSE = {
    "alert_id": "alert-0091",
    "geofence_id": "sector-b",
    "severity": "HIGH",
    "hazard_type": "possible_flood_or_debris_flow",
    "confidence": 0.76,
    "exposed_population": 1248,
    "critical_assets": ["bridge-12", "road-4", "village-2"],
    "disease_flags": ["well-3-submerged", "well-7-encircled"],
    "recommended_action": "Verify locally and prepare downstream warning",
    "human_review_required": True,
}

# Minimal alert — fits comfortably within 250 bytes.
ALERT_MINIMAL = {
    "alert_id": "alert-0001",
    "geofence_id": "sector-a",
    "severity": "LOW",
    "hazard_type": "minor_expansion",
    "confidence": 0.45,
    "exposed_population": 100,
    "critical_assets": ["well-1"],
    "disease_flags": [],
    "recommended_action": "Monitor",
    "human_review_required": False,
}

# Compact demo alert in the spirit of the PRD §10.4 example payload
# ({"aid":"siren-04","sec":"B","haz":"GLOF_FL","lvl":3,...}). Short coded
# values keep it well under 250 bytes.
ALERT_DEMO = {
    "alert_id": "alert-0091",
    "geofence_id": "B",
    "severity": "HIGH",
    "hazard_type": "GLOF_FL",
    "confidence": 0.76,
    "exposed_population": 1240,
    "critical_assets": ["BR-12", "RD-4"],
    "disease_flags": [],
    "recommended_action": "BOIL_WATER_NOW",
    "human_review_required": True,
}

# Alerts whose encoded form fits within 250 bytes (used for round-trip).
_FITTING_ALERTS = [ALERT_MINIMAL, ALERT_DEMO]


def _expected_round_trip(alert: dict) -> dict:
    """The alert as it should look after a round trip (siren- prefix added)."""
    expected = dict(alert)
    if not expected["alert_id"].startswith("siren-"):
        expected["alert_id"] = "siren-" + expected["alert_id"]
    return expected


# ---------------------------------------------------------------------------
# Round-trip property test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alert", _FITTING_ALERTS, ids=["minimal", "demo"])
def test_round_trip(alert):
    """decode(encode(alert)) == alert for every field preserved by the codec."""
    payload = encode(alert)
    assert len(payload) <= 250
    decoded = decode(payload)
    assert decoded == _expected_round_trip(alert)


def test_round_trip_all_severities():
    """Every severity level maps to an int and back to its string."""
    base = dict(ALERT_MINIMAL)
    for sev in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        base["severity"] = sev
        decoded = decode(encode(base))
        assert decoded["severity"] == sev


# ---------------------------------------------------------------------------
# Oversize alert raises
# ---------------------------------------------------------------------------


def test_oversize_verbose_alert_raises():
    """The verbose PRD §10.3 alert exceeds 250 bytes and must raise."""
    with pytest.raises(PayloadTooLargeError):
        encode(ALERT_VERBOSE)


def test_oversize_many_assets_raises():
    """An alert with many critical_assets / long strings must raise."""
    big = dict(ALERT_DEMO)
    big["critical_assets"] = [f"asset-{i:04d}-longname" for i in range(40)]
    big["recommended_action"] = "x" * 200
    with pytest.raises(PayloadTooLargeError):
        encode(big)


# ---------------------------------------------------------------------------
# aid prefix test
# ---------------------------------------------------------------------------


def test_aid_prefix_added():
    """alert_id without siren- gets the prefix in the encoded payload."""
    payload = encode(ALERT_DEMO)
    obj = json.loads(payload)
    assert obj["aid"].startswith("siren-")
    assert b"siren-" in payload


def test_aid_prefix_preserved_when_present():
    """An alert_id already starting with siren- is left unchanged."""
    alert = dict(ALERT_DEMO)
    alert["alert_id"] = "siren-04"
    obj = json.loads(encode(alert))
    assert obj["aid"] == "siren-04"


# ---------------------------------------------------------------------------
# Size validation
# ---------------------------------------------------------------------------


def test_validate_size_within_limit():
    validate_size(b"x" * 250)
    validate_size(b"x" * 250, max_bytes=250)


def test_validate_size_exceeds_limit():
    with pytest.raises(PayloadTooLargeError):
        validate_size(b"x" * 251)
    with pytest.raises(PayloadTooLargeError):
        validate_size(b"x" * 300, max_bytes=250)


def test_validate_size_custom_limit():
    validate_size(b"x" * 100, max_bytes=100)
    with pytest.raises(PayloadTooLargeError):
        validate_size(b"x" * 101, max_bytes=100)


# ---------------------------------------------------------------------------
# Exact byte count test (PRD §10.4 demo)
# ---------------------------------------------------------------------------


def test_demo_payload_within_budget():
    """The PRD §10.4 demo alert encodes to <=250 bytes."""
    payload = encode(ALERT_DEMO)
    assert len(payload) <= 250
    # Sanity: it is valid JSON with the expected short keys.
    obj = json.loads(payload)
    assert set(obj.keys()) == {
        "aid", "sec", "haz", "lvl", "conf",
        "exp_pop", "crit", "med_act", "act", "req",
    }
    assert obj["lvl"] == 3  # HIGH


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_encoding_is_deterministic():
    """Same alert -> identical bytes (no unseeded randomness)."""
    assert encode(ALERT_DEMO) == encode(ALERT_DEMO)
    assert encode(ALERT_MINIMAL) == encode(ALERT_MINIMAL)
