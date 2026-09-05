"""Alert payload codec — serialize alerts to <=250-byte packets (PRD §10.4).

Maps verbose Alert fields (PRD §10.3 / API_CONTRACT.md) to short keys and
emits compact JSON encoded as UTF-8 bytes, suitable for LoRa mesh, satellite
messenger, or low-bandwidth SMS transport (Track 7.ii).

Key mapping (verbose -> compact):
    alert_id             -> aid   (always prefixed with "siren-")
    geofence_id          -> sec
    hazard_type          -> haz
    severity             -> lvl   (CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1)
    confidence           -> conf
    exposed_population   -> exp_pop
    critical_assets      -> crit
    disease_flags        -> med_act
    recommended_action   -> act
    human_review_required-> req

Encoding is deterministic: the same alert always produces the same bytes
(fixed key order, shortest JSON separators, no unseeded randomness).
"""

from __future__ import annotations

import json

from .validate import PayloadTooLargeError, validate_size

__all__ = ["encode", "decode", "PayloadTooLargeError"]

# Severity <-> integer level mapping (PRD §10.4: HIGH=3, MEDIUM=2, LOW=1,
# CRITICAL=4).
SEVERITY_TO_LVL: dict[str, int] = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}
LVL_TO_SEVERITY: dict[int, str] = {v: k for k, v in SEVERITY_TO_LVL.items()}

# Fixed key order for deterministic encoding (same alert -> same bytes).
_KEY_ORDER = (
    "aid",
    "sec",
    "haz",
    "lvl",
    "conf",
    "exp_pop",
    "crit",
    "med_act",
    "act",
    "req",
)

_SIREN_PREFIX = "siren-"


def _ensure_siren_prefix(alert_id: str) -> str:
    """Return ``alert_id`` guaranteed to start with the ``siren-`` prefix."""
    if not isinstance(alert_id, str) or not alert_id:
        raise ValueError("alert_id must be a non-empty string")
    if alert_id.startswith(_SIREN_PREFIX):
        return alert_id
    return _SIREN_PREFIX + alert_id


def encode(alert: dict) -> bytes:
    """Encode an alert dict to a <=250-byte compressed payload.

    The alert dict has fields per PRD §10.3 / API_CONTRACT.md Alert model:
        alert_id, geofence_id, severity, hazard_type, confidence,
        exposed_population, critical_assets, disease_flags,
        recommended_action, human_review_required

    Returns:
        UTF-8 encoded JSON bytes, <=250 bytes, with short keys and the
        ``aid`` field guaranteed to start with ``siren-``.

    Raises:
        PayloadTooLargeError: If the encoded payload exceeds 250 bytes.
        ValueError: If a required field is missing or severity is unknown.
        KeyError: If a required field is absent from ``alert``.
    """
    severity = alert["severity"]
    if severity not in SEVERITY_TO_LVL:
        raise ValueError(f"unknown severity: {severity!r}")

    compact = {
        "aid": _ensure_siren_prefix(alert["alert_id"]),
        "sec": alert["geofence_id"],
        "haz": alert["hazard_type"],
        "lvl": SEVERITY_TO_LVL[severity],
        "conf": alert["confidence"],
        "exp_pop": alert["exposed_population"],
        "crit": list(alert["critical_assets"]),
        "med_act": list(alert["disease_flags"]),
        "act": alert["recommended_action"],
        "req": bool(alert["human_review_required"]),
    }
    # Deterministic, fixed key order -> identical bytes for identical alerts.
    ordered = {k: compact[k] for k in _KEY_ORDER}
    payload = json.dumps(
        ordered, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    validate_size(payload, 250)
    return payload


def decode(payload: bytes) -> dict:
    """Decode a compressed payload back to an alert dict.

    Returns:
        dict with the same fields that were encoded (PRD §10.3 Alert model),
        with ``severity`` restored from the integer level and ``alert_id``
        carrying the ``siren-`` prefix that encode() guaranteed.

    Raises:
        ValueError: If the payload is not valid JSON or has an unknown level.
    """
    if isinstance(payload, (bytes, bytearray)):
        text = payload.decode("utf-8")
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError("payload must be bytes or str")

    obj = json.loads(text)

    lvl = obj["lvl"]
    if lvl not in LVL_TO_SEVERITY:
        raise ValueError(f"unknown severity level: {lvl!r}")

    return {
        "alert_id": obj["aid"],
        "geofence_id": obj["sec"],
        "hazard_type": obj["haz"],
        "severity": LVL_TO_SEVERITY[lvl],
        "confidence": obj["conf"],
        "exposed_population": obj["exp_pop"],
        "critical_assets": list(obj["crit"]),
        "disease_flags": list(obj["med_act"]),
        "recommended_action": obj["act"],
        "human_review_required": bool(obj["req"]),
    }
