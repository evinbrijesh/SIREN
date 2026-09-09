"""Tests for alerting/dispatch.py — dual-path hardware dispatch (V3 §4.4).

Tests cover the dual-path dispatch engine, payload size enforcement,
simulated receipts, and the production stubs.
"""

from __future__ import annotations

import pytest

from siren.alerting.dispatch import (
    DispatchEngine,
    DispatchPath,
    DispatchStatus,
    DispatchReceipt,
    DispatchResult,
)


# --------------------------------------------------------------------------- #
# Enums and dataclasses
# --------------------------------------------------------------------------- #

def test_dispatch_path_values():
    """DispatchPath has the two V3 §4.4 paths."""
    assert DispatchPath.SMS.value == "sms"
    assert DispatchPath.SATELLITE.value == "satellite"


def test_dispatch_status_values():
    """DispatchStatus has the expected statuses."""
    assert DispatchStatus.PENDING.value == "pending"
    assert DispatchStatus.DELIVERED.value == "delivered"
    assert DispatchStatus.FAILED.value == "failed"
    assert DispatchStatus.SIMULATED.value == "simulated"


def test_dispatch_receipt_to_dict():
    """DispatchReceipt.to_dict produces a serializable dict."""
    r = DispatchReceipt(
        path=DispatchPath.SMS,
        status=DispatchStatus.SIMULATED,
        message_id="sim-sms-disp-001",
        timestamp="2026-09-08T12:00:00Z",
        payload_size=180,
    )
    d = r.to_dict()
    assert d["path"] == "sms"
    assert d["status"] == "simulated"
    assert d["message_id"] == "sim-sms-disp-001"
    assert d["payload_size"] == 180
    assert d["error"] is None


def test_dispatch_result_to_dict():
    """DispatchResult.to_dict produces a serializable dict."""
    r = DispatchResult(
        dispatch_id="disp-001",
        receipts=[
            DispatchReceipt(
                path=DispatchPath.SMS,
                status=DispatchStatus.SIMULATED,
                message_id="sim-sms-disp-001",
                timestamp="2026-09-08T12:00:00Z",
                payload_size=180,
            ),
        ],
        delivered=True,
        payload="siren-001...",
    )
    d = r.to_dict()
    assert d["dispatch_id"] == "disp-001"
    assert d["delivered"] is True
    assert d["payload"] == "siren-001..."
    assert d["payload_size"] == len("siren-001...")
    assert len(d["receipts"]) == 1


# --------------------------------------------------------------------------- #
# DispatchEngine — simulated mode (demo/offline)
# --------------------------------------------------------------------------- #

def test_dispatch_engine_simulated_returns_two_receipts():
    """Simulated dispatch returns receipts from both paths."""
    engine = DispatchEngine(simulate=True)
    result = engine.dispatch("disp-001", "siren-payload-001", "default")
    assert len(result.receipts) == 2
    assert result.receipts[0].path == DispatchPath.SMS
    assert result.receipts[1].path == DispatchPath.SATELLITE


def test_dispatch_engine_simulated_status():
    """Simulated dispatch returns SIMULATED status on both paths."""
    engine = DispatchEngine(simulate=True)
    result = engine.dispatch("disp-001", "siren-payload-001", "default")
    assert result.receipts[0].status == DispatchStatus.SIMULATED
    assert result.receipts[1].status == DispatchStatus.SIMULATED


def test_dispatch_engine_simulated_delivered():
    """Simulated dispatch marks the alert as delivered."""
    engine = DispatchEngine(simulate=True)
    result = engine.dispatch("disp-001", "siren-payload-001", "default")
    assert result.delivered is True


def test_dispatch_engine_simulated_message_ids():
    """Simulated dispatch produces unique message IDs per path."""
    engine = DispatchEngine(simulate=True)
    result = engine.dispatch("disp-001", "siren-payload-001", "default")
    assert "sim-sms-disp-001" == result.receipts[0].message_id
    assert "sim-sat-disp-001" == result.receipts[1].message_id


def test_dispatch_engine_simulated_payload_size():
    """Simulated dispatch records the payload size."""
    engine = DispatchEngine(simulate=True)
    payload = "siren-001-flood-critical"
    result = engine.dispatch("disp-001", payload, "default")
    for receipt in result.receipts:
        assert receipt.payload_size == len(payload)


# --------------------------------------------------------------------------- #
# DispatchEngine — payload size enforcement (Hard Rule 4)
# --------------------------------------------------------------------------- #

def test_dispatch_engine_rejects_oversize_payload():
    """Dispatch raises ValueError on payload > 250 bytes (Hard Rule 4)."""
    engine = DispatchEngine(simulate=True)
    oversize_payload = "x" * 251
    with pytest.raises(ValueError, match="250-byte limit"):
        engine.dispatch("disp-001", oversize_payload, "default")


def test_dispatch_engine_accepts_250_byte_payload():
    """Dispatch accepts a payload exactly at the 250-byte limit."""
    engine = DispatchEngine(simulate=True)
    payload = "x" * 250
    result = engine.dispatch("disp-001", payload, "default")
    assert result.delivered is True
    assert result.receipts[0].payload_size == 250


def test_dispatch_engine_accepts_small_payload():
    """Dispatch accepts a small payload."""
    engine = DispatchEngine(simulate=True)
    payload = "siren-001"
    result = engine.dispatch("disp-001", payload, "default")
    assert result.delivered is True


# --------------------------------------------------------------------------- #
# DispatchEngine — production stubs
# --------------------------------------------------------------------------- #

def test_dispatch_engine_twilio_without_keys_fails():
    """Twilio dispatch without API keys returns FAILED receipt."""
    engine = DispatchEngine(
        sms_provider="twilio",
        simulate=False,
        sms_config={"account_sid": "test", "auth_token": "test"},
    )
    result = engine.dispatch("disp-001", "siren-payload", "default")
    sms_receipt = result.receipts[0]
    assert sms_receipt.status == DispatchStatus.FAILED
    assert "Twilio" in sms_receipt.error


def test_dispatch_engine_aws_sns_without_creds_fails():
    """AWS SNS dispatch without credentials returns FAILED receipt."""
    engine = DispatchEngine(
        sms_provider="aws_sns",
        simulate=False,
    )
    result = engine.dispatch("disp-001", "siren-payload", "default")
    sms_receipt = result.receipts[0]
    assert sms_receipt.status == DispatchStatus.FAILED
    assert "AWS SNS" in sms_receipt.error


def test_dispatch_engine_iridium_without_hardware_fails():
    """Iridium SBD dispatch without hardware returns FAILED receipt."""
    engine = DispatchEngine(
        satellite_provider="iridium_sbd",
        simulate=False,
    )
    result = engine.dispatch("disp-001", "siren-payload", "default")
    sat_receipt = result.receipts[1]
    assert sat_receipt.status == DispatchStatus.FAILED
    assert "Iridium" in sat_receipt.error


def test_dispatch_engine_lora_without_hardware_fails():
    """LoRa dispatch without hardware returns FAILED receipt."""
    engine = DispatchEngine(
        satellite_provider="lora",
        simulate=False,
    )
    result = engine.dispatch("disp-001", "siren-payload", "default")
    sat_receipt = result.receipts[1]
    assert sat_receipt.status == DispatchStatus.FAILED
    assert "LoRa" in sat_receipt.error


def test_dispatch_engine_unknown_sms_provider_fails():
    """Unknown SMS provider returns a FAILED receipt."""
    engine = DispatchEngine(
        sms_provider="unknown_provider",
        simulate=False,
    )
    result = engine.dispatch("disp-001", "siren-payload", "default")
    sms_receipt = result.receipts[0]
    assert sms_receipt.status == DispatchStatus.FAILED
    assert "unknown SMS provider" in sms_receipt.error


def test_dispatch_engine_unknown_satellite_provider_fails():
    """Unknown satellite provider returns a FAILED receipt."""
    engine = DispatchEngine(
        satellite_provider="unknown_provider",
        simulate=False,
    )
    result = engine.dispatch("disp-001", "siren-payload", "default")
    sat_receipt = result.receipts[1]
    assert sat_receipt.status == DispatchStatus.FAILED
    assert "unknown satellite provider" in sat_receipt.error


# --------------------------------------------------------------------------- #
# DispatchEngine — mixed mode (one simulated, one real)
# --------------------------------------------------------------------------- #

def test_dispatch_engine_mixed_mode_one_simulated_one_failed():
    """Mixed mode: simulated SMS + failed satellite → delivered=True."""
    engine = DispatchEngine(
        sms_provider="simulated",
        satellite_provider="unknown",
        simulate=False,  # don't force simulate for all
    )
    result = engine.dispatch("disp-001", "siren-payload", "default")
    assert result.receipts[0].status == DispatchStatus.SIMULATED  # SMS simulated
    assert result.receipts[1].status == DispatchStatus.FAILED  # satellite unknown
    assert result.delivered is True  # at least one path delivered


def test_dispatch_engine_both_failed_not_delivered():
    """Both paths failed → delivered=False."""
    engine = DispatchEngine(
        sms_provider="unknown1",
        satellite_provider="unknown2",
        simulate=False,
    )
    result = engine.dispatch("disp-001", "siren-payload", "default")
    assert result.receipts[0].status == DispatchStatus.FAILED
    assert result.receipts[1].status == DispatchStatus.FAILED
    assert result.delivered is False
