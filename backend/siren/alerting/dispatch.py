"""Dual-path hardware dispatch engine (V3 §4.4, Phase 4.4).

Replaces the hackathon-era ntfy.sh browser-side push with a dual-path
hardware dispatch engine:

  Path 1: Terrestrial SMS via AWS SNS / Twilio
    - Low latency, wide coverage in inhabited valleys
    - Fails when cellular infrastructure is down (the disaster scenario)

  Path 2: Satellite SMS via Iridium SBD (RockBLOCK 9603) or LoRa (SX1262)
    - High latency (~60s), works when cellular is down
    - Serial bridge to the modem hardware

The ≤250-byte payload codec is preserved (Hard Rule 4). The dispatch engine
sends via both paths simultaneously; delivery receipts from either path mark
the alert as delivered.

Offline-safe (Hard Rule 2): the dispatch engine is a simulation in the
demo/development environment. Real hardware dispatch requires the production
deployment with connected modems. The simulation records what *would* be
sent and returns a simulated delivery receipt.

Human gate (Hard Rule 3): no dispatch without a recorded confirm review.
This is enforced by the repository's dispatch trigger, not by this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DispatchPath(str, Enum):
    """The two hardware dispatch paths (V3 §4.4)."""
    SMS = "sms"          # Terrestrial: AWS SNS / Twilio
    SATELLITE = "satellite"  # Satellite: Iridium SBD / LoRa


class DispatchStatus(str, Enum):
    """Status of a dispatch attempt."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    SIMULATED = "simulated"  # Demo/offline mode — no real hardware


@dataclass
class DispatchReceipt:
    """Receipt from a dispatch path attempt.

    Attributes:
        path: which dispatch path was used.
        status: delivery status.
        message_id: provider message ID (or simulated ID).
        timestamp: ISO 8601 delivery timestamp.
        error: error message if status is FAILED.
        payload_size: size of the dispatched payload in bytes.
    """

    path: DispatchPath
    status: DispatchStatus
    message_id: str
    timestamp: str
    error: str | None = None
    payload_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.value,
            "status": self.status.value,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "error": self.error,
            "payload_size": self.payload_size,
        }


@dataclass
class DispatchResult:
    """Result of a dual-path dispatch attempt.

    Attributes:
        dispatch_id: unique dispatch identifier.
        receipts: list of receipts from each path.
        delivered: True if at least one path delivered successfully.
        payload: the ≤250-byte payload that was dispatched.
    """

    dispatch_id: str
    receipts: list[DispatchReceipt] = field(default_factory=list)
    delivered: bool = False
    payload: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "receipts": [r.to_dict() for r in self.receipts],
            "delivered": self.delivered,
            "payload": self.payload,
            "payload_size": len(self.payload),
        }


class DispatchEngine:
    """Dual-path hardware dispatch engine (V3 §4.4).

    Sends the ≤250-byte alert payload via both terrestrial SMS and satellite
    paths simultaneously. Delivery from either path marks the alert as delivered.

    In demo/offline mode (no hardware connected), both paths return
    SIMULATED receipts — recording what would be sent without actual
    hardware interaction.

    Args:
        sms_provider: SMS provider name ("twilio", "aws_sns", or "simulated").
        satellite_provider: satellite provider name ("iridium_sbd", "lora", or "simulated").
        sms_config: provider-specific config (API keys, phone numbers).
        satellite_config: provider-specific config (serial port, modem ID).
        simulate: if True, always return SIMULATED receipts (demo mode).
    """

    def __init__(
        self,
        sms_provider: str = "simulated",
        satellite_provider: str = "simulated",
        sms_config: dict[str, Any] | None = None,
        satellite_config: dict[str, Any] | None = None,
        simulate: bool = True,
    ) -> None:
        self.sms_provider = sms_provider
        self.satellite_provider = satellite_provider
        self.sms_config = sms_config or {}
        self.satellite_config = satellite_config or {}
        self.simulate = simulate

    def dispatch(
        self,
        dispatch_id: str,
        payload: str,
        recipient_group: str = "default",
    ) -> DispatchResult:
        """Dispatch a payload via both hardware paths simultaneously.

        Args:
            dispatch_id: unique dispatch identifier.
            payload: the ≤250-byte compressed alert payload (Hard Rule 4).
            recipient_group: recipient group name (for routing).

        Returns:
            DispatchResult with receipts from both paths.

        Raises:
            ValueError: if the payload exceeds 250 bytes (Hard Rule 4).
        """
        # Hard Rule 4: payload must be ≤ 250 bytes
        if len(payload) > 250:
            raise ValueError(
                f"Payload exceeds 250-byte limit: {len(payload)} bytes (Hard Rule 4)"
            )

        logger.info(
            "Dispatch %s: payload=%d bytes, recipient_group=%s",
            dispatch_id, len(payload), recipient_group,
        )

        receipts: list[DispatchReceipt] = []

        # Path 1: Terrestrial SMS
        sms_receipt = self._dispatch_sms(dispatch_id, payload, recipient_group)
        receipts.append(sms_receipt)

        # Path 2: Satellite (Iridium SBD / LoRa)
        sat_receipt = self._dispatch_satellite(dispatch_id, payload, recipient_group)
        receipts.append(sat_receipt)

        # Delivered if at least one path succeeded
        delivered = any(
            r.status in (DispatchStatus.DELIVERED, DispatchStatus.SIMULATED)
            for r in receipts
        )

        result = DispatchResult(
            dispatch_id=dispatch_id,
            receipts=receipts,
            delivered=delivered,
            payload=payload,
        )

        logger.info(
            "Dispatch %s: delivered=%s (sms=%s, sat=%s)",
            dispatch_id, delivered,
            sms_receipt.status.value, sat_receipt.status.value,
        )

        return result

    def _dispatch_sms(
        self,
        dispatch_id: str,
        payload: str,
        recipient_group: str,
    ) -> DispatchReceipt:
        """Dispatch via terrestrial SMS (AWS SNS / Twilio)."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        if self.simulate or self.sms_provider == "simulated":
            return DispatchReceipt(
                path=DispatchPath.SMS,
                status=DispatchStatus.SIMULATED,
                message_id=f"sim-sms-{dispatch_id}",
                timestamp=now,
                payload_size=len(payload),
            )

        # Real SMS dispatch would go here (Twilio/AWS SNS SDK calls)
        # This is production infrastructure — not active in the demo build
        try:
            if self.sms_provider == "twilio":
                return self._dispatch_twilio(dispatch_id, payload, recipient_group, now)
            elif self.sms_provider == "aws_sns":
                return self._dispatch_aws_sns(dispatch_id, payload, recipient_group, now)
            else:
                return DispatchReceipt(
                    path=DispatchPath.SMS,
                    status=DispatchStatus.FAILED,
                    message_id="",
                    timestamp=now,
                    error=f"unknown SMS provider: {self.sms_provider}",
                    payload_size=len(payload),
                )
        except Exception as e:
            logger.exception("SMS dispatch failed for %s", dispatch_id)
            return DispatchReceipt(
                path=DispatchPath.SMS,
                status=DispatchStatus.FAILED,
                message_id="",
                timestamp=now,
                error=str(e),
                payload_size=len(payload),
            )

    def _dispatch_satellite(
        self,
        dispatch_id: str,
        payload: str,
        recipient_group: str,
    ) -> DispatchReceipt:
        """Dispatch via satellite (Iridium SBD / LoRa)."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        if self.simulate or self.satellite_provider == "simulated":
            return DispatchReceipt(
                path=DispatchPath.SATELLITE,
                status=DispatchStatus.SIMULATED,
                message_id=f"sim-sat-{dispatch_id}",
                timestamp=now,
                payload_size=len(payload),
            )

        # Real satellite dispatch would go here (serial bridge to RockBLOCK)
        try:
            if self.satellite_provider == "iridium_sbd":
                return self._dispatch_iridium_sbd(dispatch_id, payload, recipient_group, now)
            elif self.satellite_provider == "lora":
                return self._dispatch_lora(dispatch_id, payload, recipient_group, now)
            else:
                return DispatchReceipt(
                    path=DispatchPath.SATELLITE,
                    status=DispatchStatus.FAILED,
                    message_id="",
                    timestamp=now,
                    error=f"unknown satellite provider: {self.satellite_provider}",
                    payload_size=len(payload),
                )
        except Exception as e:
            logger.exception("Satellite dispatch failed for %s", dispatch_id)
            return DispatchReceipt(
                path=DispatchPath.SATELLITE,
                status=DispatchStatus.FAILED,
                message_id="",
                timestamp=now,
                error=str(e),
                payload_size=len(payload),
            )

    def _dispatch_twilio(
        self, dispatch_id: str, payload: str, recipient_group: str, now: str
    ) -> DispatchReceipt:
        """Dispatch via Twilio SMS (production — requires API keys)."""
        # Production implementation would use the Twilio SDK:
        #   from twilio.rest import Client
        #   client = Client(self.sms_config["account_sid"], self.sms_config["auth_token"])
        #   message = client.messages.create(body=payload, to=..., from_=...)
        # For now, this is a stub that documents the production path.
        raise NotImplementedError(
            "Twilio dispatch requires production API keys — use simulate=True for demo"
        )

    def _dispatch_aws_sns(
        self, dispatch_id: str, payload: str, recipient_group: str, now: str
    ) -> DispatchReceipt:
        """Dispatch via AWS SNS (production — requires AWS credentials)."""
        raise NotImplementedError(
            "AWS SNS dispatch requires production credentials — use simulate=True for demo"
        )

    def _dispatch_iridium_sbd(
        self, dispatch_id: str, payload: str, recipient_group: str, now: str
    ) -> DispatchReceipt:
        """Dispatch via Iridium SBD serial bridge (production — requires hardware)."""
        # Production implementation would use pyserial:
        #   import serial
        #   ser = serial.Serial(self.satellite_config["port"], 19200)
        #   ser.write(payload.encode())
        #   ser.close()
        raise NotImplementedError(
            "Iridium SBD dispatch requires connected hardware — use simulate=True for demo"
        )

    def _dispatch_lora(
        self, dispatch_id: str, payload: str, recipient_group: str, now: str
    ) -> DispatchReceipt:
        """Dispatch via SX1262 LoRa gateway (production — requires hardware)."""
        raise NotImplementedError(
            "LoRa dispatch requires connected hardware — use simulate=True for demo"
        )
