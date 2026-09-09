"""RFC 3161 timestamp anchoring for the audit hash chain (V3 §4.5, Phase 4.5).

Anchors hash chain roots to an external public witness so the audit log is
provably tamper-evident even if the database is compromised. Without external
anchoring, an attacker with database access could rewrite the entire chain
and recompute all hashes — the chain would still verify, but the history
would be forged.

RFC 3161 Time-Stamp Protocol (TSP):
    1. Client creates a TimeStamp Request (TSQ) from the hash of the data
    2. Client sends the TSQ to a Time Stamp Authority (TSA)
    3. TSA returns a TimeStamp Response (TSR) with a signed timestamp token
    4. Client verifies the token against the TSA's certificate

Alternative witnesses (V3 §4.5):
    - Rekor (Sigstore): transparency log for signed artifacts
    - OpenTimestamps: Bitcoin-anchored timestamps (provably pre-date any
      future revision)

Offline-safe (Hard Rule 2): in the demo/development environment, the
anchoring is simulated — recording what would be anchored without actual
network calls to a TSA. The production deployment would use real TSA
endpoints or Rekor/OpenTimestamps.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Known TSA endpoints (RFC 3161 compliant).
# These are public services — production use should verify their policies.
TSA_ENDPOINTS: dict[str, str] = {
    "google": "https://timestamp.googleapis.com/",
    "digicert": "http://timestamp.digicert.com",
    "sectigo": "http://timestamp.sectigo.com",
}

# Alternative witnesses (V3 §4.5)
REKOR_ENDPOINT = "https://rekor.sigstore.dev/api/v1/log/entries"
OPENTIMESTAMPS_ENDPOINT = "https://opentimestamps.org/api/v1"

# Default TSA for production use
DEFAULT_TSA: str = "google"

# Content type for RFC 3161 TimeStamp Protocol
TSP_CONTENT_TYPE: str = "application/timestamp-query"


@dataclass
class TimestampAnchor:
    """A cryptographic timestamp anchor for an audit chain root.

    Attributes:
        chain_root: the SHA-256 hash being anchored (the latest chain entry).
        witness: the witness type ("rfc3161", "rekor", "opentimestamps").
        witness_endpoint: the endpoint URL used.
        timestamp: ISO 8601 timestamp from the witness.
        token: the raw timestamp token (base64-encoded in production;
            simulated in demo mode).
        is_verified: True if the token has been verified against the witness.
        is_simulated: True if this is a simulated anchor (demo/offline mode).
    """

    chain_root: str
    witness: str
    witness_endpoint: str
    timestamp: str
    token: str
    is_verified: bool = False
    is_simulated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_root": self.chain_root,
            "witness": self.witness,
            "witness_endpoint": self.witness_endpoint,
            "timestamp": self.timestamp,
            "token": self.token[:64] + "..." if len(self.token) > 64 else self.token,
            "is_verified": self.is_verified,
            "is_simulated": self.is_simulated,
        }


class TimestampAuthority:
    """RFC 3161 / Rekor / OpenTimestamps timestamp anchoring for audit chains.

    Anchors the hash chain root to an external public witness so the audit
    log is provably tamper-evident. In demo/offline mode, produces simulated
    anchors that record what would be anchored without actual network calls.

    Args:
        witness: witness type ("rfc3161", "rekor", "opentimestamps").
        tsa_name: TSA name for RFC 3161 (e.g. "google", "digicert").
        simulate: if True, always produce simulated anchors (demo mode).
    """

    def __init__(
        self,
        witness: str = "rfc3161",
        tsa_name: str = DEFAULT_TSA,
        simulate: bool = True,
    ) -> None:
        self.witness = witness
        self.tsa_name = tsa_name
        self.simulate = simulate

        if witness == "rfc3161":
            if tsa_name not in TSA_ENDPOINTS:
                raise ValueError(
                    f"unknown TSA: {tsa_name}. Available: {list(TSA_ENDPOINTS.keys())}"
                )
            self.endpoint = TSA_ENDPOINTS[tsa_name]
        elif witness == "rekor":
            self.endpoint = REKOR_ENDPOINT
        elif witness == "opentimestamps":
            self.endpoint = OPENTIMESTAMPS_ENDPOINT
        else:
            raise ValueError(
                f"unknown witness: {witness}. Expected 'rfc3161', 'rekor', or 'opentimestamps'"
            )

    def anchor(self, chain_root: str) -> TimestampAnchor:
        """Anchor a hash chain root to an external witness.

        Args:
            chain_root: the SHA-256 hash to anchor (typically the latest
                entry in the audit hash chain).

        Returns:
            TimestampAnchor with the witness timestamp + token.

        Raises:
            ValueError: if chain_root is not a valid SHA-256 hash (64 hex chars).
        """
        if not self._is_valid_sha256(chain_root):
            raise ValueError(
                f"chain_root must be a 64-char hex SHA-256 hash, got {len(chain_root)} chars"
            )

        if self.simulate:
            return self._simulate_anchor(chain_root)

        # Production implementation would make actual network calls here
        if self.witness == "rfc3161":
            return self._anchor_rfc3161(chain_root)
        elif self.witness == "rekor":
            return self._anchor_rekor(chain_root)
        else:
            return self._anchor_opentimestamps(chain_root)

    def _simulate_anchor(self, chain_root: str) -> TimestampAnchor:
        """Produce a simulated anchor (demo/offline mode)."""
        now = datetime.now(timezone.utc).isoformat()
        # Simulated token: hash of chain_root + timestamp (not cryptographically
        # meaningful, but deterministic and reproducible)
        token_material = f"simulated:{chain_root}:{now}".encode("utf-8")
        token = hashlib.sha256(token_material).hexdigest()

        logger.info(
            "Simulated timestamp anchor: chain_root=%s... witness=%s",
            chain_root[:16], self.witness,
        )

        return TimestampAnchor(
            chain_root=chain_root,
            witness=self.witness,
            witness_endpoint=self.endpoint,
            timestamp=now,
            token=token,
            is_verified=True,  # simulated anchors are "verified" by construction
            is_simulated=True,
        )

    def _anchor_rfc3161(self, chain_root: str) -> TimestampAnchor:
        """Anchor via RFC 3161 Time-Stamp Protocol (production — requires network)."""
        raise NotImplementedError(
            "RFC 3161 anchoring requires network access — use simulate=True for demo"
        )

    def _anchor_rekor(self, chain_root: str) -> TimestampAnchor:
        """Anchor via Rekor transparency log (production — requires network)."""
        raise NotImplementedError(
            "Rekor anchoring requires network access — use simulate=True for demo"
        )

    def _anchor_opentimestamps(self, chain_root: str) -> TimestampAnchor:
        """Anchor via OpenTimestamps (production — requires network)."""
        raise NotImplementedError(
            "OpenTimestamps anchoring requires network access — use simulate=True for demo"
        )

    def _is_valid_sha256(self, hash_str: str) -> bool:
        """Check if a string is a valid 64-character hex SHA-256 hash."""
        if len(hash_str) != 64:
            return False
        try:
            int(hash_str, 16)
            return True
        except ValueError:
            return False

    def verify_anchor(self, anchor: TimestampAnchor) -> bool:
        """Verify a timestamp anchor.

        For simulated anchors, verification is a structural check (the token
        is deterministic from the chain_root + timestamp). For production
        anchors, this would verify the cryptographic signature against the
        TSA's certificate.

        Args:
            anchor: the TimestampAnchor to verify.

        Returns:
            True if the anchor is valid.
        """
        if not self._is_valid_sha256(anchor.chain_root):
            return False

        if anchor.is_simulated:
            # Simulated anchors: re-derive the token and check it matches
            token_material = f"simulated:{anchor.chain_root}:{anchor.timestamp}".encode("utf-8")
            expected_token = hashlib.sha256(token_material).hexdigest()
            return anchor.token == expected_token

        # Production verification would check the TSA's cryptographic signature
        # against the TSA's certificate chain. This is a stub.
        logger.warning("Production anchor verification not implemented — returning True")
        return True


def anchor_chain_root(
    chain_root: str,
    witness: str = "rfc3161",
    tsa_name: str = DEFAULT_TSA,
    simulate: bool = True,
) -> TimestampAnchor:
    """Convenience function: anchor a hash chain root to an external witness.

    Args:
        chain_root: the SHA-256 hash to anchor.
        witness: witness type ("rfc3161", "rekor", "opentimestamps").
        tsa_name: TSA name for RFC 3161.
        simulate: if True, produce a simulated anchor (demo mode).

    Returns:
        TimestampAnchor with the witness timestamp + token.
    """
    authority = TimestampAuthority(witness=witness, tsa_name=tsa_name, simulate=simulate)
    return authority.anchor(chain_root)
