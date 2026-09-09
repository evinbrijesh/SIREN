"""Tests for audit/timestamp.py — RFC 3161 timestamp anchoring (V3 §4.5, Phase 4.5).

Tests cover the timestamp authority, simulated anchoring, chain root
validation, anchor verification, and the production stubs.
"""

from __future__ import annotations

import hashlib
import pytest
from datetime import datetime

from siren.audit.timestamp import (
    TimestampAuthority,
    TimestampAnchor,
    anchor_chain_root,
    TSA_ENDPOINTS,
    DEFAULT_TSA,
)
from siren.audit.hash_chain import GENESIS_HASH, event_hash


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

def test_tsa_endpoints_available():
    """Known TSA endpoints are configured."""
    assert "google" in TSA_ENDPOINTS
    assert "digicert" in TSA_ENDPOINTS
    assert "sectigo" in TSA_ENDPOINTS


def test_default_tsa_is_google():
    """Default TSA is Google's RFC 3161 endpoint."""
    assert DEFAULT_TSA == "google"


# --------------------------------------------------------------------------- #
# TimestampAnchor dataclass
# --------------------------------------------------------------------------- #

def test_timestamp_anchor_to_dict():
    """TimestampAnchor.to_dict truncates long tokens."""
    anchor = TimestampAnchor(
        chain_root="a" * 64,
        witness="rfc3161",
        witness_endpoint="https://timestamp.googleapis.com/",
        timestamp="2026-09-08T12:00:00Z",
        token="x" * 100,
        is_verified=True,
        is_simulated=True,
    )
    d = anchor.to_dict()
    assert d["chain_root"] == "a" * 64
    assert d["witness"] == "rfc3161"
    assert d["is_verified"] is True
    assert d["is_simulated"] is True
    # Token is truncated in the dict representation
    assert d["token"].endswith("...")
    assert len(d["token"]) < 100


# --------------------------------------------------------------------------- #
# TimestampAuthority — simulated mode (demo/offline)
# --------------------------------------------------------------------------- #

def test_authority_simulated_anchor_returns_valid_anchor():
    """Simulated anchoring produces a valid TimestampAnchor."""
    authority = TimestampAuthority(simulate=True)
    chain_root = "a" * 64
    anchor = authority.anchor(chain_root)
    assert anchor.chain_root == chain_root
    assert anchor.witness == "rfc3161"
    assert anchor.is_simulated is True
    assert anchor.is_verified is True
    assert len(anchor.token) == 64  # SHA-256 hex


def test_authority_simulated_anchor_deterministic():
    """Simulated anchors with the same chain_root produce the same token
    if the timestamp is the same."""
    authority = TimestampAuthority(simulate=True)
    chain_root = "b" * 64
    anchor1 = authority.anchor(chain_root)
    # Manually construct the same token with the same timestamp
    import hashlib
    token_material = f"simulated:{chain_root}:{anchor1.timestamp}".encode("utf-8")
    expected_token = hashlib.sha256(token_material).hexdigest()
    assert anchor1.token == expected_token


def test_authority_simulated_anchor_uses_utc_timestamp():
    """Simulated anchor timestamp is ISO 8601 UTC."""
    authority = TimestampAuthority(simulate=True)
    anchor = authority.anchor("c" * 64)
    assert anchor.timestamp.endswith("Z") or "+" in anchor.timestamp
    # Should be parseable as ISO 8601
    datetime.fromisoformat(anchor.timestamp.replace("Z", "+00:00"))


def test_authority_simulated_anchor_different_roots_different_tokens():
    """Different chain roots produce different tokens."""
    authority = TimestampAuthority(simulate=True)
    anchor1 = authority.anchor("d" * 64)
    anchor2 = authority.anchor("e" * 64)
    assert anchor1.token != anchor2.token


# --------------------------------------------------------------------------- #
# TimestampAuthority — chain root validation
# --------------------------------------------------------------------------- #

def test_authority_rejects_short_hash():
    """Anchoring rejects a hash that's too short."""
    authority = TimestampAuthority(simulate=True)
    with pytest.raises(ValueError, match="64-char hex"):
        authority.anchor("a" * 32)


def test_authority_rejects_non_hex_hash():
    """Anchoring rejects a non-hex string."""
    authority = TimestampAuthority(simulate=True)
    with pytest.raises(ValueError, match="64-char hex"):
        authority.anchor("g" * 64)  # 'g' is not hex


def test_authority_rejects_empty_hash():
    """Anchoring rejects an empty string."""
    authority = TimestampAuthority(simulate=True)
    with pytest.raises(ValueError, match="64-char hex"):
        authority.anchor("")


def test_authority_accepts_genesis_hash():
    """Anchoring accepts the genesis hash (all zeros)."""
    authority = TimestampAuthority(simulate=True)
    anchor = authority.anchor(GENESIS_HASH)
    assert anchor.chain_root == GENESIS_HASH


def test_authority_accepts_real_event_hash():
    """Anchoring accepts a real event_hash from the hash chain."""
    h = event_hash(GENESIS_HASH, "2026-09-08T12:00:00Z", "test payload")
    authority = TimestampAuthority(simulate=True)
    anchor = authority.anchor(h)
    assert anchor.chain_root == h


# --------------------------------------------------------------------------- #
# TimestampAuthority — witness types
# --------------------------------------------------------------------------- #

def test_authority_rfc3161_witness():
    """RFC 3161 witness uses the configured TSA endpoint."""
    authority = TimestampAuthority(witness="rfc3161", tsa_name="google", simulate=True)
    anchor = authority.anchor("a" * 64)
    assert anchor.witness == "rfc3161"
    assert "googleapis" in anchor.witness_endpoint


def test_authority_rekor_witness():
    """Rekor witness uses the Rekor endpoint."""
    authority = TimestampAuthority(witness="rekor", simulate=True)
    anchor = authority.anchor("a" * 64)
    assert anchor.witness == "rekor"
    assert "rekor" in anchor.witness_endpoint


def test_authority_opentimestamps_witness():
    """OpenTimestamps witness uses the OpenTimestamps endpoint."""
    authority = TimestampAuthority(witness="opentimestamps", simulate=True)
    anchor = authority.anchor("a" * 64)
    assert anchor.witness == "opentimestamps"
    assert "opentimestamps" in anchor.witness_endpoint


def test_authority_unknown_witness_raises():
    """Unknown witness type raises ValueError."""
    with pytest.raises(ValueError, match="unknown witness"):
        TimestampAuthority(witness="unknown", simulate=True)


def test_authority_unknown_tsa_raises():
    """Unknown TSA name raises ValueError."""
    with pytest.raises(ValueError, match="unknown TSA"):
        TimestampAuthority(witness="rfc3161", tsa_name="unknown", simulate=True)


# --------------------------------------------------------------------------- #
# TimestampAuthority — production stubs
# --------------------------------------------------------------------------- #

def test_authority_rfc3161_production_raises():
    """RFC 3161 production anchoring raises NotImplementedError."""
    authority = TimestampAuthority(witness="rfc3161", tsa_name="google", simulate=False)
    with pytest.raises(NotImplementedError, match="RFC 3161"):
        authority.anchor("a" * 64)


def test_authority_rekor_production_raises():
    """Rekor production anchoring raises NotImplementedError."""
    authority = TimestampAuthority(witness="rekor", simulate=False)
    with pytest.raises(NotImplementedError, match="Rekor"):
        authority.anchor("a" * 64)


def test_authority_opentimestamps_production_raises():
    """OpenTimestamps production anchoring raises NotImplementedError."""
    authority = TimestampAuthority(witness="opentimestamps", simulate=False)
    with pytest.raises(NotImplementedError, match="OpenTimestamps"):
        authority.anchor("a" * 64)


# --------------------------------------------------------------------------- #
# TimestampAuthority — anchor verification
# --------------------------------------------------------------------------- #

def test_verify_simulated_anchor_returns_true():
    """Verifying a simulated anchor returns True (deterministic token)."""
    authority = TimestampAuthority(simulate=True)
    anchor = authority.anchor("a" * 64)
    assert authority.verify_anchor(anchor) is True


def test_verify_anchor_with_invalid_chain_root_returns_false():
    """Verifying an anchor with an invalid chain root returns False."""
    authority = TimestampAuthority(simulate=True)
    anchor = TimestampAnchor(
        chain_root="invalid",
        witness="rfc3161",
        witness_endpoint="https://example.com",
        timestamp="2026-09-08T12:00:00Z",
        token="x" * 64,
        is_simulated=True,
    )
    assert authority.verify_anchor(anchor) is False


def test_verify_simulated_anchor_with_tampered_token_returns_false():
    """Verifying a simulated anchor with a tampered token returns False."""
    authority = TimestampAuthority(simulate=True)
    anchor = authority.anchor("a" * 64)
    # Tamper with the token
    tampered = TimestampAnchor(
        chain_root=anchor.chain_root,
        witness=anchor.witness,
        witness_endpoint=anchor.witness_endpoint,
        timestamp=anchor.timestamp,
        token="0" * 64,  # wrong token
        is_verified=anchor.is_verified,
        is_simulated=anchor.is_simulated,
    )
    assert authority.verify_anchor(tampered) is False


# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #

def test_anchor_chain_root_convenience_function():
    """anchor_chain_root produces a valid anchor."""
    anchor = anchor_chain_root("a" * 64, simulate=True)
    assert anchor.chain_root == "a" * 64
    assert anchor.is_simulated is True


def test_anchor_chain_root_with_rekor():
    """anchor_chain_root with rekor witness."""
    anchor = anchor_chain_root("a" * 64, witness="rekor", simulate=True)
    assert anchor.witness == "rekor"


# --------------------------------------------------------------------------- #
# Integration with the audit hash chain
# --------------------------------------------------------------------------- #

def test_anchor_real_chain_entry():
    """Anchoring a real hash chain entry produces a valid anchor."""
    # Build a small chain: genesis -> entry1 -> entry2
    h1 = event_hash(GENESIS_HASH, "2026-09-08T12:00:00Z", "run started")
    h2 = event_hash(h1, "2026-09-08T12:01:00Z", "observation processed")

    # Anchor the latest entry
    anchor = anchor_chain_root(h2, simulate=True)
    assert anchor.chain_root == h2
    assert anchor.is_verified is True

    # Verify the anchor
    authority = TimestampAuthority(simulate=True)
    assert authority.verify_anchor(anchor) is True
