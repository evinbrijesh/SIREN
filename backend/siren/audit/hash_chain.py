from __future__ import annotations

import hashlib

GENESIS_HASH = "0" * 64


def event_hash(previous_hash: str, timestamp: str, payload: str) -> str:
    material = f"{previous_hash}{timestamp}{payload}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def verify_chain(entries: list[dict]) -> bool:
    """Verify the integrity of an audit hash chain.

    Each entry must contain 'prev_hash', 'event_hash', 'created_at', and
    'detail' (or 'detail_json') fields. Returns True if every entry's
    event_hash matches the recomputed hash from its prev_hash + timestamp +
    payload, and the chain links are unbroken.
    """
    expected_prev = GENESIS_HASH
    for entry in entries:
        prev_hash = entry.get("prev_hash", "")
        event_digest = entry.get("event_hash", "")
        timestamp = entry.get("created_at", "")
        payload = entry.get("detail", entry.get("detail_json", ""))

        # Check chain linkage
        if prev_hash != expected_prev:
            return False

        # Recompute hash
        if isinstance(payload, (dict, list)):
            import json
            payload = json.dumps(payload, sort_keys=True)

        recomputed = event_hash(prev_hash, timestamp, str(payload))
        if recomputed != event_digest:
            return False

        expected_prev = event_digest

    return True
