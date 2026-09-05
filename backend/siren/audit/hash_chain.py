from __future__ import annotations

import hashlib

GENESIS_HASH = "0" * 64


def event_hash(previous_hash: str, timestamp: str, payload: str) -> str:
    material = f"{previous_hash}{timestamp}{payload}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()
