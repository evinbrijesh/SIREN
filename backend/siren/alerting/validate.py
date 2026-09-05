"""Size validation for the compressed alert payload (PRD §10.4, Hard Rule 4).

The <=250-byte constraint is enforced here in code and mirrored by a CHECK
constraint on the ``dispatches`` table in ``backend/siren/db/schema.sql``
(``payload_bytes <= 250``).
"""

from __future__ import annotations


class PayloadTooLargeError(Exception):
    """Raised when a compressed payload exceeds the maximum allowed size."""


def validate_size(payload: bytes, max_bytes: int = 250) -> None:
    """Validate that ``payload`` is within the allowed byte budget.

    Args:
        payload: Encoded payload bytes.
        max_bytes: Maximum allowed size in bytes (default 250 per PRD §10.4).

    Raises:
        PayloadTooLargeError: If ``len(payload) > max_bytes``.
        TypeError: If ``payload`` is not bytes-like.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    size = len(payload)
    if size > max_bytes:
        raise PayloadTooLargeError(
            f"compressed payload is {size} bytes, exceeds limit of {max_bytes}"
        )
