"""Append-only audit log repository (PRD §7.8).

Preserves every run, model version, input snapshot, risk result, reviewer
decision, and alert action so a later user can reconstruct why an alert was
created and how it was handled.

Append-only is enforced at the database level by triggers in
``backend/siren/db/schema.sql`` (``audit_log_no_update`` and
``audit_log_no_delete``). Any UPDATE or DELETE on ``audit_log`` raises
``sqlite3.OperationalError`` with an "append-only" message.

This repository deliberately exposes ONLY ``INSERT`` and ``SELECT`` operations.
There are no ``update``, ``delete``, ``edit``, or ``modify`` methods — by
design. Do not add them.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from siren.audit.hash_chain import GENESIS_HASH, event_hash

__all__ = ["AuditLog", "utc_now_iso"]


class AuditLog:
    """Append-only audit log repository (PRD §7.8).

    Enforced append-only: only INSERT and SELECT methods exist.
    The schema triggers (schema.sql) enforce this at the DB level —
    any UPDATE or DELETE on audit_log raises ABORT.
    """

    def __init__(self, conn: sqlite3.Connection):
        """Initialize with a sqlite3 connection.

        IMPORTANT: the connection MUST have ``PRAGMA foreign_keys = ON``
        executed on it before use (SQLite quirk — see schema.sql comment).
        """
        self.conn = conn

    def append(
        self,
        actor: str,
        action: str,
        detail: dict,
        alert_id: str | None = None,
    ) -> int:
        """Append a new audit entry. Returns the entry_id.

        Args:
            actor: who performed the action (e.g. 'pipeline', 'coordinator-01')
            action: what happened (run|score|review|dispatch|reject)
            detail: full snapshot of the event (serialized to JSON)
            alert_id: optional lineage key to group entries for one alert

        Returns:
            entry_id (auto-incremented integer)

        Raises:
            sqlite3.OperationalError: if the insert violates a constraint.
        """
        if not isinstance(actor, str) or not actor:
            raise ValueError("actor must be a non-empty string")
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string")
        if not isinstance(detail, dict):
            raise ValueError("detail must be a dict (event snapshot)")

        detail_json = json.dumps(detail, sort_keys=True, separators=(",", ":"), default=str)
        created_at = utc_now_iso()
        previous_row = self.conn.execute(
            "SELECT event_hash FROM audit_log ORDER BY entry_id DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous_row[0] if previous_row and previous_row[0] else GENESIS_HASH
        digest = event_hash(previous_hash, created_at, detail_json)
        cur = self.conn.execute(
            """
            INSERT INTO audit_log
            (alert_id, actor, action, detail_json, created_at, prev_hash, event_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (alert_id, actor, action, detail_json, created_at, previous_hash, digest),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def _row_to_entry(self, row: tuple) -> dict:
        """Convert a raw DB row into an entry dict with parsed detail_json."""
        entry_id, alert_id, actor, action, detail_json, created_at, prev_hash, digest = row
        try:
            detail = json.loads(detail_json) if detail_json is not None else None
        except (json.JSONDecodeError, TypeError):
            detail = None
        return {
            "entry_id": entry_id,
            "alert_id": alert_id,
            "actor": actor,
            "action": action,
            "detail_json": detail,
            "created_at": created_at,
            "prev_hash": prev_hash,
            "event_hash": digest,
        }

    def query_by_alert(self, alert_id: str) -> list[dict]:
        """Query full lineage for an alert (all entries with matching alert_id).

        Returns:
            list of entry dicts, ordered by entry_id (chronological).
            Each dict: {entry_id, alert_id, actor, action, detail_json (parsed),
            created_at}
        """
        cur = self.conn.execute(
            """
            SELECT entry_id, alert_id, actor, action, detail_json, created_at, prev_hash, event_hash
            FROM audit_log
            WHERE alert_id = ?
            ORDER BY entry_id ASC
            """,
            (alert_id,),
        )
        return [self._row_to_entry(r) for r in cur.fetchall()]

    def query_by_action(self, action: str, limit: int = 100) -> list[dict]:
        """Query entries by action type. Returns most recent ``limit`` entries.

        Args:
            action: the action type to filter on (run|score|review|dispatch|reject)
            limit: maximum number of entries to return (most recent first)

        Returns:
            list of entry dicts ordered by entry_id DESC.
        """
        cur = self.conn.execute(
            """
            SELECT entry_id, alert_id, actor, action, detail_json, created_at, prev_hash, event_hash
            FROM audit_log
            WHERE action = ?
            ORDER BY entry_id DESC
            LIMIT ?
            """,
            (action, limit),
        )
        return [self._row_to_entry(r) for r in cur.fetchall()]

    def query_all(self, limit: int = 100) -> list[dict]:
        """Query all entries (most recent first).

        Args:
            limit: maximum number of entries to return.

        Returns:
            list of entry dicts ordered by entry_id DESC.
        """
        cur = self.conn.execute(
            """
            SELECT entry_id, alert_id, actor, action, detail_json, created_at, prev_hash, event_hash
            FROM audit_log
            ORDER BY entry_id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._row_to_entry(r) for r in cur.fetchall()]


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string with ``Z`` suffix.

    Convenience helper for callers that want to stamp a detail snapshot with
    an explicit event time distinct from the DB ``created_at``.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
