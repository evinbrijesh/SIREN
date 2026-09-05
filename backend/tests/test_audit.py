"""Tests for the append-only audit log repository (D6 / PRD §7.8).

These tests execute the authoritative schema.sql so the append-only triggers
are present, then exercise the AuditLog repository's INSERT + SELECT surface
and verify that UPDATE/DELETE are blocked at the DB level.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from siren.audit.writer import AuditLog

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "siren" / "db" / "schema.sql"


@pytest.fixture()
def conn():
    """In-memory SQLite with the authoritative schema + triggers loaded."""
    with open(SCHEMA_PATH) as f:
        schema = f.read()
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    c.executescript(schema)
    yield c
    c.close()


@pytest.fixture()
def audit(conn):
    return AuditLog(conn)


# ---------------------------------------------------------------------------
# Append + query
# ---------------------------------------------------------------------------
def test_append_and_query_by_alert_chronological(audit):
    """Append 3 entries for alert-0091; query_by_alert returns all 3 in order."""
    e1 = audit.append("pipeline", "run", {"run_id": "run-0007"}, alert_id="alert-0091")
    e2 = audit.append("coordinator-01", "review", {"decision": "confirm"}, alert_id="alert-0091")
    e3 = audit.append("coordinator-01", "dispatch", {"channel": "sms"}, alert_id="alert-0091")

    entries = audit.query_by_alert("alert-0091")
    assert len(entries) == 3
    # chronological (ascending entry_id)
    assert [e["entry_id"] for e in entries] == [e1, e2, e3]
    assert [e["action"] for e in entries] == ["run", "review", "dispatch"]
    assert [e["actor"] for e in entries] == ["pipeline", "coordinator-01", "coordinator-01"]
    # every entry carries the lineage key
    assert all(e["alert_id"] == "alert-0091" for e in entries)


# ---------------------------------------------------------------------------
# Append-only enforcement (schema triggers)
# ---------------------------------------------------------------------------
def test_update_is_aborted_by_trigger(conn, audit):
    """Direct UPDATE on audit_log must raise ABORT (trigger fires)."""
    audit.append("pipeline", "run", {"x": 1}, alert_id="alert-0091")
    # SQLite RAISE(ABORT, ...) maps to IntegrityError in Python's sqlite3.
    with pytest.raises((sqlite3.OperationalError, sqlite3.IntegrityError)) as exc:
        conn.execute("UPDATE audit_log SET actor='hacker' WHERE entry_id=1")
    msg = str(exc.value)
    assert "append-only" in msg or "UPDATE forbidden" in msg, f"unexpected msg: {msg}"


def test_delete_is_aborted_by_trigger(conn, audit):
    """Direct DELETE on audit_log must raise ABORT (trigger fires)."""
    audit.append("pipeline", "run", {"x": 1}, alert_id="alert-0091")
    # SQLite RAISE(ABORT, ...) maps to IntegrityError in Python's sqlite3.
    with pytest.raises((sqlite3.OperationalError, sqlite3.IntegrityError)) as exc:
        conn.execute("DELETE FROM audit_log WHERE entry_id=1")
    msg = str(exc.value)
    assert "append-only" in msg or "DELETE forbidden" in msg, f"unexpected msg: {msg}"


# ---------------------------------------------------------------------------
# No update/delete methods on the repository (design guarantee)
# ---------------------------------------------------------------------------
def test_no_mutation_methods_exist():
    """AuditLog must not expose update/delete/edit/modify/patch methods."""
    forbidden = {"update", "delete", "edit", "modify", "patch", "remove", "set", "put"}
    members = {name for name in dir(AuditLog) if not name.startswith("_")}
    offenders = members & forbidden
    assert not offenders, f"AuditLog must not expose mutation methods: {offenders}"
    # positive: the allowed surface is present
    assert hasattr(AuditLog, "append")
    assert hasattr(AuditLog, "query_by_alert")
    assert hasattr(AuditLog, "query_by_action")
    assert hasattr(AuditLog, "query_all")


# ---------------------------------------------------------------------------
# Lineage reconstruction across multiple alerts
# ---------------------------------------------------------------------------
def test_lineage_grouping_by_alert(audit):
    """Two alerts; querying each returns only its own entries."""
    audit.append("pipeline", "run", {"run_id": "run-A"}, alert_id="alert-A")
    audit.append("coordinator-01", "review", {"decision": "confirm"}, alert_id="alert-A")
    audit.append("pipeline", "run", {"run_id": "run-B"}, alert_id="alert-B")
    audit.append("coordinator-02", "reject", {"reason": "cloud"}, alert_id="alert-B")
    audit.append("coordinator-02", "dispatch", {"channel": "sms"}, alert_id="alert-B")

    a = audit.query_by_alert("alert-A")
    b = audit.query_by_alert("alert-B")
    assert len(a) == 2
    assert len(b) == 3
    assert {e["alert_id"] for e in a} == {"alert-A"}
    assert {e["alert_id"] for e in b} == {"alert-B"}
    assert [e["action"] for e in b] == ["run", "reject", "dispatch"]


# ---------------------------------------------------------------------------
# Detail JSON round-trip
# ---------------------------------------------------------------------------
def test_detail_json_roundtrip_complex(audit):
    """A complex nested detail dict survives a serialize->parse round-trip."""
    detail = {
        "run_id": "run-0007",
        "observation_id": "obs-003",
        "scores": {"hazard": 0.72, "exposure": 0.55, "disease_risk": 0.31},
        "reasons": ["water expansion +18%", "rainfall 7d 142mm", "bridge BR-12 inundated"],
        "nested": {"a": [1, 2, 3], "b": {"c": True, "d": None}},
        "flag": True,
        "nothing": None,
    }
    eid = audit.append("pipeline", "score", detail, alert_id="alert-0091")
    entries = audit.query_by_alert("alert-0091")
    assert len(entries) == 1
    got = entries[0]
    assert got["entry_id"] == eid
    assert got["detail_json"] == detail


# ---------------------------------------------------------------------------
# query_by_action / query_all ordering
# ---------------------------------------------------------------------------
def test_query_by_action_most_recent_first(audit):
    audit.append("pipeline", "run", {"i": 1}, alert_id="alert-A")
    audit.append("pipeline", "run", {"i": 2}, alert_id="alert-B")
    audit.append("coordinator-01", "review", {"d": "confirm"}, alert_id="alert-A")
    audit.append("pipeline", "run", {"i": 3}, alert_id="alert-C")

    runs = audit.query_by_action("run")
    assert [e["detail_json"]["i"] for e in runs] == [3, 2, 1]  # DESC
    assert all(e["action"] == "run" for e in runs)


def test_query_all_most_recent_first(audit):
    audit.append("pipeline", "run", {"i": 1}, alert_id="alert-A")
    audit.append("coordinator-01", "review", {"i": 2}, alert_id="alert-A")
    audit.append("coordinator-01", "dispatch", {"i": 3}, alert_id="alert-A")

    all_entries = audit.query_all()
    assert [e["entry_id"] for e in all_entries] == [3, 2, 1]


def test_query_by_alert_empty_for_unknown(audit):
    assert audit.query_by_alert("does-not-exist") == []


def test_append_without_alert_id(audit):
    """alert_id is nullable — entries with no lineage key still store."""
    eid = audit.append("pipeline", "run", {"k": "v"})
    entries = audit.query_all()
    assert len(entries) == 1
    assert entries[0]["entry_id"] == eid
    assert entries[0]["alert_id"] is None


def test_created_at_is_utc_iso8601(audit):
    """created_at defaults to a UTC ISO-8601 timestamp from the schema."""
    audit.append("pipeline", "run", {"x": 1}, alert_id="alert-0091")
    [entry] = audit.query_by_alert("alert-0091")
    ts = entry["created_at"]
    # strftime('%Y-%m-%dT%H:%M:%SZ','now') shape
    assert ts.endswith("Z")
    assert len(ts) == 20  # YYYY-MM-DDTHH:MM:SSZ
    assert ts[10] == "T" and ts[19] == "Z"
