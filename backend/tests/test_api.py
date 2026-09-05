"""Smoke tests for the SIREN FastAPI app (D4).

Uses fastapi.testclient.TestClient (httpx-backed). Each test builds a fresh app
against an isolated temp SQLite file so runs are deterministic and independent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from siren.api import create_app
from siren.api.models import BasinConfig, ObservationList, RunList


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    app = create_app(db_path=tmp_path / "siren-test.db")
    return TestClient(app)


def test_basin(client: TestClient) -> None:
    resp = client.get("/basin")
    assert resp.status_code == 200
    BasinConfig.model_validate(resp.json())
    body = resp.json()
    assert body["basin_id"] == "dudh-koshi-demo-01"
    assert body["crs"] == "EPSG:4326"


def test_observations(client: TestClient) -> None:
    resp = client.get("/observations")
    assert resp.status_code == 200
    ObservationList.model_validate(resp.json())
    obs = resp.json()["observations"]
    assert len(obs) >= 1
    # newest first -> obs-003 is first
    assert obs[0]["observation_id"] == "obs-003"


def test_observation_by_id(client: TestClient) -> None:
    resp = client.get("/observations/obs-003")
    assert resp.status_code == 200
    assert resp.json()["observation_id"] == "obs-003"


def test_runs(client: TestClient) -> None:
    resp = client.get("/runs")
    assert resp.status_code == 200
    RunList.model_validate(resp.json())
    runs = resp.json()["runs"]
    assert len(runs) >= 1
    run = runs[0]
    assert run["run_id"] == "run-0001"
    assert run["score"] is not None
    assert run["score"]["severity"] == "elevated"
    # PRD §9.5: >= 3 reasons on elevated+
    assert len(run["score"]["reasons"]) >= 3


def test_dispatch_without_confirm_returns_409(client: TestClient) -> None:
    # run-0001 has no review -> dispatch must be blocked by the human gate
    # (schema trigger `dispatches_require_existing_review` aborts the insert).
    resp = client.post(
        "/runs/run-0001/dispatch",
        json={"channel": "sms", "recipient_group": "sector-b"},
    )
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "human_gate"


def test_dispatch_after_confirm_succeeds(client: TestClient) -> None:
    rev = client.post(
        "/runs/run-0001/review",
        json={"reviewer": "coordinator-01", "decision": "confirm", "note": "verified"},
    )
    assert rev.status_code == 200
    assert rev.json()["decision"] == "confirm"

    disp = client.post(
        "/runs/run-0001/dispatch",
        json={"channel": "sms", "recipient_group": "sector-b"},
    )
    assert disp.status_code == 200
    body = disp.json()
    assert body["status"] == "sent"
    assert body["payload_bytes"] <= 250  # Hard Rule 4 / PRD §10.4


def test_audit_lineage(client: TestClient) -> None:
    # confirm + dispatch, then check audit lineage is queryable by alert_id
    client.post(
        "/runs/run-0001/review",
        json={"reviewer": "coordinator-01", "decision": "confirm", "note": "ok"},
    )
    disp = client.post(
        "/runs/run-0001/dispatch",
        json={"channel": "sms", "recipient_group": "sector-b"},
    )
    assert disp.status_code == 200
    alert_id = disp.json()["alert_id"]

    audit = client.get("/audit", params={"alert_id": alert_id})
    assert audit.status_code == 200
    entries = audit.json()["entries"]
    assert len(entries) >= 1
    actions = [e["action"] for e in entries]
    assert "dispatch" in actions

    # Query by run_id returns entries whose detail_json references run-0001
    # (review + dispatch both carry run_id in detail_json). The pipeline-run
    # entries (run/score) are only present when POST /runs triggers the pipeline.
    audit_by_run = client.get("/audit", params={"run_id": "run-0001"})
    assert audit_by_run.status_code == 200
    run_entries = audit_by_run.json()["entries"]
    assert len(run_entries) >= 2
    run_actions = {e["action"] for e in run_entries}
    assert {"review", "dispatch"}.issubset(run_actions)


def test_exposures(client: TestClient) -> None:
    resp = client.get("/runs/run-0001/exposures")
    assert resp.status_code == 200
    exposures = resp.json()["exposures"]
    assert len(exposures) >= 1
    assert any(e["asset_id"] == "village-2" for e in exposures)


def test_review_reject(client: TestClient) -> None:
    resp = client.post(
        "/runs/run-0001/review",
        json={"reviewer": "coordinator-01", "decision": "reject", "note": "false alarm"},
    )
    assert resp.status_code == 200
    assert resp.json()["decision"] == "reject"

    # dispatch after reject should still 409 (no confirm review)
    disp = client.post(
        "/runs/run-0001/dispatch",
        json={"channel": "sms", "recipient_group": "sector-b"},
    )
    assert disp.status_code == 409


def test_create_run(client: TestClient) -> None:
    resp = client.post("/runs", json={"observation_id": "obs-001"})
    assert resp.status_code == 202
    body = resp.json()
    # Pipeline now runs synchronously — status is "processed", not "queued"
    assert body["status"] == "processed"
    assert body["observation_id"] == "obs-001"
    assert body["run_id"].startswith("run-")


def test_sar_priority(client: TestClient) -> None:
    """SAR Priority Layer endpoint (PRD §15 stretch goal)."""
    resp = client.get("/runs/run-0001/sar-priority")
    assert resp.status_code == 200
    body = resp.json()
    assert "sectors" in body
    assert "summary" in body
    assert len(body["sectors"]) > 0
    # Sectors should be sorted by sar_priority descending
    priorities = [s["sar_priority"] for s in body["sectors"]]
    assert priorities == sorted(priorities, reverse=True)
    # Top priority should be the first sector
    if body["top_priority"]:
        assert body["top_priority"]["sector_id"] == body["sectors"][0]["sector_id"]


def test_sar_priority_run_not_found(client: TestClient) -> None:
    resp = client.get("/runs/run-nonexistent/sar-priority")
    assert resp.status_code == 404
    assert resp.json()["error"] == "not_found"
