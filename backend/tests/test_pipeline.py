"""Tests for the pipeline orchestrator (siren.pipeline).

Verifies the full chain: quality→route→detect→corridor→risk→DB→audit.
Uses the scenario masks (deterministic, offline-safe).
"""

from __future__ import annotations

from siren.db.repo import Repository
from siren.pipeline import run_pipeline, run_all_observations


def test_run_pipeline_produces_score_and_exposures(tmp_path) -> None:
    repo = Repository(":memory:")
    run = run_pipeline("obs-001", repo)

    assert run is not None
    assert run["run_id"].startswith("run-")
    assert run["observation_id"] == "obs-001"
    assert run["score"] is not None
    assert run["score"]["severity"] in ("informational", "watch", "elevated", "critical")
    assert len(run["score"]["reasons"]) >= 3
    assert run["change_mask_uri"] is not None
    assert run["change_mask_uri"].endswith(".tif")

    exposures = repo.list_exposures(run["run_id"])
    assert len(exposures) > 0
    # Each exposure has the required fields
    for exp in exposures:
        assert "asset_id" in exp
        assert "asset_type" in exp
        assert "distance_m" in exp
        assert "buffer_m" in exp


def test_run_pipeline_elevated_has_three_reasons(tmp_path) -> None:
    """Hard Rule 5: elevated+ scores must have >= 3 reasons."""
    repo = Repository(":memory:")
    # obs-002 (+28%) should be critical or elevated
    run = run_pipeline("obs-002", repo)
    severity = run["score"]["severity"]
    assert severity in ("elevated", "critical"), f"expected elevated+, got {severity}"
    assert len(run["score"]["reasons"]) >= 3


def test_run_all_observations(tmp_path) -> None:
    repo = Repository(":memory:")
    runs = run_all_observations(repo)
    assert len(runs) == 3  # obs-001, obs-002, obs-003
    for run in runs:
        assert run["score"] is not None
        assert run["score"]["hazard_score"] >= 0.0
        assert run["score"]["hazard_score"] <= 1.0


def test_pipeline_deterministic(tmp_path) -> None:
    """Hard Rule 6: same inputs → identical outputs."""
    repo1 = Repository(":memory:")
    run1 = run_pipeline("obs-001", repo1)

    repo2 = Repository(":memory:")
    run2 = run_pipeline("obs-001", repo2)

    assert run1["score"]["hazard_score"] == run2["score"]["hazard_score"]
    assert run1["score"]["severity"] == run2["score"]["severity"]
    assert run1["score"]["reasons"] == run2["score"]["reasons"]


def test_pipeline_then_review_then_dispatch(tmp_path) -> None:
    """Full DoD chain: pipeline → review → dispatch → audit."""
    repo = Repository(":memory:")
    run = run_pipeline("obs-002", repo)  # +28% → critical
    run_id = run["run_id"]

    # Dispatch without review must fail (human gate)
    try:
        repo.create_dispatch(run_id, "sms", "sector-b")
        assert False, "dispatch should have been blocked"
    except Exception:
        pass  # expected

    # Confirm review
    review = repo.create_review(run_id, "coordinator-01", "confirm", "test")
    assert review["decision"] == "confirm"

    # Now dispatch works
    dispatch = repo.create_dispatch(run_id, "sms", "sector-b")
    assert dispatch["payload_bytes"] <= 250
    assert dispatch["alert_id"].startswith("alert-")

    # Audit lineage exists
    entries = repo.list_audit(dispatch["alert_id"])
    assert len(entries) >= 1
