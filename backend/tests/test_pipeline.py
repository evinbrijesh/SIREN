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
    # obs-003 is elevated+ under either expansion source: scripted +43%
    # (deterministic) or the neural Δp measurement (operational-primary,
    # in-scope) with rain/slope/trend keeping H in the elevated band.
    run = run_pipeline("obs-003", repo)
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


def test_in_scope_observation_not_flagged() -> None:
    """Monsoon-window observations are annotated in-scope, no scope reason."""
    repo = Repository(":memory:")
    run = run_pipeline("obs-001", repo)  # 2026-07-23 — in window
    scope = run["change_stats_json"]["operational_scope"]
    assert scope["in_scope"] is True
    assert not any(
        "Outside the certified monsoon window" in r
        for r in run["score"]["reasons"]
    )


def test_out_of_scope_observation_annotated_and_reasoned(tmp_path) -> None:
    """Outside the monsoon window: scope annotation + review reason.

    The deterministic baseline stays authoritative and the scope note
    surfaces as a review reason — never silently dropped.
    """
    import numpy as np
    import rasterio

    from siren.detect.scenario import scenario_expansion_mask

    mask_dir = tmp_path / "processed"
    mask_dir.mkdir()
    mask_path = mask_dir / "oos-001_expansion_mask.tif"
    mask, meta = scenario_expansion_mask(0.15, seed=7)
    with rasterio.open(
        str(mask_path), "w", driver="GTiff",
        height=mask.shape[0], width=mask.shape[1],
        count=1, dtype="uint8", crs="EPSG:4326",
        transform=meta["transform"],
    ) as dst:
        dst.write(mask.astype(np.uint8), 1)

    repo = Repository(":memory:")
    repo.register_observation(
        observation_id="oos-001",
        basin_id="dudh-koshi-demo-01",
        acquired_at="2026-11-15T12:00:00Z",  # outside the Jun-Sep window
        source="sentinel-1-grd-nrt",
        raster_uri=str(mask_path),
        water_area_km2=3.0,
        water_area_change_percent=10.0,
        rainfall_24h_mm=0.0,
        rainfall_7d_mm=0.0,
    )
    run = run_pipeline("oos-001", repo)

    scope = run["change_stats_json"]["operational_scope"]
    assert scope["in_scope"] is False
    assert scope["month"] == 11
    assert any(
        "Outside the certified monsoon window" in r
        for r in run["score"]["reasons"]
    )


def test_pipeline_then_review_then_dispatch(tmp_path) -> None:
    """Full DoD chain: pipeline → review → dispatch → audit."""
    repo = Repository(":memory:")
    run = run_pipeline("obs-003", repo)
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


def test_neural_expansion_primary_when_in_scope() -> None:
    """ADR-014-am1 operational-primary: in-scope runs with promoted neural
    evidence use the gated Δp expansion for water_area_change_percent and
    record the deterministic registry value as the labeled cross-check."""
    import pytest

    pytest.importorskip("torch")
    repo = Repository(":memory:")
    run = run_pipeline("obs-002", repo)
    cs = run["change_stats_json"]
    if cs.get("ml_expansion_km2") is None:
        pytest.skip("ML evidence unavailable (no checkpoint/SAR data)")
    assert cs["expansion_pct_source"] == "neural_primary"
    assert "expansion_pct_neural" in cs
    assert cs["expansion_pct_deterministic"] == 28.0
    assert run["score"]["method"] == "mixed"
    assert any("neural-primary" in r for r in run["score"]["reasons"])


def test_neural_expansion_carries_conformal_interval() -> None:
    """Level 2.4: when the promoted checkpoint ships a calibrated
    conformal sidecar, the neural expansion measurement records a 90%
    interval in change_stats; an interval reaching zero sets the
    scene-level "uncertain expansion" flag + review reason."""
    import pytest

    pytest.importorskip("torch")
    repo = Repository(":memory:")
    run = run_pipeline("obs-002", repo)
    cs = run["change_stats_json"]
    if cs.get("expansion_pct_source") != "neural_primary":
        pytest.skip("neural expansion not primary (no checkpoint/SAR data)")
    if cs.get("uncertainty_conformal_quantile") is None:
        pytest.skip("no conformal sidecar for the promoted checkpoint")
    lo, hi = cs["expansion_pct_ci90"]
    assert lo <= cs["expansion_pct_neural"] <= hi
    assert cs["expansion_trend_uncertain"] is (lo <= 0.0)
    if lo <= 0.0:
        assert any("trend uncertain" in r for r in run["score"]["reasons"])


def test_demotion_forces_deterministic_expansion(monkeypatch) -> None:
    """SIREN_ML_DEMOTE reverses the promotion at runtime — the registry
    value is load-bearing again and the demotion is not silent."""
    import pytest

    pytest.importorskip("torch")
    monkeypatch.setenv("SIREN_ML_DEMOTE", "sar_segmentation_expansion")
    repo = Repository(":memory:")
    run = run_pipeline("obs-002", repo)
    cs = run["change_stats_json"]
    if cs.get("ml_source") is None:
        pytest.skip("ML evidence unavailable (no checkpoint/SAR data)")
    assert cs["expansion_pct_source"] == "deterministic_fallback"
    assert cs["expansion_pct_demoted"] is True
    assert run["score"]["method"] == "deterministic_fallback"
    assert any("runtime-demoted" in r for r in run["score"]["reasons"])
