"""Tests for the DL-primary readiness dashboard.

The readiness report exposes which neural components are shadow-only,
advisory-primary, or operational-primary, and must not claim the system is
DL-primary ready before the gates are actually passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from siren.api.app import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(db_path=tmp_path / "siren.db"))


def test_get_ml_readiness_report_shape() -> None:
    from siren.ml.promotion import get_ml_readiness_report

    report = get_ml_readiness_report()
    summary = report.pop("_summary")

    assert "dl_primary_ready" in summary
    assert summary["dl_primary_ready"] is False
    assert summary["operational_primary_components"] == 0
    assert summary["advisory_primary_components"] >= 3
    assert summary["shadow_components"] >= 1
    assert summary["total_components"] == len(report)

    expected_components = {
        "sar_segmentation_expansion",
        "bayesian_uncertainty",
        "neural_bathymetry",
        "multi_modal_fusion",
        "latent_fno",
        "susceptibility",
        "dynamic_escalation",
        "learned_risk_fusion",
    }
    assert set(report.keys()) == expected_components

    for comp in expected_components:
        rec = report[comp]
        assert rec["status"] in {"shadow", "advisory_primary", "operational_primary"}
        assert rec["display"]
        assert "gate" in rec
        assert "blocker" in rec


def test_readiness_endpoint_returns_expected_shape(client) -> None:
    """GET /system/ml-status returns the readiness dashboard JSON."""
    response = client.get("/system/ml-status")
    assert response.status_code == 200
    data = response.json()

    assert "components" in data
    assert "summary" in data
    summary = data["summary"]
    assert summary["dl_primary_ready"] is False
    assert summary["operational_primary_components"] == 0
    assert summary["advisory_primary_components"] >= 3

    components = data["components"]
    assert "sar_segmentation_expansion" in components
    assert components["sar_segmentation_expansion"]["status"] == "advisory_primary"
    assert components["neural_bathymetry"]["status"] == "shadow"
    assert components["latent_fno"]["status"] == "shadow"
