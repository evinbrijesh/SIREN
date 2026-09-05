"""Tests for the Search & Rescue Priority Layer (PRD §15 stretch goal).

Pure-function tests against synthetic exposure lists — no real basin data needed.
Verifies deterministic ranking, access-loss logic, and the summary output.
"""

from __future__ import annotations

from siren.risk.sar_priority import compute_sar_priority


def test_empty_exposures() -> None:
    result = compute_sar_priority([])
    assert result["sectors"] == []
    assert result["top_priority"] is None
    assert "No sectors" in result["summary"]


def test_village_with_cut_access_ranks_highest() -> None:
    """A village behind an inundated bridge should be top priority."""
    exposures = [
        {"asset_id": "village-1", "asset_type": "village", "name": "Benkar",
         "population": 800, "distance_m": 150.0, "inundated": False},
        {"asset_id": "BR-1", "asset_type": "bridge", "name": "Hillary Bridge",
         "distance_m": 50.0, "inundated": True},
        {"asset_id": "well-1", "asset_type": "well", "name": "Well 1",
         "distance_m": 80.0, "inundated": False},
    ]
    result = compute_sar_priority(exposures)
    assert result["top_priority"] is not None
    assert result["top_priority"]["name"] == "Benkar"
    assert result["top_priority"]["access_label"] == "CUT"
    assert result["top_priority"]["sar_priority"] > 0
    # Village should rank above water points and access routes
    assert result["sectors"][0]["asset_type"] == "village"


def test_accessible_village_has_low_priority() -> None:
    """When no routes are inundated or buffered, access is ACCESSIBLE."""
    exposures = [
        {"asset_id": "village-1", "asset_type": "village", "name": "Benkar",
         "population": 500, "distance_m": None, "inundated": False},
    ]
    result = compute_sar_priority(exposures)
    assert result["top_priority"]["access_label"] == "ACCESSIBLE"
    # pop_vuln(500) = 0.25, access=0.3, urgency=1.0 → 0.075
    assert result["top_priority"]["sar_priority"] == 0.075


def test_deterministic_same_inputs_same_output() -> None:
    """Hard Rule 6: same inputs → identical outputs."""
    exposures = [
        {"asset_id": "village-1", "asset_type": "village", "name": "Benkar",
         "population": 1200, "distance_m": 200.0, "inundated": False},
        {"asset_id": "BR-1", "asset_type": "bridge", "name": "Bridge 1",
         "distance_m": 60.0, "inundated": True},
    ]
    r1 = compute_sar_priority(exposures)
    r2 = compute_sar_priority(exposures)
    assert r1 == r2


def test_wells_grouped_into_single_sector() -> None:
    exposures = [
        {"asset_id": "well-1", "asset_type": "well", "name": "Well 1",
         "distance_m": 80.0, "inundated": True},
        {"asset_id": "well-2", "asset_type": "well", "name": "Well 2",
         "distance_m": 120.0, "inundated": False},
        {"asset_id": "village-1", "asset_type": "village", "name": "Chhukung",
         "population": 1000, "distance_m": 200.0, "inundated": False},
    ]
    result = compute_sar_priority(exposures)
    well_sectors = [s for s in result["sectors"] if s["asset_type"] == "well"]
    assert len(well_sectors) == 1
    assert well_sectors[0]["name"] == "Water Points"
    assert len(well_sectors[0]["assets"]) == 2


def test_access_routes_sector_present_when_bridges_exist() -> None:
    exposures = [
        {"asset_id": "BR-1", "asset_type": "bridge", "name": "Bridge 1",
         "distance_m": 60.0, "inundated": False},
        {"asset_id": "RD-1", "asset_type": "road", "name": "Road 1",
         "distance_m": 40.0, "inundated": False},
        {"asset_id": "village-1", "asset_type": "village", "name": "Benkar",
         "population": 500, "distance_m": 200.0, "inundated": False},
    ]
    result = compute_sar_priority(exposures)
    route_sectors = [s for s in result["sectors"] if s["sector_id"] == "access-routes"]
    assert len(route_sectors) == 1
    assert len(route_sectors[0]["assets"]) == 2


def test_sectors_sorted_by_priority_descending() -> None:
    exposures = [
        {"asset_id": "village-big", "asset_type": "village", "name": "Big Village",
         "population": 1800, "distance_m": 200.0, "inundated": False},
        {"asset_id": "village-small", "asset_type": "village", "name": "Small Village",
         "population": 200, "distance_m": 200.0, "inundated": False},
        {"asset_id": "BR-1", "asset_type": "bridge", "name": "Bridge 1",
         "distance_m": 60.0, "inundated": True},
    ]
    result = compute_sar_priority(exposures)
    priorities = [s["sar_priority"] for s in result["sectors"]]
    assert priorities == sorted(priorities, reverse=True)


def test_reason_string_contains_population_and_access() -> None:
    exposures = [
        {"asset_id": "village-1", "asset_type": "village", "name": "Benkar",
         "population": 800, "distance_m": 150.0, "inundated": False},
        {"asset_id": "BR-1", "asset_type": "bridge", "name": "Bridge 1",
         "distance_m": 50.0, "inundated": True},
    ]
    result = compute_sar_priority(exposures)
    village = result["sectors"][0]
    assert "800" in village["reason"]
    assert "cut off" in village["reason"]


def test_summary_mentions_cut_sectors() -> None:
    exposures = [
        {"asset_id": "village-1", "asset_type": "village", "name": "Benkar",
         "population": 800, "distance_m": 150.0, "inundated": False},
        {"asset_id": "BR-1", "asset_type": "bridge", "name": "Bridge 1",
         "distance_m": 50.0, "inundated": True},
    ]
    result = compute_sar_priority(exposures)
    assert "cut" in result["summary"].lower()
    assert "Benkar" in result["summary"]
