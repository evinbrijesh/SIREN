"""Search & Rescue Priority Layer (PRD §15 stretch goal).

Ranks downstream sectors by `population × access-loss` — bridges/roads cut =
harder to reach = higher SAR priority. Reuses corridor and asset/exposure data
already computed by the pipeline. Pure deterministic function over the
exposure list; no new data ingestion required.

  SAR_priority(sector) = population_vuln × access_loss_factor × urgency_weight

Where:
  - population_vuln = _norm(population, 0, 2000)
  - access_loss_factor = 1.0 if any bridge/road is inundated, else
                         0.6 if any bridge/road is buffered (at risk),
                         else 0.3 (accessible)
  - urgency_weight = 1.0 for villages, 0.7 for wells (water points),
                     0.5 for bridges/roads themselves

Sectors are derived by grouping exposures by asset_type. Each village is its
own sector (named after the village). Wells are grouped into a single
"water-points" sector. Bridges/roads are grouped into an "access-routes"
sector.
"""

from __future__ import annotations

from typing import Any

from siren.risk.fusion import _norm

# Urgency weights by asset type (PRD §15: population is the primary driver)
URGENCY_WEIGHTS: dict[str, float] = {
    "village": 1.0,
    "well": 0.7,
    "bridge": 0.5,
    "road": 0.5,
    "clinic": 1.2,  # clinics outrank villages if present
}


def _access_loss_factor(exposures_of_type: list[dict[str, Any]]) -> float:
    """Determine access-loss factor from bridge/road exposure states.

    1.0 = at least one access route is inundated (cut)
    0.6 = at least one is buffered (at risk but passable)
    0.3 = all accessible
    """
    if not exposures_of_type:
        return 0.3
    if any(e.get("inundated") for e in exposures_of_type):
        return 1.0
    if any(e.get("distance_m") is not None for e in exposures_of_type):
        return 0.6
    return 0.3


def compute_sar_priority(exposures: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute SAR priority ranking from a run's exposure list.

    Returns a dict with:
      - sectors: list of {sector_id, name, asset_type, population,
                          access_loss, sar_priority, reason, assets}
      - top_priority: the highest-priority sector dict (or None)
      - summary: human-readable one-liner for the review card

    Deterministic: same exposures → identical output (Hard Rule 6).
    """
    # Separate access-route exposures (bridges + roads) for access-loss calc
    access_exposures = [
        e for e in exposures if e.get("asset_type") in ("bridge", "road")
    ]
    global_access_loss = _access_loss_factor(access_exposures)

    # Group exposures into sectors
    sectors: list[dict[str, Any]] = []

    # Villages: each village is its own sector
    villages = [e for e in exposures if e.get("asset_type") == "village"]
    for v in villages:
        pop = v.get("population") or 0
        pop_vuln = _norm(pop, 0.0, 2000.0)
        urgency = URGENCY_WEIGHTS["village"]
        # Village SAR priority uses global access loss (are routes to it cut?)
        sar = round(pop_vuln * global_access_loss * urgency, 3)
        name = v.get("name") or v.get("asset_id", "unknown")
        reason = _village_reason(v, pop, global_access_loss, sar)
        sectors.append({
            "sector_id": v.get("asset_id", name),
            "name": name,
            "asset_type": "village",
            "population": pop,
            "access_loss": global_access_loss,
            "access_label": _access_label(global_access_loss),
            "sar_priority": sar,
            "reason": reason,
            "assets": [v.get("asset_id", name)],
        })

    # Wells: grouped into a single "water-points" sector
    wells = [e for e in exposures if e.get("asset_type") == "well"]
    if wells:
        well_pop = sum((w.get("population") or 0) for w in wells)
        # Wells serve surrounding villages — estimate served population
        # from the nearest village population if available, else use 0
        served_pop = max(well_pop, _estimate_served_population(wells, villages))
        pop_vuln = _norm(served_pop, 0.0, 2000.0)
        urgency = URGENCY_WEIGHTS["well"]
        sar = round(pop_vuln * global_access_loss * urgency, 3)
        inundated_wells = sum(1 for w in wells if w.get("inundated"))
        reason = (
            f"{len(wells)} water points ({inundated_wells} inundated) "
            f"serve ~{served_pop} people; access {'cut' if global_access_loss == 1.0 else 'at risk'}"
        )
        sectors.append({
            "sector_id": "water-points",
            "name": "Water Points",
            "asset_type": "well",
            "population": served_pop,
            "access_loss": global_access_loss,
            "access_label": _access_label(global_access_loss),
            "sar_priority": sar,
            "reason": reason,
            "assets": [w.get("asset_id", "?") for w in wells],
        })

    # Access routes: grouped into a single "access-routes" sector
    if access_exposures:
        # Population served by these routes = sum of village populations
        # that depend on them (approximation: all villages in the corridor)
        route_pop = sum((v.get("population") or 0) for v in villages)
        pop_vuln = _norm(route_pop, 0.0, 2000.0)
        urgency = URGENCY_WEIGHTS["bridge"]
        sar = round(pop_vuln * global_access_loss * urgency, 3)
        cut_routes = [e for e in access_exposures if e.get("inundated")]
        reason = (
            f"{len(access_exposures)} access routes ({len(cut_routes)} cut) "
            f"affect {route_pop} people downstream"
        )
        sectors.append({
            "sector_id": "access-routes",
            "name": "Access Routes",
            "asset_type": "bridge",
            "population": route_pop,
            "access_loss": global_access_loss,
            "access_label": _access_label(global_access_loss),
            "sar_priority": sar,
            "reason": reason,
            "assets": [e.get("asset_id", "?") for e in access_exposures],
        })

    # Sort by SAR priority descending
    sectors.sort(key=lambda s: s["sar_priority"], reverse=True)

    top = sectors[0] if sectors else None
    summary = _summary(sectors, top)

    return {
        "sectors": sectors,
        "top_priority": top,
        "summary": summary,
    }


def _access_label(factor: float) -> str:
    if factor >= 1.0:
        return "CUT"
    if factor >= 0.6:
        return "AT_RISK"
    return "ACCESSIBLE"


def _village_reason(
    v: dict[str, Any], pop: int, access_loss: float, sar: float
) -> str:
    name = v.get("name") or v.get("asset_id", "unknown")
    status = "inundated" if v.get("inundated") else "buffered" if v.get("distance_m") is not None else "safe"
    access = "cut off" if access_loss >= 1.0 else "at risk" if access_loss >= 0.6 else "accessible"
    return (
        f"{name} ({pop} people, {status}) — access {access}, "
        f"SAR priority {sar:.2f}"
    )


def _estimate_served_population(
    wells: list[dict[str, Any]], villages: list[dict[str, Any]]
) -> int:
    """Estimate population served by wells = nearest village population per well."""
    if not villages:
        return 0
    total = 0
    for _w in wells:
        # Approximation: each well serves the largest village
        total += max((v.get("population") or 0) for v in villages)
    return total


def _summary(sectors: list[dict[str, Any]], top: dict[str, Any] | None) -> str:
    if not top:
        return "No sectors require SAR prioritization."
    cut = [s for s in sectors if s["access_loss"] >= 1.0]
    if cut:
        names = ", ".join(s["name"] for s in cut)
        return (
            f"{len(cut)} sector(s) with cut access: {names}. "
            f"Top SAR priority: {top['name']} ({top['sar_priority']:.2f})."
        )
    at_risk = [s for s in sectors if s["access_loss"] >= 0.6]
    if at_risk:
        return (
            f"{len(at_risk)} sector(s) at risk. "
            f"Top SAR priority: {top['name']} ({top['sar_priority']:.2f})."
        )
    return f"All sectors accessible. Top priority: {top['name']} ({top['sar_priority']:.2f})."
