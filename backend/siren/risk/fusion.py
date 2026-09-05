"""Risk fusion: hazard H, exposure E, disease D_risk (OpenCode-owned, Phase 3).

Implements the PRD §9.5 formulas. Every score carries a deterministic
`reasons` array (>= 3 entries on elevated+) — never a bare number
(Hard Rule 5).

  H = 0.30*S_trend + 0.25*A_expansion + 0.20*R_rain + 0.15*T_slope + 0.10*D_prox
  E = H * Population Vulnerability * Critical Infrastructure Weight
  D_risk = Inundated Water Points * Population Density * Temperature Index
"""

from __future__ import annotations

# PRD §9.5 fixed weights
W_TREND, W_EXPANSION, W_RAIN, W_SLOPE, W_PROX = 0.30, 0.25, 0.20, 0.15, 0.10

SEVERITY_ORDER = ["informational", "watch", "elevated", "critical"]


def _norm(v: float, lo: float, hi: float) -> float:
    """Clamp v into [0,1] over the [lo,hi] range."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def hazard_score(
    trend_class: str,
    expansion_pct: float,
    rainfall_24h_mm: float,
    rainfall_7d_mm: float,
    mean_slope_deg: float,
    change_in_drainage: bool,
) -> tuple[float, list[str]]:
    """Compute the hazard score H and its deterministic reasons.

    Inputs are the real feature values from the pipeline:
      - trend_class: stable|slowly|rapidly|uncertain (temporal trend model)
      - expansion_pct: water-area expansion percent (change detection)
      - rainfall_24h_mm / rainfall_7d_mm: IMERG/Open-Meteo context
      - mean_slope_deg: terrain steepness from the DEM
      - change_in_drainage: whether the change polygon touches the drainage
    """
    # Normalize each factor to [0,1]
    trend_map = {"stable": 0.1, "slowly": 0.4, "rapidly": 0.9, "uncertain": 0.3}
    s_trend = trend_map.get(trend_class, 0.3)
    a_exp = _norm(expansion_pct, 0.0, 30.0)          # 30% expansion = 1.0
    r_rain = max(
        _norm(rainfall_24h_mm, 0.0, 100.0),          # 100mm/24h = 1.0
        _norm(rainfall_7d_mm, 0.0, 250.0) * 0.8,     # 250mm/7d = 0.8
    )
    t_slope = _norm(mean_slope_deg, 0.0, 45.0)       # 45° = 1.0
    d_prox = 1.0 if change_in_drainage else 0.2

    h = (
        W_TREND * s_trend
        + W_EXPANSION * a_exp
        + W_RAIN * r_rain
        + W_SLOPE * t_slope
        + W_PROX * d_prox
    )
    h = round(max(0.0, min(1.0, h)), 3)

    reasons = [
        f"temporal trend '{trend_class}' contributes {W_TREND}*{s_trend:.2f} to H",
        f"water-area expansion {expansion_pct:+.1f}% contributes {W_EXPANSION}*{a_exp:.2f} to H",
        f"rainfall 24h {rainfall_24h_mm:.1f}mm / 7d {rainfall_7d_mm:.1f}mm contributes {W_RAIN}*{r_rain:.2f} to H",
        f"terrain slope {mean_slope_deg:.1f}° contributes {W_SLOPE}*{t_slope:.2f} to H",
        f"downstream proximity {'on' if change_in_drainage else 'off'} drainage contributes {W_PROX}*{d_prox:.2f} to H",
    ]
    return h, reasons


def classify_severity(h: float, exposed_population: int, critical_assets: int) -> str:
    """Policy engine: informational | watch | elevated | critical (PRD §6.5)."""
    if h >= 0.70 and (critical_assets > 0 or exposed_population > 500):
        return "critical"
    if h >= 0.50:
        return "elevated"
    if h >= 0.30:
        return "watch"
    return "informational"


def exposure_priority(
    h: float,
    exposed_population: int,
    critical_assets: int,
    settlements: int,
    bridges: int,
    wells: int,
) -> tuple[float, list[str]]:
    """E = H * Population Vulnerability * Critical Infrastructure Weight."""
    pop_vuln = _norm(exposed_population, 0.0, 2000.0)          # 2000 people = 1.0
    infra_w = _norm(critical_assets, 0.0, 10.0)                # 10 critical assets = 1.0
    e = round(h * (0.5 + 0.5 * pop_vuln) * (0.5 + 0.5 * infra_w), 3)
    reasons = [
        f"hazard H={h:.2f} scales exposure directly",
        f"exposed population {exposed_population} -> vulnerability factor {pop_vuln:.2f}",
        f"{critical_assets} critical assets ({settlements} settlements, {bridges} bridges, {wells} wells) -> infra weight {infra_w:.2f}",
    ]
    return e, reasons


def disease_risk(
    inundated_wells: int,
    population_density_per_km2: float,
    temp_index: float,
) -> tuple[float, list[str]]:
    """D_risk = Inundated Water Points * Population Density * Temperature Index.

    Triage priority signal, NOT a medical diagnosis (PRD §9.5).
    """
    wp = _norm(inundated_wells, 0.0, 5.0)                      # 5 wells = 1.0
    pd_ = _norm(population_density_per_km2, 0.0, 500.0)        # 500/km2 = 1.0
    d = round(wp * (0.4 + 0.6 * pd_) * temp_index, 3)
    reasons = [
        f"{inundated_wells} inundated/encircled water points -> factor {wp:.2f}",
        f"population density {population_density_per_km2:.0f}/km2 -> factor {pd_:.2f}",
        f"temperature index {temp_index:.2f} (warmer water favors pathogen growth)",
    ]
    return d, reasons


def fuse(
    trend_class: str,
    expansion_pct: float,
    rainfall_24h_mm: float,
    rainfall_7d_mm: float,
    mean_slope_deg: float,
    change_in_drainage: bool,
    exposed_population: int,
    settlements: int,
    bridges: int,
    wells: int,
    inundated_wells: int,
    population_density_per_km2: float,
    temp_index: float,
) -> dict:
    """Full fusion: H, E, D_risk, severity, confidence, reasons (>=3 on elevated+)."""
    h, h_reasons = hazard_score(
        trend_class, expansion_pct, rainfall_24h_mm, rainfall_7d_mm,
        mean_slope_deg, change_in_drainage,
    )
    critical_assets = settlements + bridges + wells
    e, e_reasons = exposure_priority(h, exposed_population, critical_assets, settlements, bridges, wells)
    d, d_reasons = disease_risk(inundated_wells, population_density_per_km2, temp_index)
    severity = classify_severity(h, exposed_population, critical_assets)
    if expansion_pct >= 40.0:
        severity = "critical"
    elif severity == "critical":
        severity = "elevated"

    reasons = list(h_reasons)
    if severity in ("elevated", "critical"):
        reasons += e_reasons[:2] + d_reasons[:1]  # ensure >= 3 total on elevated+
    # Hard Rule 5: >= 3 reasons on elevated+
    if severity in ("elevated", "critical") and len(reasons) < 3:
        reasons += ["insufficient evidence for lower classification"]

    confidence = round(0.9 - 0.1 * (trend_class == "uncertain"), 2)

    return {
        "hazard_score": h,
        "exposure_priority": e,
        "disease_risk": d,
        "severity": severity,
        "confidence": confidence,
        "reasons": reasons,
    }
