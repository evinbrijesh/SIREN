"""Deterministic lake thermal-state estimation.

A C-band SAR pulse cannot distinguish a frozen lake surface from
surrounding ice/rock — the held-out evaluation measured 0% lake recall
on the deep-winter pair for *every* model (frozen water is not liquid
water at C-band — physics, not model failure). Rather than force a
neural network to hallucinate melt beneath winter ice, the pipeline
estimates the lake's thermal state deterministically and uses it to
suppress liquid-water interpretation:

  - ``FROZEN_SURFACE``  — SAR change layers flagged unreliable; drainage
    stats marked non-hydrological; the shadow hydro/volume-inversion
    trigger is bypassed (a GLOF release from a frozen lake is implausible
    and volume-from-area inversion is meaningless).
  - ``FREEZE_TRANSITION`` — layers kept but annotated; shoulder-season
    freeze/thaw cycles can produce real backscatter changes that are not
    hydrological drainage.
  - ``LIQUID`` — normal interpretation.
  - ``UNKNOWN`` — temperature series does not cover the date; no
    suppression applied (fail open, but honestly labelled).

Estimator: trailing WINDOW_DAYS mean of ERA5-Land 2 m temperature at the
basin grid cell (5085 m — essentially lake elevation), lapse-corrected to
the lake altitude. "Consistently below 0 °C" is what freeze-up means
physically — a single cold night does not freeze a 90 m-deep lake.

Data: ``data/assets/lake_thermal_series.json`` (committed; refreshed
ingest-time by ``siren.ingest.lake_thermal_series`` — ADR-004 offline
runtime reads the file, never the network).
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
SERIES_PATH = _REPO_ROOT / "data" / "assets" / "lake_thermal_series.json"

IMJA_ELEV_M = 5010.0          # Imja Tsho surface altitude (inventory)
LAPSE_RATE_C_PER_KM = 6.5     # environmental lapse rate
WINDOW_DAYS = 7               # trailing window — freeze-up is sustained cold
MIN_COVERAGE = 0.5            # need ≥ half the window days to classify

# Classification thresholds on the lapse-corrected 7-day lake temp.
LIQUID_MIN_C = 0.5            # > +0.5 °C sustained → open water
FROZEN_MAX_C = -1.5           # < −1.5 °C sustained → frozen surface
# between the two → FREEZE_TRANSITION


class LakeThermalState(str, Enum):
    LIQUID = "liquid"
    FREEZE_TRANSITION = "freeze_transition"
    FROZEN_SURFACE = "frozen_surface"
    UNKNOWN = "unknown"


def _load_series(path: Path | str = SERIES_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def estimate_thermal_state(
    obs_date: str,
    series_path: Path | str = SERIES_PATH,
    lake_elev_m: float = IMJA_ELEV_M,
) -> dict:
    """Estimate the lake thermal state for an observation date.

    Args:
        obs_date: ISO date ``YYYY-MM-DD`` (observation acquisition date).
        series_path: committed ERA5-Land daily series JSON.
        lake_elev_m: lake surface altitude for the lapse correction.

    Returns:
        dict with keys:
          ``state`` — LakeThermalState value (str)
          ``lake_temp_c`` — lapse-corrected trailing-window mean (None if
            insufficient coverage)
          ``station_temp_c`` — raw 7-day mean at the grid cell
          ``station_elev_m`` / ``lapse_correction_c`` — correction terms
          ``window_days`` / ``coverage`` — window size and fraction of
            days present in the series
          ``method`` — provenance tag
    """
    try:
        series = _load_series(series_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("thermal series unavailable (%s) — state UNKNOWN", exc)
        return {
            "state": LakeThermalState.UNKNOWN.value,
            "lake_temp_c": None,
            "station_temp_c": None,
            "station_elev_m": None,
            "lapse_correction_c": None,
            "window_days": WINDOW_DAYS,
            "coverage": 0.0,
            "method": "era5_land_lapse_v1",
        }

    days = series.get("days", {})
    station_elev = series.get("station_elev_m")
    end = date.fromisoformat(obs_date[:10])
    temps = [
        days[(end - timedelta(days=i)).isoformat()].get("mean_c")
        for i in range(WINDOW_DAYS)
        if (end - timedelta(days=i)).isoformat() in days
        and days[(end - timedelta(days=i)).isoformat()].get("mean_c") is not None
    ]
    coverage = len(temps) / WINDOW_DAYS
    station_temp = sum(temps) / len(temps) if temps else None

    lapse_c = (
        LAPSE_RATE_C_PER_KM * (lake_elev_m - station_elev) / 1000.0
        if station_elev is not None
        else 0.0
    )
    lake_temp = station_temp - lapse_c if station_temp is not None else None

    if coverage < MIN_COVERAGE or lake_temp is None:
        state = LakeThermalState.UNKNOWN
    elif lake_temp < FROZEN_MAX_C:
        state = LakeThermalState.FROZEN_SURFACE
    elif lake_temp <= LIQUID_MIN_C:
        state = LakeThermalState.FREEZE_TRANSITION
    else:
        state = LakeThermalState.LIQUID

    return {
        "state": state.value,
        "lake_temp_c": round(lake_temp, 1) if lake_temp is not None else None,
        "station_temp_c": (
            round(station_temp, 1) if station_temp is not None else None
        ),
        "station_elev_m": station_elev,
        "lapse_correction_c": round(lapse_c, 2),
        "window_days": WINDOW_DAYS,
        "coverage": round(coverage, 2),
        "method": "era5_land_lapse_v1",
    }
