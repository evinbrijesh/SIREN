"""Pull a daily 2 m temperature series for the Imja thermal-state gate.

Writes a committed asset JSON (default:
``data/assets/lake_thermal_series.json``) consumed at runtime by
``siren.detect.thermal_state`` — the deterministic estimator that decides
whether the monitored lake surface is LIQUID, FREEZE_TRANSITION, or
FROZEN_SURFACE for a given observation date. A SAR change mask over a
frozen lake is measuring ice, not water, so the state gate suppresses
liquid-water interpretation (drainage alarms, volume-inversion trigger)
rather than letting the model hallucinate melt beneath winter ice.

The series is fetched for the Open-Meteo grid cell containing the basin
centroid; the estimator applies an elevation lapse-rate correction to the
lake's altitude (~5010 m) since the grid cell sits far lower.

Offline-safe (ADR-004): exits cleanly without network; the committed JSON
is read at runtime, never fetched.

Usage:
    python -m siren.ingest.lake_thermal_series
    python -m siren.ingest.lake_thermal_series --start 2025-10-01 --end 2026-09-01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

# Basin centroid of data/assets/dudh_koshi_aoi.geojson (same as openmeteo.py)
DEFAULT_LAT, DEFAULT_LON = 27.815, 86.825
DEFAULT_START = "2025-10-01"
DEFAULT_END = "2026-09-01"
DEFAULT_OUT_PATH = Path("data/assets/lake_thermal_series.json")

API = "https://archive-api.open-meteo.com/v1/archive"


def fetch_daily_series(
    start: str, end: str, lat: float, lon: float
) -> dict:
    """Query Open-Meteo archive for a continuous daily temperature series.

    Returns the raw API payload; the ``elevation`` field is the grid-cell
    altitude the lapse-rate correction is applied against.
    """
    url = (
        f"{API}?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily=temperature_2m_mean,temperature_2m_min"
        f"&timezone=UTC"
    )
    with urlopen(url, timeout=60) as resp:
        return json.load(resp)


def build_series(data: dict, lat: float, lon: float) -> dict:
    """Shape the API payload into the committed asset format."""
    daily = data.get("daily", {})
    times = daily.get("time", [])
    t_mean = daily.get("temperature_2m_mean", [])
    t_min = daily.get("temperature_2m_min", [])
    if not times:
        raise ValueError("Open-Meteo returned no daily records")

    days = {}
    for t, m, mn in zip(times, t_mean, t_min):
        days[t] = {"mean_c": m, "min_c": mn}

    return {
        "provenance": (
            "Open-Meteo historical archive (ERA5-Land reanalysis), "
            "daily temperature_2m_mean/min for the Dudh Koshi basin "
            "centroid grid cell"
        ),
        "lat": lat,
        "lon": lon,
        "station_elev_m": data.get("elevation"),
        "timezone": data.get("timezone", "UTC"),
        "fetched_at": date.today().isoformat(),
        "days": days,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lat", type=float, default=DEFAULT_LAT)
    p.add_argument("--lon", type=float, default=DEFAULT_LON)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    p.add_argument("--strict", action="store_true")
    args = p.parse_args(argv)

    try:
        data = fetch_daily_series(args.start, args.end, args.lat, args.lon)
        series = build_series(data, args.lat, args.lon)
    except (URLError, OSError) as exc:
        print(f"✗ Network unavailable: {exc}", file=sys.stderr)
        return 1 if args.strict else 0
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Open-Meteo error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(series, indent=1) + "\n")
    print(
        f"✓ {len(series['days'])} days  station_elev={series['station_elev_m']} m"
        f"  → {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
