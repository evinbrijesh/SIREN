"""Open-Meteo historical rainfall fetcher — free, no-auth alternative to IMERG.

Fetches daily precipitation and temperature data from the Open-Meteo Archive
API (ERA5 reanalysis) for the Dudh Koshi basin. Computes 24h and 7d antecedent
rainfall for each observation date and writes the result to a weather series
JSON file. As of Sprint 1 Step 8, the pipeline reads weather from the
observations DB table (not this file at runtime); this script is an ingest-time
tool for refreshing values to wire into register_observation().

Usage:
    python -m siren.ingest.open_meteo --bbox 86.65,27.65,87.00,27.98 --out data/assets/weather_series.json

Offline-safe: if the network is unavailable, prints a message and exits 0.
The existing weather file is preserved (not overwritten) on failure.

Open-Meteo Archive API:
    - URL: https://archive-api.open-meteo.com/v1/archive
    - No authentication required
    - Data source: ERA5 reanalysis (ECMWF)
    - Variables: precipitation_sum, temperature_2m_mean
    - Resolution: daily, 0.25° grid
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Basin center for the Dudh Koshi / Imja AOI
BASIN_CENTER = (86.826, 27.888)  # (lon, lat)

# Observation dates (must match DEMO_OBSERVATIONS in pipeline.py)
OBSERVATION_DATES = {
    "obs-001": "2026-07-23",
    "obs-002": "2026-08-04",
    "obs-003": "2026-08-12",
}


def fetch_daily_precipitation(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timeout: int = 30,
) -> dict:
    """Fetch daily precipitation and temperature from Open-Meteo Archive API.

    Args:
        lat: Latitude of the basin center.
        lon: Longitude of the basin center.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        timeout: HTTP timeout in seconds.

    Returns:
        Dict with 'dates' (list of str), 'precipitation' (list of float|None),
        and 'temperature' (list of float|None).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum,temperature_2m_mean",
        "timezone": "UTC",
    }
    url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())

    daily = data.get("daily", {})
    return {
        "dates": daily.get("time", []),
        "precipitation": daily.get("precipitation_sum", []),
        "temperature": daily.get("temperature_2m_mean", []),
    }


def compute_antecedent_rainfall(
    dates: list[str],
    precipitation: list[float | None],
    target_date: str,
) -> tuple[float, float]:
    """Compute 24h and 7d antecedent rainfall for a target date.

    Args:
        dates: List of date strings (YYYY-MM-DD).
        precipitation: List of daily precipitation values (mm), aligned with dates.
        target_date: The date to compute antecedent rainfall for.

    Returns:
        (rainfall_24h, rainfall_7d) in mm.
    """
    if target_date not in dates:
        return 0.0, 0.0

    idx = dates.index(target_date)

    # 24h: precipitation on the target date
    r24 = precipitation[idx] if precipitation[idx] is not None else 0.0

    # 7d: sum of precipitation over the 7 days ending on (and including) the target date
    start_idx = max(0, idx - 6)
    r7 = sum(
        precipitation[i] if precipitation[i] is not None else 0.0
        for i in range(start_idx, idx + 1)
    )

    return float(r24), float(r7)


def compute_temp_index(temp_mean: float | None) -> float:
    """Compute temperature index for disease risk (PRD §9.5).

    Maps mean temperature to a [0, 1] index:
      - < 5°C → 0.3 (cold, low disease risk)
      - 5-15°C → 0.4-0.6 (moderate)
      - 15-25°C → 0.6-0.8 (warm, elevated)
      - > 25°C → 0.8-1.0 (hot, high disease risk)
    """
    if temp_mean is None:
        return 0.5  # neutral default
    if temp_mean < 5:
        return 0.3
    elif temp_mean < 15:
        return 0.4 + (temp_mean - 5) * 0.02
    elif temp_mean < 25:
        return 0.6 + (temp_mean - 15) * 0.02
    else:
        return min(1.0, 0.8 + (temp_mean - 25) * 0.01)


def build_weather_series(
    lat: float = BASIN_CENTER[1],
    lon: float = BASIN_CENTER[0],
    observation_dates: dict[str, str] | None = None,
) -> dict:
    """Fetch real rainfall data and build the weather series dict.

    Returns a dict in the same format as data/assets/weather_series.json:
        {
          "basin_id": "dudh-koshi-demo-01",
          "source": "open-meteo-era5",
          "series": [
            {
              "date": "2026-07-23",
              "observation_id": "obs-001",
              "rainfall_24h_mm": 3.2,
              "rainfall_7d_mm": 58.8,
              "temp_mean_c": 8.5,
              "temp_index": 0.47
            },
            ...
          ]
        }
    """
    if observation_dates is None:
        observation_dates = OBSERVATION_DATES

    # Fetch data covering 7 days before the earliest observation through the latest
    all_dates = list(observation_dates.values())
    earliest = min(all_dates)
    latest = max(all_dates)
    # Go back 7 days before earliest to compute 7d antecedent
    start = (date.fromisoformat(earliest) - timedelta(days=7)).isoformat()
    end = latest

    data = fetch_daily_precipitation(lat, lon, start, end)

    series = []
    for obs_id, obs_date in sorted(observation_dates.items(), key=lambda x: x[1]):
        r24, r7 = compute_antecedent_rainfall(
            data["dates"], data["precipitation"], obs_date
        )
        if obs_date in data["dates"]:
            idx = data["dates"].index(obs_date)
            temp_mean = data["temperature"][idx]
        else:
            temp_mean = None
        temp_idx = compute_temp_index(temp_mean)

        series.append({
            "date": obs_date,
            "observation_id": obs_id,
            "rainfall_24h_mm": round(r24, 1),
            "rainfall_7d_mm": round(r7, 1),
            "temp_mean_c": round(temp_mean, 1) if temp_mean is not None else 12.0,
            "temp_index": round(temp_idx, 2),
        })

    return {
        "basin_id": "dudh-koshi-demo-01",
        "source": "open-meteo-era5",
        "series": series,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m siren.ingest.open_meteo",
        description="Fetch real rainfall data from Open-Meteo Archive API (ERA5).",
    )
    parser.add_argument(
        "--out", default="data/assets/weather_series.json",
        help="output path (default: data/assets/weather_series.json)",
    )
    parser.add_argument("--lat", type=float, default=BASIN_CENTER[1])
    parser.add_argument("--lon", type=float, default=BASIN_CENTER[0])
    args = parser.parse_args(argv)

    out_path = Path(args.out)

    try:
        weather = build_weather_series(lat=args.lat, lon=args.lon)
    except (urllib.error.URLError, OSError) as exc:
        print(f"✗ Network unavailable or Open-Meteo error: {exc}", file=sys.stderr)
        print("  Offline-safe exit (existing weather_series.json preserved).", file=sys.stderr)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(weather, indent=2))
    print(f"✓ Wrote real ERA5 rainfall data to {out_path}")
    for entry in weather["series"]:
        print(f"  {entry['observation_id']} ({entry['date']}): "
              f"24h={entry['rainfall_24h_mm']}mm, 7d={entry['rainfall_7d_mm']}mm, "
              f"temp={entry['temp_mean_c']}°C")
    return 0


if __name__ == "__main__":
    sys.exit(main())
