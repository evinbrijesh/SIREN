"""Pull real rainfall + temperature context from Open-Meteo for observation dates.

Writes data/assets/weather_series.json in the shape consumed by the risk engine.

Usage:
    python -m siren.ingest.openmeteo                          # demo dates, default output
    python -m siren.ingest.openmeteo --lat 27.815 --lon 86.825 --date 2026-07-23,2026-08-04,2026-08-12 --out data/assets/weather_series.json

Offline-safe (ADR-004): if the network is unavailable, exits cleanly with a
message and does NOT write a file. The committed weather_series.json is the
runtime source; this script only refreshes it. Use --strict to get non-zero
exit codes on any failure (for scheduler use).

Provenance: values are REAL Open-Meteo historical archive data for the basin
centroid. temp_index is derived as a 0..1 normalized value from
temperature_2m_mean (see PRD §9.5 D_risk).

Bug fix (2026-09-07): the 7-day antecedent rainfall window was walking FORWARD
from obs_date instead of BACKWARD. The window is now [obs_date - 6, obs_date].
Missing precipitation is recorded as null, not 0.0.
"""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

# Basin centroid of data/assets/dudh_koshi_aoi.geojson
DEFAULT_LAT, DEFAULT_LON = 27.815, 86.825
# Demo observation dates (must match the Sentinel-1 scenes in data/raw/)
DEFAULT_DATES = ["2026-07-23", "2026-08-04", "2026-08-12"]
DEFAULT_OUT_PATH = Path("data/assets/weather_series.json")

API = "https://archive-api.open-meteo.com/v1/archive"
# 7-day window before each observation date for rainfall_7d_mm
WINDOW_DAYS = 7


def fetch_daily(start: str, end: str, lat: float, lon: float) -> dict:
    """Query Open-Meteo archive for daily precipitation + temperature."""
    url = (
        f"{API}?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily=precipitation_sum,temperature_2m_mean"
        f"&timezone=UTC"
    )
    with urlopen(url, timeout=30) as resp:
        return json.load(resp)


def temp_index(temp_mean_c: float) -> float:
    """Normalize mean temperature to a 0..1 disease-risk factor (PRD §9.5).

    Warmer water favors pathogen growth. Linear ramp: 0C -> 0.0, 20C -> 1.0.
    """
    return max(0.0, min(1.0, temp_mean_c / 20.0))


def _days_before(date_str: str, n: int) -> str:
    """Return the ISO date n days BEFORE date_str (positive n = past).

    Bug fix: the previous implementation used -offset which walked FORWARD.
    The 7-day antecedent window must look BACKWARD from the observation date.
    """
    d = date.fromisoformat(date_str)
    return (d - timedelta(days=n)).isoformat()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m siren.ingest.openmeteo",
        description="Fetch rainfall + temperature from Open-Meteo historical archive.",
    )
    p.add_argument("--lat", type=float, default=DEFAULT_LAT,
                   help=f"latitude (default: {DEFAULT_LAT})")
    p.add_argument("--lon", type=float, default=DEFAULT_LON,
                   help=f"longitude (default: {DEFAULT_LON})")
    p.add_argument("--date", default=",".join(DEFAULT_DATES),
                   help="comma-separated observation dates YYYY-MM-DD")
    p.add_argument("--out", default=str(DEFAULT_OUT_PATH),
                   help="output JSON path")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on any failure (for scheduler use)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    obs_dates = [d.strip() for d in args.date.split(",") if d.strip()]
    out_path = Path(args.out)

    try:
        # Fetch a window covering all dates + WINDOW_DAYS prior to the first
        start = _days_before(obs_dates[0], WINDOW_DAYS - 1)
        end = obs_dates[-1]
        data = fetch_daily(start, end, args.lat, args.lon)
    except (URLError, OSError) as exc:
        print(f"✗ Network unavailable or API error: {exc}", file=sys.stderr)
        print("  Keeping existing weather_series.json (offline-safe, ADR-004).", file=sys.stderr)
        return 1 if args.strict else 0
    except Exception as exc:  # noqa: BLE001 — JSON parse errors etc.
        print(f"✗ Open-Meteo API error: {exc}", file=sys.stderr)
        return 1

    daily = data.get("daily", {})
    times = daily.get("time", [])
    precip = daily.get("precipitation_sum", [])
    temps = daily.get("temperature_2m_mean", [])

    # Build a date -> (precip, temp) lookup; keep None for missing values
    # (do NOT coerce nulls to 0.0 — missing data must be distinguishable)
    lookup: dict[str, tuple[float | None, float | None]] = {}
    for t, p, tmp in zip(times, precip, temps):
        lookup[t] = (p, tmp)

    series = []
    for i, obs_date in enumerate(obs_dates, start=1):
        if obs_date not in lookup:
            print(f"✗ No data for {obs_date}", file=sys.stderr)
            return 1
        p24, tmp = lookup[obs_date]

        # rainfall_7d_mm = sum of precipitation over the 7 days ENDING on
        # obs_date: [obs_date - 6 days, obs_date]. Skip missing days.
        # Bug fix: walk BACKWARD (positive offset), not forward.
        p7 = 0.0
        p7_missing = 0
        for offset in range(WINDOW_DAYS):
            day = _days_before(obs_date, offset)  # offset 0 = obs_date, 6 = a week ago
            if day in lookup:
                day_precip = lookup[day][0]
                if day_precip is not None:
                    p7 += day_precip
                else:
                    p7_missing += 1
            else:
                p7_missing += 1

        if tmp is None:
            print(f"✗ No temperature data for {obs_date}", file=sys.stderr)
            return 1

        # Use null for missing 24h precipitation, not 0.0
        rainfall_24h = round(p24, 1) if p24 is not None else None
        # If more than half the 7-day window is missing, flag it
        rainfall_7d = round(p7, 1)
        rainfall_7d_complete = p7_missing == 0

        series.append({
            "date": obs_date,
            "observation_id": f"obs-00{i}",
            "rainfall_24h_mm": rainfall_24h,
            "rainfall_7d_mm": rainfall_7d,
            "rainfall_7d_days_missing": p7_missing,
            "rainfall_7d_complete": rainfall_7d_complete,
            "temp_mean_c": round(tmp, 1),
            "temp_index": round(temp_index(tmp), 2),
        })

    payload = {
        "basin_id": "dudh-koshi-demo-01",
        "source": "open-meteo-archive",
        "provenance": f"Open-Meteo historical archive, centroid ({args.lat},{args.lon}), UTC",
        "series": series,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"✓ Wrote {out_path} with real Open-Meteo data:")
    for s in series:
        p24_str = f"{s['rainfall_24h_mm']}mm" if s['rainfall_24h_mm'] is not None else "null"
        missing = f" ({s['rainfall_7d_days_missing']} days missing)" if not s['rainfall_7d_complete'] else ""
        print(f"  {s['date']}: 24h={p24_str} 7d={s['rainfall_7d_mm']}mm{missing} "
              f"temp={s['temp_mean_c']}C idx={s['temp_index']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
