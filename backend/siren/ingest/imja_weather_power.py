"""Fetch Imja's full NASA POWER daily weather series -> committed asset.

The dynamic-escalation runtime scorer must run fully offline (Hard Rule
2), so its trailing-30d feature window needs a committed weather series
covering the observation dates plus ~10 years of climatology. This
script is the ingest-side writer for that asset — the runtime reads
`data/assets/imja_power_series.json`, never the network.

POWER daily (MERRA-2, ~0.5deg x 0.625deg) covers 1981-01-01 to present.
Recorded provenance: source, fetch time, station (grid-cell) elevation.

Usage:
    python -m siren.ingest.imja_weather_power
    python -m siren.ingest.imja_weather_power --offline   # no-op check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = REPO_ROOT / "data" / "assets" / "imja_power_series.json"

IMJA_LAT, IMJA_LON = 27.8983, 86.9282
START = date(1981, 1, 1)   # POWER daily coverage start


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--offline", action="store_true",
                   help="verify the asset exists without fetching")
    p.add_argument("--output", type=Path, default=OUT_PATH)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.offline:
        if args.output.exists():
            d = json.loads(args.output.read_text())
            n = len(d.get("daily", {}).get("time", []))
            logger.info("Asset present: %s (%d daily rows)", args.output, n)
            return 0
        logger.error("Asset missing: %s — run networked once", args.output)
        return 1

    from siren.ml.dataset_dynamic_escalation import _fetch_series_power

    end = datetime.now(UTC).date()
    logger.info("Fetching POWER daily series for Imja %s..%s", START, end)
    series = _fetch_series_power(IMJA_LAT, IMJA_LON, START, end)

    payload = {
        "provenance": {
            "source": "NASA POWER daily point API (MERRA-2)",
            "endpoint": "power.larc.nasa.gov/api/temporal/daily/point",
            "resolution_deg": 0.5,
            "variables": ["PRECTOTCORR", "T2M", "T2M_MIN", "T2M_MAX"],
            "caveat": (
                "~0.5deg reanalysis grid — smooths convective extremes "
                "in steep terrain; pre-1981 unavailable"
            ),
        },
        "lake": "Imja Tsho",
        "lat": IMJA_LAT,
        "lon": IMJA_LON,
        "station_elev_m": series.get("elevation"),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "daily": series["daily"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    n = len(series["daily"].get("time", []))
    logger.info("Wrote %s (%d daily rows, %s..%s)", args.output, n,
                series["daily"]["time"][0], series["daily"]["time"][-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
