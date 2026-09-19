"""South Lhonak hindcast — score the real 2023 GLOF with the trained
dynamic-escalation model.

South Lhonak Lake (Sikkim, ~27.912N 88.190E) breached 2023-10-03 after a
glacier-flank collapse displaced the lake onto its moraine dam — the
worst-documented Himalayan GLOF of the decade.

This script fetches the real trailing 30-day NASA POWER window before
the event, computes the same feature vector as the training pipeline,
and scores it with the deployed checkpoint + isotonic calibrator.

Honest framing (recorded in the report):
    - South Lhonak IS in the HMAGLOFDB positive set — the final
      ``--save-model`` fit trains on all rows, so this is an
      **in-sample** check, not held-out evidence. Its out-of-fold
      behaviour was covered by the spatial CV in the eval report.
    - The actual breach trigger was a landslide-impact displacement
      wave — a process our weather/morphology features do not observe.
      A high score = "lake of this type under these conditions", not
      "we predicted the landslide".
    - POWER is ~0.5deg MERRA-2 — coarse for a steep Himalayan valley.

Usage:
    python -m siren.ml.hindcast_south_lhonak            # fetch + score
    python -m siren.ml.hindcast_south_lhonak --offline  # cached only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from siren.ml import train_susceptibility_spatial as sus
from siren.ml.dataset_dynamic_escalation import (
    CLIM_YEARS,
    MIN_POWER_DATE,
    WINDOW_DAYS,
    _fetch_series_power,
    _window_features,
)
from siren.ml.score_known_lakes import KNOWN_LAKES, _match_known_lakes

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = (
    REPO_ROOT / "models" / "checkpoints" / "south_lhonak_hindcast.json"
)
CACHE_PATH = (
    REPO_ROOT / "data" / "datasets" / "dynamic_escalation"
    / "south_lhonak_window.json"
)
BREACH_DATE = date(2023, 10, 3)
WARNING_THRESHOLD = 0.65


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--offline", action="store_true")
    p.add_argument("--output", type=Path, default=REPORT_PATH)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sl = next(k for k in KNOWN_LAKES if "lhonak" in k["name"].lower())

    # static morphometrics — ICIMOD match + RGI glacier context
    icimod = sus._load_icimod()
    matched = _match_known_lakes([sl], icimod)
    glaciers = sus._load_rgi_glaciers()
    matched = sus._glacier_features(matched, glaciers)
    r = matched.iloc[0]

    # weather window — cached after first networked fetch
    if CACHE_PATH.exists():
        series = json.loads(CACHE_PATH.read_text())
        logger.info("Using cached window %s", CACHE_PATH)
    elif args.offline:
        print("No cached window and --offline set — run networked once.")
        return 1
    else:
        start = max(BREACH_DATE - timedelta(days=WINDOW_DAYS + CLIM_YEARS * 366),
                    MIN_POWER_DATE)
        series = _fetch_series_power(r["lat"], r["lon"], start, BREACH_DATE)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(series))
        logger.info("Cached window -> %s", CACHE_PATH)

    feats = _window_features(series["daily"], BREACH_DATE,
                             series["elevation"], r["lake_elev_m"])
    if feats is None:
        print("Window features could not be computed (incomplete coverage).")
        return 1

    model_input = {
        **feats,
        "lake_elev_m": float(r["lake_elev_m"]),
        "log_lake_area_km2": float(np.log1p(r["lake_area_km2"])),
        "log_dist_glacier_m": float(np.log1p(max(r["dist_glacier_m"], 0))),
        "log_glacier_area_10km": float(
            np.log1p(r["glacier_area_10km_m2"] / 1e6)),
    }

    from siren.risk.dynamic_escalation import DynamicEscalationScorer
    scorer = DynamicEscalationScorer()
    if not scorer.load_checkpoint():
        print("No trained checkpoint — run train_dynamic_escalation "
              "--save-model first.")
        return 1
    out = scorer.predict(model_input)

    report = {
        "status": "hindcast_scored",
        "lake": "South Lhonak (Sikkim)",
        "breach_date": str(BREACH_DATE),
        "weather_window": f"{BREACH_DATE - timedelta(days=WINDOW_DAYS)} .. "
                          f"{BREACH_DATE}",
        "weather_source": "NASA POWER (MERRA-2, ~0.5deg)",
        "icimod_match_km": float(r["match_km"]),
        "features": model_input,
        "p_dynamic_raw": out["p_dynamic_raw"],
        "p_dynamic": out["p_dynamic"],
        "warning_threshold": WARNING_THRESHOLD,
        "would_warn": bool(out["p_dynamic"] >= WARNING_THRESHOLD),
        "reasons": out.get("reasons", []),
        "feature_contributions": out.get("feature_contributions", {}),
        "caveats": [
            (
                "in-sample check — South Lhonak is in the HMAGLOFDB "
                "positive set and the saved checkpoint trains on all "
                "rows; out-of-fold behaviour is in the CV eval report"
            ),
            (
                "actual trigger was a glacier-flank collapse displacement "
                "wave — not observable by weather/morphology features"
            ),
            "POWER ~0.5deg grid is coarse for a steep Himalayan valley",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("South Lhonak hindcast — 2023-10-03 breach")
    print("=" * 60)
    print(f"  p_dynamic = {out['p_dynamic']:.3f} "
          f"(raw {out['p_dynamic_raw']:.3f})")
    print(f"  threshold {WARNING_THRESHOLD} → "
          f"{'WOULD HAVE WARNED' if report['would_warn'] else 'no warning'}")
    for reason in out.get("reasons", [])[:4]:
        print(f"    {reason}")
    print("=" * 60)
    logger.info("Report written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
