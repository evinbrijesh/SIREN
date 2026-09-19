"""Tier-2 dynamic escalation dataset builder — antecedent weather windows.

For each date-resolved HMAGLOFDB GLOF event, pulls a trailing 30-day
hydrometeorological window (plus a multi-year climatology for anomaly
features) from the Open-Meteo archive (ERA5/ERA5-Land reanalysis) and
engineers physically motivated antecedent-trigger features. Matched
negative windows are sampled from stable ICIMOD lakes with pseudo-dates
drawn from the empirical (year, month) distribution of the positives —
same seasons, same era, no spatial block bias.

Why this is leakage-free (unlike the disqualified susceptibility path):
every feature is a measured reanalysis quantity or a static morphometric
derived identically for both classes; nothing is conditioned on the label.

Data sources:
    Positives: HMAGLOFDB rows with Year_exact+Month+Day resolved AND
        event_date >= MIN_ARCHIVE_DATE (reanalysis coverage).
    Negatives: ICIMOD lakes >=1 km from every breach centroid (same
        exclusion as train_susceptibility_spatial), stratified by the
        spatial blocks defined by breach geography.
    Weather:   Open-Meteo archive API — daily precipitation_sum +
        temperature_2m_mean/min/max. NOTE: this is ERA5 reanalysis
        precipitation, not IMERG satellite precipitation — a documented
        substitution (IMERG requires Earthdata auth; ERA5 is a
        defensible reanalysis driver for moraine saturation/melt).

Feature set (per sample, window = [T0-30d, T0]):

    Dynamic (meteorological, lapse-corrected to lake elevation):
        precip_30d_mm         total precipitation in window
        precip_7d_mm          last-7-day precipitation
        max_daily_precip_mm   single-day maximum (cloudburst proxy)
        heavy_rain_days       days with >25 mm
        api_30                antecedent precipitation index at T0,
                              API_t = P_t + 0.9 * API_{t-1}
        rain_anom_30d_mm      precip_30d minus same-calendar-window mean
                              over the preceding CLIM_YEARS years
        mdd_30                melt-degree days: sum max(0, T_lake)
        mdd_anom_30           MDD anomaly vs the same climatology
        ft_cycles_14          freeze-thaw days (Tmin<0<Tmax, lapse-
                              corrected) in the trailing 14 days
    Static (identical derivation as train_susceptibility_spatial):
        lake_elev_m, log_lake_area_km2, log_dist_glacier_m,
        log_glacier_area_10km

Design:
    Stage 1 (network): fetch one continuous daily series per sample
        [T0 - 30d - CLIM_YEARS*366d, T0], cached to
        data/datasets/dynamic_escalation/cache/*.json — resumable.
    Stage 2 (offline): feature extraction reads only the cache.
    --offline skips stage 1 entirely (builds from whatever is cached).

Usage:
    python -m siren.ml.dataset_dynamic_escalation --offline --limit 5
    python -m siren.ml.dataset_dynamic_escalation --n-negatives 800
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from siren.detect.thermal_state import LAPSE_RATE_C_PER_KM
from siren.ml import train_susceptibility_spatial as sus

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
HMAGLOFDB_CSV = sus.HMAGLOFDB_CSV
CACHE_DIR = REPO_ROOT / "data" / "datasets" / "dynamic_escalation" / "cache"
DEFAULT_OUT = REPO_ROOT / "data" / "datasets" / "dynamic_escalation_train.parquet"
REPORT_PATH = REPO_ROOT / "models" / "checkpoints" / "dynamic_escalation_dataset_report.json"

API = "https://archive-api.open-meteo.com/v1/archive"
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
MIN_ARCHIVE_DATE = date(1940, 2, 1)   # archive starts 1940-01; +31d window
MIN_POWER_DATE = date(1981, 1, 1)     # NASA POWER daily coverage start
WINDOW_DAYS = 30
FT_WINDOW_DAYS = 14
CLIM_YEARS = 10                       # climatology depth for anomalies
API_DECAY = 0.9                       # antecedent precipitation index decay
HEAVY_RAIN_MM = 25.0
REQUEST_DELAY_S = 0.4
MAX_RETRIES = 3

DYNAMIC_FEATURES = [
    "precip_30d_mm", "precip_7d_mm", "max_daily_precip_mm",
    "heavy_rain_days", "api_30", "rain_anom_30d_mm",
    "mdd_30", "mdd_anom_30", "ft_cycles_14",
]
STATIC_FEATURES = [
    "lake_elev_m", "log_lake_area_km2", "log_dist_glacier_m",
    "log_glacier_area_10km",
]
FEATURE_NAMES = DYNAMIC_FEATURES + STATIC_FEATURES


# --------------------------------------------------------------------------- #
# Sample frame
# --------------------------------------------------------------------------- #

def _assign_lake_ids(df: pd.DataFrame, radius_m: float = 1_000.0) -> np.ndarray:
    """Greedy spherical clustering — repeat events of one lake share an id."""
    coords = np.radians(df[["lat", "lon"]].values)
    R = 6_371_000.0
    labels = np.full(len(df), -1, dtype=int)
    centres: list[np.ndarray] = []
    for i, (lat_r, lon_r) in enumerate(coords):
        p = np.array([
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r),
        ])
        for cid, c in enumerate(centres):
            if 2 * R * np.arcsin(min(np.linalg.norm(p - c) / 2, 1.0)) <= radius_m:
                labels[i] = cid
                break
        if labels[i] < 0:
            labels[i] = len(centres)
            centres.append(p)
    return labels


def _load_dated_events() -> pd.DataFrame:
    """HMAGLOFDB events with resolved dates inside the archive era."""
    df = pd.read_csv(str(HMAGLOFDB_CSV), encoding="latin-1")
    for c in ("Lat_lake", "Lon_lake", "Elev_lake", "Area",
              "Year_exact", "Month", "Day"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["Lat_lake"].notna() & df["Lon_lake"].notna()]
    df = df[df["Year_exact"].notna() & df["Month"].notna() & df["Day"].notna()]
    df = df.rename(columns={"Lat_lake": "lat", "Lon_lake": "lon"})
    df["event_date"] = pd.to_datetime(
        {"year": df["Year_exact"].astype(int),
         "month": df["Month"].astype(int),
         "day": df["Day"].astype(int)},
        errors="coerce",
    )
    df = df[df["event_date"].notna()]
    df = df[df["event_date"].dt.date >= MIN_ARCHIVE_DATE].copy()
    df["lake_id"] = _assign_lake_ids(df)
    df["hmaglofdb_area_km2"] = df["Area"] / 1e6
    df.loc[df["hmaglofdb_area_km2"] <= 0, "hmaglofdb_area_km2"] = np.nan
    df["breached"] = 1
    logger.info("Dated archive-era events: %d rows across %d unique lakes",
                len(df), df["lake_id"].nunique())
    return df.reset_index(drop=True)


def _sample_negative_windows(
    pos: pd.DataFrame, icimod, kmeans, n_negatives: int, seed: int,
) -> pd.DataFrame:
    """Stable lakes + pseudo-event dates matched to positive seasonality."""
    neg = sus._sample_negatives(pos, icimod, kmeans, n_negatives, seed)
    rng = np.random.default_rng(seed + 1)
    # Draw (year, month, day) jointly from the empirical positive
    # distribution — preserves both seasonal and era coverage.
    pos_dates = pos["event_date"].dt.date.values
    neg["event_date"] = pd.to_datetime(
        rng.choice(pos_dates, size=len(neg), replace=True)
    )
    neg["lake_id"] = -1 - np.arange(len(neg))   # negative lake ids distinct
    neg["lake_area_km2"] = neg["icimod_area_km2"]
    return neg


# --------------------------------------------------------------------------- #
# Stage 1 — fetch (network; cached, resumable)
# --------------------------------------------------------------------------- #

def _cache_path(sample_id: str, source: str = "openmeteo") -> Path:
    """Cache file for a sample. Legacy flat layout doubles as the
    openmeteo dir; power caches live in cache/power/ so a source switch
    never silently reuses another product's data."""
    if source == "power":
        return CACHE_DIR / "power" / f"{sample_id}.json"
    return CACHE_DIR / f"{sample_id}.json"


def _fetch_series(lat: float, lon: float, start: date, end: date) -> dict:
    url = (
        f"{API}?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily=precipitation_sum,temperature_2m_mean,"
        f"temperature_2m_min,temperature_2m_max"
        f"&timezone=UTC"
    )
    with urlopen(url, timeout=60) as resp:
        return json.load(resp)


def _fetch_series_power(lat: float, lon: float,
                        start: date, end: date) -> dict:
    """NASA POWER daily point query, normalised to the Open-Meteo schema.

    POWER (MERRA-2-based, ~0.5°x0.625°) has no API key and a far more
    generous rate limit — the practical source for the ~1,100-sample
    corpus, since Open-Meteo weights ~10-year windows at ~500 calls each.
    Returns {"daily": {...open-meteo keys...}, "elevation": ...}.
    """
    start = max(start, MIN_POWER_DATE)
    url = (
        f"{POWER_API}?parameters=PRECTOTCORR,T2M,T2M_MIN,T2M_MAX"
        f"&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start:%Y%m%d}&end={end:%Y%m%d}"
        f"&format=JSON&time-standard=UTC"
    )
    with urlopen(url, timeout=120) as resp:
        raw = json.load(resp)

    par = raw.get("properties", {}).get("parameter", {})
    if not par:
        raise ValueError("empty POWER response")

    times = [
        f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in sorted(par.get(
            "PRECTOTCORR", {}).keys())
    ]
    ordered = sorted(par.get("PRECTOTCORR", {}).keys())
    daily = {
        "time": times,
        "precipitation_sum": [
            (None if par["PRECTOTCORR"][d] <= -900
             else float(par["PRECTOTCORR"][d])) for d in ordered],
        "temperature_2m_mean": [
            (None if par["T2M"][d] <= -900
             else float(par["T2M"][d])) for d in ordered],
        "temperature_2m_min": [
            (None if par["T2M_MIN"][d] <= -900
             else float(par["T2M_MIN"][d])) for d in ordered],
        "temperature_2m_max": [
            (None if par["T2M_MAX"][d] <= -900
             else float(par["T2M_MAX"][d])) for d in ordered],
    }
    coords = raw.get("geometry", {}).get("coordinates", [])
    elev = float(coords[2]) if len(coords) > 2 else None
    return {"daily": daily, "elevation": elev}


RATE_LIMIT_WAIT_S = 65.0  # Open-Meteo minutely limit — wait a full window


def _fetch_with_retry(lat: float, lon: float, start: date, end: date,
                      delay_s: float,
                      source: str = "openmeteo") -> tuple[dict | None, str]:
    """Fetch with 429-aware retry. Returns (data, error_reason)."""
    fetcher = _fetch_series_power if source == "power" else _fetch_series
    last_err = "unknown"
    for attempt in range(MAX_RETRIES):
        try:
            return fetcher(lat, lon, start, end), ""
        except HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            if exc.code == 429:
                logger.info("rate limited — waiting %.0fs", RATE_LIMIT_WAIT_S)
                time.sleep(RATE_LIMIT_WAIT_S)
            else:
                time.sleep(delay_s * (2 ** attempt + 1))
        except (URLError, OSError, ValueError) as exc:
            last_err = str(exc.reason if hasattr(exc, "reason") else exc)
            time.sleep(delay_s * (2 ** attempt + 1))
    return None, last_err


def fetch_missing(
    samples: pd.DataFrame, delay_s: float = REQUEST_DELAY_S,
    offline: bool = False, source: str = "openmeteo",
) -> dict:
    """Fetch per-sample daily series into the cache. Returns stats."""
    stats = {"cached": 0, "fetched": 0, "failed": 0, "skipped_offline": 0,
             "no_coverage_pre1981": 0}
    for _, row in samples.iterrows():
        sid = row["sample_id"]
        end = row["event_date"].date()
        if _cache_path(sid, source).exists():
            stats["cached"] += 1
            continue
        # POWER daily coverage starts 1981 — pre-1981 events are
        # unreachable there; reuse any legacy openmeteo cache, else skip
        # (no retries — the request can never succeed).
        if source == "power" and end < MIN_POWER_DATE:
            if _cache_path(sid).exists():
                stats["cached"] += 1
            else:
                stats["no_coverage_pre1981"] += 1
            continue
        if offline:
            stats["skipped_offline"] += 1
            continue

        start = max(
            end - timedelta(days=WINDOW_DAYS + CLIM_YEARS * 366),
            MIN_POWER_DATE if source == "power" else date(1940, 1, 1),
        )
        data, err = _fetch_with_retry(row["lat"], row["lon"], start, end,
                                      delay_s, source)
        if data is None:
            stats["failed"] += 1
            logger.warning("fetch failed: %s (%s)", sid, err)
            continue

        cache = _cache_path(sid, source)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({
            "sample_id": sid, "lat": row["lat"], "lon": row["lon"],
            "weather_source": source,
            "station_elev_m": data.get("elevation"),
            "daily": data.get("daily", {}),
        }))
        stats["fetched"] += 1
        if stats["fetched"] % 50 == 0:
            logger.info("fetched %d series...", stats["fetched"])
        time.sleep(delay_s)
    return stats


# --------------------------------------------------------------------------- #
# Stage 2 — features (offline)
# --------------------------------------------------------------------------- #

def _window_features(daily: dict, end: date, station_elev: float | None,
                     lake_elev: float) -> dict | None:
    """Compute antecedent features for window [end-30d, end].

    Temperatures are lapse-corrected to the lake elevation (same
    convention as detect/thermal_state.py). Returns None if coverage of
    the 30-day window is incomplete.
    """
    days = {}
    for i, t in enumerate(daily.get("time", [])):
        days[t] = i

    lapse_c = (
        LAPSE_RATE_C_PER_KM * (lake_elev - station_elev) / 1000.0
        if station_elev is not None else 0.0
    )

    def series_for(d0: date, d1: date, key: str):
        vals, n = [], (d1 - d0).days + 1
        for i in range(n):
            ds = (d0 + timedelta(days=i)).isoformat()
            v = daily.get(key, [None] * len(daily.get("time", [])))[
                days[ds]] if ds in days else None
            vals.append(v)
        return vals

    w_start = end - timedelta(days=WINDOW_DAYS - 1)
    precip = series_for(w_start, end, "precipitation_sum")
    tmean = series_for(w_start, end, "temperature_2m_mean")
    if sum(v is None for v in precip + tmean) > 4:
        return None
    precip = [v if v is not None else 0.0 for v in precip]
    t_lake = [(v - lapse_c) if v is not None else np.nan for v in tmean]

    ft_start = end - timedelta(days=FT_WINDOW_DAYS - 1)
    tmin = series_for(ft_start, end, "temperature_2m_min")
    tmax = series_for(ft_start, end, "temperature_2m_max")
    ft = sum(
        1 for a, b in zip(tmin, tmax)
        if a is not None and b is not None
        and (a - lapse_c) < 0 < (b - lapse_c)
    )

    api = 0.0
    for p in precip:
        api = p + API_DECAY * api

    # Climatology: same calendar window over preceding CLIM_YEARS years
    rain_clim, mdd_clim = [], []
    for y in range(1, CLIM_YEARS + 1):
        c_end = end - timedelta(days=365 * y)
        c_start = c_end - timedelta(days=WINDOW_DAYS - 1)
        cp = series_for(c_start, c_end, "precipitation_sum")
        ct = series_for(c_start, c_end, "temperature_2m_mean")
        if cp and all(v is not None for v in cp):
            rain_clim.append(sum(cp))
        if ct and all(v is not None for v in ct):
            mdd_clim.append(sum(max(0.0, v - lapse_c) for v in ct))

    precip_30 = float(sum(precip))
    mdd_30 = float(sum(v for v in t_lake if not np.isnan(v) and v > 0))

    return {
        "precip_30d_mm": precip_30,
        "precip_7d_mm": float(sum(precip[-7:])),
        "max_daily_precip_mm": float(max(precip)),
        "heavy_rain_days": int(sum(p > HEAVY_RAIN_MM for p in precip)),
        "api_30": float(api),
        "rain_anom_30d_mm": (
            precip_30 - float(np.mean(rain_clim)) if rain_clim else np.nan
        ),
        "mdd_30": mdd_30,
        "mdd_anom_30": (
            mdd_30 - float(np.mean(mdd_clim)) if mdd_clim else np.nan
        ),
        "ft_cycles_14": int(ft),
    }


def build_features(samples: pd.DataFrame,
                   source: str = "openmeteo") -> pd.DataFrame:
    """Compute feature rows for every sample with a cache entry."""
    rows = []
    for _, row in samples.iterrows():
        path = _cache_path(row["sample_id"], source)
        if not path.exists():
            # fallback to the legacy flat cache (openmeteo) — the only
            # option for pre-1981 events under source=power
            path = _cache_path(row["sample_id"])
        if not path.exists():
            continue
        try:
            cached = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("corrupt cache: %s", path.name)
            continue
        feats = _window_features(
            cached["daily"], row["event_date"].date(),
            cached.get("station_elev_m"), row["lake_elev_m"],
        )
        if feats is None:
            continue
        rows.append({**row.to_dict(), **feats})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build_dataset(
    n_negatives: int = 800,
    n_blocks: int = 5,
    seed: int = 42,
    offline: bool = False,
    delay_s: float = REQUEST_DELAY_S,
    limit: int | None = None,
    source: str = "openmeteo",
) -> tuple[pd.DataFrame, dict]:
    """Full build: sample frame -> fetch -> features -> spatial blocks."""
    pos = _load_dated_events()
    icimod = sus._load_icimod()
    pos = sus._match_icimod(pos, icimod)
    pos["lake_area_km2"] = pos["icimod_area_km2"].fillna(pos["hmaglofdb_area_km2"])
    pos["lake_elev_m"] = pos["icimod_elev_m"].fillna(pos["Elev_lake"])
    # lake_elev_m is required (lapse correction); lake_area_km2 stays NaN
    # where neither source measured it — XGBoost handles missing natively.
    pos = pos[pos["lake_elev_m"].notna()].reset_index(drop=True)
    pos["sample_id"] = ["pos_" + str(i) for i in range(len(pos))]

    unique_lakes = pos.drop_duplicates("lake_id")
    kmeans = sus._fit_blocks(
        unique_lakes["lat"].values, unique_lakes["lon"].values,
        n_blocks, seed,
    )
    neg = _sample_negative_windows(pos, icimod, kmeans, n_negatives, seed)
    neg["sample_id"] = ["neg_" + str(i) for i in range(len(neg))]
    neg["breached"] = 0

    keep = ["sample_id", "lake_id", "event_date", "lat", "lon",
            "lake_elev_m", "lake_area_km2", "breached"]
    samples = pd.concat([pos[keep], neg[keep]], ignore_index=True)
    if limit:
        samples = samples.head(limit)

    fetch_stats = fetch_missing(samples, delay_s=delay_s, offline=offline,
                                source=source)
    df = build_features(samples, source=source)
    if df.empty:
        return df, {"fetch": fetch_stats, "note": "no cached series"}

    glaciers = sus._load_rgi_glaciers()
    df = sus._glacier_features(df, glaciers)
    df["log_lake_area_km2"] = np.log1p(df["lake_area_km2"])
    df["log_dist_glacier_m"] = np.log1p(df["dist_glacier_m"].clip(lower=0))
    df["log_glacier_area_10km"] = np.log1p(df["glacier_area_10km_m2"] / 1e6)
    df["block"] = kmeans.predict(
        sus._to_cartesian(df["lat"].values, df["lon"].values)
    )

    meta = {
        "weather_source": source,
        "fetch": fetch_stats,
        "n_samples": len(df),
        "n_breached": int(df["breached"].sum()),
        "n_stable": int((df["breached"] == 0).sum()),
        "n_unique_breach_lakes": int(
            df[df["breached"] == 1]["lake_id"].nunique()),
        "n_breached_with_area": int(
            df[df["breached"] == 1]["lake_area_km2"].notna().sum()),
        "features": FEATURE_NAMES,
    }
    return df, meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-negatives", type=int, default=800)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offline", action="store_true",
                   help="skip fetching; build features from cache only")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--delay", type=float, default=REQUEST_DELAY_S)
    p.add_argument("--source", choices=["openmeteo", "power"],
                   default="power",
                   help=(
                       "weather source: 'power' (NASA POWER, no key, "
                       "generous limits, ~0.5deg grid) is the default; "
                       "'openmeteo' (ERA5-Land, 9-11km) is quota-limited "
                       "to ~19 heavy-window requests/day on free tier"
                   ))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df, meta = build_dataset(
        n_negatives=args.n_negatives, n_blocks=args.n_blocks,
        seed=args.seed, offline=args.offline, delay_s=args.delay,
        limit=args.limit, source=args.source,
    )
    logger.info("fetch stats: %s", meta["fetch"])

    if df.empty:
        print("No feature rows (cache empty / fetch unavailable). "
              "Run without --offline on a networked host to populate "
              f"{CACHE_DIR}.")
        return 1

    df.to_parquet(args.out, index=False)
    logger.info("Dataset written to %s (%d rows)", args.out, len(df))

    try:
        out_display = str(args.out.relative_to(REPO_ROOT))
    except ValueError:
        out_display = str(args.out)

    report = {
        "status": "dataset_built",
        "dataset": out_display,
        "weather_source": (
            "NASA POWER (MERRA-2, ~0.5deg)" if args.source == "power"
            else "Open-Meteo archive (ERA5/ERA5-Land reanalysis)"
        ) + " — IMERG substitution documented: satellite precipitation "
            "requires Earthdata auth",
        **meta,
        "known_limitations": [
            (
                "reanalysis precipitation, not IMERG satellite precip "
                "— coarse grid smooths convective extremes "
                "(9-11km ERA5 / ~0.5deg MERRA-2)"
            ),
            (
                "negative windows may coincide with unrecorded events "
                "(label noise)"
            ),
            (
                "events before ~1991 lose the 10-yr climatology under "
                "NASA POWER (coverage starts 1981) — anomaly features "
                "are NaN there (XGBoost handles missing natively)"
            ),
            "ICIMOD 2022 area is post-event for historical breaches",
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report written to %s", REPORT_PATH)

    print(f"\nDataset: {len(df)} windows "
          f"({meta['n_breached']} event / {meta['n_stable']} non-event), "
          f"{meta['n_unique_breach_lakes']} unique breach lakes")
    print(f"Features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
