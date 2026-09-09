"""STAC polling daemon for autonomous Sentinel-1 ingestion (Sprint 1 Step 3).

Polls the Copernicus Data Space Ecosystem (CDSE) STAC API on a schedule for
new Sentinel-1 GRD scenes intersecting the target AOI. When a new scene is
detected that hasn't been ingested, it:
  1. Registers an acquisition_job (idempotent via UNIQUE(source, provider_product_id))
  2. Dispatches the calibration + pipeline worker

Two execution modes:
  - **Celery mode** (production): ``celery -A siren.ingest.stac_daemon worker``
    Polls every 6 hours via a Celery beat schedule. Requires redis + celery
    (``pip install -e ".[production]"``).
  - **CLI mode** (dev/cron): ``python -m siren.ingest.stac_daemon poll``
    Runs a single poll cycle. Schedule via cron or systemd timer. No celery
    or redis required.

Auth: set ``CDSE_TOKEN`` or ``CDSE_USERNAME`` + ``CDSE_PASSWORD`` (free
account at dataspace.copernicus.eu). Public STAC search works without auth
but downloads require it.

AOI: Dudh Koshi / Imja basin — [86.65, 27.65, 87.00, 27.98]
Product: Sentinel-1 GRD (IW_GRDH_1S), polarization VV+VH
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from siren.ingest.cdse import (
    STAC_URL,
    _auth_token,
    _http_post_json,
    search_all_scenes,
)

logger = logging.getLogger(__name__)

# Default AOI: Dudh Koshi / Imja basin
DEFAULT_BBOX: tuple[float, float, float, float] = (86.65, 27.65, 87.00, 27.98)
DEFAULT_BASIN_ID = "dudh-koshi-demo-01"
DEFAULT_SENSOR = "s1"
DEFAULT_LOOKBACK_DAYS = 30  # search the last N days for new scenes
DEFAULT_POLL_INTERVAL_HOURS = 6

# Sentinel-1 GRD product type filter for CDSE STAC
S1_PRODUCT_TYPE = "IW_GRDH_1S"
S1_POLARIZATION = "VV+VH"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _date_range(days: int = DEFAULT_LOOKBACK_DAYS) -> tuple[str, str]:
    """Return (start, end) YYYY-MM-DD strings for the last N days."""
    end = _utcnow()
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _extract_scene_metadata(item: dict) -> dict[str, Any]:
    """Extract ingestion-relevant metadata from a STAC item.

    Returns a dict with: scene_id, acquired_at, footprint, download_url,
    orbit, product_type, polarization.
    """
    props = item.get("properties", {})
    geometry = item.get("geometry", {})
    scene_id = item.get("id", "")

    # Extract download URL from assets
    download_url = None
    assets = item.get("assets", {})
    for key in ("data", "product", "download"):
        a = assets.get(key)
        if a and "href" in a:
            download_url = a["href"]
            break

    # Orbit info from properties (CDSE provides these)
    orbit = props.get("sat:relative_orbit", props.get("relativeOrbitNumber"))
    product_type = props.get("sar:product_type", S1_PRODUCT_TYPE)
    polarization = props.get("sar:polarizations", S1_POLARIZATION)

    return {
        "scene_id": scene_id,
        "acquired_at": props.get("datetime", ""),
        "geometry": geometry,
        "download_url": download_url,
        "orbit": orbit,
        "product_type": product_type,
        "polarization": polarization,
    }


def poll_stac(
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    sensor: str = DEFAULT_SENSOR,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Search CDSE STAC for recent scenes in the AOI.

    Returns a list of scene metadata dicts (from _extract_scene_metadata).
    Does NOT register or download — just searches. The caller decides what
    to do with the results.
    """
    date_range = _date_range(lookback_days)
    logger.info(
        "Polling CDSE STAC: bbox=%s sensor=%s date=%s",
        bbox, sensor, date_range,
    )
    features = search_all_scenes(
        bbox=bbox,
        sensor=sensor,
        date_range=date_range,
        max_scenes=100,
        token=token,
    )
    logger.info("Found %d scene(s) from CDSE STAC", len(features))
    return [_extract_scene_metadata(f) for f in features]


def register_new_scenes(
    scenes: list[dict[str, Any]],
    repo=None,
    basin_id: str = DEFAULT_BASIN_ID,
) -> list[dict[str, Any]]:
    """Register new scenes as acquisition_jobs in the database.

    Idempotent: if a scene's (source, provider_product_id) already exists
    in acquisition_jobs, it is skipped (not re-registered).

    Returns the list of newly registered scenes (empty list if all were
    already known).
    """
    if repo is None:
        from siren.db.repo import get_repository
        repo = get_repository()

    newly_registered: list[dict[str, Any]] = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        if not scene_id:
            logger.warning("Skipping scene with no ID: %s", scene)
            continue

        source = "cdse-s1"
        # Check idempotency: does this scene already exist?
        existing = repo.find_acquisition_job(source, scene_id)
        if existing is not None:
            logger.debug("Scene %s already registered (status=%s), skipping", scene_id, existing["status"])
            continue

        # Register the new scene
        job = repo.create_acquisition_job(
            source=source,
            provider_product_id=scene_id,
            download_url=scene.get("download_url"),
            acquired_at=scene.get("acquired_at"),
        )
        logger.info("Registered new scene %s as job %s", scene_id, job["job_id"])
        newly_registered.append(scene)

    logger.info(
        "Registered %d new scene(s) (%d already known)",
        len(newly_registered), len(scenes) - len(newly_registered),
    )
    return newly_registered


def dispatch_calibration(
    scenes: list[dict[str, Any]],
    repo=None,
    basin_id: str = DEFAULT_BASIN_ID,
) -> list[str]:
    """Dispatch calibration workers for newly registered scenes.

    In Celery mode, this enqueues ``calibrate_scene_task`` for each scene.
    In CLI mode, this returns the list of scene IDs that would be processed
    (the caller runs calibration directly or via a separate worker).

    Returns the list of scene IDs dispatched.
    """
    dispatched: list[str] = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        if not scene_id:
            continue

        # Try Celery mode first (if celery is installed and configured)
        if os.environ.get("CELERY_BROKER_URL") or os.environ.get("REDIS_URL"):
            try:
                from siren.ingest.stac_daemon_celery import calibrate_scene_task
                calibrate_scene_task.delay(scene_id)
                logger.info("Dispatched Celery task for scene %s", scene_id)
                dispatched.append(scene_id)
                continue
            except ImportError:
                logger.debug("Celery not available, falling back to direct dispatch")

        # CLI mode: just log — the caller handles actual processing
        logger.info("Scene %s queued for calibration (direct mode)", scene_id)
        dispatched.append(scene_id)

    return dispatched


def run_poll_cycle(
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    sensor: str = DEFAULT_SENSOR,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    repo=None,
    basin_id: str = DEFAULT_BASIN_ID,
) -> dict[str, Any]:
    """Run a single poll cycle: search → register → dispatch.

    This is the main entry point for both Celery beat and cron scheduling.
    Returns a summary dict with counts.
    """
    token = _auth_token()

    try:
        scenes = poll_stac(bbox=bbox, sensor=sensor, lookback_days=lookback_days, token=token)
    except Exception as exc:
        logger.error("STAC poll failed: %s", exc)
        return {"error": str(exc), "found": 0, "registered": 0, "dispatched": 0}

    new_scenes = register_new_scenes(scenes, repo=repo, basin_id=basin_id)
    dispatched = dispatch_calibration(new_scenes, repo=repo, basin_id=basin_id)

    summary = {
        "found": len(scenes),
        "registered": len(new_scenes),
        "dispatched": len(dispatched),
        "already_known": len(scenes) - len(new_scenes),
        "timestamp": _utcnow().isoformat(),
    }
    logger.info("Poll cycle complete: %s", summary)
    return summary


# --------------------------------------------------------------------------- #
# Celery app (lazy — only imported when celery is installed)
# --------------------------------------------------------------------------- #
def _get_celery_app():
    """Create and return the Celery app for the STAC daemon.

    Requires celery + redis installed (``pip install -e ".[production]"``).
    The broker URL is read from CELERY_BROKER_URL or REDIS_URL env vars.
    """
    try:
        from celery import Celery
    except ImportError as exc:
        raise ImportError(
            "celery is required for daemon mode. "
            "Install with: pip install -e '.[production]'"
        ) from exc

    broker = os.environ.get("CELERY_BROKER_URL") or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    app = Celery("siren-stac", broker=broker, backend=broker)

    # Beat schedule: poll every 6 hours
    app.conf.beat_schedule = {
        "poll-cdse-stac": {
            "task": "siren.ingest.stac_daemon.poll_task",
            "schedule": timedelta(hours=DEFAULT_POLL_INTERVAL_HOURS),
        },
    }
    app.conf.timezone = "UTC"
    return app


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the STAC daemon.

    Usage:
        python -m siren.ingest.stac_daemon poll          # single poll cycle
        python -m siren.ingest.stac_daemon poll --lookback 60
        python -m siren.ingest.stac_daemon worker        # start Celery worker
        python -m siren.ingest.stac_daemon beat           # start Celery beat scheduler
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m siren.ingest.stac_daemon",
        description="STAC polling daemon for autonomous Sentinel-1 ingestion.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # poll subcommand
    poll_p = sub.add_parser("poll", help="Run a single poll cycle (cron mode)")
    poll_p.add_argument("--bbox", default=None,
                        help="bbox 'lon_min,lat_min,lon_max,lat_max' (default: Dudh Koshi)")
    poll_p.add_argument("--sensor", default=DEFAULT_SENSOR, choices=("s1", "s2"),
                        help="Sensor (default: s1)")
    poll_p.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help=f"Lookback days (default: {DEFAULT_LOOKBACK_DAYS})")
    poll_p.add_argument("--basin", default=DEFAULT_BASIN_ID,
                        help="Basin ID (default: dudh-koshi-demo-01)")

    # worker subcommand (Celery)
    sub.add_parser("worker", help="Start Celery worker (requires redis)")

    # beat subcommand (Celery scheduler)
    sub.add_parser("beat", help="Start Celery beat scheduler (requires redis)")

    args = p.parse_args(argv)

    if args.command == "poll":
        bbox = DEFAULT_BBOX
        if args.bbox:
            from siren.ingest.cdse import parse_bbox
            bbox = parse_bbox(args.bbox)

        summary = run_poll_cycle(
            bbox=bbox,
            sensor=args.sensor,
            lookback_days=args.lookback,
            basin_id=args.basin,
        )
        print(json.dumps(summary, indent=2))
        return 0 if "error" not in summary else 1

    elif args.command == "worker":
        app = _get_celery_app()
        # Register tasks on the app
        _register_celery_tasks(app)
        app.worker_main(["worker", "--loglevel=info"])
        return 0

    elif args.command == "beat":
        app = _get_celery_app()
        _register_celery_tasks(app)
        app.start(["beat", "--loglevel=info"])
        return 0

    return 1


def _register_celery_tasks(app):
    """Register poll_task and calibrate_scene_task on the Celery app."""
    import celery

    @app.task(name="siren.ingest.stac_daemon.poll_task")
    def poll_task():
        """Celery task: run a single STAC poll cycle."""
        return run_poll_cycle()

    @app.task(name="siren.ingest.stac_daemon.calibrate_scene_task")
    def calibrate_scene_task(scene_id: str):
        """Celery task: calibrate and process a single scene.

        Applies Radiometric Terrain Correction (γ⁰) when the scene's SAFE
        archive and a DEM are available locally (Sprint 1 Step 6). The
        calibrated 2-band γ⁰ dB GeoTIFF is written to
        ``data/processed/rtc/<scene_id>.tif`` and the acquisition job is
        marked ``calibrated`` with the local path recorded.

        When the SAFE archive or DEM is not yet on disk (e.g. the scene was
        only registered from STAC metadata and not downloaded — Step 5
        /vsicurl/ COG reading will close that gap), the job is marked
        ``pending`` so a future worker can pick it up once the data lands.
        This preserves the offline-demo contract (Hard Rule 2): no network
        calls at runtime, no silent failures.
        """
        from siren.db.repo import get_repository
        repo = get_repository()
        job = repo.find_acquisition_job("cdse-s1", scene_id)
        if job is None:
            logger.warning("Scene %s not found in acquisition_jobs", scene_id)
            return {"scene_id": scene_id, "status": "not_found"}

        # Locate the SAFE archive and DEM on disk.
        from pathlib import Path
        from siren.preprocess.sar_calibrate import find_safe_for_scene_id

        raw_dir = Path(os.environ.get("SIREN_RAW_DIR", "data/raw"))
        processed_dir = Path(os.environ.get("SIREN_PROCESSED_DIR", "data/processed"))
        dem_path = Path(os.environ.get("SIREN_DEM_PATH", str(raw_dir / "srtm_30m.tif")))

        safe_zip = find_safe_for_scene_id(scene_id, raw_dir)
        if safe_zip is None or not dem_path.exists():
            logger.info(
                "Scene %s awaiting data (safe=%s, dem=%s) — marking pending",
                scene_id, safe_zip, dem_path if dem_path.exists() else None,
            )
            repo.update_acquisition_job(job["job_id"], status="pending")
            return {"scene_id": scene_id, "status": "pending", "rtc_applied": False}

        # Real RTC calibration path. Use the AOI-windowed reader
        # (Production Roadmap §2.2) when a bbox is configured, else fall
        # back to the full-scene reader.
        from siren.preprocess.rtc_pipeline import (
            calibrate_scene_with_rtc,
            calibrate_scene_with_rtc_windowed,
        )

        out_path = processed_dir / "rtc" / f"{scene_id}.tif"
        bbox_env = os.environ.get("SIREN_AOI_BBOX", "")
        try:
            if bbox_env:
                lon_min, lat_min, lon_max, lat_max = (
                    float(v) for v in bbox_env.split(",")
                )
                meta = calibrate_scene_with_rtc_windowed(
                    safe_zip, str(dem_path), out_path,
                    bbox=(lon_min, lat_min, lon_max, lat_max),
                    decimation=int(os.environ.get("SIREN_DECIMATION", "10")),
                )
            else:
                meta = calibrate_scene_with_rtc(
                    safe_zip, str(dem_path), out_path,
                    decimation=int(os.environ.get("SIREN_DECIMATION", "10")),
                )
        except Exception as exc:
            logger.error("RTC calibration failed for scene %s: %s", scene_id, exc)
            repo.update_acquisition_job(
                job["job_id"], status="failed", last_error=str(exc)
            )
            return {"scene_id": scene_id, "status": "failed", "error": str(exc)}

        repo.update_acquisition_job(
            job["job_id"], status="calibrated", local_path=str(out_path)
        )
        logger.info(
            "Scene %s calibrated to γ⁰ dB (IoU gate pending Sprint 2): %s",
            scene_id, meta["out_path"],
        )
        return {
            "scene_id": scene_id,
            "status": "calibrated",
            "rtc_applied": True,
            "out_path": meta["out_path"],
            "incidence_deg": meta["incidence_deg"],
            "look_azimuth_deg": meta["look_azimuth_deg"],
        }

    return poll_task, calibrate_scene_task


if __name__ == "__main__":
    sys.exit(main())
