"""Celery task wrappers for the STAC daemon.

This module is imported lazily by stac_daemon when Celery is available.
It re-exports the tasks registered on the Celery app so they can be
imported by the worker:

    from siren.ingest.stac_daemon_celery import calibrate_scene_task
"""

from __future__ import annotations

import os

try:
    from siren.ingest.stac_daemon import _get_celery_app, _register_celery_tasks

    app = _get_celery_app()
    poll_task, calibrate_scene_task = _register_celery_tasks(app)
except ImportError:
    # Celery not installed — this module should only be imported when
    # CELERY_BROKER_URL or REDIS_URL is set.
    app = None  # type: ignore[assignment]
    poll_task = None  # type: ignore[assignment]
    calibrate_scene_task = None  # type: ignore[assignment]
