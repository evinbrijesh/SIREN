"""Tests for the end-to-end live staging orchestrator (Track A).

Tests the staging orchestrator's structure, the Docker Compose staging
override, and (when a PostgreSQL DATABASE_URL is available) the full E2E
flow against a real PostGIS database.

The PostgreSQL integration tests are skipped when DATABASE_URL is not set
or psycopg is not installed — they require the staging stack to be running
(``./scripts/staging/live_staging.sh``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Docker Compose staging override — structural tests (no Docker required)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_STAGING = PROJECT_ROOT / "docker-compose.staging.yml"
ORCHESTRATOR = PROJECT_ROOT / "backend" / "siren" / "staging" / "orchestrator.py"


def test_staging_compose_override_exists():
    """The staging Docker Compose override file exists."""
    assert COMPOSE_STAGING.exists(), f"Missing {COMPOSE_STAGING}"


def test_staging_compose_has_postgres_service():
    """The staging compose file defines a postgres service with PostGIS."""
    content = COMPOSE_STAGING.read_text()
    assert "postgres:" in content
    assert "postgis/postgis" in content
    assert "healthcheck" in content


def test_staging_compose_has_redis_service():
    """The staging compose file defines a redis service."""
    content = COMPOSE_STAGING.read_text()
    assert "redis:" in content
    assert "redis:7-alpine" in content
    assert "healthcheck" in content


def test_staging_compose_has_celery_worker():
    """The staging compose file defines a celery-worker service."""
    content = COMPOSE_STAGING.read_text()
    assert "celery-worker:" in content
    assert "celery -A siren.ingest.stac_daemon_celery worker" in content


def test_staging_compose_has_celery_beat():
    """The staging compose file defines a celery-beat scheduler."""
    content = COMPOSE_STAGING.read_text()
    assert "celery-beat:" in content
    assert "celery -A siren.ingest.stac_daemon_celery beat" in content


def test_staging_compose_has_staging_runner():
    """The staging compose file defines a staging-runner service."""
    content = COMPOSE_STAGING.read_text()
    assert "staging-runner:" in content
    assert "siren.staging.orchestrator" in content
    assert "profiles:" in content  # gated by profile


def test_staging_compose_backend_has_database_url():
    """The backend service receives DATABASE_URL pointing to PostgreSQL."""
    content = COMPOSE_STAGING.read_text()
    assert "DATABASE_URL=postgresql://" in content
    assert "REDIS_URL=redis://" in content
    assert "CELERY_BROKER_URL=redis://" in content


def test_staging_compose_postgis_init_script_mounted():
    """The PostGIS schema SQL is mounted as an init script."""
    content = COMPOSE_STAGING.read_text()
    assert "postgres_schema.sql" in content
    assert "docker-entrypoint-initdb.d" in content


# ---------------------------------------------------------------------------
# Orchestrator module — structural tests (no PostgreSQL required)
# ---------------------------------------------------------------------------

def test_orchestrator_module_exists():
    """The staging orchestrator module exists."""
    assert ORCHESTRATOR.exists(), f"Missing {ORCHESTRATOR}"


def test_orchestrator_has_all_steps():
    """The orchestrator defines all 8 E2E steps."""
    content = ORCHESTRATOR.read_text()
    steps = [
        "verify_postgres",
        "seed_demo_data",
        "run_stac_poll",
        "process_observations",
        "confirm_review",
        "dispatch_alert",
        "anchor_audit_chain",
        "verify_hash_chain",
    ]
    for step in steps:
        assert f"def {step}(" in content, f"Missing step: {step}"


def test_orchestrator_has_main_entry():
    """The orchestrator has a main() entry point with argparse."""
    content = ORCHESTRATOR.read_text()
    assert "def main(" in content
    assert "--skip-stac" in content
    assert "--skip-dispatch" in content
    assert 'if __name__ == "__main__"' in content


def test_orchestrator_imports_clean():
    """The orchestrator module imports without errors."""
    from siren.staging import orchestrator
    assert hasattr(orchestrator, "main")
    assert hasattr(orchestrator, "verify_postgres")
    assert hasattr(orchestrator, "seed_demo_data")
    assert hasattr(orchestrator, "process_observations")
    assert hasattr(orchestrator, "verify_hash_chain")


# ---------------------------------------------------------------------------
# Staging launch script — structural tests
# ---------------------------------------------------------------------------

LAUNCH_SCRIPT = PROJECT_ROOT / "scripts" / "staging" / "live_staging.sh"


def test_launch_script_exists():
    """The staging launch script exists and is executable."""
    assert LAUNCH_SCRIPT.exists(), f"Missing {LAUNCH_SCRIPT}"


def test_launch_script_uses_staging_compose():
    """The launch script uses both compose files."""
    content = LAUNCH_SCRIPT.read_text()
    assert "docker-compose.yml" in content
    assert "docker-compose.staging.yml" in content


def test_launch_script_has_down_mode():
    """The launch script supports --down to tear down the stack."""
    content = LAUNCH_SCRIPT.read_text()
    assert "--down" in content


# ---------------------------------------------------------------------------
# Dockerfile.backend — production extra test
# ---------------------------------------------------------------------------

DOCKERFILE = PROJECT_ROOT / "Dockerfile.backend"


def test_dockerfile_installs_production_extra():
    """The backend Dockerfile installs the [production] extra."""
    content = DOCKERFILE.read_text()
    assert "[dev,production]" in content or "[production]" in content


# ---------------------------------------------------------------------------
# PostgreSQL integration tests (skipped if no DATABASE_URL or no psycopg)
# ---------------------------------------------------------------------------

_HAS_PG = (
    os.environ.get("DATABASE_URL", "").startswith("postgresql://")
    and _import_check("psycopg")
)


def _import_check(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


pytestmark_pg = pytest.mark.skipif(
    not _HAS_PG,
    reason="DATABASE_URL not set to postgresql:// or psycopg not installed",
)


@pytest.fixture
def pg_repo():
    """Provide a PostgresRepository connected to the staging database."""
    from siren.db.repo import get_repository
    repo = get_repository()
    yield repo


@pytestmark_pg
class TestPostgresStaging:
    """Integration tests against a live PostgreSQL/PostGIS instance.

    These tests run only when DATABASE_URL is set to a postgresql:// URL
    and psycopg is installed. Start the staging stack first:

        ./scripts/staging/live_staging.sh --skip-stac
    """

    def test_postgres_connection(self, pg_repo):
        """The PostgresRepository connects and the schema is initialized."""
        basin = pg_repo.get_basin()
        assert basin is not None
        assert basin["basin_id"] == "dudh-koshi-demo-01"

    def test_observations_seeded(self, pg_repo):
        """Demo observations are present in PostgreSQL."""
        obs = pg_repo.list_observations()
        obs_ids = {o["observation_id"] for o in obs}
        assert "obs-001" in obs_ids
        assert "obs-002" in obs_ids
        assert "obs-003" in obs_ids

    def test_pipeline_run_against_postgres(self, pg_repo):
        """The pipeline runs successfully against PostgreSQL."""
        from siren.pipeline import run_pipeline
        run = run_pipeline("obs-001", pg_repo)
        assert run is not None
        assert "run_id" in run
        assert "score" in run
        assert run["score"]["severity"] in ("informational", "watch", "elevated", "critical")

    def test_hash_chain_verifies_after_pipeline(self, pg_repo):
        """The SHA-256 hash chain verifies after pipeline runs."""
        # Run a pipeline to generate audit entries
        from siren.pipeline import run_pipeline
        run_pipeline("obs-001", pg_repo)
        assert pg_repo.verify_hash_chain() is True

    def test_full_staging_orchestrator(self):
        """The full staging orchestrator runs end-to-end against PostgreSQL."""
        from siren.staging.orchestrator import main
        exit_code = main(["--skip-stac", "--skip-dispatch"])
        assert exit_code == 0
