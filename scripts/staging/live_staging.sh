#!/usr/bin/env bash
# SIREN — Live Staging Launcher
# Starts the full production stack: PostgreSQL/PostGIS + Redis + Celery + Backend + Frontend
# Then runs the E2E staging orchestrator.
#
# Usage:
#   ./scripts/staging/live_staging.sh              # full E2E with live STAC poll
#   ./scripts/staging/live_staging.sh --skip-stac  # skip STAC poll (seeded data only)
#   ./scripts/staging/live_staging.sh --down       # tear down the staging stack
set -euo pipefail

cd "$(dirname "$0")/../.."

COMPOSE_FILES="-f docker-compose.yml -f docker-compose.staging.yml"

# ---------------------------------------------------------------------------
# Tear down mode
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--down" ]]; then
    echo "Tearing down SIREN staging stack..."
    docker compose $COMPOSE_FILES down -v
    echo "Done."
    exit 0
fi

# ---------------------------------------------------------------------------
# Start the staging stack
# ---------------------------------------------------------------------------
echo "Starting SIREN live staging stack..."
echo "  PostgreSQL/PostGIS:  localhost:5433"
echo "  Redis:               localhost:6380"
echo "  Backend API:          http://localhost:8010"
echo "  Frontend:             http://localhost:5175"
echo ""

# Build and start all services (except staging-runner, which is gated by profile)
docker compose $COMPOSE_FILES up -d --build

# Wait for PostgreSQL to be healthy
echo "Waiting for PostgreSQL to be ready..."
for i in $(seq 1 30); do
    if docker compose $COMPOSE_FILES exec -T postgres pg_isready -U siren -d siren >/dev/null 2>&1; then
        echo "  PostgreSQL ready (attempt $i)"
        break
    fi
    sleep 2
    if [[ $i -eq 30 ]]; then
        echo "ERROR: PostgreSQL did not become ready in 60s"
        exit 1
    fi
done

# Wait for Redis to be healthy
echo "Waiting for Redis to be ready..."
for i in $(seq 1 15); do
    if docker compose $COMPOSE_FILES exec -T redis redis-cli ping >/dev/null 2>&1; then
        echo "  Redis ready (attempt $i)"
        break
    fi
    sleep 1
done

echo ""
echo "=== Running E2E Staging Orchestrator ==="
echo ""

# Run the staging orchestrator (passes through args like --skip-stac)
docker compose $COMPOSE_FILES run --rm \
    -e SIREN_STAGING_MODE=live \
    staging-runner \
    python -m siren.staging.orchestrator "$@"

EXIT_CODE=$?

echo ""
echo "=== Staging orchestrator exited with code $EXIT_CODE ==="
echo ""
echo "Services are still running:"
echo "  Backend API:  http://localhost:8010"
echo "  Frontend:     http://localhost:5175"
echo "  PostgreSQL:   localhost:5433 (siren/siren_staging)"
echo "  Redis:        localhost:6380"
echo ""
echo "To tear down: ./scripts/staging/live_staging.sh --down"

exit $EXIT_CODE
