#!/usr/bin/env bash
# SIREN — Docker Compose launcher with port conflict resolution
# Kills anything on ports 8010 (backend) and 5175 (frontend) before starting.
set -euo pipefail

cd "$(dirname "$0")"

PORTS=(8010 5175)

for port in "${PORTS[@]}"; do
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "Port $port in use by PID(s): $pids — killing..."
    kill -9 $pids 2>/dev/null || true
    sleep 1
    if lsof -ti:"$port" >/dev/null 2>&1; then
      echo "WARNING: Port $port still in use after kill"
    else
      echo "Port $port freed"
    fi
  fi
done

echo "Starting Docker Compose..."
docker compose up -d --build

echo ""
echo "SIREN is live:"
echo "  Frontend: http://localhost:5175"
echo "  Backend:  http://localhost:8010"
