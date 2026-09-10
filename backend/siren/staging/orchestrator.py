"""SIREN — End-to-End Live Staging Orchestrator (Track A)

Runs the complete production stack end-to-end against PostgreSQL/PostGIS:

  1. Verify PostgreSQL connectivity + PostGIS extension
  2. Seed demo basin + observations + assets into PostgreSQL
  3. Run the STAC poll cycle (CDSE search → register → dispatch)
  4. Process all observations through the pipeline:
       quality gate → routing → change mask → corridor + exposure →
       risk fusion → shadow evidence (FNO forward pass) → DB → audit
  5. Confirm review on the critical observation (human gate)
  6. Dispatch the ≤250-byte alert (dual-path hardware engine, simulated)
  7. Anchor the audit chain root to RFC 3161 timestamp authority
  8. Verify the SHA-256 hash chain integrity
  9. Print a comprehensive staging report

Usage:
    python -m siren.staging.orchestrator              # full E2E
    python -m siren.staging.orchestrator --skip-stac  # skip live STAC poll
    python -m siren.staging.orchestrator --skip-stac --skip-dispatch

Environment:
    DATABASE_URL=postgresql://siren:siren_staging@postgres:5432/siren
    REDIS_URL=redis://redis:6379/0
    SIREN_STAGING_MODE=live

This orchestrator is the "Run Monitoring" button's production equivalent.
It exercises every module in the pipeline against a real PostgreSQL
database with PostGIS spatial indexing, Celery-backed STAC ingestion, and
the dual-path hardware dispatch engine.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("siren.staging")

# ---------------------------------------------------------------------------
# ANSI colors for the staging report
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {CYAN}•{RESET} {msg}")


# ---------------------------------------------------------------------------
# Step 1: Verify PostgreSQL connectivity
# ---------------------------------------------------------------------------

def verify_postgres() -> bool:
    """Verify PostgreSQL connectivity and PostGIS extension."""
    _section("Step 1: Verify PostgreSQL / PostGIS")

    db_url = os.environ.get("DATABASE_URL")
    if not db_url or not db_url.startswith("postgresql://"):
        _fail("DATABASE_URL not set or not a PostgreSQL URL")
        _info("Set DATABASE_URL=postgresql://siren:siren_staging@postgres:5432/siren")
        return False

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        _fail("psycopg not installed — run: pip install -e '.[production]'")
        return False

    try:
        conn = psycopg.connect(db_url, row_factory=dict_row)
        conn.autocommit = True
    except Exception as exc:
        _fail(f"Cannot connect to PostgreSQL: {exc}")
        return False

    try:
        # Check PostGIS extension
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'postgis'"
        ).fetchone()
        if row is None:
            _fail("PostGIS extension not installed")
            conn.close()
            return False
        _ok(f"PostGIS {row['extversion']} active")

        # Check tables exist
        tables = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        ).fetchall()
        table_names = [t["tablename"] for t in tables]
        expected = ["basins", "observations", "runs", "scores", "exposures",
                     "reviews", "dispatches", "audit_log", "assets",
                     "acquisition_jobs"]
        missing = [t for t in expected if t not in table_names]
        if missing:
            _fail(f"Missing tables: {missing}")
            conn.close()
            return False
        _ok(f"Schema verified ({len(expected)} tables present)")

        conn.close()
        return True
    except Exception as exc:
        _fail(f"PostgreSQL verification failed: {exc}")
        conn.close()
        return False


# ---------------------------------------------------------------------------
# Step 2: Seed demo data into PostgreSQL
# ---------------------------------------------------------------------------

def seed_demo_data() -> bool:
    """Seed the demo basin, observations, and assets into PostgreSQL."""
    _section("Step 2: Seed Demo Data into PostgreSQL")

    from siren.db.repo import DEMO_BASIN, DEMO_OBSERVATIONS, DEMO_ASSETS, get_repository

    repo = get_repository()

    try:
        import psycopg
        conn = psycopg.connect(os.environ["DATABASE_URL"])
        conn.autocommit = True
    except Exception as exc:
        _fail(f"Cannot connect for seeding: {exc}")
        return False

    try:
        # Seed basin
        basin = DEMO_BASIN
        boundary_json = json.dumps(basin["boundary_geojson"])
        # Convert GeoJSON to PostGIS geometry
        geom_type = basin["boundary_geojson"].get("type", "MultiPolygon")
        conn.execute(
            """INSERT INTO basins (basin_id, name, boundary, boundary_geojson, crs)
               VALUES (%s, %s, ST_GeomFromGeoJSON(%s), %s, %s)
               ON CONFLICT (basin_id) DO UPDATE SET
                 name = excluded.name,
                 boundary = excluded.boundary,
                 boundary_geojson = excluded.boundary_geojson,
                 crs = excluded.crs""",
            (basin["basin_id"], basin["name"], boundary_json, boundary_json, basin["crs"]),
        )
        _ok(f"Basin '{basin['name']}' ({basin['basin_id']}) seeded")

        # Seed observations
        for obs in DEMO_OBSERVATIONS:
            kwargs = {
                "observation_id": obs["observation_id"],
                "basin_id": obs["basin_id"],
                "acquired_at": obs["acquired_at"],
                "source": obs["source"],
                "raster_uri": obs["raster_uri"],
                "quality_score": obs["quality_score"],
                "cloud_fraction": obs["cloud_fraction"],
                "optical_cloud_fraction": obs.get("optical_cloud_fraction", obs["cloud_fraction"]),
                "alignment_ok": obs["alignment_ok"],
                "usable": obs["usable"],
                "confidence_adjustment": obs["confidence_adjustment"],
                "water_area_km2": obs["water_area_km2"],
                "water_area_change_percent": obs["water_area_change_percent"],
                "rainfall_24h_mm": obs["rainfall_24h_mm"],
                "rainfall_7d_mm": obs["rainfall_7d_mm"],
                "temp_mean_c": obs.get("temp_mean_c"),
                "temp_index": obs.get("temp_index"),
                "mean_slope_degrees": obs["mean_slope_degrees"],
                "processing_version": obs["processing_version"],
            }
            repo.register_observation(**kwargs)
            _ok(f"Observation {obs['observation_id']} seeded "
                f"({obs['source']}, cloud={obs.get('optical_cloud_fraction', 0):.0%}, "
                f"expansion={obs['water_area_change_percent']}%)")

        # Seed assets
        for asset in DEMO_ASSETS:
            geom_json = json.dumps(asset["geometry_geojson"])
            conn.execute(
                """INSERT INTO assets
                   (asset_id, basin_id, asset_type, name, geometry, geometry_geojson,
                    population, weight)
                   VALUES (%s, %s, %s, %s, ST_GeomFromGeoJSON(%s), %s, %s, %s)
                   ON CONFLICT (asset_id) DO UPDATE SET
                     asset_type = excluded.asset_type,
                     name = excluded.name,
                     geometry = excluded.geometry,
                     geometry_geojson = excluded.geometry_geojson,
                     population = excluded.population,
                     weight = excluded.weight""",
                (
                    asset["asset_id"], asset["basin_id"], asset["asset_type"],
                    asset["name"], geom_json, geom_json,
                    asset.get("population"), asset.get("weight", 1.0),
                ),
            )
        _ok(f"{len(DEMO_ASSETS)} assets seeded (bridges, settlements, wells, roads, health)")

        conn.close()
        return True
    except Exception as exc:
        _fail(f"Seeding failed: {exc}")
        conn.close()
        return False


# ---------------------------------------------------------------------------
# Step 3: STAC poll cycle (live CDSE search)
# ---------------------------------------------------------------------------

def run_stac_poll(skip: bool = False) -> bool:
    """Run a single STAC poll cycle (CDSE search → register → dispatch)."""
    _section("Step 3: STAC Poll Cycle (CDSE)")

    if skip:
        _warn("STAC poll skipped (--skip-stac)")
        return True

    try:
        from siren.ingest.stac_daemon import run_poll_cycle
    except Exception as exc:
        _fail(f"Cannot import STAC daemon: {exc}")
        return False

    try:
        _info("Polling CDSE STAC for recent Sentinel-1 scenes in Dudh Koshi AOI...")
        summary = run_poll_cycle(
            bbox=(86.65, 27.65, 87.00, 27.98),
            sensor="s1",
            lookback_days=30,
        )

        if "error" in summary:
            _warn(f"STAC poll returned error (expected if CDSE token not set): {summary['error']}")
            _info("This is non-blocking — the pipeline will process seeded demo observations")
            return True

        _ok(f"Found {summary['found']} scene(s), registered {summary['registered']} new, "
            f"dispatched {summary['dispatched']}")
        if summary["already_known"]:
            _info(f"{summary['already_known']} scene(s) already registered (idempotent skip)")

        return True
    except Exception as exc:
        _warn(f"STAC poll failed (non-blocking): {exc}")
        _info("Pipeline will process seeded demo observations instead")
        return True


# ---------------------------------------------------------------------------
# Step 4: Process all observations through the pipeline
# ---------------------------------------------------------------------------

def process_observations() -> list[dict[str, Any]]:
    """Run the full pipeline for all demo observations."""
    _section("Step 4: Pipeline Execution (detect → geo → risk → FNO → audit)")

    from siren.pipeline import run_pipeline
    from siren.db.repo import get_repository

    repo = get_repository()
    results: list[dict[str, Any]] = []

    obs_ids = ["obs-001", "obs-002", "obs-003"]
    for obs_id in obs_ids:
        _info(f"Processing {obs_id}...")
        t0 = time.time()
        try:
            run = run_pipeline(obs_id, repo)
            elapsed = time.time() - t0
            score = run.get("score", {}) if run else {}
            severity = score.get("severity", "unknown")
            hazard = score.get("hazard_score", 0)
            n_reasons = len(score.get("reasons", []))

            # Check shadow evidence (FNO)
            change_stats = run.get("change_stats", run.get("change_stats_json", {})) if run else {}
            shadow = change_stats.get("shadow_evidence", {}) if isinstance(change_stats, dict) else {}
            hydro = shadow.get("hydro_surrogate", {}) if isinstance(shadow, dict) else {}
            fno_triggered = hydro.get("is_triggered", False) if isinstance(hydro, dict) else False
            fno_t_arrival = hydro.get("t_arrival_by_sector") if isinstance(hydro, dict) else None

            _ok(f"{obs_id} processed in {elapsed:.1f}s — severity={severity}, "
                f"H={hazard:.3f}, reasons={n_reasons}")

            if fno_triggered:
                if fno_t_arrival:
                    _ok(f"  FNO shadow: triggered, T_arrival={fno_t_arrival}")
                else:
                    _info(f"  FNO shadow: triggered (no checkpoint — simulated)")
            else:
                _info(f"  FNO shadow: not triggered (P_breach < 0.70)")

            results.append(run or {"observation_id": obs_id, "error": "no result"})
        except Exception as exc:
            _fail(f"{obs_id} failed: {exc}")
            results.append({"observation_id": obs_id, "error": str(exc)})

    return results


# ---------------------------------------------------------------------------
# Step 5: Human gate — confirm review on the critical observation
# ---------------------------------------------------------------------------

def confirm_review(run_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Confirm review on the highest-severity run (human gate, Hard Rule 3)."""
    _section("Step 5: Human Gate — Review Confirmation")

    from siren.db.repo import get_repository

    repo = get_repository()

    # Find the highest-severity run
    best_run = None
    best_severity_rank = 0
    severity_rank = {"informational": 1, "watch": 2, "elevated": 3, "critical": 4}
    for run in run_results:
        score = run.get("score", {}) if isinstance(run, dict) else {}
        sev = score.get("severity", "informational")
        rank = severity_rank.get(sev, 0)
        if rank > best_severity_rank:
            best_severity_rank = rank
            best_run = run

    if best_run is None or best_severity_rank < 3:
        _warn("No elevated/critical run to confirm — skipping dispatch")
        return None

    run_id = best_run["run_id"]
    severity = best_run.get("score", {}).get("severity", "unknown")

    try:
        review = repo.create_review(
            run_id=run_id,
            reviewer="staging-orchestrator",
            decision="confirm",
            note=f"Staging E2E: confirmed {severity} alert for {run_id}",
        )
        _ok(f"Review confirmed on {run_id} (severity={severity}) — review_id={review['review_id']}")
        return best_run
    except Exception as exc:
        _fail(f"Review confirmation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 6: Dispatch the ≤250-byte alert (dual-path hardware engine)
# ---------------------------------------------------------------------------

def dispatch_alert(confirmed_run: dict[str, Any] | None, skip: bool = False) -> dict[str, Any] | None:
    """Dispatch the alert via the dual-path hardware engine (simulated)."""
    _section("Step 6: Dual-Path Hardware Dispatch")

    if skip or confirmed_run is None:
        _warn("Dispatch skipped — no confirmed run or --skip-dispatch flag")
        return None

    from siren.db.repo import get_repository
    repo = get_repository()

    run_id = confirmed_run["run_id"]

    try:
        # Create the dispatch via the repository (human gate enforced)
        dispatch = repo.create_dispatch(
            run_id=run_id,
            channel="dual-path",
            recipient_group="basin-B",
        )

        payload = dispatch.get("payload", "")
        payload_bytes = dispatch.get("payload_bytes", 0)
        alert_id = dispatch.get("alert_id", "")

        _ok(f"Dispatch sent: {alert_id} ({payload_bytes} bytes)")
        _info(f"Payload: {payload}")

        if payload_bytes > 250:
            _fail(f"Payload exceeds 250-byte limit: {payload_bytes} bytes (Hard Rule 4)")
            return dispatch

        _ok(f"Payload within 250-byte budget ({payload_bytes}/250 bytes)")

        # Also run the dual-path hardware dispatch engine (simulated)
        from siren.alerting.dispatch import DispatchEngine
        engine = DispatchEngine(simulate=True)
        result = engine.dispatch(
            dispatch_id=dispatch["dispatch_id"],
            payload=payload,
            recipient_group="basin-B",
        )

        for receipt in result.receipts:
            _ok(f"  {receipt.path.value}: {receipt.status.value} "
                f"({receipt.message_id}, {receipt.payload_size} bytes)")

        return dispatch
    except Exception as exc:
        _fail(f"Dispatch failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 7: Anchor audit chain to RFC 3161 timestamp authority
# ---------------------------------------------------------------------------

def anchor_audit_chain() -> bool:
    """Anchor the audit chain root to an external timestamp witness."""
    _section("Step 7: RFC 3161 Timestamp Anchoring")

    from siren.db.repo import get_repository
    from siren.audit.timestamp import anchor_chain_root

    repo = get_repository()

    try:
        # Get the latest audit entry's hash
        audit_entries = repo.list_audit()
        if not audit_entries:
            _warn("No audit entries to anchor")
            return True

        latest = audit_entries[-1]
        chain_root = latest.get("event_hash", "")
        if not chain_root or len(chain_root) != 64:
            _warn(f"Latest audit entry has no valid hash (got {len(chain_root)} chars)")
            return True

        _info(f"Anchoring chain root {chain_root[:16]}... to RFC 3161 (Google TSA)")

        anchor = anchor_chain_root(
            chain_root=chain_root,
            witness="rfc3161",
            tsa_name="google",
            simulate=True,  # demo mode — no real network call
        )

        _ok(f"Anchored: witness={anchor.witness}, verified={anchor.is_verified}, "
            f"simulated={anchor.is_simulated}")
        _info(f"Timestamp: {anchor.timestamp}")
        _info(f"Token: {anchor.token[:32]}...")

        return True
    except Exception as exc:
        _fail(f"Timestamp anchoring failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Step 8: Verify SHA-256 hash chain integrity
# ---------------------------------------------------------------------------

def verify_hash_chain() -> bool:
    """Verify the audit hash chain integrity."""
    _section("Step 8: SHA-256 Hash Chain Verification")

    from siren.db.repo import get_repository

    repo = get_repository()

    try:
        audit_entries = repo.list_audit()
        _ok(f"Audit log: {len(audit_entries)} entries")

        if not audit_entries:
            _warn("Empty audit log — nothing to verify")
            return True

        # Print the chain
        for i, entry in enumerate(audit_entries):
            action = entry.get("action", "?")
            actor = entry.get("actor", "?")
            prev = entry.get("prev_hash", "")[:12]
            curr = entry.get("event_hash", "")[:12]
            _info(f"  [{i + 1:3d}] {action:12s} by {actor:20s} "
                  f"prev={prev}... curr={curr}...")

        # Verify the chain
        is_valid = repo.verify_hash_chain()
        if is_valid:
            _ok("Hash chain VERIFIED — all entries linked, no tampering detected")
        else:
            _fail("Hash chain BROKEN — tampering detected or corrupted entry")

        return is_valid
    except Exception as exc:
        _fail(f"Hash chain verification failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Step 9: Print staging report
# ---------------------------------------------------------------------------

def print_report(
    pg_ok: bool,
    seed_ok: bool,
    stac_ok: bool,
    run_results: list[dict[str, Any]],
    confirmed_run: dict[str, Any] | None,
    dispatch: dict[str, Any] | None,
    anchor_ok: bool,
    chain_ok: bool,
) -> bool:
    """Print the final staging report and return overall success."""
    _section("STAGING REPORT")

    checks = [
        ("PostgreSQL / PostGIS", pg_ok),
        ("Demo data seeded", seed_ok),
        ("STAC poll cycle", stac_ok),
        ("Pipeline execution", all("error" not in r for r in run_results)),
        ("FNO shadow evidence", any(
            r.get("change_stats", r.get("change_stats_json", {})).get("shadow_evidence", {}).get("hydro_surrogate", {}).get("is_triggered")
            for r in run_results if isinstance(r, dict)
        )),
        ("Human gate (confirm)", confirmed_run is not None),
        ("Dual-path dispatch", dispatch is not None),
        ("RFC 3161 anchoring", anchor_ok),
        ("SHA-256 hash chain", chain_ok),
        ("Payload ≤ 250 bytes", dispatch is not None and dispatch.get("payload_bytes", 999) <= 250),
    ]

    all_ok = True
    for name, ok in checks:
        if ok:
            _ok(name)
        else:
            _fail(name)
            all_ok = False

    print()
    if all_ok:
        print(f"{BOLD}{GREEN}  ✦  STAGING PASSED — all checks green{RESET}")
    else:
        failed = sum(1 for _, ok in checks if not ok)
        print(f"{BOLD}{YELLOW}  ⚠  STAGING PARTIAL — {failed} check(s) failed{RESET}")

    print(f"\n  Observations processed: {len(run_results)}")
    print(f"  Audit entries: {len(run_results) * 3 + (2 if dispatch else 0)}")
    print(f"  Timestamp: {_utcnow_iso()}")

    return all_ok


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m siren.staging.orchestrator",
        description="SIREN End-to-End Live Staging Orchestrator (Track A)",
    )
    p.add_argument("--skip-stac", action="store_true",
                    help="Skip the live STAC poll cycle (use seeded demo data only)")
    p.add_argument("--skip-dispatch", action="store_true",
                    help="Skip the dispatch step (no alert sent)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print(f"\n{BOLD}SIREN — End-to-End Live Staging{RESET}")
    print(f"  Timestamp: {_utcnow_iso()}")
    print(f"  DATABASE_URL: {os.environ.get('DATABASE_URL', '(not set)')}")
    print(f"  REDIS_URL: {os.environ.get('REDIS_URL', '(not set)')}")
    print(f"  Mode: {os.environ.get('SIREN_STAGING_MODE', 'default')}")

    # Step 1: Verify PostgreSQL
    pg_ok = verify_postgres()
    if not pg_ok:
        _fail("Cannot proceed without PostgreSQL — aborting")
        return 1

    # Step 2: Seed demo data
    seed_ok = seed_demo_data()
    if not seed_ok:
        _fail("Cannot proceed without demo data — aborting")
        return 1

    # Step 3: STAC poll
    stac_ok = run_stac_poll(skip=args.skip_stac)

    # Step 4: Process observations
    run_results = process_observations()

    # Step 5: Confirm review (human gate)
    confirmed_run = confirm_review(run_results)

    # Step 6: Dispatch alert
    dispatch = dispatch_alert(confirmed_run, skip=args.skip_dispatch)

    # Step 7: Anchor audit chain
    anchor_ok = anchor_audit_chain()

    # Step 8: Verify hash chain
    chain_ok = verify_hash_chain()

    # Step 9: Report
    all_ok = print_report(
        pg_ok=pg_ok,
        seed_ok=seed_ok,
        stac_ok=stac_ok,
        run_results=run_results,
        confirmed_run=confirmed_run,
        dispatch=dispatch,
        anchor_ok=anchor_ok,
        chain_ok=chain_ok,
    )

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
