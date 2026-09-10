"""PostgreSQL / PostGIS repository for SIREN (production path).

Implements the same public interface as ``siren.db.repo.Repository`` but uses
psycopg3 + PostGIS. Activated when ``DATABASE_URL=postgresql://...`` is set
at runtime (ADR-011). When unset, the SQLite ``Repository`` is used instead.

Key differences from the SQLite path:
  - ``?`` placeholders → ``%s``
  - ``INTEGER`` booleans → native ``BOOLEAN``
  - ``TEXT`` JSON → ``JSONB``
  - ``strftime(...)`` → ``now()``
  - ``PRAGMA foreign_keys`` → not needed (Postgres enforces FKs)
  - ``AUTOINCREMENT`` → ``BIGSERIAL``
  - Spatial queries use ``ST_Intersects``, ``ST_DWithin``, ``ST_GeomFromGeoJSON``

The SQLite schema (``schema.sql``) and PostGIS schema (``postgres_schema.sql``)
share the same table/column names (except geometry columns). The public method
signatures are identical so callers do not need to know which backend is active.
"""

from __future__ import annotations

import json
import zlib
from datetime import datetime, timezone
from typing import Any

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]

from siren.audit.hash_chain import GENESIS_HASH, event_hash

POSTGRES_SCHEMA_PATH = (
    __import__("pathlib").Path(__file__).resolve().parent / "postgres_schema.sql"
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PostgresRepository:
    """PostgreSQL + PostGIS repository.

    Requires ``psycopg`` installed (``pip install -e ".[production]"``).
    Uses a single synchronous connection with ``autocommit=False`` — each
    method commits its own transaction. The connection row factory is
    ``dict_row`` so rows are plain dicts (matching the SQLite ``Row`` pattern).
    """

    def __init__(self, database_url: str) -> None:
        if psycopg is None:
            raise ImportError(
                "psycopg is required for PostgreSQL mode. "
                "Install with: pip install -e '.[production]'"
            )
        self.database_url = database_url
        self._conn = psycopg.connect(database_url, row_factory=dict_row)
        self._conn.autocommit = False
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    # --- schema ---

    def _init_schema(self) -> None:
        self._conn.execute(POSTGRES_SCHEMA_PATH.read_text())
        self._conn.commit()

    # --- basins ---

    def get_basin(self, basin_id: str = "dudh-koshi-demo-01") -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT basin_id, name, boundary_geojson, crs FROM basins WHERE basin_id=%s",
            (basin_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "basin_id": row["basin_id"],
            "name": row["name"],
            "boundary_geojson": row["boundary_geojson"] if isinstance(row["boundary_geojson"], dict)
            else json.loads(row["boundary_geojson"]),
            "crs": row["crs"],
        }

    # --- observations ---

    def list_observations(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM observations ORDER BY acquired_at DESC"
        ).fetchall()
        return [self._observation_row(row) for row in rows]

    def get_observation(self, observation_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM observations WHERE observation_id=%s",
            (observation_id,),
        ).fetchone()
        return None if row is None else self._observation_row(row)

    @staticmethod
    def _observation_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "observation_id": row["observation_id"],
            "basin_id": row["basin_id"],
            "acquired_at": row["acquired_at"].isoformat() if hasattr(row["acquired_at"], "isoformat") else row["acquired_at"],
            "source": row["source"],
            "raster_uri": row["raster_uri"],
            "crs": row["crs"],
            "quality_score": row["quality_score"],
            "cloud_fraction": row["cloud_fraction"],
            "optical_cloud_fraction": row["optical_cloud_fraction"],
            "alignment_ok": row["alignment_ok"],
            "usable": row["usable"],
            "confidence_adjustment": row["confidence_adjustment"],
            "water_area_km2": row["water_area_km2"],
            "water_area_change_percent": row["water_area_change_percent"],
            "rainfall_24h_mm": row["rainfall_24h_mm"],
            "rainfall_7d_mm": row["rainfall_7d_mm"],
            "temp_mean_c": row["temp_mean_c"],
            "temp_index": row["temp_index"],
            "mean_slope_degrees": row["mean_slope_degrees"],
            "processing_version": row["processing_version"],
            "status": row["status"],
        }

    # --- runs ---

    def run_exists(self, run_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM runs WHERE run_id=%s", (run_id,)
            ).fetchone()
            is not None
        )

    def create_run(self, observation_id: str, processing_version: str = "0.1.0") -> dict[str, Any]:
        count = self._conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
        run_id = f"run-{count + 1:04d}"
        started_at = _utcnow_iso()
        self._conn.execute(
            "INSERT INTO runs(run_id, observation_id, processing_version) VALUES(%s,%s,%s)",
            (run_id, observation_id, processing_version),
        )
        self._conn.commit()
        return {
            "run_id": run_id,
            "observation_id": observation_id,
            "status": "queued",
            "started_at": started_at,
        }

    def complete_run(
        self,
        run_id: str,
        change_mask_uri: str,
        corridor_geojson: dict[str, Any],
        change_stats_json: dict[str, Any],
    ) -> None:
        # Insert corridor as PostGIS geometry + GeoJSON
        corridor_wkb = None
        if corridor_geojson.get("coordinates"):
            corridor_wkb = f"ST_GeomFromGeoJSON(%s)"
            self._conn.execute(
                f"""UPDATE runs
                    SET change_mask_uri=%s, corridor={corridor_wkb},
                        corridor_geojson=%s, change_stats=%s, finished_at=now()
                    WHERE run_id=%s""",
                (
                    change_mask_uri,
                    json.dumps(corridor_geojson),
                    json.dumps(corridor_geojson),
                    json.dumps(change_stats_json),
                    run_id,
                ),
            )
        else:
            self._conn.execute(
                """UPDATE runs
                   SET change_mask_uri=%s, corridor_geojson=%s, change_stats=%s,
                       finished_at=now()
                   WHERE run_id=%s""",
                (
                    change_mask_uri,
                    json.dumps(corridor_geojson),
                    json.dumps(change_stats_json),
                    run_id,
                ),
            )
        self._conn.commit()

    def add_score(
        self,
        run_id: str,
        hazard_score: float,
        exposure_priority: float,
        disease_risk: float | None,
        confidence: float,
        severity: str,
        reasons: list[str],
    ) -> str:
        count = self._conn.execute("SELECT COUNT(*) AS c FROM scores").fetchone()["c"]
        score_id = f"score-{count + 1:04d}"
        self._conn.execute(
            """INSERT INTO scores
               (score_id, run_id, hazard_score, exposure_priority,
                disease_risk, confidence, severity, reasons)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                score_id, run_id, hazard_score, exposure_priority,
                disease_risk, confidence, severity, json.dumps(reasons),
            ),
        )
        self._audit(None, "pipeline", "score", {
            "run_id": run_id, "score_id": score_id, "severity": severity,
        })
        self._conn.commit()
        return score_id

    def add_exposures(
        self, run_id: str, exposures: list[dict[str, Any]]
    ) -> None:
        for i, exp in enumerate(exposures):
            exp_id = f"exp-{run_id}-{i + 1:03d}"
            asset_id = exp["asset_id"]

            exists = self._conn.execute(
                "SELECT 1 FROM assets WHERE asset_id=%s", (asset_id,)
            ).fetchone()
            if not exists:
                geom = exp.get("geometry_geojson", {"type": "Point", "coordinates": [0, 0]})
                self._conn.execute(
                    """INSERT INTO assets
                       (asset_id, basin_id, asset_type, name, geometry, geometry_geojson, population, weight)
                       VALUES(%s,%s,%s,%s,ST_GeomFromGeoJSON(%s),%s,%s,%s)""",
                    (
                        asset_id, "dudh-koshi-demo-01",
                        exp.get("asset_type", "other"),
                        exp.get("name", ""),
                        json.dumps(geom),
                        json.dumps(geom),
                        None, 1.0,
                    ),
                )

            self._conn.execute(
                """INSERT INTO exposures
                   (exposure_id, run_id, asset_id, distance_m, buffer_m, inundated)
                   VALUES(%s,%s,%s,%s,%s,%s)""",
                (
                    exp_id, run_id, asset_id,
                    exp.get("distance_m"), exp.get("buffer_m"),
                    bool(exp.get("inundated", False)),
                ),
            )
        self._conn.commit()

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC"
        ).fetchall()
        return [self._run_row(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id=%s", (run_id,)
        ).fetchone()
        return None if row is None else self._run_row(row)

    def _run_row(self, row: dict[str, Any]) -> dict[str, Any]:
        score_row = self._conn.execute(
            "SELECT * FROM scores WHERE run_id=%s", (row["run_id"],)
        ).fetchone()
        score: dict[str, Any] | None = None
        if score_row is not None:
            reasons = score_row["reasons"]
            if isinstance(reasons, str):
                reasons = json.loads(reasons)
            score = {
                "hazard_score": score_row["hazard_score"],
                "exposure_priority": score_row["exposure_priority"],
                "disease_risk": score_row["disease_risk"],
                "confidence": score_row["confidence"],
                "severity": score_row["severity"],
                "reasons": reasons,
            }
        review_row = self._conn.execute(
            """SELECT r.reviewer, r.decision, r.decided_at FROM reviews r
               JOIN scores s ON r.score_id=s.score_id
               WHERE s.run_id=%s ORDER BY r.decided_at DESC, r.review_id DESC LIMIT 1""",
            (row["run_id"],),
        ).fetchone()
        corridor = row["corridor_geojson"]
        if isinstance(corridor, str):
            corridor = json.loads(corridor)
        change_stats = row["change_stats"]
        if isinstance(change_stats, str):
            change_stats = json.loads(change_stats)
        return {
            "run_id": row["run_id"],
            "observation_id": row["observation_id"],
            "processing_version": row["processing_version"],
            "change_mask_uri": row["change_mask_uri"],
            "corridor_geojson": corridor,
            "change_stats_json": change_stats,
            "score": score,
            "status": "processed" if row["finished_at"] else "running",
            "started_at": row["started_at"].isoformat() if hasattr(row["started_at"], "isoformat") else row["started_at"],
            "finished_at": row["finished_at"].isoformat() if row["finished_at"] and hasattr(row["finished_at"], "isoformat") else row["finished_at"],
            "decision": review_row["decision"] if review_row else None,
            "reviewer": review_row["reviewer"] if review_row else None,
            "decided_at": review_row["decided_at"].isoformat() if review_row and hasattr(review_row["decided_at"], "isoformat") else (review_row["decided_at"] if review_row else None),
        }

    # --- exposures ---

    def list_exposures(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT e.*, a.asset_type, a.name, a.population, a.geometry_geojson
               FROM exposures e JOIN assets a ON e.asset_id = a.asset_id
               WHERE e.run_id=%s ORDER BY e.exposure_id""",
            (run_id,),
        ).fetchall()
        out = []
        for row in rows:
            geom = row["geometry_geojson"]
            if isinstance(geom, str):
                geom = json.loads(geom)
            out.append({
                "asset_id": row["asset_id"],
                "asset_type": row["asset_type"],
                "name": row["name"],
                "distance_m": row["distance_m"],
                "buffer_m": row["buffer_m"],
                "inundated": bool(row["inundated"]),
                "population": row["population"],
                "geometry_geojson": geom,
            })
        return out

    # --- reviews (human-in-the-loop gate) ---

    def get_score_for_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM scores WHERE run_id=%s", (run_id,)
        ).fetchone()
        if row is None:
            return None
        reasons = row["reasons"]
        if isinstance(reasons, str):
            reasons = json.loads(reasons)
        return {
            "score_id": row["score_id"],
            "run_id": row["run_id"],
            "hazard_score": row["hazard_score"],
            "exposure_priority": row["exposure_priority"],
            "disease_risk": row["disease_risk"],
            "confidence": row["confidence"],
            "severity": row["severity"],
            "reasons": reasons,
        }

    def create_review(self, run_id: str, reviewer: str, decision: str, note: str | None) -> dict[str, Any]:
        score = self.get_score_for_run(run_id)
        if score is None:
            from siren.db.repo import NotFoundError
            raise NotFoundError(f"no score for run {run_id}")
        count = self._conn.execute("SELECT COUNT(*) AS c FROM reviews").fetchone()["c"]
        review_id = f"rev-{count + 1:04d}"
        decided_at = _utcnow_iso()
        self._conn.execute(
            """INSERT INTO reviews(review_id, score_id, reviewer, decision, note)
            VALUES(%s,%s,%s,%s,%s)""",
            (review_id, score["score_id"], reviewer, decision, note),
        )
        if decision == "escalate":
            self._conn.execute(
                "UPDATE scores SET severity = 'elevated' WHERE score_id = %s",
                (score["score_id"],),
            )
        self._audit(None, reviewer, "review", {"run_id": run_id, "decision": decision, "note": note})
        self._conn.commit()
        return {
            "review_id": review_id,
            "score_id": score["score_id"],
            "reviewer": reviewer,
            "decision": decision,
            "decided_at": decided_at,
        }

    def _confirm_review_for_run(self, run_id: str) -> str | None:
        row = self._conn.execute(
            """SELECT r.review_id, r.decision FROM reviews r
               JOIN scores s ON r.score_id = s.score_id
               WHERE s.run_id=%s
               ORDER BY r.decided_at DESC, r.review_id DESC LIMIT 1""",
            (run_id,),
        ).fetchone()
        if row is None or row["decision"] != "confirm":
            return None
        return row["review_id"]

    # --- dispatches (human gate enforced by PL/pgSQL trigger, NOT bypassed) ---

    def create_dispatch(self, run_id: str, channel: str, recipient_group: str) -> dict[str, Any]:
        from siren.db.repo import HumanGateError, NotFoundError

        score = self.get_score_for_run(run_id)
        if score is None:
            raise NotFoundError(f"no score for run {run_id}")
        exposures = self.list_exposures(run_id)

        alert_id = f"alert-{zlib.crc32(run_id.encode('utf-8')) % 10000:04d}"

        exposed_pop = sum(e.get("population", 0) or 0 for e in exposures)
        critical_assets = [
            e["asset_id"] for e in exposures
            if e.get("asset_type") in ("bridge", "road") and e.get("inundated")
        ][:5]
        disease_flags = [
            e["asset_id"] for e in exposures
            if e.get("asset_type") == "well" and e.get("inundated")
        ][:3]
        med_act = "BOIL_WATER_NOW" if disease_flags else "MONITOR"

        alert = {
            "alert_id": alert_id,
            "geofence_id": recipient_group[-1].upper() if recipient_group else "B",
            "severity": score["severity"],
            "hazard_type": "GLOF_FL",
            "exposed_population": exposed_pop,
            "critical_assets": critical_assets,
            "disease_flags": [med_act],
        }

        from siren.alerting.codec import encode
        payload_bytes_obj = encode(alert)
        payload = payload_bytes_obj.decode("utf-8")
        payload_bytes = len(payload_bytes_obj)

        review_id = self._confirm_review_for_run(run_id) or "rev-none"

        count = self._conn.execute("SELECT COUNT(*) AS c FROM dispatches").fetchone()["c"]
        dispatch_id = f"disp-{count + 1:04d}"
        sent_at = _utcnow_iso()
        try:
            self._conn.execute(
                """INSERT INTO dispatches
                (dispatch_id, review_id, alert_id, geofence_id, payload,
                 payload_bytes, channel, recipient_group, status)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    dispatch_id, review_id, alert_id, recipient_group, payload,
                    payload_bytes, channel, recipient_group, "sent",
                ),
            )
        except Exception as exc:
            # Human gate trigger aborted the insert (PRD §7.6) — do NOT bypass.
            self._conn.rollback()
            raise HumanGateError(
                f"dispatch requires a confirm review for run {run_id} (PRD §7.6)"
            ) from exc

        self._audit(alert_id, "coordinator-01", "dispatch", {"run_id": run_id, "channel": channel, "recipient_group": recipient_group})
        self._conn.commit()
        return {
            "dispatch_id": dispatch_id,
            "alert_id": alert_id,
            "geofence_id": recipient_group,
            "payload": payload,
            "payload_bytes": payload_bytes,
            "channel": channel,
            "status": "sent",
            "sent_at": sent_at,
        }

    # --- acquisition jobs (ADR-008) ---

    def create_acquisition_job(
        self,
        source: str,
        provider_product_id: str,
        download_url: str | None = None,
        acquired_at: str | None = None,
    ) -> dict[str, Any]:
        job_id = f"acq-{zlib.crc32(f'{source}:{provider_product_id}'.encode()) % 100000:05d}"
        self._conn.execute(
            """INSERT INTO acquisition_jobs
               (job_id, source, provider_product_id, status, download_url, acquired_at)
               VALUES(%s,%s,%s,%s,%s,%s)
               ON CONFLICT(source, provider_product_id) DO UPDATE SET
               download_url=excluded.download_url,
               acquired_at=excluded.acquired_at""",
            (job_id, source, provider_product_id, "pending", download_url, acquired_at),
        )
        self._conn.commit()
        return self.get_acquisition_job(job_id) or {
            "job_id": job_id, "source": source, "provider_product_id": provider_product_id,
            "status": "pending",
        }

    def update_acquisition_job(
        self,
        job_id: str,
        status: str,
        local_path: str | None = None,
        last_error: str | None = None,
    ) -> None:
        self._conn.execute(
            """UPDATE acquisition_jobs
               SET status=%s, local_path=COALESCE(%s, local_path),
                   last_error=%s, attempts=attempts+1
               WHERE job_id=%s""",
            (status, local_path, last_error, job_id),
        )
        self._conn.commit()

    def get_acquisition_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM acquisition_jobs WHERE job_id=%s", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return self._acquisition_job_row(row)

    def find_acquisition_job(self, source: str, provider_product_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM acquisition_jobs WHERE source=%s AND provider_product_id=%s",
            (source, provider_product_id),
        ).fetchone()
        return None if row is None else self._acquisition_job_row(row)

    def list_acquisition_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        if status is not None:
            rows = self._conn.execute(
                "SELECT * FROM acquisition_jobs WHERE status=%s ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM acquisition_jobs ORDER BY created_at DESC"
            ).fetchall()
        return [self._acquisition_job_row(row) for row in rows]

    @staticmethod
    def _acquisition_job_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "source": row["source"],
            "provider_product_id": row["provider_product_id"],
            "status": row["status"],
            "download_url": row["download_url"],
            "local_path": row["local_path"],
            "acquired_at": row["acquired_at"].isoformat() if row["acquired_at"] and hasattr(row["acquired_at"], "isoformat") else row["acquired_at"],
            "attempts": row["attempts"],
            "last_error": row["last_error"],
            "created_at": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else row["created_at"],
            "updated_at": row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else row["updated_at"],
        }

    def register_observation(
        self,
        observation_id: str,
        basin_id: str,
        acquired_at: str,
        source: str,
        raster_uri: str,
        cloud_fraction: float = 0.0,
        optical_cloud_fraction: float = 0.0,
        alignment_ok: bool = True,
        usable: bool = True,
        quality_score: float = 0.9,
        confidence_adjustment: float = 1.0,
        processing_version: str = "0.1.0",
        water_area_km2: float | None = None,
        water_area_change_percent: float | None = None,
        rainfall_24h_mm: float | None = None,
        rainfall_7d_mm: float | None = None,
        temp_mean_c: float | None = None,
        temp_index: float | None = None,
        mean_slope_degrees: float | None = None,
    ) -> dict[str, Any]:
        self._conn.execute(
            """INSERT INTO observations
            (observation_id, basin_id, acquired_at, source, raster_uri, crs,
             quality_score, cloud_fraction, optical_cloud_fraction, alignment_ok, usable, confidence_adjustment,
             water_area_km2, water_area_change_percent, rainfall_24h_mm, rainfall_7d_mm,
             temp_mean_c, temp_index,
             mean_slope_degrees, processing_version, status)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(observation_id) DO UPDATE SET
            acquired_at=excluded.acquired_at, source=excluded.source,
            raster_uri=excluded.raster_uri,
            cloud_fraction=excluded.cloud_fraction,
            optical_cloud_fraction=excluded.optical_cloud_fraction,
            alignment_ok=excluded.alignment_ok, usable=excluded.usable,
            confidence_adjustment=excluded.confidence_adjustment,
            quality_score=excluded.quality_score,
            water_area_km2=excluded.water_area_km2,
            water_area_change_percent=excluded.water_area_change_percent,
            rainfall_24h_mm=excluded.rainfall_24h_mm,
            rainfall_7d_mm=excluded.rainfall_7d_mm,
            temp_mean_c=excluded.temp_mean_c,
            temp_index=excluded.temp_index,
            mean_slope_degrees=excluded.mean_slope_degrees,
            processing_version=excluded.processing_version, status=excluded.status""",
            (
                observation_id, basin_id, acquired_at, source, raster_uri, "EPSG:4326",
                quality_score, cloud_fraction, optical_cloud_fraction,
                alignment_ok, usable, confidence_adjustment,
                water_area_km2, water_area_change_percent, rainfall_24h_mm, rainfall_7d_mm,
                temp_mean_c, temp_index,
                mean_slope_degrees, processing_version, "ingested",
            ),
        )
        self._conn.commit()
        return self.get_observation(observation_id) or {}

    # --- audit (append-only: INSERT + SELECT only; no update/delete methods) ---

    def _audit(self, alert_id: str | None, actor: str, action: str, detail: dict[str, Any]) -> None:
        detail_json = json.dumps(detail, sort_keys=True, separators=(",", ":"), default=str)
        created_at = _utcnow_iso()
        row = self._conn.execute(
            "SELECT event_hash FROM audit_log ORDER BY entry_id DESC LIMIT 1"
        ).fetchone()
        prev_hash = row["event_hash"] if row and row["event_hash"] else GENESIS_HASH
        digest = event_hash(prev_hash, created_at, detail_json)
        self._conn.execute(
            """INSERT INTO audit_log
               (alert_id, actor, action, detail, created_at, prev_hash, event_hash)
               VALUES(%s,%s,%s,%s,%s,%s,%s)""",
            (alert_id, actor, action, json.dumps(detail, sort_keys=True, separators=(",", ":"), default=str), created_at, prev_hash, digest),
        )
        self._conn.commit()

    def list_audit(self, alert_id: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT entry_id, alert_id, actor, action, detail, created_at,
                      prev_hash, event_hash
               FROM audit_log ORDER BY entry_id"""
        ).fetchall()
        entries = []
        for row in rows:
            detail = row["detail"]
            if isinstance(detail, str):
                detail = json.loads(detail)
            entries.append({
                "entry_id": row["entry_id"],
                "alert_id": row["alert_id"],
                "actor": row["actor"],
                "action": row["action"],
                "detail_json": json.dumps(detail) if isinstance(detail, dict) else detail,
                "created_at": row["created_at"].strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(row["created_at"], "strftime") else row["created_at"],
                "prev_hash": row["prev_hash"],
                "event_hash": row["event_hash"],
            })
        if alert_id is not None:
            entries = [e for e in entries if e["alert_id"] == alert_id]
        if run_id is not None:
            entries = [e for e in entries if f'"{run_id}"' in (e["detail_json"] or "")]
        return entries

    def verify_hash_chain(self) -> bool:
        rows = self._conn.execute(
            "SELECT prev_hash, event_hash, created_at, detail FROM audit_log ORDER BY entry_id"
        ).fetchall()
        expected_prev = GENESIS_HASH
        for row in rows:
            if row["prev_hash"] != expected_prev:
                return False
            detail_str = row["detail"]
            if isinstance(detail_str, dict):
                detail_str = json.dumps(detail_str, sort_keys=True, separators=(",", ":"), default=str)
            # Normalize created_at to match _utcnow_iso() format: "%Y-%m-%dT%H:%M:%SZ"
            # PostgreSQL returns a datetime object; isoformat() gives +00:00, not Z.
            created_at = row["created_at"]
            if hasattr(created_at, "strftime"):
                created_at = created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            elif hasattr(created_at, "isoformat"):
                created_at = created_at.isoformat()
            recomputed = event_hash(row["prev_hash"], created_at, detail_str)
            if recomputed != row["event_hash"]:
                return False
            expected_prev = row["event_hash"]
        return True
