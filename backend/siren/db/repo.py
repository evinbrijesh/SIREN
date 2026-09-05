"""SQLite repository for SIREN.

Executes schema.sql on startup, seeds demo basins + observations, and exposes
per-table INSERT/SELECT methods. The audit_log table exposes ONLY insert +
select (append-only, enforced by schema triggers — no update/delete methods
exist). Dispatches are gated by the `dispatches_require_confirm` /
`dispatches_require_existing_review` triggers (human gate, PRD §7.6): the
repository attempts the insert and lets the trigger abort -> 409; it does NOT
bypass the trigger.

IMPORTANT (schema.sql W7): `PRAGMA foreign_keys = ON` is PER-CONNECTION in
SQLite. The repository holds a single persistent connection and executes the
pragma once on creation, satisfying the requirement for every connection used.
"""

from __future__ import annotations

import json
import os
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from siren.audit.hash_chain import GENESIS_HASH, event_hash

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# --- Deterministic demo/fixture data (matches API_CONTRACT.md examples) ---


def _project_root() -> Path:
    env = os.environ.get("SIREN_PROJECT_ROOT")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "assets").is_dir():
            return parent
    return Path(__file__).resolve().parents[3]


def _basin_boundary() -> dict[str, Any]:
    path = _project_root() / "data" / "assets" / "dudh_koshi_aoi.geojson"
    if not path.exists():
        return {"type": "Polygon", "coordinates": []}
    data = json.loads(path.read_text())
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        return features[0]["geometry"] if features else {"type": "Polygon", "coordinates": []}
    if data.get("type") == "Feature":
        return data["geometry"]
    return data


DEMO_BASIN = {
    "basin_id": "dudh-koshi-demo-01",
    "name": "Dudh Koshi / Imja",
    "boundary_geojson": _basin_boundary(),
    "crs": "EPSG:4326",
}

DEMO_OBSERVATIONS = [
    {
        "observation_id": "obs-001",
        "basin_id": "dudh-koshi-demo-01",
        "acquired_at": "2026-07-23T12:00:00Z",
        "source": "sentinel-1-grd-nrt",
        "raster_uri": "data/processed/obs-001.tif",
        "crs": "EPSG:4326",
        "quality_score": 0.94,
        "cloud_fraction": 0.0,
        "optical_cloud_fraction": 0.0,
        "alignment_ok": True,
        "usable": True,
        "confidence_adjustment": 1.0,
        "water_area_km2": 3.2,
        "water_area_change_percent": 8.0,
        "rainfall_24h_mm": 18.2,
        "rainfall_7d_mm": 64.0,
        "mean_slope_degrees": 31.0,
        "processing_version": "0.1.0",
        "status": "processed",
    },
    {
        "observation_id": "obs-002",
        "basin_id": "dudh-koshi-demo-01",
        "acquired_at": "2026-08-04T12:00:00Z",
        "source": "sentinel-1-grd-nrt",
        "raster_uri": "data/processed/obs-002.tif",
        "crs": "EPSG:4326",
        "quality_score": 0.90,
        "cloud_fraction": 0.0,
        "optical_cloud_fraction": 0.95,
        "alignment_ok": True,
        "usable": True,
        "confidence_adjustment": 1.0,
        "water_area_km2": 4.1,
        "water_area_change_percent": 28.0,
        "rainfall_24h_mm": 84.6,
        "rainfall_7d_mm": 192.4,
        "mean_slope_degrees": 31.0,
        "processing_version": "0.1.0",
        "status": "processed",
    },
    {
        "observation_id": "obs-003",
        "basin_id": "dudh-koshi-demo-01",
        "acquired_at": "2026-08-12T12:00:00Z",
        "source": "sentinel-1-grd-nrt",
        "raster_uri": "data/processed/obs-003.tif",
        "crs": "EPSG:4326",
        "quality_score": 0.88,
        "cloud_fraction": 0.0,
        "optical_cloud_fraction": 0.90,
        "alignment_ok": True,
        "usable": True,
        "confidence_adjustment": 0.95,
        "water_area_km2": 4.3,
        "water_area_change_percent": 43.0,
        "rainfall_24h_mm": 60.0,
        "rainfall_7d_mm": 160.0,
        "mean_slope_degrees": 31.0,
        "processing_version": "0.1.0",
        "status": "processed",
    },
]

DEMO_ASSETS = [
    {
        "asset_id": "village-2",
        "basin_id": "dudh-koshi-demo-01",
        "asset_type": "village",
        "name": "Chhukung",
        "geometry_geojson": {"type": "Point", "coordinates": [86.86, 27.90]},
        "population": 1240,
        "weight": 1.0,
    },
    {
        "asset_id": "BR-12",
        "basin_id": "dudh-koshi-demo-01",
        "asset_type": "bridge",
        "name": "Bridge 12",
        "geometry_geojson": {"type": "Point", "coordinates": [86.85, 27.91]},
        "population": None,
        "weight": 1.5,
    },
    {
        "asset_id": "RD-4",
        "basin_id": "dudh-koshi-demo-01",
        "asset_type": "road",
        "name": "Road 4",
        "geometry_geojson": {"type": "LineString", "coordinates": [[86.84, 27.90], [86.87, 27.92]]},
        "population": None,
        "weight": 1.2,
    },
    {
        "asset_id": "well-3",
        "basin_id": "dudh-koshi-demo-01",
        "asset_type": "well",
        "name": "Well 3",
        "geometry_geojson": {"type": "Point", "coordinates": [86.86, 27.89]},
        "population": 1240,
        "weight": 1.0,
    },
]

DEMO_RUN = {
    "run_id": "run-0001",
    "observation_id": "obs-003",
    "processing_version": "0.1.0",
    "change_mask_uri": "data/processed/obs-003_expansion_mask.tif",
    "corridor_geojson": {"type": "LineString", "coordinates": []},
    "change_stats_json": {
        "water_area_km2": 4.3,
        "expansion_percent": 43.0,
        "rainfall_24h_mm": 60.0,
        "rainfall_7d_mm": 160.0,
        "source": "sentinel-1-grd-nrt",
        "routing": {"path": "sar", "sar_primary": True, "cloud_fraction_reported": 0.9},
    },
}

DEMO_SCORE = {
    "score_id": "score-0001",
    "run_id": "run-0001",
    "hazard_score": 0.84,
    "exposure_priority": 0.51,
    "disease_risk": 0.08,
    "confidence": 0.90,
    "severity": "critical",
    "reasons": [
        "Water area expanded 43.0% vs baseline",
        "60.0 mm rainfall in 24h increases runoff pressure",
        "Optical cloud 90% triggered the Sentinel-1 SAR path",
    ],
}

DEMO_EXPOSURES = [
    {"exposure_id": "exp-001", "run_id": "run-0001", "asset_id": "village-2", "distance_m": 210.0, "buffer_m": 100.0, "inundated": False},
    {"exposure_id": "exp-002", "run_id": "run-0001", "asset_id": "BR-12", "distance_m": 60.0, "buffer_m": 75.0, "inundated": False},
    {"exposure_id": "exp-003", "run_id": "run-0001", "asset_id": "RD-4", "distance_m": 40.0, "buffer_m": 50.0, "inundated": False},
    {"exposure_id": "exp-004", "run_id": "run-0001", "asset_id": "well-3", "distance_m": 90.0, "buffer_m": 100.0, "inundated": True},
]


class HumanGateError(Exception):
    """Raised when a dispatch is attempted without a confirm review (PRD §7.6)."""


class NotFoundError(Exception):
    """Raised when a requested resource does not exist."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bool(v: int | None) -> bool | None:
    if v is None:
        return None
    return bool(v)


class Repository:
    """SQLite repository backed by a single persistent connection.

    The connection executes `PRAGMA foreign_keys = ON` on creation (schema W7).
    Works for both file paths and `:memory:` (re-seeds on every construction).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection = sqlite3.connect(
            self.db_path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")  # per-connection (schema W7)
        self._init_schema_and_seed()

    def close(self) -> None:
        self._conn.close()

    # --- schema + seed ---

    def _init_schema_and_seed(self) -> None:
        self._conn.executescript(SCHEMA_PATH.read_text())
        self._migrate_schema()
        self._seed()
        self._conn.commit()

    def _migrate_schema(self) -> None:
        observation_columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(observations)").fetchall()
        }
        if "optical_cloud_fraction" not in observation_columns:
            self._conn.execute("ALTER TABLE observations ADD COLUMN optical_cloud_fraction REAL")
        audit_columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(audit_log)").fetchall()
        }
        if "prev_hash" not in audit_columns:
            self._conn.execute("ALTER TABLE audit_log ADD COLUMN prev_hash TEXT NOT NULL DEFAULT ''")
        if "event_hash" not in audit_columns:
            self._conn.execute("ALTER TABLE audit_log ADD COLUMN event_hash TEXT NOT NULL DEFAULT ''")

    def _seed(self) -> None:
        self._conn.execute(
            """INSERT INTO basins(basin_id, name, boundary_geojson, crs) VALUES(?,?,?,?)
               ON CONFLICT(basin_id) DO UPDATE SET
               name=excluded.name, boundary_geojson=excluded.boundary_geojson, crs=excluded.crs""",
            (
                DEMO_BASIN["basin_id"],
                DEMO_BASIN["name"],
                json.dumps(DEMO_BASIN["boundary_geojson"]),
                DEMO_BASIN["crs"],
            ),
        )

        for o in DEMO_OBSERVATIONS:
            self._conn.execute(
                """INSERT INTO observations
                (observation_id, basin_id, acquired_at, source, raster_uri, crs,
                 quality_score, cloud_fraction, optical_cloud_fraction, alignment_ok, usable, confidence_adjustment,
                 water_area_km2, water_area_change_percent, rainfall_24h_mm, rainfall_7d_mm,
                 mean_slope_degrees, processing_version, status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(observation_id) DO UPDATE SET
                acquired_at=excluded.acquired_at, source=excluded.source,
                raster_uri=excluded.raster_uri, crs=excluded.crs,
                quality_score=excluded.quality_score, cloud_fraction=excluded.cloud_fraction,
                optical_cloud_fraction=excluded.optical_cloud_fraction,
                alignment_ok=excluded.alignment_ok, usable=excluded.usable,
                confidence_adjustment=excluded.confidence_adjustment,
                water_area_km2=excluded.water_area_km2,
                water_area_change_percent=excluded.water_area_change_percent,
                rainfall_24h_mm=excluded.rainfall_24h_mm,
                rainfall_7d_mm=excluded.rainfall_7d_mm,
                mean_slope_degrees=excluded.mean_slope_degrees,
                processing_version=excluded.processing_version, status=excluded.status""",
                (
                    o["observation_id"], o["basin_id"], o["acquired_at"], o["source"],
                    o["raster_uri"], o["crs"], o["quality_score"], o["cloud_fraction"],
                    o["optical_cloud_fraction"], int(o["alignment_ok"]), int(o["usable"]), o["confidence_adjustment"],
                    o["water_area_km2"], o["water_area_change_percent"], o["rainfall_24h_mm"],
                    o["rainfall_7d_mm"], o["mean_slope_degrees"], o["processing_version"],
                    o["status"],
                ),
            )

        for a in DEMO_ASSETS:
            self._conn.execute(
                """INSERT INTO assets
                (asset_id, basin_id, asset_type, name, geometry_geojson, population, weight)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(asset_id) DO UPDATE SET
                asset_type=excluded.asset_type, name=excluded.name,
                geometry_geojson=excluded.geometry_geojson,
                population=excluded.population, weight=excluded.weight""",
                (
                    a["asset_id"], a["basin_id"], a["asset_type"], a["name"],
                    json.dumps(a["geometry_geojson"]), a["population"], a["weight"],
                ),
            )

        if self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0:
            r = DEMO_RUN
            self._conn.execute(
                """INSERT INTO runs
                (run_id, observation_id, processing_version, change_mask_uri,
                 corridor_geojson, change_stats_json, finished_at)
                VALUES(?,?,?,?,?,?,?)""",
                (
                    r["run_id"], r["observation_id"], r["processing_version"],
                    r["change_mask_uri"], json.dumps(r["corridor_geojson"]),
                    json.dumps(r["change_stats_json"]), _utcnow_iso(),
                ),
            )

        if self._conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0:
            s = DEMO_SCORE
            self._conn.execute(
                """INSERT INTO scores
                (score_id, run_id, hazard_score, exposure_priority, disease_risk,
                 confidence, severity, reasons_json)
                VALUES(?,?,?,?,?,?,?,?)""",
                (
                    s["score_id"], s["run_id"], s["hazard_score"], s["exposure_priority"],
                    s["disease_risk"], s["confidence"], s["severity"],
                    json.dumps(s["reasons"]),
                ),
            )

        if self._conn.execute("SELECT COUNT(*) FROM exposures").fetchone()[0] == 0:
            for e in DEMO_EXPOSURES:
                self._conn.execute(
                    """INSERT INTO exposures
                    (exposure_id, run_id, asset_id, distance_m, buffer_m, inundated)
                    VALUES(?,?,?,?,?,?)""",
                    (
                        e["exposure_id"], e["run_id"], e["asset_id"], e["distance_m"],
                        e["buffer_m"], int(e["inundated"]),
                    ),
                )

    # --- basins ---

    def get_basin(self, basin_id: str = "dudh-koshi-demo-01") -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM basins WHERE basin_id=?", (basin_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "basin_id": row["basin_id"],
            "name": row["name"],
            "boundary_geojson": json.loads(row["boundary_geojson"]),
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
            "SELECT * FROM observations WHERE observation_id=?", (observation_id,)
        ).fetchone()
        return None if row is None else self._observation_row(row)

    @staticmethod
    def _observation_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "observation_id": row["observation_id"],
            "basin_id": row["basin_id"],
            "acquired_at": row["acquired_at"],
            "source": row["source"],
            "raster_uri": row["raster_uri"],
            "crs": row["crs"],
            "quality_score": row["quality_score"],
            "cloud_fraction": row["cloud_fraction"],
            "optical_cloud_fraction": row["optical_cloud_fraction"],
            "alignment_ok": _bool(row["alignment_ok"]),
            "usable": _bool(row["usable"]),
            "confidence_adjustment": row["confidence_adjustment"],
            "water_area_km2": row["water_area_km2"],
            "water_area_change_percent": row["water_area_change_percent"],
            "rainfall_24h_mm": row["rainfall_24h_mm"],
            "rainfall_7d_mm": row["rainfall_7d_mm"],
            "mean_slope_degrees": row["mean_slope_degrees"],
            "processing_version": row["processing_version"],
            "status": row["status"],
        }

    # --- runs ---

    def run_exists(self, run_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            is not None
        )

    def create_run(self, observation_id: str, processing_version: str = "0.1.0") -> dict[str, Any]:
        count = self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        run_id = f"run-{count + 1:04d}"
        started_at = _utcnow_iso()
        self._conn.execute(
            "INSERT INTO runs(run_id, observation_id, processing_version) VALUES(?,?,?)",
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
        """Update a run with pipeline results (mask, corridor, stats).

        Only audit_log has UPDATE triggers; runs is updatable.
        """
        self._conn.execute(
            """UPDATE runs
               SET change_mask_uri=?, corridor_geojson=?, change_stats_json=?,
                   finished_at=?
               WHERE run_id=?""",
            (
                change_mask_uri,
                json.dumps(corridor_geojson),
                json.dumps(change_stats_json),
                _utcnow_iso(),
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
        """Insert a score row for a run. Returns the score_id."""
        count = self._conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        score_id = f"score-{count + 1:04d}"
        self._conn.execute(
            """INSERT INTO scores
               (score_id, run_id, hazard_score, exposure_priority,
                disease_risk, confidence, severity, reasons_json)
               VALUES(?,?,?,?,?,?,?,?)""",
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
        """Insert exposure rows for a run."""
        for i, exp in enumerate(exposures):
            exp_id = f"exp-{run_id}-{i + 1:03d}"
            self._conn.execute(
                """INSERT INTO exposures
                   (exposure_id, run_id, asset_id, distance_m, buffer_m, inundated)
                   VALUES(?,?,?,?,?,?)""",
                (
                    exp_id, run_id, exp["asset_id"],
                    exp.get("distance_m"), exp.get("buffer_m"),
                    bool(exp.get("inundated", False)),
                ),
            )
        self._conn.commit()

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC"
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(self._run_row(row))
        return out

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return None if row is None else self._run_row(row)

    def _run_row(self, row: sqlite3.Row) -> dict[str, Any]:
        score_row = self._conn.execute(
            "SELECT * FROM scores WHERE run_id=?", (row["run_id"],)
        ).fetchone()
        score: dict[str, Any] | None = None
        if score_row is not None:
            score = {
                "hazard_score": score_row["hazard_score"],
                "exposure_priority": score_row["exposure_priority"],
                "disease_risk": score_row["disease_risk"],
                "confidence": score_row["confidence"],
                "severity": score_row["severity"],
                "reasons": json.loads(score_row["reasons_json"]),
            }
        review_row = self._conn.execute(
            """SELECT r.reviewer, r.decision, r.decided_at FROM reviews r
               JOIN scores s ON r.score_id=s.score_id
               WHERE s.run_id=? ORDER BY r.decided_at DESC, r.review_id DESC LIMIT 1""",
            (row["run_id"],),
        ).fetchone()
        return {
            "run_id": row["run_id"],
            "observation_id": row["observation_id"],
            "processing_version": row["processing_version"],
            "change_mask_uri": row["change_mask_uri"],
            "corridor_geojson": json.loads(row["corridor_geojson"]) if row["corridor_geojson"] else None,
            "change_stats_json": json.loads(row["change_stats_json"]) if row["change_stats_json"] else None,
            "score": score,
            "status": "processed" if row["finished_at"] else "running",
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "decision": review_row["decision"] if review_row else None,
            "reviewer": review_row["reviewer"] if review_row else None,
            "decided_at": review_row["decided_at"] if review_row else None,
        }

    # --- exposures ---

    def list_exposures(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT e.*, a.asset_type, a.name, a.population, a.geometry_geojson
               FROM exposures e JOIN assets a ON e.asset_id = a.asset_id
               WHERE e.run_id=? ORDER BY e.exposure_id""",
            (run_id,),
        ).fetchall()
        return [
            {
                "asset_id": row["asset_id"],
                "asset_type": row["asset_type"],
                "name": row["name"],
                "distance_m": row["distance_m"],
                "buffer_m": row["buffer_m"],
                "inundated": bool(row["inundated"]),
                "population": row["population"],
                "geometry_geojson": json.loads(row["geometry_geojson"]),
            }
            for row in rows
        ]

    # --- reviews (human-in-the-loop gate) ---

    def get_score_for_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM scores WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "score_id": row["score_id"],
            "run_id": row["run_id"],
            "hazard_score": row["hazard_score"],
            "exposure_priority": row["exposure_priority"],
            "disease_risk": row["disease_risk"],
            "confidence": row["confidence"],
            "severity": row["severity"],
            "reasons": json.loads(row["reasons_json"]),
        }

    def create_review(self, run_id: str, reviewer: str, decision: str, note: str | None) -> dict[str, Any]:
        score = self.get_score_for_run(run_id)
        if score is None:
            raise NotFoundError(f"no score for run {run_id}")
        count = self._conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        review_id = f"rev-{count + 1:04d}"
        decided_at = _utcnow_iso()
        self._conn.execute(
            """INSERT INTO reviews(review_id, score_id, reviewer, decision, note)
            VALUES(?,?,?,?,?)""",
            (review_id, score["score_id"], reviewer, decision, note),
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
            """SELECT r.review_id FROM reviews r
               JOIN scores s ON r.score_id = s.score_id
               WHERE s.run_id=? AND r.decision='confirm'
               ORDER BY r.decided_at DESC LIMIT 1""",
            (run_id,),
        ).fetchone()
        return row["review_id"] if row is not None else None

    # --- dispatches (human gate enforced by schema trigger, NOT bypassed) ---

    def create_dispatch(self, run_id: str, channel: str, recipient_group: str) -> dict[str, Any]:
        score = self.get_score_for_run(run_id)
        if score is None:
            raise NotFoundError(f"no score for run {run_id}")
        # deterministic alert_id (no unseeded randomness — Hard Rule 6)
        alert_id = f"alert-{zlib.crc32(run_id.encode('utf-8')) % 10000:04d}"
        # <250-byte resilient payload (PRD §10.4)
        payload_obj = {
            "aid": "siren-04",
            "sec": recipient_group[-1].upper() if recipient_group else "B",
            "haz": "GLOF_FL",
            "lvl": 3,
            "exp_pop": 1240,
            "crit": ["BR-12", "RD-4"],
            "med_act": "BOIL_WATER_NOW",
        }
        payload = json.dumps(payload_obj, separators=(",", ":"))
        payload_bytes = len(payload.encode("utf-8"))

        # Use the confirm review if one exists; otherwise attempt the insert with
        # a non-existent review_id and let the schema trigger abort -> 409.
        review_id = self._confirm_review_for_run(run_id) or "rev-none"

        count = self._conn.execute("SELECT COUNT(*) FROM dispatches").fetchone()[0]
        dispatch_id = f"disp-{count + 1:04d}"
        sent_at = _utcnow_iso()
        try:
            self._conn.execute(
                """INSERT INTO dispatches
                (dispatch_id, review_id, alert_id, geofence_id, payload,
                 payload_bytes, channel, recipient_group, status)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    dispatch_id, review_id, alert_id, recipient_group, payload,
                    payload_bytes, channel, recipient_group, "sent",
                ),
            )
        except sqlite3.IntegrityError as exc:
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

    # --- audit (append-only: INSERT + SELECT only; no update/delete methods) ---

    def _audit(self, alert_id: str | None, actor: str, action: str, detail: dict[str, Any]) -> None:
        detail_json = json.dumps(detail, sort_keys=True, separators=(",", ":"), default=str)
        created_at = _utcnow_iso()
        existing = self._audit_rows_with_hashes()
        prev_hash = existing[-1]["event_hash"] if existing else GENESIS_HASH
        digest = event_hash(prev_hash, created_at, detail_json)
        self._conn.execute(
            """INSERT INTO audit_log
               (alert_id, actor, action, detail_json, created_at, prev_hash, event_hash)
               VALUES(?,?,?,?,?,?,?)""",
            (alert_id, actor, action, detail_json, created_at, prev_hash, digest),
        )

    def _audit_rows_with_hashes(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM audit_log ORDER BY entry_id").fetchall()
        entries: list[dict[str, Any]] = []
        previous = GENESIS_HASH
        for row in rows:
            digest = event_hash(previous, row["created_at"], row["detail_json"])
            entries.append({
                "entry_id": row["entry_id"],
                "alert_id": row["alert_id"],
                "actor": row["actor"],
                "action": row["action"],
                "detail_json": row["detail_json"],
                "created_at": row["created_at"],
                "prev_hash": previous,
                "event_hash": digest,
            })
            previous = digest
        return entries

    def list_audit(self, alert_id: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        """Query audit log by alert_id and/or run_id.

        When run_id is provided, matches entries whose detail_json contains the
        run_id (the run/score/review entries carry run_id in detail_json, not in
        the alert_id column). When alert_id is provided, matches the alert_id
        column directly. Both filters are AND-ed when both are given.
        """
        entries = self._audit_rows_with_hashes()
        if alert_id is not None:
            entries = [entry for entry in entries if entry["alert_id"] == alert_id]
        if run_id is not None:
            entries = [entry for entry in entries if f'"{run_id}"' in entry["detail_json"]]
        return entries


def default_db_path() -> str:
    """Default DB path. `:memory:` re-seeds on every boot (offline demo)."""
    return os.environ.get("SIREN_DB_PATH", ":memory:")


_repo: Repository | None = None


def get_repository(db_path: str | None = None) -> Repository:
    global _repo
    if db_path is not None:
        return Repository(db_path)
    if _repo is None:
        _repo = Repository(default_db_path())
    return _repo


def init_db(db_path: str | None = None) -> Repository:
    return get_repository(db_path)
