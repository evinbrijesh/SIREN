"""FastAPI application for SIREN.

Routes are thin and delegate to the SQLite repository (siren.db.repo) and
the pipeline orchestrator (siren.pipeline). Implements the endpoints
defined in docs/API_CONTRACT.md exactly — field names/types are not renamed.

POST /runs triggers the full pipeline (quality→route→detect→corridor→risk→DB)
synchronously and returns the completed run with scores and exposures.

Boot with:  uvicorn siren.api:app --port 8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


def _project_root() -> Path:
    """Resolve the project root directory.

    In Docker, SIREN_PROJECT_ROOT is set to /app. In dev, we walk up from
    this file (backend/siren/api/app.py → project root = parents[4]).
    """
    env = os.environ.get("SIREN_PROJECT_ROOT")
    if env:
        p = Path(env)
        if p.exists():
            return p
    # Dev: backend/siren/api/app.py → parents[4] = project root
    # Docker: /app/siren/api/__init__.py → parents[2] = /app
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "processed").is_dir() or (parent / "backend").is_dir():
            return parent
    return here.parents[4] if len(here.parents) > 4 else here.parent

from siren.api import models
from siren.db.repo import HumanGateError, NotFoundError, Repository, default_db_path


def create_app(db_path: str | Path | None = None) -> FastAPI:
    repo = Repository(str(db_path) if db_path is not None else default_db_path())
    app = FastAPI(title="SIREN API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:5175"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.repo = repo

    # Map HTTPException to the API_CONTRACT error shape: {"error", "detail"}.
    @app.exception_handler(HTTPException)
    async def _http_exc_handler(_: Any, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.detail["error"], "detail": exc.detail.get("detail", "")},
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "error", "detail": str(exc.detail)},
        )

    # GET /basin
    @app.get("/basin", response_model=models.BasinConfig)
    def get_basin() -> Any:
        basin = repo.get_basin()
        if basin is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": "no basin configured"},
            )
        # Attach basemap if it exists on disk
        basemap_json = _project_root() / "data" / "processed" / "basemap_bounds.json"
        if basemap_json.exists():
            try:
                bounds_data = json.loads(basemap_json.read_text())
                basin["basemap_uri"] = bounds_data.get("uri")
                basin["basemap_bounds"] = bounds_data.get("bounds")
            except Exception:
                pass
        return basin

    # GET /observations
    @app.get("/observations", response_model=models.ObservationList)
    def list_observations() -> Any:
        return {"observations": repo.list_observations()}

    # GET /observations/{observation_id}
    @app.get("/observations/{observation_id}", response_model=models.Observation)
    def get_observation(observation_id: str) -> Any:
        obs = repo.get_observation(observation_id)
        if obs is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"observation {observation_id} not found"},
            )
        return obs

    # POST /runs — triggers the full pipeline synchronously
    @app.post(
        "/runs",
        response_model=models.RunCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_run(body: models.RunCreateRequest) -> Any:
        if repo.get_observation(body.observation_id) is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"observation {body.observation_id} not found"},
            )
        # Run the full pipeline (quality→route→detect→corridor→risk→DB)
        from siren.pipeline import run_pipeline
        try:
            run = run_pipeline(body.observation_id, repo)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": "pipeline_error", "detail": str(exc)},
            )
        # started_at is set by the pipeline from the initial create_run
        started_at = run.get("started_at") if run else ""
        return {
            "run_id": run["run_id"],
            "observation_id": body.observation_id,
            "status": "processed",
            "started_at": started_at,
        }

    # GET /runs
    @app.get("/runs", response_model=models.RunList)
    def list_runs() -> Any:
        return {"runs": repo.list_runs()}

    # GET /runs/{run_id}/exposures
    @app.get("/runs/{run_id}/exposures", response_model=models.ExposureList)
    def list_exposures(run_id: str) -> Any:
        if not repo.run_exists(run_id):
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"run {run_id} not found"},
            )
        return {"exposures": repo.list_exposures(run_id)}

    # GET /runs/{run_id}/sar-priority — Search & Rescue priority ranking (PRD §15)
    @app.get("/runs/{run_id}/sar-priority", response_model=models.SarPriorityList)
    def get_sar_priority(run_id: str) -> Any:
        if not repo.run_exists(run_id):
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"run {run_id} not found"},
            )
        from siren.risk.sar_priority import compute_sar_priority
        exposures = repo.list_exposures(run_id)
        return compute_sar_priority(exposures)

    # GET /runs/{run_id}/ml-evidence — ML change detection evidence layer (ADR-002)
    @app.get("/runs/{run_id}/ml-evidence")
    def get_ml_evidence(run_id: str) -> Any:
        if not repo.run_exists(run_id):
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"run {run_id} not found"},
            )
        run = repo.get_run(run_id)
        if run is None or run.get("change_stats_json") is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"no change stats for run {run_id}"},
            )
        stats = run["change_stats_json"]
        obs_id = run["observation_id"]
        mask_uri = f"/data/processed/{obs_id}_expansion_mask.png"
        heatmap_uri = stats.get("heatmap_uri", f"/data/processed/{obs_id}_change_heatmap.png")
        baseline_uri = "/data/processed/baseline_water_mask.png"
        return {
            "run_id": run_id,
            "observation_id": obs_id,
            "ml_source": stats.get("ml_source", "deterministic_fallback"),
            "ml_confidence_mean": stats.get("ml_confidence_mean", 0.0),
            "ml_consensus_pixels": stats.get("ml_consensus_pixels", 0),
            "heatmap_uri": heatmap_uri,
            "heatmap_bounds": _image_bounds(heatmap_uri),
            "mask_uri": mask_uri,
            "mask_bounds": _image_bounds(mask_uri),
            "baseline_mask_uri": baseline_uri,
            "baseline_mask_bounds": _image_bounds(baseline_uri),
            "model_available": stats.get("ml_source", "deterministic_fallback") != "deterministic_fallback",
            "change_polygon": stats.get("change_polygon"),
        }

    # POST /runs/{run_id}/review
    @app.post("/runs/{run_id}/review", response_model=models.ReviewResponse)
    def create_review(run_id: str, body: models.ReviewRequest) -> Any:
        if not repo.run_exists(run_id):
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"run {run_id} not found"},
            )
        if body.decision not in ("confirm", "reject", "postpone"):
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid", "detail": "decision must be confirm|reject|postpone"},
            )
        try:
            return repo.create_review(run_id, body.reviewer, body.decision, body.note)
        except NotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": str(exc)},
            )

    # POST /runs/{run_id}/dispatch  (human gate: 409 without a confirm review)
    @app.post("/runs/{run_id}/dispatch", response_model=models.DispatchResponse)
    def create_dispatch(run_id: str, body: models.DispatchRequest) -> Any:
        if not repo.run_exists(run_id):
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "detail": f"run {run_id} not found"},
            )
        try:
            return repo.create_dispatch(run_id, body.channel, body.recipient_group)
        except HumanGateError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": "human_gate", "detail": str(exc)},
            )

    # GET /audit?alert_id={alert_id}&run_id={run_id}
    # Either parameter is optional; both are AND-ed when provided.
    @app.get("/audit", response_model=models.AuditList)
    def list_audit(alert_id: str | None = None, run_id: str | None = None) -> Any:
        return {"entries": repo.list_audit(alert_id=alert_id, run_id=run_id)}

    # POST /runs/process-all — run the full demo simulation (all observations)
    @app.post("/runs/process-all")
    def process_all() -> Any:
        from siren.pipeline import run_all_observations
        try:
            runs = run_all_observations(repo)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": "pipeline_error", "detail": str(exc)},
            )
        return {"runs": runs, "count": len(runs)}

    # Mount data/processed/ as static files so the frontend can fetch rasters
    data_processed = _project_root() / "data" / "processed"
    if data_processed.exists():
        app.mount("/data/processed", StaticFiles(directory=str(data_processed)), name="processed")

    return app


def _image_bounds(uri: str) -> list[list[float]] | None:
    """Return [[top-left], [top-right], [bottom-right], [bottom-left]] in [lon,lat].

    MapLibre image sources need the four corners in this order. The .png may not
    be georeferenced, so we look for the matching .tif and read its bounds, then
    convert to EPSG:4326 if necessary. For heatmaps (no matching .tif), fall back
    to the obs-NNN_expansion_mask.tif which shares the same extent.
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except Exception:
        return None

    png = _project_root() / uri.lstrip("/")
    tif = png.with_suffix(".tif")
    # Heatmap PNGs don't have a matching TIF; use the expansion_mask TIF instead
    if not tif.exists() and "change_heatmap" in png.name:
        tif = png.parent / png.name.replace("change_heatmap", "expansion_mask").replace(".png", ".tif")
    if not tif.exists():
        return None
    try:
        with rasterio.open(tif) as src:
            left, bottom, right, top = transform_bounds(
                src.crs, "EPSG:4326", *src.bounds
            )
        return [[left, top], [right, top], [right, bottom], [left, bottom]]
    except Exception:
        return None


# Module-level app so `uvicorn siren.api:app` boots.
app = create_app()
