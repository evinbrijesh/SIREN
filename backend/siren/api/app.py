"""FastAPI application for SIREN.

Routes are thin and delegate to the SQLite repository (siren.db.repo). Demo /
fixture data is seeded by the repository on construction. Implements the
endpoints defined in docs/API_CONTRACT.md exactly — field names/types are not
renamed.

Boot with:  uvicorn siren.api:app --port 8000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from siren.api import models
from siren.db.repo import HumanGateError, NotFoundError, Repository, default_db_path


def create_app(db_path: str | Path | None = None) -> FastAPI:
    repo = Repository(str(db_path) if db_path is not None else default_db_path())
    app = FastAPI(title="SIREN API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
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

    # POST /runs
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
        return repo.create_run(body.observation_id)

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

    # GET /audit?alert_id={alert_id}
    @app.get("/audit", response_model=models.AuditList)
    def list_audit(alert_id: str) -> Any:
        return {"entries": repo.list_audit(alert_id)}

    return app


# Module-level app so `uvicorn siren.api:app` boots.
app = create_app()
