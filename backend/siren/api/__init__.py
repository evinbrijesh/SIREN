"""SIREN HTTP surface — FastAPI app and Pydantic models.

Exposes the `app` ASGI application so `uvicorn siren.api:app` boots.
"""

from siren.api.app import app, create_app

__all__ = ["app", "create_app"]
