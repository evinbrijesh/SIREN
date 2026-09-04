"""Fixture path helper for SIREN tests.

Usage:
    from tests.fixtures import fixture_path
    path = fixture_path("rasters/baseline.tif")
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent


def fixture_path(rel: str) -> Path:
    """Return an absolute path to a fixture file, e.g. fixture_path('rasters/baseline.tif')."""
    return FIXTURES_DIR / rel