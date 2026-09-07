"""Tests for Open-Meteo rainfall fetcher (ingest/open_meteo.py).

Tests the antecedent rainfall computation and temperature index logic
without requiring network access. Network-dependent functions are tested
with mocked responses.
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from siren.ingest.open_meteo import (
    compute_antecedent_rainfall,
    compute_temp_index,
    build_weather_series,
)


def test_compute_antecedent_rainfall_basic():
    """24h and 7d antecedent rainfall computed correctly."""
    dates = ["2026-07-16", "2026-07-17", "2026-07-18", "2026-07-19",
             "2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23"]
    precip = [10.0, 5.0, 0.0, 3.0, 8.0, 2.0, 15.0, 3.2]

    r24, r7 = compute_antecedent_rainfall(dates, precip, "2026-07-23")
    assert r24 == 3.2
    # 7d = sum of last 7 days (indices 1-7): 5+0+3+8+2+15+3.2 = 36.2
    assert abs(r7 - 36.2) < 0.01


def test_compute_antecedent_rainfall_handles_none():
    """None precipitation values are treated as 0."""
    dates = ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23"]
    precip = [None, 5.0, None, 3.0]

    r24, r7 = compute_antecedent_rainfall(dates, precip, "2026-07-23")
    assert r24 == 3.0
    # 7d = 5.0 + 0 + 3.0 = 8.0 (only 3 days available, not 7)
    assert r7 == 8.0


def test_compute_antecedent_rainfall_missing_date():
    """Returns (0, 0) when the target date is not in the list."""
    dates = ["2026-07-20", "2026-07-21"]
    precip = [5.0, 3.0]
    r24, r7 = compute_antecedent_rainfall(dates, precip, "2026-08-01")
    assert r24 == 0.0
    assert r7 == 0.0


def test_compute_temp_index_cold():
    """Cold temperatures map to low disease risk index."""
    assert compute_temp_index(2.0) == 0.3
    assert compute_temp_index(-5.0) == 0.3


def test_compute_temp_index_moderate():
    """Moderate temperatures map to mid-range index."""
    idx = compute_temp_index(10.0)
    assert 0.4 <= idx <= 0.6


def test_compute_temp_index_warm():
    """Warm temperatures map to higher index."""
    idx = compute_temp_index(20.0)
    assert 0.6 <= idx <= 0.8


def test_compute_temp_index_none():
    """None temperature returns neutral default."""
    assert compute_temp_index(None) == 0.5


def test_build_weather_series_with_mocked_network():
    """build_weather_series produces correct output with mocked API response."""
    mock_response = {
        "daily": {
            "time": ["2026-07-16", "2026-07-17", "2026-07-18", "2026-07-19",
                      "2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23",
                      "2026-07-24", "2026-07-25", "2026-07-26", "2026-07-27",
                      "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31",
                      "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04",
                      "2026-08-05", "2026-08-06", "2026-08-07", "2026-08-08",
                      "2026-08-09", "2026-08-10", "2026-08-11", "2026-08-12"],
            "precipitation_sum": [15.3, 10.4, 13.1, 7.5, 6.7, 4.9, 13.1, 3.2,
                                   2.1, 2.4, 12.5, 1.9, 5.5, 6.5, 10.4, 2.5,
                                   5.8, 3.1, 8.1, 12.1, 25.2, 9.5, 2.4, 2.4,
                                   9.2, 25.9, 8.2, 3.7],
            "temperature_2m_mean": [9.1, 9.9, 9.2, 8.1, 8.0, 8.1, 8.0, 8.5,
                                    8.7, 9.3, 9.5, 9.4, 9.4, 9.4, 8.1, 8.5,
                                    9.0, 9.3, 9.2, 9.3, 8.5, 9.6, 9.9, 9.6,
                                    9.9, 8.1, 9.4, 10.6],
        }
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_response).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = build_weather_series()

    assert result["basin_id"] == "dudh-koshi-demo-01"
    assert result["source"] == "open-meteo-era5"
    assert len(result["series"]) == 3

    # obs-001 (2026-07-23)
    obs1 = result["series"][0]
    assert obs1["observation_id"] == "obs-001"
    assert obs1["rainfall_24h_mm"] == 3.2
    # 7d = sum of 2026-07-17 through 2026-07-23
    # 10.4+13.1+7.5+6.7+4.9+13.1+3.2 = 58.9
    assert abs(obs1["rainfall_7d_mm"] - 58.9) < 0.1

    # obs-002 (2026-08-04)
    obs2 = result["series"][1]
    assert obs2["observation_id"] == "obs-002"
    assert obs2["rainfall_24h_mm"] == 12.1

    # obs-003 (2026-08-12)
    obs3 = result["series"][2]
    assert obs3["observation_id"] == "obs-003"
    assert obs3["rainfall_24h_mm"] == 3.7
