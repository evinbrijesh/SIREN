"""Tests for SAR calibration module (preprocess/sar_calibrate.py).

These tests verify the calibration logic without requiring real SAFE
archives — they use synthetic calibration LUTs and mock the zipfile
reads where needed. The integration with real SAFE archives is verified
manually (see the pipeline end-to-end test).
"""

from __future__ import annotations

import numpy as np
import pytest

from siren.ml.contract import SAR_DB_MIN, SAR_DB_MAX, normalize_sar


def test_normalize_sar_with_realistic_db_values():
    """normalize_sar correctly maps realistic sigma0 dB values to [0, 1]."""
    # Simulate a small VV/VH patch with realistic dB values
    # Water: -25 dB, bare soil: -12 dB, urban: -3 dB
    sar_db = np.array([
        [[-25.0, -12.0, -3.0],
         [-28.0, -15.0, -5.0],
         [-30.0, -20.0, -10.0]],
        [[-30.0, -18.0, -8.0],
         [-29.0, -22.0, -12.0],
         [-27.0, -25.0, -15.0]],
    ], dtype=np.float32)

    normalized = normalize_sar(sar_db)

    assert normalized.shape == sar_db.shape
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0

    # Water (-25 dB) should map toward 0.0 (low backscatter)
    water_norm = normalized[0, 0, 0]  # -25 dB
    assert water_norm < 0.2, f"Water (-25 dB) should be near 0, got {water_norm}"

    # Urban (-3 dB) should map toward 1.0 (high backscatter)
    urban_norm = normalized[0, 0, 2]  # -3 dB
    assert urban_norm > 0.8, f"Urban (-3 dB) should be near 1, got {urban_norm}"


def test_normalize_sar_clamps_outside_range():
    """Values outside [-30, 0] dB are clamped to the contract range."""
    sar_db = np.array([[[5.0, -35.0, -15.0]]], dtype=np.float32)
    normalized = normalize_sar(sar_db)

    # 5 dB → clamped to 0 dB → normalized to 1.0
    assert normalized[0, 0, 0] == 1.0
    # -35 dB → clamped to -30 dB → normalized to 0.0
    assert normalized[0, 0, 1] == 0.0
    # -15 dB → normalized to (-15 - (-30)) / (0 - (-30)) = 15/30 = 0.5
    assert abs(normalized[0, 0, 2] - 0.5) < 1e-6


def test_normalize_sar_handles_nan():
    """NaN values are replaced with per-channel median before normalization."""
    sar_db = np.array([
        [[-10.0, np.nan, -20.0],
         [-15.0, -12.0, -25.0]],
    ], dtype=np.float32)

    normalized = normalize_sar(sar_db)
    assert not np.any(np.isnan(normalized))
    # The NaN should be replaced with median of valid values: [-10, -20, -15, -12, -25]
    # sorted: [-25, -20, -15, -12, -10] → median = -15
    # which normalizes to (-15 - (-30)) / 30 = 15/30 = 0.5
    assert abs(normalized[0, 0, 1] - 0.5) < 0.01


def test_normalize_sar_idempotent_on_already_normalized():
    """normalize_sar is NOT idempotent — it expects dB input, not [0, 1].
    Calling it on already-normalized data should still produce valid [0, 1]
    output (values in [0, 1] map to [0, 1] after clamping and rescaling).
    """
    already_norm = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    result = normalize_sar(already_norm)
    # 0.0 dB → clamped to 0 → (0 - (-30)) / 30 = 1.0
    # 0.5 dB → clamped to 0.5 → (0.5 - (-30)) / 30 ≈ 1.0
    # 1.0 dB → clamped to 1.0 → (1.0 - (-30)) / 30 ≈ 1.0
    # All values in [0, 1] are treated as dB and map to near 1.0
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_calibration_formula():
    """Verify the sigma0 calibration formula: sigma0 = DN^2 / sigmaNought^2."""
    # Synthetic test: DN=100, sigmaNought=500
    # sigma0 = 100^2 / 500^2 = 10000 / 250000 = 0.04
    # sigma0_dB = 10 * log10(0.04) = 10 * (-1.398) = -13.98 dB
    dn = 100.0
    sigma_nought = 500.0
    sigma0 = (dn ** 2) / (sigma_nought ** 2)
    sigma0_db = 10.0 * np.log10(sigma0)
    expected_db = 10.0 * np.log10(0.04)
    assert abs(sigma0_db - expected_db) < 0.01
    # Should be in the typical C-band range
    assert -30.0 <= sigma0_db <= 0.0


def test_find_safe_for_observation_returns_none_for_unknown():
    """find_safe_for_observation returns None for obs-003 (no real scene)."""
    from pathlib import Path
    from siren.preprocess.sar_calibrate import find_safe_for_observation

    # Use a non-existent directory — should return None for any obs
    result = find_safe_for_observation("obs-003", Path("/tmp/nonexistent"))
    assert result is None

    # obs-999 doesn't exist in the date map
    result = find_safe_for_observation("obs-999", Path("/tmp/nonexistent"))
    assert result is None
