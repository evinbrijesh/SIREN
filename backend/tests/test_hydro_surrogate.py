"""Tests for geo/hydro_surrogate.py — FNO-2D hydrodynamic surrogate (V3 §4).

Tests cover the FNO architecture, spectral convolution, trigger gate,
synthetic HEC-RAS data generation, and the HydroSurrogate wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from siren.geo.hydro_surrogate import (
    FNO2D,
    SpectralConv2d,
    HydroSurrogate,
    HydroSurrogateResult,
    generate_synthetic_hecras_run,
    generate_training_dataset,
    FNO_TRIGGER_GATE,
    DEFAULT_NAMED_POINTS,
    DEFAULT_GRID_SIZE,
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

def test_fno_trigger_gate():
    """FNO trigger gate is 0.70 per V3 §4.3."""
    assert FNO_TRIGGER_GATE == 0.70


def test_default_named_points():
    """Default named points match V3 §4.1 (Dudh Koshi)."""
    assert DEFAULT_NAMED_POINTS == ("Hillary Bridge", "Benkar", "Jorsale")


# --------------------------------------------------------------------------- #
# SpectralConv2d
# --------------------------------------------------------------------------- #

def test_spectral_conv2d_output_shape():
    """SpectralConv2d produces the correct output shape."""
    conv = SpectralConv2d(in_channels=4, out_channels=8, modes1=8, modes2=8)
    x = torch.randn(2, 4, 32, 32)
    out = conv(x)
    assert out.shape == (2, 8, 32, 32)


def test_spectral_conv2d_gradients_flow():
    """SpectralConv2d produces gradients on the weights."""
    conv = SpectralConv2d(in_channels=4, out_channels=8, modes1=8, modes2=8)
    x = torch.randn(1, 4, 16, 16, requires_grad=True)
    out = conv(x)
    out.sum().backward()
    assert conv.weights.grad is not None


# --------------------------------------------------------------------------- #
# FNO2D architecture
# --------------------------------------------------------------------------- #

def test_fno2d_forward_pass():
    """FNO2D accepts (B, 2, H, W) input and produces h_water + t_arrival."""
    model = FNO2D(modes=8, width=16, n_points=3, n_layers=2)
    x = torch.randn(2, 2, 32, 32)
    output = model(x)
    assert "h_water" in output
    assert "t_arrival" in output
    assert output["h_water"].shape == (2, 1, 32, 32)
    assert output["t_arrival"].shape == (2, 3, 32, 32)


def test_fno2d_h_water_nonnegative():
    """h_water output is non-negative (physical constraint)."""
    model = FNO2D(modes=8, width=16, n_points=3, n_layers=2)
    model.eval()
    x = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        output = model(x)
    assert output["h_water"].min() >= 0.0


def test_fno2d_t_arrival_nonnegative():
    """T_arrival output is non-negative (time constraint)."""
    model = FNO2D(modes=8, width=16, n_points=3, n_layers=2)
    model.eval()
    x = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        output = model(x)
    assert output["t_arrival"].min() >= 0.0


def test_fno2d_param_budget():
    """FNO2D stays within a reasonable parameter budget (< 5M)."""
    model = FNO2D(modes=12, width=32, n_points=3, n_layers=4)
    n_params = model.num_parameters()
    assert n_params < 5_000_000, f"FNO2D has {n_params} params (>5M)"


def test_fno2d_deterministic_in_eval_mode():
    """FNO2D is deterministic in eval mode."""
    model = FNO2D(modes=8, width=16, n_points=3, n_layers=2)
    model.eval()
    x = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1["h_water"], out2["h_water"])


def test_fno2d_rejects_wrong_channels():
    """FNO2D raises on wrong input channel count."""
    model = FNO2D(modes=8, width=16, n_points=3, n_layers=2)
    x = torch.randn(1, 3, 16, 16)  # wrong: 3 channels
    with pytest.raises(AssertionError, match="2 channels"):
        model(x)


# --------------------------------------------------------------------------- #
# HydroSurrogate wrapper + trigger gate
# --------------------------------------------------------------------------- #

def test_hydro_surrogate_should_trigger_above_gate():
    """should_trigger returns True when P_breach ≥ 0.70."""
    surrogate = HydroSurrogate(model=None)
    assert surrogate.should_trigger(0.70) is True
    assert surrogate.should_trigger(0.85) is True


def test_hydro_surrogate_should_not_trigger_below_gate():
    """should_trigger returns False when P_breach < 0.70."""
    surrogate = HydroSurrogate(model=None)
    assert surrogate.should_trigger(0.69) is False
    assert surrogate.should_trigger(0.50) is False


def test_hydro_surrogate_predict_below_gate_returns_empty():
    """predict() below the gate returns empty result (not triggered)."""
    surrogate = HydroSurrogate(model=None)
    dem = np.zeros((16, 16), dtype=np.float32)
    result = surrogate.predict(dem, v_breach=1e6, p_breach=0.50)
    assert result.is_triggered is False
    assert result.h_water.size == 0
    assert result.t_arrival == {}


def test_hydro_surrogate_predict_above_gate_untrained_raises():
    """predict() above the gate with no model raises RuntimeError."""
    surrogate = HydroSurrogate(model=None)
    dem = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(RuntimeError, match="not trained"):
        surrogate.predict(dem, v_breach=1e6, p_breach=0.80)


def test_hydro_surrogate_predict_with_model():
    """predict() with a trained model returns h_water + t_arrival."""
    model = FNO2D(modes=8, width=16, n_points=3, n_layers=2)
    surrogate = HydroSurrogate(model=model)
    dem = np.random.RandomState(42).uniform(0, 1, (16, 16)).astype(np.float32)
    point_coords = {
        "Hillary Bridge": (10, 5),
        "Benkar": (12, 8),
        "Jorsale": (14, 10),
    }
    result = surrogate.predict(
        dem, v_breach=1e6, p_breach=0.80, point_coords=point_coords
    )
    assert result.is_triggered is True
    assert result.h_water.shape == (16, 16)
    assert len(result.t_arrival) == 3
    assert "Hillary Bridge" in result.t_arrival
    assert all(v >= 0 for v in result.t_arrival.values())


def test_hydro_surrogate_result_to_dict():
    """HydroSurrogateResult.to_dict produces a serializable dict."""
    r = HydroSurrogateResult(
        h_water=np.array([[0.0, 1.5], [2.0, 0.5]]),
        t_arrival={"Hillary Bridge": 15.5, "Benkar": 32.1},
        is_triggered=True,
        p_breach=0.85,
    )
    d = r.to_dict()
    assert d["h_water_shape"] == [2, 2]
    assert d["h_water_max"] == 2.0
    assert d["t_arrival"]["Hillary Bridge"] == 15.5
    assert d["is_triggered"] is True


def test_hydro_surrogate_predict_invalid_dem_dims():
    """predict() raises on non-2D DEM."""
    model = FNO2D(modes=8, width=16, n_points=3, n_layers=2)
    surrogate = HydroSurrogate(model=model)
    with pytest.raises(ValueError, match="2D"):
        surrogate.predict(np.zeros((16,), dtype=np.float32), 1e6, 0.80)


# --------------------------------------------------------------------------- #
# Synthetic HEC-RAS data generation (V3 §4.2)
# --------------------------------------------------------------------------- #

def test_generate_synthetic_hecras_run_shapes():
    """generate_synthetic_hecras_run produces correct output shapes."""
    dem = np.random.RandomState(42).uniform(100, 500, (32, 32)).astype(np.float32)
    run = generate_synthetic_hecras_run(dem, v_breach=1e6, grid_size=32, random_state=42)
    assert run["h_water"].shape == (32, 32)
    assert run["t_arrival"].shape == (3, 32, 32)
    assert "source" in run


def test_generate_synthetic_hecras_run_h_water_nonnegative():
    """h_water is non-negative (physical constraint)."""
    dem = np.random.RandomState(42).uniform(100, 500, (16, 16)).astype(np.float32)
    run = generate_synthetic_hecras_run(dem, v_breach=1e6, grid_size=16, random_state=42)
    assert run["h_water"].min() >= 0.0


def test_generate_synthetic_hecras_run_t_arrival_nonnegative():
    """T_arrival is non-negative (time constraint)."""
    dem = np.random.RandomState(42).uniform(100, 500, (16, 16)).astype(np.float32)
    run = generate_synthetic_hecras_run(dem, v_breach=1e6, grid_size=16, random_state=42)
    assert run["t_arrival"].min() >= 0.0


def test_generate_synthetic_hecras_run_deterministic():
    """Same random_state → same output (Hard Rule 6)."""
    dem = np.random.RandomState(42).uniform(100, 500, (16, 16)).astype(np.float32)
    run1 = generate_synthetic_hecras_run(dem, v_breach=1e6, grid_size=16, random_state=42)
    run2 = generate_synthetic_hecras_run(dem, v_breach=1e6, grid_size=16, random_state=42)
    assert np.allclose(run1["h_water"], run2["h_water"])
    assert np.allclose(run1["t_arrival"], run2["t_arrival"])


def test_generate_synthetic_hecras_run_resizes_dem():
    """generate_synthetic_hecras_run resizes the DEM to grid_size."""
    dem = np.random.RandomState(42).uniform(100, 500, (50, 50)).astype(np.float32)
    run = generate_synthetic_hecras_run(dem, v_breach=1e6, grid_size=32, random_state=42)
    assert run["h_water"].shape == (32, 32)


def test_generate_synthetic_hecras_run_larger_v_breach_higher_water():
    """Larger V_breach produces higher peak water depth."""
    dem = np.random.RandomState(42).uniform(100, 500, (32, 32)).astype(np.float32)
    run_small = generate_synthetic_hecras_run(dem, v_breach=1e5, grid_size=32, random_state=42)
    run_large = generate_synthetic_hecras_run(dem, v_breach=1e8, grid_size=32, random_state=42)
    assert run_large["h_water"].max() > run_small["h_water"].max()


def test_generate_training_dataset_shapes():
    """generate_training_dataset produces correct output shapes."""
    dem = np.random.RandomState(42).uniform(100, 500, (32, 32)).astype(np.float32)
    data = generate_training_dataset(dem, n_runs=10, grid_size=32, random_state=42)
    assert data["inputs"].shape == (10, 2, 32, 32)
    assert data["h_water"].shape == (10, 1, 32, 32)
    assert data["t_arrival"].shape == (10, 3, 32, 32)


def test_generate_training_dataset_deterministic():
    """Same random_state → same dataset (Hard Rule 6)."""
    dem = np.random.RandomState(42).uniform(100, 500, (16, 16)).astype(np.float32)
    data1 = generate_training_dataset(dem, n_runs=5, grid_size=16, random_state=42)
    data2 = generate_training_dataset(dem, n_runs=5, grid_size=16, random_state=42)
    assert np.allclose(data1["inputs"], data2["inputs"])
    assert np.allclose(data1["h_water"], data2["h_water"])


def test_generate_training_dataset_dem_normalized():
    """Training dataset inputs have DEM channel normalised to [0, 1]."""
    dem = np.random.RandomState(42).uniform(100, 500, (16, 16)).astype(np.float32)
    data = generate_training_dataset(dem, n_runs=5, grid_size=16, random_state=42)
    dem_channel = data["inputs"][:, 0]
    assert dem_channel.min() >= 0.0
    assert dem_channel.max() <= 1.0
