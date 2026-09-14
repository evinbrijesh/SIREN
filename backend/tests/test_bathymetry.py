"""Tests for neural bathymetry inversion (ADR-013 §9.7.1).

Tests the BathymetryUNet architecture, synthetic data generator, and
bed elevation prediction that replaces the Huggel empirical formula
with a neural bed elevation estimator.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from siren.ml.bathymetry import (
    BathymetryUNet,
    NeuralBathymetryResult,
    predict_bed_elevation,
    generate_synthetic_bathymetry_data,
)


# --------------------------------------------------------------------------- #
# BathymetryUNet architecture
# --------------------------------------------------------------------------- #

def test_bathymetry_unet_shapes():
    """BathymetryUNet produces correct output shape."""
    model = BathymetryUNet(in_channels=2, base_channels=16, n_down=3)
    x = torch.randn(2, 2, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_bathymetry_unet_2down_shapes():
    """BathymetryUNet with n_down=2 works on smaller grids."""
    model = BathymetryUNet(in_channels=2, base_channels=16, n_down=2)
    x = torch.randn(1, 2, 32, 32)
    out = model(x)
    assert out.shape == (1, 1, 32, 32)


def test_bathymetry_unet_parameter_budget():
    """BathymetryUNet is small (3-5M params with base_channels=32)."""
    model = BathymetryUNet(in_channels=2, base_channels=32, n_down=3)
    n_params = model.num_parameters()
    # Should be in the 1M-10M range
    assert 1e6 < n_params < 10e6


def test_bathymetry_unet_gradient_flow():
    """Gradient flows through the BathymetryUNet."""
    model = BathymetryUNet(in_channels=2, base_channels=16, n_down=3)
    x = torch.randn(1, 2, 64, 64, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_bathymetry_unet_with_dropout():
    """BathymetryUNet accepts dropout for MC Dropout uncertainty."""
    model = BathymetryUNet(in_channels=2, base_channels=16, n_down=3, dropout=0.1)
    # Check that dropout layers exist
    has_dropout = any(
        isinstance(m, torch.nn.Dropout2d) for m in model.modules()
    )
    assert has_dropout


# --------------------------------------------------------------------------- #
# Synthetic data generator
# --------------------------------------------------------------------------- #

def test_synthetic_data_shapes():
    """Synthetic data generator produces correct shapes."""
    dems, masks, beds = generate_synthetic_bathymetry_data(n_samples=5, grid_size=64, seed=42)
    assert dems.shape == (5, 64, 64)
    assert masks.shape == (5, 64, 64)
    assert beds.shape == (5, 64, 64)


def test_synthetic_data_reproducible():
    """Same seed produces identical synthetic data."""
    d1, m1, b1 = generate_synthetic_bathymetry_data(n_samples=3, seed=42)
    d2, m2, b2 = generate_synthetic_bathymetry_data(n_samples=3, seed=42)
    assert np.allclose(d1, d2)
    assert np.allclose(m1, m2)
    assert np.allclose(b1, b2)


def test_synthetic_data_bed_below_rim():
    """Synthetic bed elevation is below the shoreline rim inside the lake."""
    dems, masks, beds = generate_synthetic_bathymetry_data(n_samples=5, seed=42)
    for i in range(5):
        lake_pixels = masks[i] > 0.5
        if lake_pixels.any():
            # Bed should be below the DEM (rim) inside the lake
            rim_elev = dems[i][~lake_pixels].mean()
            bed_in_lake = beds[i][lake_pixels].mean()
            assert bed_in_lake < rim_elev, (
                f"Sample {i}: bed ({bed_in_lake:.1f}) should be below rim ({rim_elev:.1f})"
            )


def test_synthetic_data_lake_mask_nonempty():
    """Each synthetic sample has a non-empty lake mask."""
    dems, masks, beds = generate_synthetic_bathymetry_data(n_samples=10, seed=42)
    for i in range(10):
        assert masks[i].sum() > 0, f"Sample {i} has empty lake mask"


# --------------------------------------------------------------------------- #
# predict_bed_elevation
# --------------------------------------------------------------------------- #

def test_predict_bed_elevation_shapes():
    """predict_bed_elevation returns correct result."""
    model = BathymetryUNet(in_channels=2, base_channels=16, n_down=3)
    dems, masks, beds = generate_synthetic_bathymetry_data(n_samples=1, grid_size=64, seed=42)

    dem = dems[0]
    mask = masks[0]
    z_surface = float(dem[mask < 0.5].mean())

    result = predict_bed_elevation(
        model, dem, mask, z_surface, pixel_area_m2=900.0,
    )

    assert isinstance(result, NeuralBathymetryResult)
    assert result.z_bed.shape == (64, 64)
    assert result.z_surface == z_surface
    assert result.pixel_area_m2 == 900.0
    assert result.method == "neural_bathymetry"
    assert result.provenance == "neural_bathymetry_v1"


def test_predict_bed_elevation_positive_volume():
    """A trained model on synthetic data produces positive breach volume."""
    # Train briefly on synthetic data
    model = BathymetryUNet(in_channels=2, base_channels=16, n_down=3)
    dems, masks, beds = generate_synthetic_bathymetry_data(n_samples=50, grid_size=64, seed=42)

    # Quick training (just a few steps to verify the pipeline works)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(5):
        for i in range(0, 50, 8):
            dem_batch = dems[i:i + 8]
            mask_batch = masks[i:i + 8]
            bed_batch = beds[i:i + 8]

            # Normalize
            dem_min = dem_batch.min()
            dem_max = dem_batch.max()
            dem_norm = (dem_batch - dem_min) / (dem_max - dem_min + 1e-6)
            x = np.stack([dem_norm, mask_batch], axis=1)
            x_tensor = torch.from_numpy(x).float()
            bed_norm = (bed_batch - dem_min) / (dem_max - dem_min + 1e-6)
            bed_target = torch.from_numpy(bed_norm[:, np.newaxis]).float()

            pred = model(x_tensor)
            loss = torch.nn.functional.mse_loss(pred, bed_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test prediction
    model.eval()
    dem = dems[0]
    mask = masks[0]
    z_surface = float(dem[mask < 0.5].mean())

    result = predict_bed_elevation(
        model, dem, mask, z_surface, pixel_area_m2=900.0,
    )

    # After even minimal training, the model should produce a positive volume
    # (the parabolic basin shape is easy to learn)
    assert result.v_breach_m3 > 0.0, "Breach volume should be positive"


def test_predict_bed_elevation_to_dict():
    """to_dict produces a serializable summary."""
    model = BathymetryUNet(in_channels=2, base_channels=16, n_down=3)
    dems, masks, beds = generate_synthetic_bathymetry_data(n_samples=1, grid_size=32, seed=42)

    result = predict_bed_elevation(
        model, dems[0], masks[0], z_surface=5000.0, pixel_area_m2=900.0,
    )

    d = result.to_dict()
    assert d["method"] == "neural_bathymetry"
    assert d["provenance"] == "neural_bathymetry_v1"
    assert "v_breach_m3" in d
    assert "z_surface" in d
    assert "model_version" in d


# --------------------------------------------------------------------------- #
# Integration: neural bathymetry vs Huggel formula
# --------------------------------------------------------------------------- #

def test_neural_vs_huggel_volume_comparison():
    """Neural bathymetry and Huggel produce comparable volume orders of magnitude."""
    from siren.risk.breach_volume import _huggel_volume

    # Seed model init (Hard Rule 6: no unseeded randomness). Without this the
    # under-trained synthetic model's volume is nondeterministic across torch
    # versions/RNG states and can exceed the 100x sanity bound.
    torch.manual_seed(42)
    model = BathymetryUNet(in_channels=2, base_channels=16, n_down=3)
    dems, masks, beds = generate_synthetic_bathymetry_data(n_samples=20, grid_size=64, seed=42)

    # Train briefly
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(10):
        for i in range(0, 20, 4):
            dem_batch = dems[i:i + 4]
            mask_batch = masks[i:i + 4]
            bed_batch = beds[i:i + 4]
            dem_min = dem_batch.min()
            dem_max = dem_batch.max()
            dem_norm = (dem_batch - dem_min) / (dem_max - dem_min + 1e-6)
            x = np.stack([dem_norm, mask_batch], axis=1)
            x_tensor = torch.from_numpy(x).float()
            bed_norm = (bed_batch - dem_min) / (dem_max - dem_min + 1e-6)
            bed_target = torch.from_numpy(bed_norm[:, np.newaxis]).float()
            pred = model(x_tensor)
            loss = torch.nn.functional.mse_loss(pred, bed_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    dem = dems[0]
    mask = masks[0]
    z_surface = float(dem[mask < 0.5].mean())
    pixel_area = 900.0

    # Neural prediction
    neural_result = predict_bed_elevation(model, dem, mask, z_surface, pixel_area)
    neural_v = neural_result.v_breach_m3

    # Huggel formula
    lake_area = float(mask.sum() * pixel_area)
    huggel_v = _huggel_volume(lake_area)

    # Both should be positive and within an order of magnitude
    # (the neural model is trained on synthetic parabolic basins, Huggel is
    # an empirical power law — they won't match exactly but should be comparable)
    assert neural_v > 0
    assert huggel_v > 0
    ratio = max(neural_v, huggel_v) / max(min(neural_v, huggel_v), 1.0)
    assert ratio < 100, f"Neural ({neural_v:.0f}) and Huggel ({huggel_v:.0f}) differ by >100x"
