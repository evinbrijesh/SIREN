"""Tests for the GLOF dataset and trained susceptibility checkpoint (Level 1).

Tests cover the curated dataset, the training script, and the checkpoint
loading mechanism.
"""

from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path

from siren.ml.glof_dataset import (
    load_dataset,
    get_feature_matrix,
    get_labels,
    get_spatial_groups,
    dataset_summary,
    FEATURE_NAMES,
    save_parquet,
)
from siren.risk.susceptibility import (
    SusceptibilityScorer,
    DEFAULT_CHECKPOINT_PATH,
)


# --------------------------------------------------------------------------- #
# GLOF dataset
# --------------------------------------------------------------------------- #

def test_dataset_loads():
    """load_dataset returns a non-empty DataFrame."""
    df = load_dataset()
    assert len(df) > 0
    assert "breached" in df.columns


def test_dataset_has_both_classes():
    """Dataset contains both breached and stable lakes."""
    df = load_dataset()
    assert df["breached"].sum() > 0  # at least one breach
    assert (df["breached"] == 0).sum() > 0  # at least one stable


def test_dataset_feature_names_match():
    """Dataset features match FEATURE_NAMES in susceptibility.py."""
    df = load_dataset()
    for name in FEATURE_NAMES:
        assert name in df.columns, f"Missing feature: {name}"


def test_dataset_has_geographic_columns():
    """Dataset has region, country, lat, lon for spatial CV."""
    df = load_dataset()
    for col in ["region", "country", "lat", "lon"]:
        assert col in df.columns


def test_get_feature_matrix_shape():
    """get_feature_matrix returns (n_samples, 6) float32."""
    df = load_dataset()
    X = get_feature_matrix(df)
    assert X.shape == (len(df), len(FEATURE_NAMES))
    assert X.dtype == np.float32


def test_get_labels():
    """get_labels returns binary {0, 1} array."""
    df = load_dataset()
    y = get_labels(df)
    assert set(np.unique(y)).issubset({0, 1})


def test_get_spatial_groups():
    """get_spatial_groups returns region strings."""
    df = load_dataset()
    groups = get_spatial_groups(df)
    assert len(groups) == len(df)
    assert len(np.unique(groups)) > 1  # multiple regions


def test_dataset_summary():
    """dataset_summary returns a non-empty string."""
    df = load_dataset()
    s = dataset_summary(df)
    assert isinstance(s, str)
    assert "Breached:" in s
    assert "Stable:" in s


def test_save_parquet(tmp_path):
    """save_parquet writes a parquet file."""
    path = save_parquet(tmp_path / "glof_training.parquet")
    assert path.exists()


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #

def test_checkpoint_exists():
    """The trained checkpoint exists at the default path."""
    assert DEFAULT_CHECKPOINT_PATH.exists(), (
        f"Checkpoint not found at {DEFAULT_CHECKPOINT_PATH}. "
        "Run: python -m siren.ml.train_susceptibility"
    )


def test_load_checkpoint():
    """SusceptibilityScorer.load_checkpoint rejects the disqualified model."""
    scorer = SusceptibilityScorer(random_state=42)
    loaded = scorer.load_checkpoint()
    assert loaded is False
    assert scorer.is_trained is False


def test_load_checkpoint_missing_file(tmp_path):
    """load_checkpoint returns False for a non-existent file."""
    scorer = SusceptibilityScorer(random_state=42)
    loaded = scorer.load_checkpoint(tmp_path / "nonexistent.json")
    assert loaded is False
    assert scorer.is_trained is False


def test_predict_with_loaded_checkpoint():
    """predict() raises RuntimeError when the checkpoint is disqualified."""
    scorer = SusceptibilityScorer(random_state=42)
    scorer.load_checkpoint()
    assert scorer.is_trained is False

    X = np.array([[0.11, 350, 12, 0.5, 20, 1.28]], dtype=np.float32)
    with pytest.raises(RuntimeError):
        scorer.predict(X)


def test_predict_breached_lake_with_checkpoint():
    """predict() is blocked for both breached and stable lakes (disqualified)."""
    scorer = SusceptibilityScorer(random_state=42)
    scorer.load_checkpoint()
    assert scorer.is_trained is False

    X_breach = np.array([[0.08, 120, 15, 1.5, 25, 0.60]], dtype=np.float32)
    X_stable = np.array([[0.11, 350, 12, 0.5, 20, 1.28]], dtype=np.float32)
    with pytest.raises(RuntimeError):
        scorer.predict(X_breach)
    with pytest.raises(RuntimeError):
        scorer.predict(X_stable)


def test_checkpoint_brier_score_loaded():
    """The Brier score is None when the checkpoint is disqualified."""
    scorer = SusceptibilityScorer(random_state=42)
    scorer.load_checkpoint()
    assert scorer.brier_score is None
    assert scorer.passes_acceptance_gate() is False


# --------------------------------------------------------------------------- #
# Training script (smoke test — doesn't save)
# --------------------------------------------------------------------------- #

def test_training_script_runs():
    """The training script is blocked (disqualified, PRD v4.7 §17.3)."""
    from siren.ml.train_susceptibility import train_susceptibility_model
    with pytest.raises(ValueError, match="disqualified"):
        train_susceptibility_model(save=False)
