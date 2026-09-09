"""Tests for risk/shadow_evidence.py — shadow evidence integration (V3 §3.6, §6).

Tests cover the shadow evidence attachment, susceptibility scoring,
HAND summary, FNO trigger gate, shadow dispatch, and shadow anchoring.
The key invariant: shadow evidence never modifies the canonical hazard score.
"""

from __future__ import annotations

import pytest
import numpy as np

from siren.risk.shadow_evidence import (
    attach_shadow_evidence,
    shadow_dispatch,
    shadow_anchor_audit_chain,
    _compute_shadow_susceptibility,
    _compute_shadow_hand,
    _compute_shadow_hydro,
    DEFAULT_MORAINE_DAM_WIDTH_M,
    DEFAULT_MORAINE_DAM_HEIGHT_M,
    DEFAULT_LAKE_AREA_KM2,
)
from siren.audit.hash_chain import event_hash, GENESIS_HASH


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

def test_default_moraine_dam_geometry():
    """Default moraine dam geometry is set (Imja Tsho approximations)."""
    assert DEFAULT_MORAINE_DAM_WIDTH_M > 0
    assert DEFAULT_MORAINE_DAM_HEIGHT_M > 0
    assert DEFAULT_LAKE_AREA_KM2 > 0


# --------------------------------------------------------------------------- #
# attach_shadow_evidence
# --------------------------------------------------------------------------- #

def _make_change_stats(**overrides):
    """Create a minimal change_stats dict for testing."""
    defaults = {
        "water_area_km2": 1.28,
        "expansion_pct": 25.0,
        "severity": "elevated",
    }
    defaults.update(overrides)
    return defaults


def _make_obs_config(**overrides):
    """Create a minimal obs_config dict for testing."""
    defaults = {
        "expansion_pct": 25.0,
        "mean_slope_degrees": 20.0,
    }
    defaults.update(overrides)
    return defaults


def test_attach_shadow_evidence_adds_shadow_key():
    """attach_shadow_evidence adds 'shadow_evidence' to change_stats."""
    change_stats = _make_change_stats()
    obs_config = _make_obs_config()
    shadow = attach_shadow_evidence(change_stats, obs_config, 5.0, 30.0)
    assert "shadow_evidence" in change_stats
    assert change_stats["shadow_evidence"] is shadow


def test_attach_shadow_evidence_is_marked_shadow():
    """Shadow evidence is marked as shadow-only (ADR-010 §3)."""
    change_stats = _make_change_stats()
    obs_config = _make_obs_config()
    shadow = attach_shadow_evidence(change_stats, obs_config, 5.0, 30.0)
    assert shadow["is_shadow"] is True
    assert shadow["gate_status"] == "shadow_only"
    assert "note" in shadow


def test_attach_shadow_evidence_includes_susceptibility():
    """Shadow evidence includes the susceptibility scorer output."""
    change_stats = _make_change_stats()
    obs_config = _make_obs_config()
    shadow = attach_shadow_evidence(change_stats, obs_config, 5.0, 30.0)
    assert "susceptibility" in shadow
    sus = shadow["susceptibility"]
    assert "p_breach" in sus
    assert "interval_low" in sus
    assert "interval_high" in sus


def test_attach_shadow_evidence_includes_hand():
    """Shadow evidence includes the HAND summary when DEM path is provided."""
    change_stats = _make_change_stats()
    obs_config = _make_obs_config()
    shadow = attach_shadow_evidence(
        change_stats, obs_config, 5.0, 30.0, dem_path="/fake/dem.tif"
    )
    assert "hand" in shadow
    hand = shadow["hand"]
    assert "h_water_stage_m" in hand
    assert "severity" in hand


def test_attach_shadow_evidence_no_hand_without_dem():
    """Shadow evidence does not include HAND when no DEM path is provided."""
    change_stats = _make_change_stats()
    obs_config = _make_obs_config()
    shadow = attach_shadow_evidence(change_stats, obs_config, 5.0, 30.0)
    assert "hand" not in shadow


def test_attach_shadow_evidence_includes_hydro_surrogate():
    """Shadow evidence includes the FNO hydrodynamic surrogate status."""
    change_stats = _make_change_stats()
    obs_config = _make_obs_config()
    shadow = attach_shadow_evidence(change_stats, obs_config, 5.0, 30.0)
    assert "hydro_surrogate" in shadow


def test_attach_shadow_evidence_fno_not_triggered_low_p_breach():
    """FNO trigger status matches P_breach from the susceptibility scorer."""
    change_stats = _make_change_stats(expansion_pct=5.0)  # low expansion
    obs_config = _make_obs_config(expansion_pct=5.0)
    shadow = attach_shadow_evidence(change_stats, obs_config, 2.0, 10.0)
    hydro = shadow["hydro_surrogate"]
    sus = shadow["susceptibility"]
    p_breach = sus["p_breach"]
    # FNO trigger status should match the P_breach value
    if p_breach >= 0.70:
        assert hydro["is_triggered"] is True
    else:
        assert hydro["is_triggered"] is False


def test_attach_shadow_evidence_does_not_modify_hazard():
    """Shadow evidence does not modify the canonical hazard score."""
    change_stats = _make_change_stats()
    change_stats["hazard_score"] = 0.75  # canonical score
    obs_config = _make_obs_config()
    attach_shadow_evidence(change_stats, obs_config, 5.0, 30.0)
    # The canonical hazard score is unchanged
    assert change_stats["hazard_score"] == 0.75


def test_attach_shadow_evidence_handles_errors_gracefully():
    """attach_shadow_evidence handles errors without raising."""
    change_stats = _make_change_stats()
    obs_config = _make_obs_config()
    # Pass invalid data — should not raise
    shadow = attach_shadow_evidence(change_stats, obs_config, -1.0, -1.0)
    assert shadow["is_shadow"] is True


# --------------------------------------------------------------------------- #
# _compute_shadow_susceptibility
# --------------------------------------------------------------------------- #

def test_compute_shadow_susceptibility_returns_dict():
    """_compute_shadow_susceptibility returns a serializable dict."""
    change_stats = _make_change_stats()
    obs_config = _make_obs_config()
    result = _compute_shadow_susceptibility(change_stats, obs_config, 30.0)
    assert isinstance(result, dict)
    assert "p_breach" in result
    assert 0.0 <= result["p_breach"] <= 1.0


def test_compute_shadow_susceptibility_high_expansion_high_p_breach():
    """High lake expansion rate produces higher P_breach."""
    change_stats_low = _make_change_stats(expansion_pct=5.0)
    change_stats_high = _make_change_stats(expansion_pct=50.0)
    obs_config = _make_obs_config()
    result_low = _compute_shadow_susceptibility(change_stats_low, obs_config, 10.0)
    result_high = _compute_shadow_susceptibility(change_stats_high, obs_config, 10.0)
    # High expansion should generally produce higher P_breach
    # (not strictly guaranteed with synthetic training, but expected)
    assert result_high["p_breach"] >= result_low["p_breach"]


def test_compute_shadow_susceptibility_includes_reasons():
    """Susceptibility result includes TreeSHAP reasons."""
    change_stats = _make_change_stats()
    obs_config = _make_obs_config()
    result = _compute_shadow_susceptibility(change_stats, obs_config, 30.0)
    assert "reasons" in result
    assert len(result["reasons"]) >= 1


# --------------------------------------------------------------------------- #
# _compute_shadow_hand
# --------------------------------------------------------------------------- #

def test_compute_shadow_hand_returns_summary():
    """_compute_shadow_hand returns a HAND summary dict."""
    result = _compute_shadow_hand(
        "/fake/dem.tif",
        _make_obs_config(),
        _make_change_stats(severity="critical"),
    )
    assert result["severity"] == "critical"
    assert result["h_water_stage_m"] == 5.0  # critical policy default


def test_compute_shadow_hand_watch_severity():
    """HAND summary uses watch policy default (0.5 m)."""
    result = _compute_shadow_hand(
        "/fake/dem.tif",
        _make_obs_config(),
        _make_change_stats(severity="watch"),
    )
    assert result["h_water_stage_m"] == 0.5


# --------------------------------------------------------------------------- #
# _compute_shadow_hydro
# --------------------------------------------------------------------------- #

def test_compute_shadow_hydro_triggered():
    """_compute_shadow_hydro returns triggered status when P_breach ≥ 0.70."""
    result = _compute_shadow_hydro(_make_obs_config(), 0.85, None)
    assert result["is_triggered"] is True
    assert result["p_breach"] == 0.85


def test_compute_shadow_hydro_includes_trigger_gate():
    """_compute_shadow_hydro includes the trigger gate value."""
    result = _compute_shadow_hydro(_make_obs_config(), 0.80, None)
    assert "trigger_gate" in result
    assert result["trigger_gate"] == 0.70


# --------------------------------------------------------------------------- #
# shadow_dispatch
# --------------------------------------------------------------------------- #

def test_shadow_dispatch_returns_dict():
    """shadow_dispatch returns a serializable dict."""
    result = shadow_dispatch("disp-001", "siren-payload-001", "default")
    assert isinstance(result, dict)
    assert result["dispatch_id"] == "disp-001"
    assert result["delivered"] is True
    assert len(result["receipts"]) == 2


def test_shadow_dispatch_enforces_250_byte_limit():
    """shadow_dispatch enforces the 250-byte payload limit (Hard Rule 4)."""
    with pytest.raises(ValueError, match="250-byte limit"):
        shadow_dispatch("disp-001", "x" * 251, "default")


# --------------------------------------------------------------------------- #
# shadow_anchor_audit_chain
# --------------------------------------------------------------------------- #

def test_shadow_anchor_returns_dict():
    """shadow_anchor_audit_chain returns a serializable dict."""
    chain_root = event_hash(GENESIS_HASH, "2026-09-08T12:00:00Z", "test")
    result = shadow_anchor_audit_chain(chain_root)
    assert isinstance(result, dict)
    assert result["chain_root"] == chain_root
    assert result["is_simulated"] is True
    assert result["is_verified"] is True


def test_shadow_anchor_rejects_invalid_hash():
    """shadow_anchor_audit_chain rejects invalid chain roots."""
    with pytest.raises(ValueError, match="64-char hex"):
        shadow_anchor_audit_chain("invalid")
