"""Neural-vs-deterministic cross-check harness (PRD §9.8 promotion gate 3).

On every run, the neural mask is compared against the deterministic
fallback mask on a common geographic grid. Material disagreement is
surfaced as a review reason — never silently resolved. The harness runs
identically in shadow mode (accumulating disagreement evidence before
promotion) and in primary mode (where it guards the promoted output —
a persistent material disagreement is a demotion trigger).

The comparison metric is geographic overlap on the SAR grid: the rule
mask is resampled through GCP geolocation (``sar_grid_sample``) so the
two masks meet on the same pixels — index-resize agreement is spurious
when the grids share no extent (documented 2026-09-17).

Both directions are measured, because each catches a different failure:

  rule_recall    = overlap / rule_px   — neural misses real change
  ml_precision   = overlap / ml_px     — neural hallucinates change

A verdict is ``material_disagreement`` when either direction falls below
``DISAGREEMENT_BOUND`` on a run where at least one mask is non-empty.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Minimum directional overlap before the run is flagged. Set
# conservatively at 50%: the deterministic mask is the calibrated
# fallback, so a promoted component that reproduces less than half of
# it (or agrees on less than half its own output) is materially
# diverging. Tighten per-component at promotion review.
DISAGREEMENT_BOUND = 0.50


def evaluate_overlap(
    ml_mask: np.ndarray,
    rule_on_sar: np.ndarray,
) -> dict[str, Any]:
    """Score neural-vs-deterministic agreement on a shared grid.

    Args:
        ml_mask: (H, W) neural detection mask on the SAR grid (gated —
            the same pixels the review card would display).
        rule_on_sar: (H, W) deterministic mask resampled onto the SAR
            grid (``sar_grid_sample(rule_mask_path, lon, lat) > 0``).

    Returns:
        verdict dict with directional overlaps, the material flag, and
        a review reason string when disagreement is material.
    """
    ml = np.asarray(ml_mask).astype(bool)
    rule = np.asarray(rule_on_sar).astype(bool)
    overlap = int((ml & rule).sum())
    ml_px = int(ml.sum())
    rule_px = int(rule.sum())

    rule_recall = overlap / rule_px if rule_px else None
    ml_precision = overlap / ml_px if ml_px else None

    if ml_px == 0 and rule_px == 0:
        # Both empty — agreement on "no change", nothing to flag.
        material = False
    elif ml_px == 0 or rule_px == 0:
        # One mask empty, the other not — maximally divergent.
        material = True
    else:
        material = (
            (rule_recall is not None and rule_recall < DISAGREEMENT_BOUND)
            or (ml_precision is not None and ml_precision < DISAGREEMENT_BOUND)
        )

    verdict: dict[str, Any] = {
        "metric": "geographic_overlap_on_sar_grid",
        "bound": DISAGREEMENT_BOUND,
        "ml_px": ml_px,
        "rule_px": rule_px,
        "overlap_px": overlap,
        "rule_recall": round(rule_recall, 4) if rule_recall is not None else None,
        "ml_precision": round(ml_precision, 4) if ml_precision is not None else None,
        "material_disagreement": material,
    }

    if material:
        verdict["reason"] = (
            "Neural/deterministic cross-check: "
            f"overlap {overlap}px of rule {rule_px}px "
            f"(recall {rule_recall if rule_recall is not None else 'n/a'}, "
            f"ML precision {ml_precision if ml_precision is not None else 'n/a'}) "
            f"below {DISAGREEMENT_BOUND:.0%} bound — neural and deterministic "
            "masks materially disagree; fallback mask remains authoritative"
        )
    return verdict
