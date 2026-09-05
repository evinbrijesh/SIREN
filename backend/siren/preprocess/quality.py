"""Quality gate for satellite observations (PRD §9.1).

Pure, deterministic assessment of an observation's usability before it enters
the change-detection pipeline. The verdict dict emitted by :func:`assess_quality`
is the exact JSON contract consumed by ``siren.detect.router`` (PRD §9.1):

    {
      "quality_score": 0.88,
      "cloud_fraction": 0.11,
      "alignment_ok": true,
      "usable": true,
      "confidence_adjustment": 0.95
    }

Routing rule (consumed downstream): an optical scene with
``cloud_fraction >= 0.20`` flips ``usable`` to ``False``, which routes the
pipeline to the SAR path. SAR is all-weather capable, so its effective cloud
fraction is treated as ``0.0``.
"""

from __future__ import annotations

# --- Tunable constants (PRD §9.1) -------------------------------------------
CLOUD_THRESHOLD: float = 0.20
"""Optical cloud fraction at or above which the observation is routed to SAR."""

ALIGNMENT_THRESHOLD_PX: float = 1.0
"""Alignment error (RMSE, in pixels) below which ``alignment_ok`` is True."""

# Sensor-freshness weights: NRT (near-real-time) sources are weighted 1.0,
# archived products 0.9. Deterministic and simple per the task spec.
NRT_FRESHNESS_WEIGHT: float = 1.0
ARCHIVED_FRESHNESS_WEIGHT: float = 0.9

# Quality-score composite weights (PRD §9.1).
CLOUD_WEIGHT: float = 0.6
ALIGNMENT_WEIGHT: float = 0.4
ALIGNMENT_OK_SCORE: float = 1.0
ALIGNMENT_BAD_SCORE: float = 0.5


def _is_sar(sensor: str) -> bool:
    """True for SAR sensors (Sentinel-1 GRD or any sensor tagged 'sar')."""
    s = sensor.lower()
    return "sentinel-1" in s or "sar" in s


def _freshness_weight(sensor: str) -> float:
    """NRT sources weigh 1.0, everything else (archived) 0.9."""
    return NRT_FRESHNESS_WEIGHT if "nrt" in sensor.lower() else ARCHIVED_FRESHNESS_WEIGHT


def assess_quality(cloud_fraction: float, alignment_error: float, sensor: str) -> dict:
    """Assess observation quality per PRD §9.1.

    Args:
        cloud_fraction: 0..1 cloud cover fraction (optical scene statistic).
            Ignored for SAR sensors, which are all-weather (effective 0.0).
        alignment_error: alignment (co-registration) error in pixels (RMSE).
        sensor: source identifier, e.g. ``'sentinel-1-grd-nrt'``,
            ``'sentinel-2-l2a'``, or ``'prepared-demo'``.

    Returns:
        dict matching the PRD §9.1 JSON contract exactly, with the five keys
        ``quality_score``, ``cloud_fraction``, ``alignment_ok``, ``usable``,
        ``confidence_adjustment``.
    """
    # SAR is all-weather: effective cloud fraction is 0.0 regardless of input.
    effective_cloud = 0.0 if _is_sar(sensor) else float(cloud_fraction)
    # Clamp to a sane [0, 1] range to defend against malformed inputs.
    effective_cloud = min(max(effective_cloud, 0.0), 1.0)

    alignment_ok = float(alignment_error) < ALIGNMENT_THRESHOLD_PX
    usable = effective_cloud < CLOUD_THRESHOLD

    # Confidence multiplier = (1 - cloud_fraction) * sensor_freshness_weight.
    confidence_adjustment = round(
        (1.0 - effective_cloud) * _freshness_weight(sensor), 2
    )

    # Composite quality score, clamped to [0, 1], rounded to 2 decimals.
    quality_score = (1.0 - effective_cloud) * CLOUD_WEIGHT + (
        ALIGNMENT_OK_SCORE if alignment_ok else ALIGNMENT_BAD_SCORE
    ) * ALIGNMENT_WEIGHT
    quality_score = min(max(quality_score, 0.0), 1.0)
    quality_score = round(quality_score, 2)

    return {
        "quality_score": quality_score,
        "cloud_fraction": round(effective_cloud, 2),
        "alignment_ok": alignment_ok,
        "usable": usable,
        "confidence_adjustment": confidence_adjustment,
    }
