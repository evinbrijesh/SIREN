"""Weather-adaptive change-detection router (OpenCode-owned, Phase 2).

Routes each observation to the optical (NDWI) or SAR (backscatter) path
based on the quality gate's cloud-fraction verdict (PRD §9.1, ADR-003).

Rule: optical cloud_fraction >= 0.20 -> SAR primary. SAR is all-weather
(cloud_fraction treated as 0.0 on that path).
"""

from __future__ import annotations

CLOUD_THRESHOLD = 0.20


def route(cloud_fraction: float, usable: bool = True) -> dict:
    """Decide the change-detection path for an observation.

    Returns a routing verdict dict matching the PRD §9.1 contract shape.
    """
    sar_primary = (not usable) or (cloud_fraction >= CLOUD_THRESHOLD)
    effective_cloud = 0.0 if sar_primary else cloud_fraction
    path = "sar" if sar_primary else "optical"
    return {
        "path": path,
        "sar_primary": sar_primary,
        "cloud_fraction_reported": cloud_fraction,
        "cloud_fraction_effective": effective_cloud,
        "reason": (
            f"optical cloud {cloud_fraction:.0%} >= {CLOUD_THRESHOLD:.0%} -> SAR path"
            if sar_primary and cloud_fraction >= CLOUD_THRESHOLD
            else ("optical unusable -> SAR path" if not usable else "optical clear -> NDWI path")
        ),
    }
