"""Promotion registry — which neural components are promoted to primary.

PRD v5.1 §9.8/§17.2: a component is promoted only when its gate passes
on held-out real deployment-domain data, promotion is component-wise
and reversible, and the deterministic module permanently remains the
labeled fallback/cross-check — never removed.

This module is the machine-readable declaration of the current
promotion set. The pipeline consults it to choose the primary
evidence source per component; the registry reports it for audit.
"""

from __future__ import annotations

from typing import Any

# Component key -> promotion record. A component absent here is
# shadow-only (its outputs are evidence, never load-bearing).
PROMOTED_COMPONENTS: dict[str, dict[str, Any]] = {
    # ADR-014-am1: expansion evidence (water_t1 & ~water_t0 under the
    # Δp decision rule) inside monitorable-lake vicinity on orbit-121
    # descending unfrozen-season pairs. Legs L1–L5 satisfied by
    # labelrefined_v2 (L1 3px ≤5, L2 28% ≥20%, L3 min P@2px 0.72,
    # L4 −90% glacier FP, L5 2 pairs). Owner-directed promotion on the
    # documented thin-evidence caveat (~30–50 verified change px per
    # eval — the verified-change ceiling of the current data window).
    "sar_segmentation_expansion": {
        "checkpoint": (
            "water_resunet_6ch_himalayan_adapter_labelrefined_v2.pt"
        ),
        "evidence_method": "expansion_dp",  # (p1>=0.5)&(p1-p0>=0.2)
        "gate": "ADR-014-am1",
        "gate_evidence": [
            "models/checkpoints/imja_gold_eval_report.json",
            "models/checkpoints/heldout_eval_report.json",
        ],
        "scope": {
            "orbit": "s1 relative-orbit-121 descending",
            "season": "unfrozen (thermal-state gate labels frozen "
                      "observations; promotion does not apply there)",
            "region": "monitorable-lake vicinity only — lakes whose "
                      "surface is water-detectable at C-band "
                      "(sar_visibility_audit 'monitorable' class); "
                      "invisible lakes are outside the detection "
                      "contract entirely",
        },
        "union_policy": (
            "deterministic change evidence stays live as labeled "
            "cross-check; material disagreement in either direction "
            "surfaces as a review reason (cross_check verdict)"
        ),
        "caveat": (
            "verified-change truth is thin (~30-50 px per eval — the "
            "ceiling of the current data window); recall measured "
            "28-41% on verified change; promote reviewed when a real "
            "in-domain event at a monitorable lake occurs"
        ),
        "promoted_at": "2026-09-19",
        "reversible": True,
    },
}


def is_promoted(component: str) -> bool:
    """True when ``component`` is promoted to primary evidence."""
    return component in PROMOTED_COMPONENTS


def promotion_record(component: str) -> dict[str, Any] | None:
    """The promotion record for audit/provenance, or None."""
    return PROMOTED_COMPONENTS.get(component)
