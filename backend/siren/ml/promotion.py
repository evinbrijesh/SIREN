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
    # Checkpoint swapped 2026-09-19 → multidate adapter: trained on
    # 3 SAR pairs (t1: 07-14/07-26/08-07) vs v2's single pair. On the
    # held-out gold pair (08-19/09-12) it doubles verified-change
    # recall for the promoted function — dp_tp 16 vs 9 of 32 px
    # (recall 50% vs 28%) — at similar precision; extent IoU is lower
    # (0.44 vs 0.55) but extent is shadow-only under this record.
    "sar_segmentation_expansion": {
        "checkpoint": (
            "water_resunet_6ch_himalayan_adapter_multidate.pt"
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
            "50% on verified change with the multidate checkpoint "
            "(28-41% under labelrefined_v2); promote reviewed when a "
            "real in-domain event at a monitorable lake occurs"
        ),
        "promoted_at": "2026-09-19",
        "reversible": True,
    },
    # Gate-evaluated 2026-09-18 (leakage-free rebuild): 4 measured features
    # (ICIMOD elevation/area, RGI v7 glacier distance + 10km context),
    # deduplicated breach events, spatial-block GroupKFold. Mean AUC 0.792;
    # the DECLARED model is booster + isotonic (OOF-calibrated) — raw
    # probabilities are scale_pos_weight-inflated (Brier 0.157), calibrated
    # Brier 0.085 passes the <0.15 gate. External validation: Imja raw 0.963
    # / 99.8th percentile out-of-sample; Thulagi + Thorthormi are misses.
    "susceptibility": {
        "checkpoint": "xgboost_susceptibility_spatial.json",
        "calibration": "isotonic_crossfit_oof",
        "evidence_method": "calibrated p_breach — static prior for the "
                           "escalation scorer and the FNO trigger gate",
        "gate": "Brier < 0.15 on calibrated score (ADR-013 / PRD §9.8)",
        "gate_evidence": [
            "models/checkpoints/xgboost_spatial_cv_report.json",
            "models/checkpoints/pdgl_external_validation.json",
        ],
        "scope": {
            "signal": (
                "static per-lake breach susceptibility — which lakes are "
                "dangerous, not when; temporal escalation is the "
                "dynamic_escalation component's job"
            ),
            "features": "measured only (ICIMOD + RGI v7); no generated "
                        "or label-contaminated features",
        },
        "union_policy": (
            "advisory prior — feeds dynamic_escalation's "
            "combine_with_static and the FNO trigger gate; does not "
            "change deterministic severity or bypass the human gate"
        ),
        "caveat": (
            "conformal interval is wide (q=0.82 on OOF residuals) — "
            "requires_manual_inspection is expected; Thulagi and "
            "Thorthormi are external-validation misses (dam geometry "
            "not in feature set)"
        ),
        "promoted_at": "2026-09-19",
        "reversible": True,
    },
    # Gate-evaluated 2026-09-18: spatio-temporal holdout (5 spatial blocks,
    # post-2015 events only) on 887 real windows — 230 dated GLOF events +
    # 657 negatives (stable-lake AND within-lake controls) — mean ROC-AUC
    # 0.838, PR-AUC 0.872, raw Brier 0.143 < 0.15 gate. The within-lake
    # control design removes the morphology shortcut: weather features
    # carry ~45% of SHAP importance. Platt calibration (isotonic overfit
    # small folds). South Lhonak hindcast correctly declines to warn —
    # landslide-triggered breach is outside the weather-feature contract.
    "dynamic_escalation": {
        "checkpoint": "xgboost_dynamic_escalation.json",
        "calibration": "platt_crossfit_oof",
        "evidence_method": "p_dynamic >= 0.65 + detected expansion",
        "gate": "Brier < 0.15 (ADR-013 / PRD §9.8)",
        "gate_evidence": [
            "models/checkpoints/dynamic_escalation_eval_report.json",
            "models/checkpoints/south_lhonak_hindcast.json",
            "models/checkpoints/dynamic_escalation_dataset_report.json",
        ],
        "scope": {
            "signal": (
                "weather-driven escalation probability for monitored "
                "lakes — trailing-30d precip/melt/freeze-thaw window "
                "vs 10-yr climatology + static morphometrics"
            ),
            "weather_source": (
                "NASA POWER (MERRA-2 daily, ~0.5deg) — committed "
                "imja_power_series.json asset keeps runtime offline"
            ),
            "out_of_scope": (
                "non-weather triggers — landslide/avalanche impact "
                "waves, seismicity, dam piping (South Lhonak 2023 was "
                "landslide-triggered and correctly does not warn)"
            ),
        },
        "union_policy": (
            "deterministic severity classification stays live as "
            "labeled cross-check; pre_breach_warning surfaces as an "
            "advisory reason on the review card — it does not change "
            "severity or dispatch, and human confirm remains mandatory"
        ),
        "caveat": (
            "454->887-row dataset; per-block AUC spread 0.74-1.0 "
            "(small test folds); ~0.5deg grid smooths convective "
            "extremes; negatives carry label noise (unrecorded events)"
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
