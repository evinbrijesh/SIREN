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

import json
import os
from pathlib import Path
from typing import Any

# Component key -> promotion record. A component absent here is
# shadow-only (its outputs are evidence, never load-bearing).
PROMOTED_COMPONENTS: dict[str, dict[str, Any]] = {
    # ADR-014-am1: expansion evidence (water_t1 & ~water_t0 under the
    # Δp decision rule) inside monitorable-lake vicinity on orbit-121
    # descending unfrozen-season pairs.
    #
    # Gate evidence (2026-09-21): operational gate evaluates the *gated*
    # model output (AOI + slope>15° + glacier-minus-lake-vicinity — the
    # same terrain gate the runtime applies to the shadow mask).
    # 2 unfrozen descending pairs pass at τ=0.30:
    #   monsoon_2025_desc (2025-08-29→09-10): IoU 0.68, P 0.97, glacFP 0%
    #   monsoon_2025_desc2 (2025-08-29→09-22): IoU 0.62, P 0.95, glacFP 0%
    #
    # Scope limitation: the model is a change detector — it detects water
    # best when the lake's backscatter increases between pre and post
    # scenes (dVV > +1 dB). This is the right behaviour for GLOF
    # monitoring (detect expansion) but means it under-performs on
    # scenes with small change (2026 eval pair IoU 0.50) and fails on
    # ascending pass (IoU 0.03) — different geometry + different
    # backscatter statistics.
    #
    # For a monsoon-focused GLOF system this is sufficient: Sentinel-1
    # provides a descending pass every 12 days, giving ~8 observations
    # per monsoon season. Ascending pass can be added later via a
    # separate pass-specific model or a domain-adversarial approach.
    #
    # Real-event OOD probe (2026-09-21, `south_lhonak_sar_eval`): the
    # model was run on the only on-disk SAR pair that brackets a real
    # GLOF — South Lhonak 2023-09-28 → 2023-10-10 (burst 2023-10-03,
    # ~40-50 MCM). Result: FAILED. Post-event IoU 0.028, recall 3.6%
    # at τ=0.3; 38,481 of 38,489 predicted pixels (99.98%) are false
    # positives elsewhere in the scene. The lake's SAR signature
    # (VV -15.4 pre / -12.3 post, dVV +3.1 dB) differs from Imja's
    # water (-16 to -19 dB), and the model does not generalise. This
    # confirms the certified scope is Imja-area only — not a
    # conservative choice but a hard requirement. Multi-lake
    # monitoring needs multi-lake training data.
    "sar_segmentation_expansion": {
        # Dropout-native retrain of the multidate adapter (E1/L2.1 —
        # dropout=0.1 active during training, the only construction the
        # uncertainty gate accepts). Deterministic (eval-mode) metrics on
        # the operational gate IMPROVED vs the dropout=0 parent: IoU
        # 0.82/0.80 on the 2025 pairs (was 0.68/0.62), 0.64 on the 2026
        # pair (was 0.50); glacier FP stays 0%. Δp change detection is
        # comparable (14/32 tp at 6 fp vs 16/32 at 9 fp on the gold pair).
        "checkpoint": (
            "water_resunet_6ch_himalayan_adapter_multidate_mc.pt"
        ),
        "evidence_method": "expansion_dp",  # (p1>=0.5)&(p1-p0>=0.2)
        "gate": "ADR-014-am1",
        "gate_evidence": [
            "models/checkpoints/imja_operational_gate_eval.json",
            "models/checkpoints/imja_gold_eval_report.json",
            "models/checkpoints/heldout_eval_report.json",
            "models/checkpoints/lake_adapter_multidate_mc_report.json",
            "models/checkpoints/imja_scene_conformal_eval.json",
        ],
        "scope": {
            "orbit": "s1 relative-orbit-121 descending",
            "season": "monsoon window Jun-Sep (siren/scope.py — 82.6% of "
                      "dated GLOF events, ~80% of annual rainfall); "
                      "thermal-state gate additionally labels frozen "
                      "observations",
            "region": "monitorable-lake vicinity only — lakes whose "
                      "surface is water-detectable at C-band "
                      "(sar_visibility_audit 'monitorable' class); "
                      "invisible lakes are outside the detection "
                      "contract entirely",
        },
        "level": "operational_primary",
        "union_policy": (
            "deterministic change evidence stays live as labeled "
            "cross-check; material disagreement in either direction "
            "surfaces as a review reason (cross_check verdict)"
        ),
        "caveat": (
            "Operational gate passes on 2 unfrozen descending pairs "
            "(2025-08-29→09-10 IoU 0.82/P 0.98 and 08-29→09-22 IoU "
            "0.80/P 0.94, shared t0). The 2026 eval pair under-performs "
            "(IoU 0.64, precision 0.73 < 0.84) and the ascending pair "
            "fails (IoU 0.03) — the model is scoped to "
            "descending unfrozen scenes with a significant "
            "backscatter-change signal. Frozen-season predictions "
            "are correctly near-zero (winter pair passes frozen "
            "gate). Real-event OOD probe (South Lhonak 2023 GLOF) "
            "FAILED: IoU 0.028, 99.98% of predicted pixels are false "
            "positives — the certified scope is Imja-area only. For "
            "year-round coverage, a separate ascending model or "
            "pass-invariant training is needed; for multi-lake "
            "coverage, multi-lake training data is needed. "
            "Uncertainty: dropout-native (p=0.1) — the per-checkpoint "
            "conformal sidecar (whole-scene LOSO, 3 in-scope gold "
            "scenes) gives q*=0.639 but the Level 2.2 gate FAILED "
            "(per-scene coverage 0.816–0.931 vs nominal 0.90 ±0.05) — "
            "the interval is advisory, not certified."
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
        "level": "advisory_primary",
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
        "level": "advisory_primary",
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
    # Gate-evaluated 2026-09-22 (DL_PRIMARY_ROADMAP Level 6): fused
    # event-probability scorer — Tier-2's 13 features + a fold-honest
    # stacked susceptibility prior + the Jun–Sep monsoon flag — on the
    # same 887-window corpus and identical spatio-temporal holdout.
    # Mean ROC-AUC 0.835 vs the deterministic five-factor baseline's
    # 0.504 on identical folds; mean raw Brier 0.142 < 0.15 — both gate
    # legs pass. Caveat: the fused model is statistically identical to
    # Tier-2 alone (AUC 0.835 vs 0.838) — the stacked prior and season
    # flag add no measurable accuracy; the component's value is the
    # unified fusion contract and the first honest head-to-head against
    # the deterministic formula.
    "learned_risk_fusion": {
        "checkpoint": "xgboost_risk_fusion.json",
        "calibration": "platt_crossfit_oof",
        "evidence_method": (
            "calibrated p_fused >= 0.65 advisory flag; TreeSHAP top-3 "
            "drivers surface as review reasons"
        ),
        "gate": (
            "Brier < 0.15 AND mean ROC-AUC > deterministic five-factor "
            "baseline on identical spatio-temporal folds "
            "(DL_PRIMARY_ROADMAP §9.2)"
        ),
        "gate_evidence": [
            "models/checkpoints/risk_fusion_eval_report.json",
        ],
        "scope": {
            "signal": (
                "fused P(event) — static morphometrics + trailing-30d "
                "weather + susceptibility prior + season flag"
            ),
            "features": (
                "measured only; the susceptibility prior is fold-honest "
                "at eval time (inner-block OOF stack); at runtime the "
                "deployed calibrated prior is used (documented shift)"
            ),
            "out_of_scope": (
                "severity classification itself — deterministic "
                "classify_severity stays authoritative; the learned "
                "score does not gate dispatch"
            ),
        },
        "level": "advisory_primary",
        "union_policy": (
            "advisory reason on the review card only — deterministic "
            "severity, expansion override, and the human gate are "
            "unchanged; promotion to operational severity fusion "
            "requires a separate severity-mapping evaluation"
        ),
        "caveat": (
            "no measured gain over Tier-2 alone (AUC 0.835 vs 0.838); "
            "historical windows lack SAR inputs, so the deterministic "
            "baseline leg is evaluated on its measurable subset only "
            "(rain + fixed slope/drainage proxies); promotion to "
            "severity-primary is a separate decision"
        ),
        "promoted_at": "2026-09-22",
        "reversible": True,
    },
}


# Runtime demotion flag (DL_PRIMARY_ROADMAP task 1.5): a comma-separated
# environment variable forces promoted components back to their
# deterministic fallback without editing the registry —
#   SIREN_ML_DEMOTE=sar_segmentation_expansion
#   SIREN_ML_DEMOTE=all
# Demotion is read at call time, so it takes effect per-run and is
# testable via monkeypatch. Promotion is component-wise and reversible
# (PRD §9.8) — demotion is the operational reversal path.
DEMOTE_ENV_VAR = "SIREN_ML_DEMOTE"


def demoted_components() -> set[str]:
    """Component names forced to deterministic fallback via the env var."""
    raw = os.environ.get(DEMOTE_ENV_VAR, "")
    return {t.strip() for t in raw.split(",") if t.strip()}


def is_demoted(component: str) -> bool:
    """True when ``component`` is runtime-demoted to deterministic fallback."""
    tokens = demoted_components()
    return "all" in tokens or component in tokens


def is_promoted(component: str) -> bool:
    """True when ``component`` is promoted and not runtime-demoted.

    A demoted component behaves exactly as if unpromoted: the
    deterministic module is load-bearing and the promotion record stays
    in the registry for audit.
    """
    return component in PROMOTED_COMPONENTS and not is_demoted(component)


def promotion_record(component: str) -> dict[str, Any] | None:
    """The promotion record for audit/provenance, or None."""
    return PROMOTED_COMPONENTS.get(component)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKPOINTS_DIR = _REPO_ROOT / "models" / "checkpoints"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def get_ml_readiness_report() -> dict[str, Any]:
    """Return the DL-primary readiness status for every neural component.

    The report distinguishes three states:

      * ``shadow`` — the component runs as advisory evidence but is not
        gate-passed on deployment-domain data.
      * ``advisory_primary`` — the component is promoted to provide advisory
        reasons on the review card, but the deterministic path remains the
        operational authority for severity/dispatch.
      * ``operational_primary`` — the component is the default load-bearing
        path for its stage and the deterministic fallback is a labeled
        cross-check. ``sar_segmentation_expansion`` holds this level for
        the water-area expansion factor within its certified scope
        (descending orbit, monsoon Jun-Sep, Imja-area monitorable lakes);
        out-of-scope runs and runtime demotion (``SIREN_ML_DEMOTE``)
        revert to the deterministic registry value.

    The ``dl_primary_ready`` flag is False until every load-bearing stage
    (segmentation, uncertainty, bathymetry, dynamics, risk fusion) is at
    least advisory_primary and the operational pipeline consumes neural
    outputs by default.
    """
    report: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. SAR segmentation (closest to operational-primary)
    # ------------------------------------------------------------------
    seg_rec = promotion_record("sar_segmentation_expansion")
    if seg_rec is not None:
        report["sar_segmentation_expansion"] = {
            "status": seg_rec.get("level", "advisory_primary"),
            "demoted": is_demoted("sar_segmentation_expansion"),
            "display": "SAR water/change segmentation",
            "gate": seg_rec.get("gate"),
            "gate_passed": True,
            "checkpoint": seg_rec.get("checkpoint"),
            "promoted_at": seg_rec.get("promoted_at"),
            "scope": seg_rec.get("scope"),
            "caveat": seg_rec.get("caveat"),
            "blocker": (
                "Operational-primary within certified scope (descending "
                "orbit, monsoon Jun-Sep, Imja-area monitorable lakes). "
                "Ascending pass, frozen season, and other lakes remain "
                "out of scope and need dedicated training data."
            ),
            "evidence_files": [str(p) for p in seg_rec.get("gate_evidence", [])],
        }
    else:
        report["sar_segmentation_expansion"] = {
            "status": "shadow",
            "display": "SAR water/change segmentation",
            "gate": "ADR-011.1: event-held-out IoU > 0.60 AND P ≥ 0.84",
            "gate_passed": False,
            "checkpoint": None,
            "blocker": "No promoted segmentation component registered.",
        }

    # ------------------------------------------------------------------
    # 2. Bayesian uncertainty / conformal calibration
    # ------------------------------------------------------------------
    conformal = _load_json(
        _CHECKPOINTS_DIR
        / "water_resunet_kuro_siwo_full"
        / "conformal_calibration.json"
    )
    report["bayesian_uncertainty"] = {
        "status": "shadow",
        "display": "MC Dropout + split-conformal uncertainty",
        "gate": "Coverage within ±5% of nominal 90% on held-out data",
        "gate_passed": bool(conformal and conformal.get("gate_passed")),
        "current_metric": {
            "empirical_coverage": conformal.get("empirical_coverage")
            if conformal else None,
            "nominal_level": conformal.get("nominal_level")
            if conformal else None,
            "coverage_error": conformal.get("coverage_error")
            if conformal else None,
        },
        "checkpoint": "water_resunet_kuro_siwo_full/conformal_calibration.json",
        "blocker": (
            "Calibrated on Kuro Siwo chips, not Imja-area whole scenes; "
            "model trained with dropout=0.0, so MC Dropout is post-hoc; "
            "per-pixel coverage ignores spatial correlation."
        ),
    }

    # ------------------------------------------------------------------
    # 3. Neural bathymetry
    # ------------------------------------------------------------------
    bathy_meta = _load_json(_CHECKPOINTS_DIR / "bathymetry_metadata_loo.json")
    bathy_neural = _load_json(_CHECKPOINTS_DIR / "bathymetry_loo_neural.json")
    bathy_transfer = _load_json(
        _CHECKPOINTS_DIR / "bathymetry_loo_neural_transfer.json"
    )
    report["neural_bathymetry"] = {
        "status": "shadow",
        "display": "Neural lake-bed elevation / volume estimator",
        "gate": "<15% MAPE on grouped leave-one-lake-out volume estimation",
        "gate_passed": bool(
            bathy_meta and bathy_meta.get("regression_passes_gate")
        ),
        "current_metric": {
            "huggel_overall_mape": bathy_meta.get("overall", {}).get(
                "huggel_mape"
            ) if bathy_meta else None,
            "regression_overall_mape": bathy_meta.get("overall", {}).get(
                "regression_mape"
            ) if bathy_meta else None,
            "neural_random_init_mape": bathy_neural.get("mean_volume_mape")
            if bathy_neural else None,
            "neural_transfer_mape": bathy_transfer.get("mean_volume_mape")
            if bathy_transfer else None,
        },
        "blocker": (
            "Best available method (Huggel) is 75.6% MAPE; neural U-Net "
            "overfits 20 lakes (676% MAPE). Needs ≥60 surveyed lakes or "
            "simpler terrain-feature regression."
        ),
    }

    # ------------------------------------------------------------------
    # 4. Multi-modal SAR + optical fusion
    # ------------------------------------------------------------------
    s2_eval = _load_json(_CHECKPOINTS_DIR / "s2_spectral_eval_20251122.json")
    report["multi_modal_fusion"] = {
        "status": "shadow",
        "display": "SAR + Sentinel-2 optical fusion segmenter",
        "gate": "Event-held-out IoU > 0.75 AND P ≥ 0.85 on real paired data",
        "gate_passed": False,
        "current_metric": {
            "mndwi_lake_vs_glacier_auc": s2_eval.get("mndwi", {}).get("auc")
            if s2_eval else None,
        },
        "blocker": (
            "Optical NDWI/MNDWI does not separate Imja lake from glacier "
            "(AUC 0.18–0.29). Need clean S1+S2 pairs and a glacier-robust "
            "optical discriminator before training is justified."
        ),
    }

    # ------------------------------------------------------------------
    # 5. Latent-conditioned FNO
    # ------------------------------------------------------------------
    report["latent_fno"] = {
        "status": "shadow",
        "display": "Latent-conditioned FNO hydrodynamic surrogate",
        "gate": "MAPE ≤ 20% on ≥2/3 real events; no event >30%",
        "gate_passed": False,
        "current_metric": None,
        "blocker": (
            "No real-terrain training corpus. Previous 12.3% MAPE was on a "
            "synthetic parametric corridor and is disqualified. Needs "
            "GeoClaw-style simulations on South Lhonak/Imja DEMs."
        ),
    }

    # ------------------------------------------------------------------
    # 6. Susceptibility (advisory-primary)
    # ------------------------------------------------------------------
    sus_rec = promotion_record("susceptibility")
    sus_report = _load_json(_CHECKPOINTS_DIR / "xgboost_spatial_cv_report.json")
    if sus_rec is not None:
        report["susceptibility"] = {
            "status": sus_rec.get("level", "advisory_primary"),
            "demoted": is_demoted("susceptibility"),
            "display": "Static lake-breach susceptibility prior",
            "gate": sus_rec.get("gate"),
            "gate_passed": bool(
                sus_report and sus_report.get("passes_brier_gate")
            ),
            "current_metric": {
                "mean_roc_auc": sus_report.get("mean_roc_auc")
                if sus_report else None,
                "mean_brier_calibrated": sus_report.get("mean_brier_calibrated")
                if sus_report else None,
            },
            "checkpoint": sus_rec.get("checkpoint"),
            "promoted_at": sus_rec.get("promoted_at"),
            "caveat": sus_rec.get("caveat"),
            "blocker": (
                "Advisory prior only; does not change deterministic severity "
                "or bypass human review."
            ),
            "evidence_files": [str(p) for p in sus_rec.get("gate_evidence", [])],
        }
    else:
        report["susceptibility"] = {
            "status": "shadow",
            "display": "Static lake-breach susceptibility prior",
            "gate": "Brier < 0.15 on calibrated score",
            "gate_passed": False,
            "blocker": "No promoted susceptibility component registered.",
        }

    # ------------------------------------------------------------------
    # 7. Dynamic escalation (advisory-primary)
    # ------------------------------------------------------------------
    esc_rec = promotion_record("dynamic_escalation")
    esc_report = _load_json(
        _CHECKPOINTS_DIR / "dynamic_escalation_eval_report.json"
    )
    if esc_rec is not None:
        report["dynamic_escalation"] = {
            "status": esc_rec.get("level", "advisory_primary"),
            "demoted": is_demoted("dynamic_escalation"),
            "display": "Tier-2 weather/morphology escalation scorer",
            "gate": esc_rec.get("gate"),
            "gate_passed": bool(
                esc_report and esc_report.get("passes_brier_gate")
            ),
            "current_metric": {
                "mean_roc_auc": esc_report.get("mean_roc_auc")
                if esc_report else None,
                "mean_brier": esc_report.get("mean_brier")
                if esc_report else None,
            },
            "checkpoint": esc_rec.get("checkpoint"),
            "promoted_at": esc_rec.get("promoted_at"),
            "caveat": esc_rec.get("caveat"),
            "blocker": (
                "Advisory reason only; landslide/avalanche triggers are "
                "outside the weather-feature contract."
            ),
            "evidence_files": [str(p) for p in esc_rec.get("gate_evidence", [])],
        }
    else:
        report["dynamic_escalation"] = {
            "status": "shadow",
            "display": "Tier-2 weather/morphology escalation scorer",
            "gate": "Brier < 0.15 on spatio-temporal holdout",
            "gate_passed": False,
            "blocker": "No promoted dynamic escalation component registered.",
        }

    # ------------------------------------------------------------------
    # 8. Learned risk fusion (advisory-primary)
    # ------------------------------------------------------------------
    fus_rec = promotion_record("learned_risk_fusion")
    fus_report = _load_json(
        _CHECKPOINTS_DIR / "risk_fusion_eval_report.json"
    )
    if fus_rec is not None:
        report["learned_risk_fusion"] = {
            "status": fus_rec.get("level", "advisory_primary"),
            "demoted": is_demoted("learned_risk_fusion"),
            "display": "Learned severity / risk-fusion classifier",
            "gate": fus_rec.get("gate"),
            "gate_passed": bool(
                fus_report and fus_report.get("gate_passed")
            ),
            "current_metric": {
                "mean_roc_auc": fus_report.get("mean_roc_auc")
                if fus_report else None,
                "mean_roc_auc_baseline": (
                    fus_report.get("mean_roc_auc_baseline")
                    if fus_report else None
                ),
                "mean_brier": fus_report.get("mean_brier")
                if fus_report else None,
            },
            "checkpoint": fus_rec.get("checkpoint"),
            "promoted_at": fus_rec.get("promoted_at"),
            "caveat": fus_rec.get("caveat"),
            "blocker": (
                "Advisory reason only; deterministic classify_severity "
                "stays authoritative. Operational promotion requires a "
                "separate severity-mapping evaluation."
            ),
            "evidence_files": [str(p) for p in fus_rec.get("gate_evidence", [])],
        }
    else:
        report["learned_risk_fusion"] = {
            "status": "shadow",
            "display": "Learned severity / risk-fusion classifier",
            "gate": "Brier < 0.15 on held-out real events; beats deterministic baseline",
            "gate_passed": False,
            "current_metric": None,
            "blocker": "Not started. Depends on upstream neural outputs being trustworthy.",
        }

    # ------------------------------------------------------------------
    # Overall readiness
    # ------------------------------------------------------------------
    operational_primary_count = sum(
        1 for r in report.values() if r.get("status") == "operational_primary"
    )
    advisory_primary_count = sum(
        1 for r in report.values() if r.get("status") == "advisory_primary"
    )
    from siren.scope import scope_summary

    report["_summary"] = {
        "dl_primary_ready": False,
        "operational_primary_components": operational_primary_count,
        "advisory_primary_components": advisory_primary_count,
        "shadow_components": len(report) - operational_primary_count - advisory_primary_count,
        "total_components": len(report),
        "next_recommended_level": 2,  # See docs/spec/DL_PRIMARY_ROADMAP.md
        "operational_scope": scope_summary(),
        "demoted_components": sorted(
            c for c in demoted_components() if c in PROMOTED_COMPONENTS
        ),
        "demote_env_var": DEMOTE_ENV_VAR,
        "note": (
            "DL-primary is the declared target architecture. "
            "sar_segmentation_expansion is operational-primary within "
            "its certified scope (descending orbit, monsoon Jun-Sep, "
            "Imja-area monitorable lakes) and supplies the load-bearing "
            "water-area expansion measurement; the deterministic "
            "pipeline remains the labeled cross-check and stays "
            "authoritative out of scope. Other components are "
            "advisory-primary or shadow."
        ),
    }
    return report
