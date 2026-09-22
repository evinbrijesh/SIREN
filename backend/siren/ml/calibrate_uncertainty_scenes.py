"""Whole-scene deployment-domain conformal calibration (E1, Level 2.2).

``calibrate_uncertainty_deployment.py`` calibrates on lake chips from a
held-out pair — chip-level, weak inventory labels. The Level 2 gate asks
for the deployment-domain quantity: MC Dropout run over WHOLE Imja-area
SAR scenes with coverage measured against the gold S2/analyst labels on
the scene grid, on ≥3 held-out scenes.

Design:
  * Scenes = the t1 date of every in-scope, gold-complete eval pair
    (monsoon window + descending orbit + unfrozen surface — the declared
    operational scope in siren/scope.py). Currently 3 scenes:
    monsoon_2025_desc, monsoon_2025_desc2, unfrozen_desc.
  * Per scene the dropout-native checkpoint runs T MC passes over the
    full swath; the conformal score is |mean_prob − gold_label| on the
    gold-labelled pixels (S2-derived water masks sampled to the SAR
    grid, nodata excluded).
  * Coverage is evaluated leave-one-scene-out: for each held-out scene
    q* is calibrated on the other scenes' labelled pixels — no scene
    contributes to its own quantile. Per-scene coverage dispersion is
    the honest scene-level evidence; the pooled q* over all scenes is
    the deployable quantile written to the per-checkpoint sidecar
    (<stem>.conformal.json, loaded by ChangeDetectionEngine).

Caveats (recorded in the report):
  * Labelled pixels are spatially correlated — the per-pixel score
    distribution has a much smaller effective sample size than its raw
    count; the LOSO scene split is the unit of honest coverage.
  * Labels are S2-SCL/analyst water masks at ~20 m resampled to the SAR
    grid — coverage is vs those labels, not literal truth.

Usage:
    python -m siren.ml.calibrate_uncertainty_scenes \
        --checkpoint models/checkpoints/\
water_resunet_6ch_himalayan_adapter_multidate_mc.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_OUT = (
    REPO_ROOT / "models" / "checkpoints" / "imja_scene_conformal_eval.json"
)
GATE_COVERAGE_TOL = 0.05   # |empirical − nominal| ≤ 0.05 (PRD §17.2)
MIN_SCENES = 3


def _in_scope_gold_pairs() -> list[dict]:
    """Eval-eligible pairs that are gold-complete AND inside the
    declared operational scope (monsoon + descending + unfrozen)."""
    from siren.ml.label_registry import eval_eligible_pairs
    from siren.ml.operational_gate_eval import _pair_in_scope

    out = []
    for entry in eval_eligible_pairs("eval"):
        if not entry["gold_complete"]:
            continue
        in_scope, note = _pair_in_scope(entry)
        if in_scope:
            out.append(entry)
        else:
            logger.info("pair %s out of scope (%s) — skipped",
                        entry["pair"], note)
    return out


def _scene_mean_prob(engine, pre_db: np.ndarray, post_db: np.ndarray,
                     n_samples: int, seed: int) -> np.ndarray:
    """T-pass MC Dropout mean water probability for the whole scene."""
    unc = engine.predict_change_uncertainty(
        pre_db, post_db, n_samples=n_samples, seed=seed
    )
    return unc["water_prob_t1"].astype(np.float64)


def _scene_labels(t1_date: str, cache_t1: Path):
    """Gold (water, valid) masks sampled onto the SAR grid."""
    from siren.detect.sar import sar_grid_lonlat
    from siren.ml.imja_gold_eval import _labels_on_grid
    from siren.ml.label_registry import GOLD_LABELS

    ll = sar_grid_lonlat(str(cache_t1))
    if ll is None:
        raise RuntimeError(f"no GCP geolocation in {cache_t1}")
    lon, lat = ll
    lab = _labels_on_grid(GOLD_LABELS.get(t1_date), {"lon": lon, "lat": lat})
    if lab is None:
        raise RuntimeError(f"no gold label raster for {t1_date}")
    return lab  # (water bool, valid bool)


def run_scene_conformal(
    checkpoint: Path,
    n_samples: int = 20,
    confidence_level: float = 0.90,
    seed: int = 42,
    dropout: float = 0.10,
    out_path: Path | None = None,
) -> dict:
    """Leave-one-scene-out conformal coverage on in-scope gold scenes."""
    from siren.ml.engine import ChangeDetectionEngine
    from siren.ml.heldout_eval import calibrated_cache

    pairs = _in_scope_gold_pairs()
    if len(pairs) < MIN_SCENES:
        return {
            "status": "insufficient_scenes",
            "reason": (
                f"{len(pairs)} in-scope gold-complete scenes < "
                f"{MIN_SCENES} required"
            ),
            "gate_passed": False,
        }

    engine = ChangeDetectionEngine(
        weights_path=checkpoint, dropout=dropout
    )
    if not engine.is_ready:
        raise RuntimeError(f"checkpoint {checkpoint} failed to load")
    if not engine.has_dropout_layers():
        raise RuntimeError(
            f"engine has no dropout layers (dropout={dropout}) — MC "
            "variance would be degenerate"
        )

    alpha = 1.0 - confidence_level
    t_start = time.time()
    scenes: list[dict] = []
    for entry in pairs:
        tag = "asc" if entry["pair"].endswith("_asc") else "desc"
        cache_t0 = calibrated_cache(entry["t0"], tag)
        cache_t1 = calibrated_cache(entry["t1"], tag)
        with rasterio.open(cache_t0) as d:
            pre_db = d.read().astype(np.float32)
        with rasterio.open(cache_t1) as d:
            post_db = d.read().astype(np.float32)

        mean_p = _scene_mean_prob(engine, pre_db, post_db, n_samples, seed)
        water, valid = _scene_labels(entry["t1"], cache_t1)
        scores = np.abs(mean_p[valid] - water[valid].astype(np.float64))
        scenes.append({
            "pair": entry["pair"],
            "t0": entry["t0"],
            "t1": entry["t1"],
            "n_labelled_px": int(valid.sum()),
            "label_water_frac": round(
                float(water[valid].mean()), 4
            ),
            "mean_prob_labelled": round(float(mean_p[valid].mean()), 4),
            "scores": scores,
        })
        logger.info(
            "%s (t1=%s): %d labelled px, water-frac %.3f",
            entry["pair"], entry["t1"], int(valid.sum()),
            float(water[valid].mean()),
        )

    # Leave-one-scene-out coverage: q* from the other scenes' scores.
    per_scene: list[dict] = []
    for i, s in enumerate(scenes):
        cal_scores = np.concatenate(
            [scenes[j]["scores"] for j in range(len(scenes)) if j != i]
        )
        n = len(cal_scores)
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        q_star = float(np.quantile(cal_scores, q_level))
        cov = float(np.mean(s["scores"] <= q_star))
        per_scene.append({
            "pair": s["pair"],
            "t1": s["t1"],
            "n_labelled_px": s["n_labelled_px"],
            "q_star_from_other_scenes": round(q_star, 6),
            "empirical_coverage": round(cov, 4),
            "coverage_error": round(abs(cov - confidence_level), 4),
        })

    # Deployable quantile: pooled calibration over all scenes.
    all_scores = np.concatenate([s["scores"] for s in scenes])
    n_all = len(all_scores)
    q_level_all = min(np.ceil((n_all + 1) * (1 - alpha)) / n_all, 1.0)
    q_star_pooled = float(np.quantile(all_scores, q_level_all))
    pooled_cov = float(np.mean(all_scores <= q_star_pooled))

    coverages = [s["empirical_coverage"] for s in per_scene]
    gate_passed = all(
        abs(c - confidence_level) <= GATE_COVERAGE_TOL for c in coverages
    )

    report = {
        "method": "split_conformal_mc_dropout_whole_scene_loso",
        "checkpoint": checkpoint.name,
        "dropout_native": True,
        "dropout_rate": dropout,
        "n_samples": n_samples,
        "confidence_level": confidence_level,
        "conformal_quantile_pooled": round(q_star_pooled, 6),
        "pooled_coverage_on_calibration": round(pooled_cov, 4),
        "per_scene": per_scene,
        "n_scenes": len(scenes),
        "min_scene_coverage": round(min(coverages), 4),
        "max_scene_coverage": round(max(coverages), 4),
        "gate_passed": gate_passed,
        "gate_criterion": (
            f"per-scene LOSO coverage within ±{GATE_COVERAGE_TOL} of "
            f"nominal {confidence_level} on ≥{MIN_SCENES} held-out "
            "Imja-area scenes (DL_PRIMARY_ROADMAP Level 2.2)"
        ),
        "split": (
            "leave-one-scene-out — no scene contributes pixels to its "
            "own conformal quantile"
        ),
        "label_semantics": (
            "gold S2-SCL/analyst water masks sampled to the SAR grid; "
            "coverage is vs those labels on labelled pixels only"
        ),
        "caveats": [
            "Labelled pixels within a scene are spatially correlated — "
            "effective sample size ≪ pixel count; per-scene coverage is "
            "the honest unit.",
            "3 scenes is the minimum the roadmap accepts — coverage "
            "dispersion across a larger scene set is unmeasured.",
        ],
        "seed": seed,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    out = out_path or REPORT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    # Per-checkpoint conformal sidecar — <stem>.conformal.json, which the
    # engine prefers over a directory-level file.
    sidecar = checkpoint.with_suffix(".conformal.json")
    sidecar.write_text(json.dumps({
        "method": "split_conformal_mc_dropout_whole_scene_loso",
        "conformal_quantile": round(q_star_pooled, 6),
        "confidence_level": confidence_level,
        "gate_passed": gate_passed,
        "calibration_domain": "imja_whole_scenes_gold_labels",
        "n_cal_scenes": len(scenes),
        "n_samples": n_samples,
        "dropout_rate": dropout,
        "report": out.name,
    }, indent=2) + "\n")
    logger.info(
        "Scene conformal %s: pooled q*=%.4f, per-scene coverage %s — "
        "sidecar %s",
        "PASSED" if gate_passed else "FAILED",
        q_star_pooled,
        [s["empirical_coverage"] for s in per_scene],
        sidecar,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--confidence-level", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPORT_OUT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = run_scene_conformal(
        checkpoint=args.checkpoint,
        n_samples=args.n_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        dropout=args.dropout,
        out_path=args.out,
    )
    print(json.dumps({
        "gate_passed": report.get("gate_passed"),
        "per_scene": report.get("per_scene"),
        "conformal_quantile_pooled": report.get("conformal_quantile_pooled"),
    }, indent=2))
    return 0 if report.get("gate_passed") else 1


if __name__ == "__main__":
    sys.exit(main())
