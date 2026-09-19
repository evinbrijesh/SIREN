"""Residual bathymetry corrector — Huggel baseline + learned correction.

Replaces the abandoned BathymetryUNet (676% LOO MAPE at N=19) with a
low-capacity residual model over the physical baseline:

    V_pred = V_huggel(A) * exp(f_theta(X))

where f_theta is a Gaussian Process regressor (or ridge fallback) on
deployable features only — log area, lake type, mountain region. Max
depth is deliberately excluded: it is a survey measurement unavailable
for unsurveyed lakes at deployment.

The GPR yields an analytic predictive std → a 95% volume interval, which
the audit/explainability contract rewards (uncertainty on the record).

Evaluation: grouped leave-one-lake-out over the global compilation —
identical protocol to run_metadata_loo_benchmark so numbers are
comparable (all survey-year rows of a lake held out together).

Usage:
    python -m siren.ml.bathymetry_residual            # benchmark + report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from siren.ml.bathymetry_benchmark import (
    _normalise_lake_name,
    huggel_volume_m3,
    run_metadata_loo_benchmark,
)
from siren.ml.bathymetry_dataset import load_global_compilation

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = (
    REPO_ROOT / "models" / "checkpoints" / "bathymetry_residual_loo.json"
)
MIN_REGION_COUNT = 5          # rare regions → "other"
GATE_TARGET_MAPE = 0.15


def _feature_matrix(entries: list) -> tuple[np.ndarray, list[str]]:
    """Deployable features: log area + one-hot lake_type + region."""
    types = sorted({e.lake_type for e in entries})
    regions = [e.mountain for e in entries]
    counts = pd.Series(regions).value_counts() if entries else pd.Series()
    region_cats = sorted(
        c for c in set(regions) if counts.get(c, 0) >= MIN_REGION_COUNT
    )
    names = (
        ["log_area_km2"]
        + [f"type_{t}" for t in types]
        + [f"region_{r}" for r in region_cats]
    )
    X = np.zeros((len(entries), len(names)))
    for i, e in enumerate(entries):
        col = 0
        X[i, col] = np.log(e.area_km2)
        col += 1
        for t in types:
            if e.lake_type == t:
                X[i, col] = 1.0
            col += 1
        for r in region_cats:
            if e.mountain == r:
                X[i, col] = 1.0
            col += 1
    return X, names


def run_residual_loo_benchmark(
    entries: list | None = None,
    gate_target_mape: float = GATE_TARGET_MAPE,
) -> dict[str, Any]:
    """Grouped-LOO benchmark of the Huggel-residual corrector.

    Folds mirror run_metadata_loo_benchmark; adds per-fold GPR-residual
    and ridge-residual predictions + predictive-interval coverage.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        WhiteKernel,
    )
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if entries is None:
        entries = load_global_compilation()

    evaluable = [
        e for e in entries
        if e.area_km2 is not None and e.area_km2 > 0
        and e.volume_mcm is not None and e.volume_mcm > 0
    ]
    groups: dict[str, list] = {}
    for e in evaluable:
        groups.setdefault(_normalise_lake_name(e.name), []).append(e)

    X_all, feature_names = _feature_matrix(evaluable)
    eval_index = {id(e): i for i, e in enumerate(evaluable)}
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3)

    folds: list[dict[str, Any]] = []
    for hold_name in sorted(groups):
        held = groups[hold_name]
        held_ids = {id(e) for e in held}
        tr_idx = [i for i, e in enumerate(evaluable) if id(e) not in held_ids]
        train = [evaluable[i] for i in tr_idx]

        X_tr = X_all[tr_idx]
        huggel_tr = np.array(
            [huggel_volume_m3(e.area_km2) for e in train])
        y_tr = np.array([e.volume_mcm * 1e6 for e in train])
        resid_tr = np.log(np.maximum(y_tr, 1.0)) - np.log(
            np.maximum(huggel_tr, 1.0))

        scaler = StandardScaler().fit(X_tr)
        Xs = scaler.transform(X_tr)

        gpr = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=42,
            n_restarts_optimizer=1,
        )
        ridge = Ridge(alpha=1.0)
        try:
            gpr.fit(Xs, resid_tr)
            gpr_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("GPR fit failed on %s fold: %s", hold_name, exc)
            gpr_ok = False
        ridge.fit(Xs, resid_tr)

        for e in held:
            i = eval_index[id(e)]
            gt = e.volume_mcm * 1e6
            hv = huggel_volume_m3(e.area_km2)
            xt = scaler.transform(X_all[i : i + 1])
            if gpr_ok:
                res, std = gpr.predict(xt, return_std=True)
                gpr_v = hv * float(np.exp(res[0]))
                lo = hv * float(np.exp(res[0] - 1.96 * std[0]))
                hi = hv * float(np.exp(res[0] + 1.96 * std[0]))
                covered = bool(lo <= gt <= hi)
            else:
                gpr_v, lo, hi, covered = np.nan, np.nan, np.nan, None
            ridge_v = hv * float(np.exp(ridge.predict(xt)[0]))
            folds.append({
                "lake_name": e.name,
                "lake_type": e.lake_type,
                "mountain": e.mountain,
                "area_km2": e.area_km2,
                "ground_truth_volume_m3": round(gt, 1),
                "huggel_ape": abs(hv - gt) / gt,
                "gpr_ape": (
                    abs(gpr_v - gt) / gt if gpr_ok else None
                ),
                "gpr_interval_covers": covered,
                "ridge_ape": abs(ridge_v - gt) / gt,
            })

    def _mape(subset, key):
        vals = [f[key] for f in subset if f[key] is not None]
        return float(np.mean(vals)) if vals else None

    def _summarise(subset):
        cov = [f["gpr_interval_covers"] for f in subset
               if f["gpr_interval_covers"] is not None]
        return {
            "n_entries": len(subset),
            "huggel_mape": _mape(subset, "huggel_ape"),
            "gpr_mape": _mape(subset, "gpr_ape"),
            "gpr_median_ape": (
                float(np.median([f["gpr_ape"] for f in subset
                                 if f["gpr_ape"] is not None]))
                if any(f["gpr_ape"] is not None for f in subset)
                else None
            ),
            "ridge_mape": _mape(subset, "ridge_ape"),
            "gpr_interval_coverage_95": float(np.mean(cov)) if cov else None,
        }

    himalaya = [f for f in folds if "himalaya" in f["mountain"].lower()]
    by_type = {
        t: _summarise([f for f in folds if f["lake_type"] == t])
        for t in sorted({f["lake_type"] for f in folds})
    }
    overall = _summarise(folds)

    return {
        "status": "benchmarked",
        "model": "huggel_residual_gpr",
        "target": "log(V) - log(V_huggel); V_pred = V_huggel * exp(f(X))",
        "features": feature_names,
        "excluded_features": [
            "max_depth_m (survey-only, unavailable at deployment)"
        ],
        "n_entries_evaluable": len(evaluable),
        "n_unique_lakes": len(groups),
        "gate_target_mape": gate_target_mape,
        "overall": overall,
        "himalaya_subset": _summarise(himalaya),
        "by_lake_type": by_type,
        "gpr_passes_gate_himalaya": bool(
            _summarise(himalaya)["gpr_mape"] is not None
            and _summarise(himalaya)["gpr_mape"] < gate_target_mape
        ),
        "folds": folds,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=REPORT_PATH)
    p.add_argument("--eval", action="store_true",
                   help="print metrics without writing the report")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    entries = load_global_compilation()
    if not entries:
        logger.error("Global compilation not loaded — no entries")
        return 1

    # Baseline reference (same folds) for side-by-side in the report
    baseline = run_metadata_loo_benchmark(entries)
    result = run_residual_loo_benchmark(entries)
    result["baseline_reference"] = {
        "huggel_mape": baseline["overall"]["huggel_mape"],
        "regression_mape": baseline["overall"]["regression_mape"],
        "himalaya_huggel_mape": baseline["himalaya_subset"]["huggel_mape"],
        "himalaya_regression_mape": (
            baseline["himalaya_subset"]["regression_mape"]
        ),
    }

    ov, hm = result["overall"], result["himalaya_subset"]
    print("\n" + "=" * 64)
    print("Bathymetry residual corrector — grouped LOO")
    print("=" * 64)
    print(f"  Overall   ({ov['n_entries']} entries): "
          f"Huggel {ov['huggel_mape']:.3f} | "
          f"GPR {ov['gpr_mape']:.3f} | ridge {ov['ridge_mape']:.3f} | "
          f"95%-interval coverage {ov['gpr_interval_coverage_95']:.2f}")
    print(f"  Himalaya  ({hm['n_entries']} entries): "
          f"Huggel {hm['huggel_mape']:.3f} | "
          f"GPR {hm['gpr_mape']:.3f} | ridge {hm['ridge_mape']:.3f} | "
          f"coverage {hm['gpr_interval_coverage_95']:.2f}")
    print(f"  Gate <{GATE_TARGET_MAPE:.2f} on Himalaya subset: "
          f"{'PASS' if result['gpr_passes_gate_himalaya'] else 'FAIL'}")
    print("=" * 64)

    if not args.eval:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        logger.info("Report written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
