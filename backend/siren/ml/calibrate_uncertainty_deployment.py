"""Deployment-domain conformal calibration of MC Dropout (E1, §9.8).

The Kuro Siwo ``conformal_calibration.json`` calibrates q* on Kuro Siwo
test chips — lowland flood events, out of the deployment domain. For the
adapter promotion gate the quantile must be re-calibrated on
deployment-domain data: Himalayan lake chips extracted from a held-out
ro-121 descending pair (``himalayan_lake_dataset.build_chip_dataset``).

Split discipline (PRD §17.1 — the deployment analog of the Kuro Siwo
event split): chips are grouped by ``lake_id`` and whole lakes are
assigned to the calibration/evaluation halves by seeded shuffle — no
lake contributes chips to both sides. Background chips are seeded-split
by chip_id.

Label caveat (honest): targets are WEAK labels — median-outlined
2022–2024 inventory polygons, not scene-date water truth. The measured
coverage is therefore coverage-vs-weak-labels; it validates that the MC
spread is calibrated to the model's deployment-domain error scale, not
to true segmentation error. Documented in the output JSON.

Usage:
    python -m siren.ml.calibrate_uncertainty_deployment \
        --checkpoint models/checkpoints/\
            water_resunet_6ch_himalayan_adapter_stratified.pt \
        --pair unfrozen_desc2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    _REPO_ROOT
    / "models"
    / "checkpoints"
    / "water_resunet_6ch_himalayan_adapter_stratified.pt"
)


def _deployment_chips(
    pair_name: str, out_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, list[dict], Path]:
    """Build (or reuse) the chip dataset for a held-out pair.

    Returns (x, y, manifest, chip_dir): float32 (N,6,C,C) inputs, uint8
    (N,C,C) weak labels, and the per-chip manifest records.
    """
    from siren.ml.heldout_eval import PAIRS, calibrated_cache
    from siren.ml.himalayan_lake_dataset import build_chip_dataset

    t0_str, t1_str = PAIRS[pair_name]
    tag = "asc" if pair_name.endswith("_asc") else "desc"
    chip_dir = out_dir or (
        _REPO_ROOT / "data" / "datasets" / f"himalayan_chips_{pair_name}"
    )
    npz = chip_dir / "chips.npz"
    manifest_path = chip_dir / "manifest.json"
    if npz.exists() and manifest_path.exists():
        logger.info("reusing chip set %s", chip_dir)
    else:
        build_chip_dataset(
            calibrated_cache(t0_str, tag),
            calibrated_cache(t1_str, tag),
            out_dir=chip_dir,
        )
    data = np.load(npz)
    manifest = json.loads(manifest_path.read_text())
    return data["x"], data["y"], manifest, chip_dir


def _split_by_lake(
    manifest: list[dict], seed: int
) -> tuple[list[int], list[int], list[str], list[str]]:
    """Seeded 50/50 split of chips by lake_id — no lake on both sides."""
    lake_ids = sorted({m["lake_id"] for m in manifest if m["lake_id"]})
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(lake_ids))
    cal_lakes = {lake_ids[i] for i in order[::2]}
    eval_lakes = {lake_ids[i] for i in order[1::2]}

    bg = [i for i, m in enumerate(manifest) if not m["lake_id"]]
    bg_order = rng.permutation(len(bg))
    cal_bg = {bg[i] for i in bg_order[::2]}

    cal_idx, eval_idx = [], []
    for i, m in enumerate(manifest):
        lid = m["lake_id"]
        if lid:
            (cal_idx if lid in cal_lakes else eval_idx).append(i)
        else:
            (cal_idx if i in cal_bg else eval_idx).append(i)
    return cal_idx, eval_idx, sorted(cal_lakes), sorted(eval_lakes)


def run_deployment_calibration(
    checkpoint: Path,
    pair_name: str = "unfrozen_desc2",
    dropout: float = 0.10,
    n_samples: int = 20,
    max_cal_chips: int = 256,
    max_eval_chips: int = 256,
    confidence_level: float = 0.90,
    seed: int = 42,
    device: str | None = None,
    out_path: Path | None = None,
) -> dict:
    """Run the deployment-domain E1 conformal evaluation."""
    import torch

    from siren.ml.engine import _detect_architecture
    from siren.ml.model import WaterResUNet
    from siren.ml.uncertainty import calibrate_conformal, evaluate_coverage

    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    arch, in_ch, base = _detect_architecture(state)
    model = WaterResUNet(
        in_channels=in_ch, base_channels=base, dropout=dropout
    ).to(device_str)
    model.load_state_dict(state)
    model.eval()

    x, y, manifest, chip_dir = _deployment_chips(pair_name)
    cal_idx, eval_idx, cal_lakes, eval_lakes = _split_by_lake(manifest, seed)
    cal_idx, eval_idx = cal_idx[:max_cal_chips], eval_idx[:max_eval_chips]
    logger.info(
        "pair=%s chips=%d (%d cal / %d eval; %d/%d lakes) device=%s",
        pair_name, len(x), len(cal_idx), len(eval_idx),
        len(cal_lakes), len(eval_lakes), device_str,
    )

    t0 = time.time()
    cal_inputs = [torch.from_numpy(x[i:i+1]).to(device_str) for i in cal_idx]
    cal_targets = [y[i].astype(np.float32) for i in cal_idx]
    eval_inputs = [torch.from_numpy(x[i:i+1]).to(device_str) for i in eval_idx]
    eval_targets = [y[i].astype(np.float32) for i in eval_idx]
    # Full-chip labels — no nodata mask needed; every pixel is labelled
    # (weakly). Kept explicit for parity with the Kuro Siwo path.
    cal_valids = [np.ones_like(t) for t in cal_targets]
    eval_valids = [np.ones_like(t) for t in eval_targets]

    torch.manual_seed(seed)
    if device_str != "cpu":
        torch.cuda.manual_seed_all(seed)
    q_star = calibrate_conformal(
        model, cal_inputs, cal_targets,
        n_samples=n_samples, confidence_level=confidence_level,
        valid_masks=cal_valids,
    )
    logger.info("Deployment conformal quantile q* = %.6f", q_star)

    torch.manual_seed(seed + 1)
    if device_str != "cpu":
        torch.cuda.manual_seed_all(seed + 1)
    coverage = evaluate_coverage(
        model, eval_inputs, eval_targets,
        conformal_quantile=q_star, n_samples=n_samples,
        confidence_level=confidence_level, valid_masks=eval_valids,
    )

    gate_passed = coverage["coverage_error"] <= 0.05
    result = {
        "method": "split_conformal_mc_dropout_deployment_domain",
        "checkpoint": checkpoint.name,
        "pair": pair_name,
        "chip_dir": str(chip_dir),
        "dropout": dropout,
        "n_samples": n_samples,
        "confidence_level": confidence_level,
        "conformal_quantile": float(q_star),
        **coverage,
        "gate_passed": gate_passed,
        "gate_criterion": "|empirical_coverage − 0.90| ≤ 0.05 (PRD §17.2)",
        "split": "by lake_id (no lake on both sides — PRD §17.1)",
        "n_cal_chips": len(cal_idx),
        "n_eval_chips": len(eval_idx),
        "n_cal_lakes": len(cal_lakes),
        "n_eval_lakes": len(eval_lakes),
        "label_semantics": (
            "WEAK: median-outlined 2022–2024 inventory polygons — "
            "coverage is vs weak labels, validating MC-spread "
            "calibration to deployment-domain error scale, not truth"
        ),
        "seed": seed,
        "device": device_str,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out = out_path or (
        checkpoint.parent / f"{checkpoint.stem}_conformal_deployment.json"
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    logger.info(
        "Deployment E1 conformal gate %s: coverage=%.4f (nominal %.2f, "
        "error %.4f) — wrote %s",
        "PASSED" if gate_passed else "FAILED",
        coverage["empirical_coverage"], confidence_level,
        coverage["coverage_error"], out,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pair", default="unfrozen_desc2",
                        help="heldout_eval PAIRS key supplying the chips")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--max-cal-chips", type=int, default=256)
    parser.add_argument("--max-eval-chips", type=int, default=256)
    parser.add_argument("--confidence-level", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run_deployment_calibration(
        checkpoint=args.checkpoint, pair_name=args.pair,
        dropout=args.dropout, n_samples=args.n_samples,
        max_cal_chips=args.max_cal_chips, max_eval_chips=args.max_eval_chips,
        confidence_level=args.confidence_level, seed=args.seed,
        device=args.device, out_path=args.out,
    )
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
