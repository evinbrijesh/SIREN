"""Split-conformal calibration of the MC Dropout uncertainty layer (E1).

Evaluates the ADR-013 / PRD §17.2 E1 promotion gate: empirical conformal
coverage within ±5% of the nominal 90% level on a held-out evaluation set.

Method (Angelopoulos & Bates 2021, split conformal):

  1. The Kuro Siwo TEST split is divided by flood event — no event
     contributes chips to both the calibration and the evaluation side
     (PRD §17.1: no adjacent-chip leakage). Events are assigned to the two
     sides by a seeded shuffle.
  2. ``calibrate_conformal`` runs T MC Dropout passes per calibration chip
     and computes the per-pixel nonconformity score |mean_prob − target|
     on valid pixels only. The conformal quantile q* is the
     ceil((n+1)(1−α))/n order statistic of the pooled scores.
  3. ``evaluate_coverage`` measures the fraction of held-out evaluation
     pixels whose ground truth falls inside [mean − q*, mean + q*].
  4. Gate: |empirical_coverage − 0.90| ≤ 0.05.

The checkpoint was trained with ``dropout=0.0``; Dropout2d is
parameter-free, so the same state_dict loads into a ``dropout>0`` model
and MC Dropout produces a real (post-hoc) uncertainty estimate. The result
is written to ``conformal_calibration.json`` next to the checkpoint —
``ChangeDetectionEngine`` picks it up as the default conformal quantile.

Usage:
    python -m siren.ml.calibrate_uncertainty \\
        --checkpoint models/checkpoints/water_resunet_kuro_siwo_full/\\
            water_resunet_6ch_kuro_siwo_v1.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2].parent
    / "models"
    / "checkpoints"
    / "water_resunet_kuro_siwo_full"
    / "water_resunet_6ch_kuro_siwo_v1.pt"
)


def _split_indices_by_event(
    dataset, seed: int
) -> tuple[list[int], list[int], list[str], list[str]]:
    """Split sample indices into (cal, eval) halves by flood event.

    Deterministic seeded shuffle over the sorted event list, alternating
    assignment — guarantees no event appears on both sides.
    """
    events = sorted(dataset.events())
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(events))
    cal_events = sorted(str(events[i]) for i in order[::2])
    eval_events = sorted(str(events[i]) for i in order[1::2])
    cal_set, eval_set = set(cal_events), set(eval_events)

    cal_idx, eval_idx = [], []
    for i in range(len(dataset)):
        ev = dataset._index[i][1]["sample_id"].split("_")[0]
        if ev in cal_set:
            cal_idx.append(i)
        elif ev in eval_set:
            eval_idx.append(i)
    return cal_idx, eval_idx, cal_events, eval_events


def _load_chips(dataset, indices: list[int], device_str: str):
    """Load chips as (inputs, targets, valid_masks) for uncertainty.py."""
    import torch

    inputs, targets, valids = [], [], []
    for i in indices:
        sample = dataset[i]
        inputs.append(sample["sar"].unsqueeze(0).to(device_str))
        targets.append(sample["water"].numpy())
        valids.append(sample["valid"].numpy())
    return inputs, targets, valids


def run_calibration(
    checkpoint: Path,
    dropout: float = 0.10,
    n_samples: int = 20,
    max_cal_chips: int = 256,
    max_eval_chips: int = 256,
    confidence_level: float = 0.90,
    seed: int = 42,
    device: str | None = None,
    out_path: Path | None = None,
) -> dict:
    """Run the E1 conformal gate evaluation; returns the result dict."""
    import torch

    from siren.ml.kuro_siwo_dataset import KuroSiwoDataset
    from siren.ml.model import WaterResUNet
    from siren.ml.uncertainty import calibrate_conformal, evaluate_coverage

    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state_dict = (
        ckpt["state_dict"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt
        else ckpt
    )
    in_channels = state_dict["enc1.conv1.weight"].shape[1]
    model = WaterResUNet(
        in_channels=in_channels, base_channels=32, dropout=dropout
    ).to(device_str)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = KuroSiwoDataset(split="test")
    cal_idx, eval_idx, cal_events, eval_events = _split_indices_by_event(
        dataset, seed
    )
    cal_idx, eval_idx = cal_idx[:max_cal_chips], eval_idx[:max_eval_chips]
    logger.info(
        "Calibrating on %d chips (%d events), evaluating on %d chips "
        "(%d events), T=%d passes, device=%s",
        len(cal_idx), len(cal_events), len(eval_idx), len(eval_events),
        n_samples, device_str,
    )

    t0 = time.time()
    cal_inputs, cal_targets, cal_valids = _load_chips(dataset, cal_idx, device_str)
    eval_inputs, eval_targets, eval_valids = _load_chips(
        dataset, eval_idx, device_str
    )
    logger.info("Loaded %d chips in %.1fs", len(cal_idx) + len(eval_idx), time.time() - t0)

    torch.manual_seed(seed)
    if device_str != "cpu":
        torch.cuda.manual_seed_all(seed)
    q_star = calibrate_conformal(
        model,
        cal_inputs,
        cal_targets,
        n_samples=n_samples,
        confidence_level=confidence_level,
        valid_masks=cal_valids,
    )
    logger.info("Conformal quantile q* = %.6f", q_star)

    torch.manual_seed(seed + 1)
    if device_str != "cpu":
        torch.cuda.manual_seed_all(seed + 1)
    coverage = evaluate_coverage(
        model,
        eval_inputs,
        eval_targets,
        conformal_quantile=q_star,
        n_samples=n_samples,
        confidence_level=confidence_level,
        valid_masks=eval_valids,
    )

    gate_passed = coverage["coverage_error"] <= 0.05
    result = {
        "method": "split_conformal_mc_dropout",
        "checkpoint": checkpoint.name,
        "dropout": dropout,
        "n_samples": n_samples,
        "confidence_level": confidence_level,
        "conformal_quantile": float(q_star),
        **coverage,
        "gate_passed": gate_passed,
        "gate_criterion": "|empirical_coverage − 0.90| ≤ 0.05 (PRD §17.2)",
        "n_cal_chips": len(cal_idx),
        "n_eval_chips": len(eval_idx),
        "n_cal_events": len(cal_events),
        "n_eval_events": len(eval_events),
        "cal_events": cal_events,
        "eval_events": eval_events,
        "seed": seed,
        "device": device_str,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out = out_path or checkpoint.parent / "conformal_calibration.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    logger.info(
        "E1 conformal gate %s: coverage=%.4f (nominal %.2f, error %.4f) — "
        "wrote %s",
        "PASSED" if gate_passed else "FAILED",
        coverage["empirical_coverage"],
        confidence_level,
        coverage["coverage_error"],
        out,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="E1 conformal calibration for MC Dropout uncertainty"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="WaterResUNet checkpoint to calibrate",
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--n-samples", type=int, default=20,
                        help="MC Dropout forward passes T per chip")
    parser.add_argument("--max-cal-chips", type=int, default=256)
    parser.add_argument("--max-eval-chips", type=int, default=256)
    parser.add_argument("--confidence-level", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON (default: conformal_calibration.json "
                             "next to the checkpoint)")
    args = parser.parse_args(argv)

    result = run_calibration(
        checkpoint=args.checkpoint,
        dropout=args.dropout,
        n_samples=args.n_samples,
        max_cal_chips=args.max_cal_chips,
        max_eval_chips=args.max_eval_chips,
        confidence_level=args.confidence_level,
        seed=args.seed,
        device=args.device,
        out_path=args.out,
    )
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
