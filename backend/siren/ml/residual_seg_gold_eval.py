"""Score the residual-seg corrector against hand-verified Imja gold labels.

Same contract as ``imja_gold_eval`` but for the 7-channel corrector:
the extra input is the deterministic rule mask sampled onto the SAR
grid, and the prediction is ``sigmoid(unet_logits + prior_logit)`` —
the learned additive correction on the rule prior.

Reported alongside the labelrefined_v2 adapter's prob map for a
same-scene comparison on tier-1 (gold) and tier-2 (auto) labels.

Usage:
    python -m siren.ml.residual_seg_gold_eval [--pair unfrozen_desc]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio

from siren.ml.heldout_eval import (
    PAIRS,
    PROCESSED_DIR,
    _prob_map,
    _scene_masks,
    calibrated_cache,
)
from siren.ml.imja_gold_eval import (
    _auto_labels,
    _gold_labels,
    _score_change,
    _score_tier,
)
from siren.ml.train_residual_seg import PRIOR_LOGIT

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT = REPO_ROOT / "models" / "checkpoints" / "residual_seg_corrector.pt"
RULE_MASK = PROCESSED_DIR / "baseline_water_mask.tif"
REPORT_OUT = (
    REPO_ROOT / "models" / "checkpoints" / "residual_seg_gold_eval.json"
)


def _corrector_prob_map(
    model, pre_db: np.ndarray, post_db: np.ndarray, rule: np.ndarray,
) -> np.ndarray:
    """p = sigmoid(corrector_logits + rule prior) on the full scene."""
    import torch

    from siren.ml.contract import build_kuro_siwo_tensor

    t6 = build_kuro_siwo_tensor(pre_db, post_db)
    x7 = np.concatenate([t6, rule[None].astype(np.float32)], axis=0)
    h, w = x7.shape[1:]
    padded = np.pad(
        x7, ((0, 0), (0, (16 - h % 16) % 16), (0, (16 - w % 16) % 16))
    )
    with torch.no_grad():
        logits = model(torch.from_numpy(padded)[None]).cpu().numpy()[0, 0]
    logits = logits[:h, :w] + np.where(
        rule > 0.5, PRIOR_LOGIT, -PRIOR_LOGIT)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def evaluate(pair_name: str, thresholds=(0.30,)) -> dict:
    import torch
    from scipy.ndimage import binary_dilation

    from siren.ml.model import WaterResUNet

    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = WaterResUNet(
        in_channels=state["in_channels"],
        base_channels=state["base_channels"],
    )
    model.load_state_dict(state["model_state"])
    model.eval()

    # Adapter comparison on the same pair (its eval is already on file,
    # but same-process scoring keeps the comparison exact)
    adp_state = torch.load(
        REPO_ROOT / "models" / "checkpoints"
        / "water_resunet_6ch_himalayan_adapter_labelrefined_v2.pt",
        map_location="cpu", weights_only=True,
    )
    adapter = WaterResUNet(in_channels=6)
    adapter.load_state_dict(adp_state.get("model_state", adp_state))
    adapter.eval()

    t0_str, t1_str = PAIRS[pair_name]
    tag = "asc" if pair_name.endswith("_asc") else "desc"
    with rasterio.open(calibrated_cache(t0_str, tag)) as d:
        pre_db = d.read().astype(np.float32)
    with rasterio.open(calibrated_cache(t1_str, tag)) as d:
        post_db = d.read().astype(np.float32)

    masks = _scene_masks(calibrated_cache(t1_str, tag))
    imja = masks["imja"]
    imja_roi = binary_dilation(imja, iterations=10) if imja.any() else imja

    from siren.detect.sar import sar_grid_sample
    rule = sar_grid_sample(str(RULE_MASK), masks["lon"], masks["lat"]) > 0

    tiers = {
        "gold": {"t0": _gold_labels(t0_str, masks),
                 "t1": _gold_labels(t1_str, masks)},
        "auto": {"t0": _auto_labels(t0_str, masks),
                 "t1": _auto_labels(t1_str, masks)},
    }

    results = {}
    for name, p_t1, p_t0 in (
        ("corrector",
         _corrector_prob_map(model, pre_db, post_db, rule),
         _corrector_prob_map(model, pre_db, pre_db, rule)),
        ("adapter_labelrefined_v2",
         _prob_map(adapter, pre_db, post_db),
         _prob_map(adapter, pre_db, pre_db)),
    ):
        per_tau = {}
        for tau in thresholds:
            entry = {}
            for tier, labs in tiers.items():
                for key, prob in (("t1", p_t1), ("t0", p_t0)):
                    entry[f"{tier}_{key}"] = _score_tier(
                        prob, tau, labs[key], imja_roi, tier)
                entry[f"{tier}_change"] = _score_change(
                    p_t0, p_t1, tau, labs, imja_roi, tier)
            per_tau[str(tau)] = entry
        results[name] = per_tau
        logger.info("%s %s done", pair_name, name)

    return {"pair": pair_name, "t0": t0_str, "t1": t1_str,
            "checkpoints": results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", choices=list(PAIRS), default="unfrozen_desc")
    ap.add_argument("--thresholds", default="0.30,0.40,0.50,0.55")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    taus = tuple(float(t) for t in args.thresholds.split(","))
    report = evaluate(args.pair, taus)
    REPORT_OUT.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", REPORT_OUT)
    print(json.dumps(report["checkpoints"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
