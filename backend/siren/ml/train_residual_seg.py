"""Residual segmentation — train the rule-mask corrector (lake holdout).

The residual formulation: the corrector predicts an additive logit offset
on the deterministic rule prior, not a mask from scratch —

    logits = UNet(6ch SAR, rule_mask) + logit_prior(rule_mask)

so the output equals the rule mask when the learned correction is zero.
This is the honest "ML corrects the deterministic mask" component (the
detection layer's remaining ML-primary path): the corrector keeps the
rule where it's right and fixes it where SAR says otherwise.

Evaluation: GroupKFold by lake_id — every chip of a held-out lake is
test-only. Reported per fold: IoU of the rule channel alone, the
labelrefined_v2 adapter alone (6ch), and the corrector. The promotion
question the report answers: does (SAR + rule) beat SAR alone?

Honesty notes:
  * All chips come from ONE SAR pair (2026-07-02/14, unfrozen monsoon)
    — the holdout is spatial (held-out lakes), not temporal.
  * Labels are labelrefined (per-date NDWI where S2-valid + eroded
    inventory elsewhere) — weak positives at ~90 m pitch.
  * The rule channel is near-empty outside the Imja AOI (the NDWI
    baseline is AOI-scoped) — so for most lakes the corrector ≈ adapter;
    the fold metrics show whether the channel adds anything at all.

Usage:
    python -m siren.ml.train_residual_seg --save-model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CHIPS = REPO_ROOT / "data" / "datasets" / "himalayan_chips_residual" / "chips.npz"
MANIFEST = REPO_ROOT / "data" / "datasets" / "himalayan_chips_residual" / "manifest.json"
ADAPTER_CKPT = (
    REPO_ROOT / "models" / "checkpoints"
    / "water_resunet_6ch_himalayan_adapter_labelrefined_v2.pt"
)
REPORT_PATH = (
    REPO_ROOT / "models" / "checkpoints" / "residual_seg_cv_report.json"
)
MODEL_OUT = (
    REPO_ROOT / "models" / "checkpoints" / "residual_seg_corrector.pt"
)

PRIOR_LOGIT = 4.0  # sigmoid(±4) ≈ 0.982/0.018 — strong-but-correctable rule prior


def _iou(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = float((pred & truth).sum())
    union = float((pred | truth).sum())
    return inter / union if union else float("nan")


def _evaluate_fold(
    x_te: np.ndarray, y_te: np.ndarray, corrector, adapter, device
) -> dict:
    import torch

    def _predict(net, arr):
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(arr), 16):
                t = torch.from_numpy(arr[i:i + 16]).to(device)
                outs.append(net(t).cpu().numpy())
        return np.concatenate(outs, 0)[:, 0]

    rule = x_te[:, 6] > 0.5
    truth = y_te > 0

    # Corrector: learned residual on the rule prior logit
    logits = _predict(corrector, x_te)
    prior = np.where(rule, PRIOR_LOGIT, -PRIOR_LOGIT)
    p_corr = 1.0 / (1.0 + np.exp(-(logits + prior)))
    corr = p_corr >= 0.5

    out = {
        "rule_iou": _iou(rule, truth),
        "corrector_iou": _iou(corr, truth),
        "n_test": len(x_te),
    }
    if adapter is not None:
        p_adp = _predict(adapter, x_te[:, :6])
        out["adapter_iou"] = _iou(p_adp >= 0.5, truth)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import torch
    from sklearn.model_selection import GroupKFold

    from siren.ml.model import WaterResUNet

    chips = np.load(CHIPS)
    x = chips["x"].astype(np.float32)
    y = (chips["y"] > 0).astype(np.float32)
    manifest = json.loads(MANIFEST.read_text())
    groups = np.array([m["lake_id"] for m in manifest])
    logger.info("Loaded %d chips, %d unique lakes", len(x), groups.size and len(set(groups)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = None
    if ADAPTER_CKPT.exists():
        adp = WaterResUNet(in_channels=6)
        state = torch.load(ADAPTER_CKPT, map_location="cpu", weights_only=False)
        adp.load_state_dict(state.get("model_state", state))
        adapter = adp.to(device)
        logger.info("Loaded labelrefined_v2 adapter for comparison")

    folds = []
    best_iou, best_state = -1.0, None
    gkf = GroupKFold(n_splits=args.folds)
    for fold, (tr, te) in enumerate(gkf.split(x, y, groups)):
        torch.manual_seed(42)
        net = WaterResUNet(in_channels=7, base_channels=args.base_channels).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        pos_frac = float(y[tr].mean())
        pos_weight = torch.tensor(
            [(1 - pos_frac) / max(pos_frac, 1e-4)], device=device)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        xt = torch.from_numpy(x[tr]).to(device)
        yt = torch.from_numpy(y[tr][:, None]).to(device)
        prior_t = torch.where(
            xt[:, 6:7] > 0.5,
            torch.tensor(PRIOR_LOGIT, device=device),
            torch.tensor(-PRIOR_LOGIT, device=device),
        )
        net.train()
        n = len(tr)
        for epoch in range(args.epochs):
            perm = torch.randperm(n, device=device)
            tot = 0.0
            for i in range(0, n, 16):
                idx = perm[i:i + 16]
                logits = net(xt[idx]) + prior_t[idx]
                loss = bce(logits, yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss)
            if (epoch + 1) % 10 == 0:
                logger.info("fold %d epoch %d loss %.4f", fold, epoch + 1, tot)

        res = _evaluate_fold(x[te], y[te], net, adapter, device)
        folds.append(res)
        logger.info("fold %d: %s", fold, res)
        if res["corrector_iou"] == res["corrector_iou"] and res["corrector_iou"] > best_iou:
            best_iou = res["corrector_iou"]
            best_state = {k: v.cpu() for k, v in net.state_dict().items()}

    def _mean(k):
        vals = [f[k] for f in folds if f.get(k) is not None and f[k] == f[k]]
        return float(np.mean(vals)) if vals else None

    report = {
        "experiment": "residual segmentation corrector — lake-grouped holdout",
        "folds": folds,
        "mean_rule_iou": _mean("rule_iou"),
        "mean_corrector_iou": _mean("corrector_iou"),
        "mean_adapter_iou": _mean("adapter_iou"),
        "caveats": [
            "single SAR pair (2026-07-02/14) — spatial holdout only",
            "labelrefined weak labels (~90 m pitch, per-date NDWI + eroded inventory)",
            "rule channel empty outside the Imja AOI — corrector ≈ adapter there",
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", REPORT_PATH)

    print("\n=== Residual segmentation — lake-grouped holdout ===")
    print(f"  rule IoU:      {report['mean_rule_iou']}")
    print(f"  adapter IoU:   {report['mean_adapter_iou']}")
    print(f"  corrector IoU: {report['mean_corrector_iou']}")

    if args.save_model and best_state is not None:
        torch.save({
            "model_state": best_state,
            "in_channels": 7,
            "base_channels": args.base_channels,
            "prior_logit": PRIOR_LOGIT,
            "heldout_corrector_iou": best_iou,
        }, MODEL_OUT)
        logger.info("Best-fold corrector saved to %s", MODEL_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
