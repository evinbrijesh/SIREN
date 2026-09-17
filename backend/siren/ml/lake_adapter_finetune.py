"""High-altitude decoder fine-tune — Himalayan lake chips (Phase 2).

Trains the bottleneck + decoder of the gate-passed 6-ch Kuro Siwo
WaterResUNet on the lake-inventory chip set built by
``ml/himalayan_lake_dataset.py`` while the encoder stays frozen — the
encoder's low-level SAR features (speckle, edges, backscatter texture)
are transferable; the failure is in the high-level decision boundary
that maps them to "water" on glacier terrain.

Honesty notes (same caveats as the weak-label experiment, extended):
  * Labels are WEAK positives — median-outlined 2022–2024 inventory
    polygons, not per-pixel scene-date water. Lake shorelines can be off
    by a pixel or two at ~90 m pitch.
  * A single scene pair (2026-07-02 / 2026-07-14, monsoon season) means
    the adapted model is tuned to liquid-water summer lakes; frozen
    winter lakes are out of scope until winter granules are acquired.
  * Full-scene metrics are reported before/after via the same
    terrain-gated evaluation as the pipeline — the honest signal is the
    glacier/steep false-positive rate, not polygon-overlap (which is
    partly circular since labels derive from the same inventory).

Stays shadow-only: the output checkpoint is NOT wired into the runtime —
promotion requires the ADR-013 gate on held-out real data.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
CHIPS_DIR = _REPO_ROOT / "data" / "datasets" / "himalayan_lake_chips"
CKPT = (
    _REPO_ROOT
    / "models"
    / "checkpoints"
    / "water_resunet_kuro_siwo_full"
    / "water_resunet_6ch_kuro_siwo_v1.pt"
)
OUT_CKPT = (
    _REPO_ROOT
    / "models"
    / "checkpoints"
    / "water_resunet_6ch_himalayan_adapter.pt"
)
REPORT_OUT = _REPO_ROOT / "models" / "checkpoints" / "lake_adapter_report.json"


def _spatial_split(chips_meta: list[dict], val_frac: float = 0.2) -> np.ndarray:
    """Block-split chips into train/val by grid row (north–south separation).

    Sorts lake chips by SAR-grid row, then assigns contiguous ~equal
    latitude bands alternately — neighbouring lakes never leak across
    the split (unlike a random chip shuffle).
    """
    lake_idx = [i for i, m in enumerate(chips_meta) if m["kind"] == "lake"]
    rows = np.array([chips_meta[i]["row"] for i in lake_idx])
    order = np.argsort(rows)
    val = np.zeros(len(chips_meta), dtype=bool)
    n_bands = 5
    band_size = max(len(order) // n_bands, 1)
    # every 5th band (1-indexed band 4) goes to val
    for rank, pos in enumerate(order):
        band = rank // band_size
        if band == n_bands - 1 or (band % 5 == 4):
            val[lake_idx[pos]] = True
    # background chips always train
    return val


def run_finetune(
    chips_dir: Path | str = CHIPS_DIR,
    epochs: int = 15,
    lr: float = 5e-5,
    batch_size: int = 8,
    threshold: float = 0.30,
    pos_weight_cap: float = 15.0,
    seed: int = 42,
    evaluate: bool = True,
) -> dict:
    import torch

    from siren.ml.engine import _detect_architecture
    from siren.ml.losses import bce_loss, dice_loss
    from siren.ml.metrics import metrics_from_counts, water_confusion_counts
    from siren.ml.model import WaterResUNet

    torch.manual_seed(seed)
    np.random.seed(seed)

    chips_dir = Path(chips_dir)
    data = np.load(chips_dir / "chips.npz")
    x = data["x"].astype(np.float32)          # (N, 6, C, C)
    y = data["y"].astype(np.uint8)            # (N, C, C)
    manifest = json.loads((chips_dir / "manifest.json").read_text())
    v = np.ones_like(y, dtype=np.float32)     # weak labels cover every px

    val_mask = _spatial_split(manifest)
    tr, va = ~val_mask, val_mask
    logger.info(
        "chips=%d  train=%d  val=%d  pos_frac=%.4f",
        len(x), int(tr.sum()), int(va.sum()),
        float(y.sum() / y.size),
    )

    state = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    arch, in_ch, base = _detect_architecture(state)
    model = WaterResUNet(in_channels=in_ch, base_channels=base)
    model.load_state_dict(state)

    # ---- before (full-scene terrain-gate metrics) ----
    before = None
    if evaluate:
        model.eval()
        before = _full_scene_eval(model, threshold)

    # ---- freeze encoder, fine-tune bottleneck + decoder + head ----
    frozen, trainable = [], []
    for name, p in model.named_parameters():
        if name.startswith("enc"):
            p.requires_grad = False
            frozen.append(name)
        else:
            trainable.append(name)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "frozen=%d tensors  trainable=%.1fM/%.1fM params",
        len(frozen), n_train / 1e6, n_total / 1e6,
    )

    pos = float((y[tr] * v[tr]).sum())
    neg = float(((1 - y[tr]) * v[tr]).sum())
    pos_weight = torch.tensor([min(neg / max(pos, 1.0), pos_weight_cap)])
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    xt = torch.from_numpy(x[tr])
    yt = torch.from_numpy(y[tr].astype(np.float32))[:, None]
    vt = torch.from_numpy(v[tr])[:, None]

    model.train()
    history = []
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        ep_loss = 0.0
        for i in range(0, len(xt), batch_size):
            j = perm[i:i + batch_size]
            logits = model(xt[j])
            loss = bce_loss(logits, yt[j], vt[j], pos_weight) + dice_loss(
                logits, yt[j], vt[j]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss)
        history.append(round(ep_loss / max(len(xt) // batch_size, 1), 4))
        logger.info("epoch %d  loss=%.4f", ep, history[-1])

    # ---- weak-label val IoU (sanity only — labels are polygons, not truth)
    model.eval()
    tp = fp = fn = tn = 0
    va_idx = np.where(va)[0]
    with torch.no_grad():
        for i in range(0, len(va_idx), batch_size):
            j = va_idx[i:i + batch_size]
            xb = torch.from_numpy(x[j])
            p = (torch.sigmoid(model(xb))[:, 0].numpy() >= threshold)
            c = water_confusion_counts(
                p.astype(np.uint8), y[j], v[j].astype(np.uint8)
            )
            tp += c[0]; fp += c[1]; fn += c[2]; tn += c[3]
    val_metrics = metrics_from_counts(tp, fp, fn, tn)

    after = _full_scene_eval(model, threshold) if evaluate else None

    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_CKPT)

    report = {
        "experiment": "high-altitude decoder fine-tune on Himalayan lake chips",
        "labels": {
            "positives": "verified lake inventory polygons (weak, 2022–2024 median outlines)",
            "negatives": "all non-polygon pixels incl. glacier/moraine/slope",
            "chips": int(len(x)),
            "train_chips": int(tr.sum()),
            "val_chips": int(va.sum()),
            "pos_pixel_frac": round(float(y.sum() / y.size), 5),
        },
        "training": {
            "epochs": epochs, "lr": lr,
            "frozen_tensors": len(frozen),
            "trainable_params": n_train,
            "total_params": n_total,
            "pos_weight": float(pos_weight),
            "loss_history": history,
        },
        "val_weaklabel_metrics": val_metrics,
        "full_scene_before": before,
        "full_scene_after": after,
        "checkpoint_out": str(OUT_CKPT),
        "limitations": [
            "Weak polygon labels — shoreline errors of ~1 px at 90 m pitch; "
            "median-outlined inventory, not per-pixel scene-date water.",
            "Single monsoon-season pair — frozen/winter lakes untrained.",
            "Polygon-overlap gains are partly circular (labels derive from "
            "the same inventory); honest metrics are glacier/steep "
            "false-positive rates on the full scene.",
            "Shadow-only — not wired into runtime pending ADR-013 gate "
            "on held-out real data.",
        ],
    }
    REPORT_OUT.write_text(json.dumps(report, indent=1))
    return report


def _full_scene_eval(model, threshold: float) -> dict:
    """Run the same terrain-gated full-scene eval as sar_domain_adapt."""
    import rasterio

    from siren.ml.sar_domain_adapt import (
        SAR_T0,
        SAR_T1,
        build_weak_labels,
        evaluate_scene,
    )
    from scipy.ndimage import binary_dilation

    with rasterio.open(SAR_T0) as d:
        pre_db = d.read().astype(np.float32)
    with rasterio.open(SAR_T1) as d:
        post_db = d.read().astype(np.float32)
    labels = build_weak_labels(SAR_T1)
    lake_vic = binary_dilation(labels["lake"], iterations=3)
    return evaluate_scene(
        model, pre_db, post_db, threshold,
        labels["lon"], labels["lat"], labels["aoi"], labels["glac"], lake_vic,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Himalayan lake-chip decoder fine-tune")
    ap.add_argument("--chips", default=str(CHIPS_DIR))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pos-weight-cap", type=float, default=15.0)
    ap.add_argument("--no-eval", action="store_true", help="skip full-scene eval")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = run_finetune(
        chips_dir=args.chips, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, pos_weight_cap=args.pos_weight_cap,
        evaluate=not args.no_eval,
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
