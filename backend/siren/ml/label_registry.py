"""Label registry for Imja-area segmentation evaluation.

Maps SAR acquisition dates to label rasters by truth tier (gold / auto /
scl_merged / inventory) and records pair membership + allowed uses.

Design goals:
  * Single source of truth for which labels exist and what they mean.
  * Prevent accidental use of training labels for held-out evaluation.
  * Surface label gaps so the acquisition plan stays visible.
  * Drive the operational-primary evaluation harness.

See docs/spec/LABEL_ACQUISITION_PLAN.md for the human workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CHECKPOINTS_DIR = REPO_ROOT / "models" / "checkpoints"

# ---------------------------------------------------------------------------
# Static declarations: what labels *should* exist for a complete operational
# gate evaluation. Filenames follow the conventions in LABEL_ACQUISITION_PLAN.
# ---------------------------------------------------------------------------

# SAR date -> gold label GeoTIFF. Existing hand-verified pairs + new
# AI-assisted preliminary gold labels (shoulder/winter/monsoon_asc) that
# still require human QA before operational certification.
GOLD_LABELS: dict[str, Path] = {
    "20250829": PROCESSED_DIR / "imja_gold_label_20250903.tif",
    "20250910": PROCESSED_DIR / "imja_gold_label_20250905.tif",
    "20250922": PROCESSED_DIR / "imja_gold_label_20250925.tif",
    "20251109": PROCESSED_DIR / "imja_gold_label_20251112.tif",
    "20251121": PROCESSED_DIR / "imja_gold_label_20251122.tif",
    "20260108": PROCESSED_DIR / "imja_gold_label_20260101.tif",
    "20260120": PROCESSED_DIR / "imja_gold_label_20260121.tif",
    "20260811": PROCESSED_DIR / "imja_gold_label_20260811.tif",
    "20260819": PROCESSED_DIR / "imja_gold_label_20260824.tif",
    "20260912": PROCESSED_DIR / "imja_gold_label_20260908.tif",
    "20260916": PROCESSED_DIR / "imja_gold_label_20260918.tif",
}

# SAR date -> auto-candidate label (unverified, tier 2)
AUTO_LABELS: dict[str, Path] = {
    "20260726": PROCESSED_DIR / "imja_autolabel_20260725.tif",
    "20260807": PROCESSED_DIR / "imja_autolabel_20260811.tif",
}

# SAR date -> merged SCL-water label (tier 2–3)
SCL_LABELS: dict[str, Path] = {
    "20251121": PROCESSED_DIR / "s2_water_label_for_20251121.tif",
    "20260726": PROCESSED_DIR / "s2_water_label_for_20260726.tif",
    "20260807": PROCESSED_DIR / "s2_water_label_for_20260807.tif",
    "20260811": PROCESSED_DIR / "s2_water_label_for_20260811.tif",
    "20260819": PROCESSED_DIR / "s2_water_label_for_20260819.tif",
    "20260912": PROCESSED_DIR / "s2_water_label_for_20260912.tif",
}

# Pairs ordered by priority for operational gate evaluation.
# `allowed_uses` controls whether a pair may be used for evaluation
# (`eval`) or only for training/inspection (`train`, `qualitative`).
PAIRS: dict[str, dict[str, Any]] = {
    "monsoon_2025_desc": {
        "t0": "20250829",
        "t1": "20250910",
        "season": "unfrozen",
        "allowed_uses": ["eval", "qualitative"],
        "priority": -1,
        "notes": (
            "Independent-year (2025) monsoon descending pair. Strong "
            "evidence for operational gate — different year from the "
            "adapter training data."
        ),
    },
    "monsoon_2025_desc2": {
        "t0": "20250829",
        "t1": "20250922",
        "season": "unfrozen",
        "allowed_uses": ["eval", "qualitative"],
        "priority": -1,
        "notes": (
            "Second 2025 monsoon descending pair (24-day interval). "
            "Shares 08-29 with monsoon_2025_desc but tests a different "
            "pre/post combination — the model predictions are independent."
        ),
    },
    "unfrozen_desc": {
        "t0": "20260819",
        "t1": "20260912",
        "season": "unfrozen",
        "allowed_uses": ["eval", "qualitative"],
        "priority": 0,
        "notes": (
            "Gold-labelled held-out pair used for the ADR-014 advisory "
            "promotion. Can certify operational gate if a second "
            "independent unfrozen pair also passes."
        ),
    },
    "unfrozen_desc2": {
        "t0": "20260726",
        "t1": "20260807",
        "season": "unfrozen",
        "allowed_uses": ["eval", "qualitative"],
        "priority": 1,
        "notes": (
            "Closest unfrozen-season pair to the adapter training data; "
            "gold labels needed to detect overfit. Auto labels exist."
        ),
    },
    "early_desc": {
        "t0": "20260702",
        "t1": "20260714",
        "season": "unfrozen",
        "allowed_uses": ["train", "qualitative"],
        "priority": 2,
        "notes": (
            "Adapter training pair — disallowed for operational gate "
            "evaluation. Gold labels useful for diagnosing training bias."
        ),
    },
    "shoulder": {
        "t0": "20251109",
        "t1": "20251121",
        "season": "frozen_shoulder",
        "allowed_uses": ["eval", "qualitative"],
        "priority": 3,
        "notes": (
            "Cold-season pair with partially ice-covered lake. NDWI-based "
            "labels overestimate liquid water; treat as glacier-FP stress "
            "test, not water-extent gate."
        ),
    },
    "monsoon_asc": {
        "t0": "20260811",
        "t1": "20260916",
        "season": "unfrozen",
        "allowed_uses": ["eval", "qualitative"],
        "priority": 4,
        "notes": (
            "Different pass geometry (ascending), unfrozen season. Gold "
            "labels exist but t1 has ~35% cloud cover over AOI."
        ),
    },
    "winter": {
        "t0": "20260108",
        "t1": "20260120",
        "season": "frozen",
        "allowed_uses": ["eval", "qualitative"],
        "priority": 5,
        "notes": (
            "Deep-winter frozen lake — no liquid water. Label should be "
            "near-zero water; evaluated for glacier-FP control only."
        ),
    },
}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _load_sidecar(path: Path) -> dict[str, Any] | None:
    sc = _sidecar(path)
    try:
        return json.loads(sc.read_text())
    except Exception:
        return None


def list_labels() -> dict[str, dict[str, Any]]:
    """Return every declared label keyed by SAR date, with disk status."""
    out: dict[str, dict[str, Any]] = {}

    def _add(date: str, tier: str, path: Path) -> None:
        exists = path.exists()
        sidecar = _load_sidecar(path) if exists else None
        out[date] = {
            "sar_date": date,
            "tier": tier,
            "path": str(path),
            "exists": exists,
            "sidecar": sidecar,
            "cloud_marginal": bool(
                sidecar and sidecar.get("cloud_marginal", False)
            ),
            "offset_days": sidecar.get("offset_days") if sidecar else None,
            "analyst": sidecar.get("analyst") if sidecar else None,
            "annotated_at": sidecar.get("annotated_at") if sidecar else None,
        }

    for date, path in GOLD_LABELS.items():
        _add(date, "gold", path)
    for date, path in AUTO_LABELS.items():
        _add(date, "auto", path)
    for date, path in SCL_LABELS.items():
        # Only add SCL if no higher-tier label exists for the date
        if date not in out:
            _add(date, "scl_merged", path)

    return out


def best_label_for_date(date: str) -> dict[str, Any] | None:
    """Highest-tier existing label for a SAR date, or None."""
    labels = list_labels()
    rec = labels.get(date)
    if rec is None or not rec["exists"]:
        return None
    return rec


def eval_eligible_pairs(use: str = "eval") -> list[dict[str, Any]]:
    """Pairs that may be used for the requested purpose and have labels.

    For ``use='eval'`` (the operational gate), a pair is eligible only if:
      - ``eval`` is in its allowed_uses, AND
      - both t0 and t1 have at least an existing label of any tier.
    A pair without gold labels on both dates is flagged as ``gold_incomplete``.
    """
    labels = list_labels()
    eligible: list[dict[str, Any]] = []
    for name, meta in sorted(PAIRS.items(), key=lambda kv: kv[1]["priority"]):
        if use not in meta.get("allowed_uses", []):
            continue
        t0 = labels.get(meta["t0"])
        t1 = labels.get(meta["t1"])
        t0_ok = t0 is not None and t0["exists"]
        t1_ok = t1 is not None and t1["exists"]
        t0_gold = t0_ok and t0["tier"] == "gold"
        t1_gold = t1_ok and t1["tier"] == "gold"
        entry = {
            "pair": name,
            **meta,
            "t0_label": t0,
            "t1_label": t1,
            "has_any_label": t0_ok and t1_ok,
            "gold_complete": t0_gold and t1_gold,
            "gold_incomplete": (t0_ok and t1_ok) and not (t0_gold and t1_gold),
        }
        if t0_ok and t1_ok:
            eligible.append(entry)
    return eligible


def operational_gate_status() -> dict[str, Any]:
    """Summary of whether the operational gate can currently be attempted."""
    eval_pairs = eval_eligible_pairs("eval")
    gold_complete = [p for p in eval_pairs if p["gold_complete"]]
    incomplete = [p for p in eval_pairs if p["gold_incomplete"]]
    missing = [name for name, meta in PAIRS.items() if name not in {p["pair"] for p in eval_pairs}]

    return {
        "gate": (
            "operational_primary: sar_segmentation_expansion "
            "(IoU ≥ 0.60, P ≥ 0.84, glacier-FP < 5% on ≥2 held-out pairs)"
        ),
        "operational_scope": "monsoon window Jun-Sep, descending, unfrozen "
                             "(siren/scope.py) — out-of-scope pairs are "
                             "documented probes, never gating",
        "can_attempt_gate": len(gold_complete) >= 2,
        "gold_complete_pairs": [p["pair"] for p in gold_complete],
        "gold_incomplete_pairs": [
            {"pair": p["pair"], "tiers": (p["t0_label"]["tier"], p["t1_label"]["tier"])}
            for p in incomplete
        ],
        "missing_pairs": missing,
        "next_action": (
            "Acquire gold labels for: " + ", ".join(
                p["pair"] for p in incomplete
            )
            if incomplete else "Run operational gate evaluation."
        ),
    }


def registry_report() -> dict[str, Any]:
    """Human-readable JSON report of label status across all pairs."""
    labels = list_labels()
    return {
        "labels_by_date": labels,
        "pairs": {
            name: {
                "t0": meta["t0"],
                "t1": meta["t1"],
                "priority": meta["priority"],
                "allowed_uses": meta["allowed_uses"],
                "t0_label": labels.get(meta["t0"]),
                "t1_label": labels.get(meta["t1"]),
            }
            for name, meta in PAIRS.items()
        },
        "operational_gate": operational_gate_status(),
    }


def write_registry_report(path: Path | str | None = None) -> Path:
    """Persist the registry report to disk (default: models/checkpoints)."""
    out = Path(path) if path else CHECKPOINTS_DIR / "label_registry_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry_report(), indent=2, default=str))
    return out
