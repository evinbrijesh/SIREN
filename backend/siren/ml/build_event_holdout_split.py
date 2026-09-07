"""Build a TRUE event-level holdout split, separate from the official
Sen1Floods11 chip-level splits.

Background (see docs/reference/DL_MODEL_AUDIT.md §4 and ADR-010):
The official Sen1Floods11 train/valid/test CSVs partition by CHIP, not by
event. All 10 flood events (Ghana, India, Mekong, Nigeria, Pakistan,
Paraguay, Somalia, Spain, Sri-Lanka, USA) appear in all three official
splits. A model can learn event-specific geography from the train chips
and recognize it in the test chips of the SAME event -- this inflates
the reported IoU relative to true generalization to unseen flood events.

This script builds a second, stricter split where ENTIRE events are held
out. No chip from a held-out event's flood is used anywhere in training
or model selection. This is the split that answers "does the model
generalize to a flood event it has never seen," which is the claim
ADR-010 actually requires before ML output is displayed as evidence.

Event assignment is fixed (not randomly sampled) for reproducibility
(Hard Rule 6). Rationale for the specific assignment:
  - Test events (never touched until final evaluation): Pakistan, Somalia
    -- chosen to include an arid/riverine flood (Pakistan, 2010 Indus
    floods) and a coastal/tropical flood (Somalia), both distinct from
    the training events' hydrology.
  - Val event (model selection only, never used for gradient updates):
    Nigeria, Spain -- one tropical delta, one Mediterranean/temperate
    event, for a validation signal that is not train-adjacent.
  - Train events (remaining 6): Ghana, India, Mekong, Paraguay,
    Sri-Lanka, USA -- covers the largest chip counts for training volume.

Usage:
    python -m siren.ml.build_event_holdout_split
"""

from __future__ import annotations

import csv
from pathlib import Path

SEN1FLOODS11_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "Sen1Floods11"

# Fixed event assignment -- do not change without re-running full evaluation.
EVENT_HOLDOUT_TEST = {"Pakistan", "Somalia"}
EVENT_HOLDOUT_VAL = {"Nigeria", "Spain"}
# Everything else (Ghana, India, Mekong, Paraguay, Sri-Lanka, USA) -> train


def _load_all_chips() -> list[tuple[str, str, str]]:
    """Load every (s1_file, label_file, physical_dir) triple across all
    three official-split physical directories (train/valid/test), since
    that is where the raw files actually live on disk."""
    chips = []
    for physical_dir, csv_name in [
        ("train", "flood_train_data.csv"),
        ("valid", "flood_valid_data.csv"),
        ("test", "flood_test_data.csv"),
    ]:
        csv_path = SEN1FLOODS11_ROOT / "splits" / csv_name
        with open(csv_path) as f:
            for row in csv.reader(f):
                if row:
                    chips.append((row[0], row[1], physical_dir))
    return chips


def build() -> None:
    chips = _load_all_chips()

    out_dir = SEN1FLOODS11_ROOT / "splits" / "event_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

    assigned = {"train": [], "val": [], "test": []}
    for s1_file, label_file, physical_dir in chips:
        event = s1_file.split("_")[0]
        if event in EVENT_HOLDOUT_TEST:
            logical_split = "test"
        elif event in EVENT_HOLDOUT_VAL:
            logical_split = "val"
        else:
            logical_split = "train"
        assigned[logical_split].append((s1_file, label_file, physical_dir))

    for logical_split, rows in assigned.items():
        csv_path = out_dir / f"{logical_split}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for s1_file, label_file, physical_dir in rows:
                writer.writerow([s1_file, label_file, physical_dir])
        events = sorted({r[0].split("_")[0] for r in rows})
        print(f"event_holdout/{logical_split}.csv: {len(rows)} chips, events={events}")

    # Sanity check: no event appears in more than one logical split
    train_events = {r[0].split("_")[0] for r in assigned["train"]}
    val_events = {r[0].split("_")[0] for r in assigned["val"]}
    test_events = {r[0].split("_")[0] for r in assigned["test"]}
    overlap = (train_events & val_events) | (train_events & test_events) | (val_events & test_events)
    if overlap:
        raise RuntimeError(f"Event leakage across event_holdout splits: {overlap}")
    print(f"OK: no event overlap. train={sorted(train_events)} val={sorted(val_events)} test={sorted(test_events)}")


if __name__ == "__main__":
    build()
