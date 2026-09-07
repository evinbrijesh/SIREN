"""One-time helper: download the missing Sen1Floods11 val/test chips.

The official train split (252 chips) is already committed under
data/raw/Sen1Floods11/train/. This script fetches the val (89) and
test (90) chips referenced in data/raw/Sen1Floods11/splits/flood_{valid,test}_data.csv
directly from the public GCS bucket over HTTPS (gsutil is not available
in this environment).

Usage:
    python -m siren.ml.download_splits
"""

from __future__ import annotations

import csv
import concurrent.futures
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled"
SEN1FLOODS11_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "Sen1Floods11"


def _download_one(url: str, dest: Path) -> tuple[str, bool, str]:
    if dest.exists() and dest.stat().st_size > 0:
        return (str(dest), True, "already present")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(resp.content)
        tmp.rename(dest)  # atomic on same filesystem
        return (str(dest), True, "downloaded")
    except Exception as exc:
        return (str(dest), False, str(exc))


def download_split(split_name: str, csv_name: str) -> None:
    """Download all S1 + Label chips for one split (valid or test)."""
    csv_path = SEN1FLOODS11_ROOT / "splits" / csv_name
    pairs = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if row:
                pairs.append((row[0], row[1]))

    split_dir = SEN1FLOODS11_ROOT / split_name
    (split_dir / "S1").mkdir(parents=True, exist_ok=True)
    (split_dir / "Label").mkdir(parents=True, exist_ok=True)

    jobs = []
    for s1_file, label_file in pairs:
        jobs.append((f"{BASE_URL}/S1Hand/{s1_file}", split_dir / "S1" / s1_file))
        jobs.append((f"{BASE_URL}/LabelHand/{label_file}", split_dir / "Label" / label_file))

    n_ok, n_fail = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_download_one, url, dest) for url, dest in jobs]
        for fut in concurrent.futures.as_completed(futures):
            path, ok, msg = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                logger.warning(f"FAILED {path}: {msg}")

    # Write pairs.csv for this split (same format as train/pairs.csv)
    pairs_csv = split_dir / "pairs.csv"
    with open(pairs_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for s1_file, label_file in pairs:
            writer.writerow([s1_file, label_file])

    print(f"{split_name}: {n_ok} ok, {n_fail} failed, {len(pairs)} pairs, pairs.csv written")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    download_split("valid", "flood_valid_data.csv")
    download_split("test", "flood_test_data.csv")


if __name__ == "__main__":
    main()
