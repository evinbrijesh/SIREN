#!/usr/bin/env python3
"""Download Kuro Siwo GRD WebDataset shards from HuggingFace.

The Kuro Siwo dataset (Orion-AI-Lab) is a global multi-temporal SAR dataset
for rapid flood mapping. The BlackBench WebDataset release provides the
labelled GRD component as WebDataset .tar shards.

Usage:
    python scripts/download_kuro_siwo.py --split train          # 5 shards, ~47 GB
    python scripts/download_kuro_siwo.py --split test           # 12 shards, ~123 GB
    python scripts/download_kuro_siwo.py --split test --shards 0 1 2 3
    python scripts/download_kuro_siwo.py --all
    python scripts/download_kuro_siwo.py --all --verify-only

Shards are written to data/datasets/Kuro Siwo/<split>_GRD/shard-NNNNN.tar.
Downloads resume on interruption (curl -C -) and verify tar integrity.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KURO_ROOT = REPO_ROOT / "data" / "datasets" / "Kuro Siwo"

HF_BASE = (
    "https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset"
    "/resolve/main"
)

# Shard inventory (verified against the HF repo listing, 2026-09)
SPLITS = {
    "train": {
        "dir": "train_GRD",
        "n_shards": 5,
        "total_gb": 46.8,
    },
    "test": {
        "dir": "test_GRD",
        "n_shards": 12,
        "total_gb": 123.0,
    },
}


def shard_url(split: str, index: int) -> str:
    return f"{HF_BASE}/{SPLITS[split]['dir']}/shard-{index:05d}.tar"


def shard_path(split: str, index: int) -> Path:
    return KURO_ROOT / SPLITS[split]["dir"] / f"shard-{index:05d}.tar"


def tar_valid(path: Path) -> bool:
    """Check that a tar archive's index is readable (not truncated)."""
    result = subprocess.run(
        ["tar", "-tf", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def download_shard(split: str, index: int, verify_only: bool = False) -> bool:
    url = shard_url(split, index)
    dest = shard_path(split, index)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0 and tar_valid(dest):
        size_gb = dest.stat().st_size / 1e9
        print(f"  [skip] {dest.name} complete ({size_gb:.2f} GB)")
        return True
    if verify_only:
        status = "missing" if not dest.exists() else "truncated"
        print(f"  [verify-fail] {dest.name} {status}")
        return False

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [resume] {dest.name} "
              f"({dest.stat().st_size / 1e9:.2f} GB so far)")

    print(f"  [download] {dest.name}")
    cmd = [
        "curl", "-f", "-L", "--retry", "5", "--retry-delay", "10",
        "-C", "-",
        "-o", str(dest),
        "--progress-bar",
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [fail] curl exit {result.returncode} for {dest.name}")
        return False
    if not tar_valid(dest):
        print(f"  [fail] {dest.name} failed tar integrity check")
        return False
    print(f"  [ok] {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
    return True


def run_split(split: str, shards: list[int] | None, verify_only: bool) -> tuple[int, int]:
    info = SPLITS[split]
    indices = shards if shards is not None else list(range(info["n_shards"]))
    print(f"=== {split}_GRD ({len(indices)} shards, "
          f"~{info['total_gb']:.0f} GB total) ===")
    ok = fail = 0
    for i in indices:
        if download_shard(split, i, verify_only=verify_only):
            ok += 1
        else:
            fail += 1
    print(f"  -> {ok} ok, {fail} failed\n")
    return ok, fail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), help="Split to download")
    parser.add_argument("--all", action="store_true", help="Download both splits")
    parser.add_argument(
        "--shards", type=int, nargs="+",
        help="Specific shard indices (default: all in split)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Check integrity without downloading",
    )
    args = parser.parse_args()

    if not args.split and not args.all:
        parser.error("specify --split or --all")

    splits = sorted(SPLITS) if args.all else [args.split]
    total_ok = total_fail = 0
    for split in splits:
        shards = args.shards if not args.all else None
        ok, fail = run_split(split, shards, args.verify_only)
        total_ok += ok
        total_fail += fail

    print(f"DONE: {total_ok} shards ok, {total_fail} failed")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
