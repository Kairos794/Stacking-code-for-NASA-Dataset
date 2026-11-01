#!/usr/bin/env python3
"""
Step 1 - Prepare / validate the cleaned ASRS data.

Checks that the cleaned train/val/test CSVs (from `prepare_dataset.py`, header
``acn,text,label,label_id``) are present in the ``data/`` folder, prints class
distributions, and writes a leakage check report. It never re-splits the data:
the stratified 70/15/15 split is already fixed and leakage-free.

Usage:
    python 01_prepare.py            # validate + report (no side effects)
    python 01_prepare.py --copy C:/path/to/cleaned_data   # copy CSVs in first
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asrs_lib import config, data_utils  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare/validate ASRS cleaned data")
    ap.add_argument(
        "--copy", type=str, default=None,
        help="optional directory containing train.csv/val.csv/test.csv to copy "
             "into the data/ folder",
    )
    args = ap.parse_args()

    if args.copy:
        src = Path(args.copy)
        missing = [n for n in ("train.csv", "val.csv", "test.csv")
                   if not (src / n).exists()]
        if missing:
            print(f"[prepare] ERROR missing in {src}: {missing}")
            return 1
        for n in ("train.csv", "val.csv", "test.csv"):
            shutil.copy(src / n, config.DATA_DIR / n)
            print(f"[prepare] copied {n}")

    print("=" * 65)
    print("ASRS cleaned-data validation & leakage check")
    print("=" * 65)
    try:
        splits = data_utils.load_all_splits()
    except FileNotFoundError as e:
        print(e)
        return 1

    train_acns = set(splits["train"]["acn"])
    val_acns = set(splits["val"]["acn"])
    test_acns = set(splits["test"]["acn"])
    pairs = {
        "train-vs-val": train_acns & val_acns,
        "train-vs-test": train_acns & test_acns,
        "val-vs-test": val_acns & test_acns,
    }

    for name, df in splits.items():
        print(f"\n--- {name} : {len(df)} records ---")
        counts = df["label_id"].value_counts().sort_index()
        for label_id, c in counts.items():
            print(f"   [{label_id}] {config.ID_TO_LABEL[int(label_id)]:<18}: {c}")

    print("\n--- leakage check (ACN overlap must be 0) ---")
    ok = True
    for k, overlap in pairs.items():
        print(f"   {k}: {len(overlap)}")
        ok = ok and len(overlap) == 0
    print("[prepare] LEAKAGE-FREE" if ok else "[prepare] WARNING: overlap found!")

    with open(config.RESULT_DIR / "data_report.txt", "w", encoding="utf-8") as fh:
        fh.write(f"sizes: train={len(train_acns)} val={len(val_acns)} "
                 f"test={len(test_acns)}\n")
        for k, overlap in pairs.items():
            fh.write(f"{k}: {len(overlap)}\n")
    print(f"[prepare] report -> {config.RESULT_DIR / 'data_report.txt'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
