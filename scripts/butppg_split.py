"""Fix the BUT PPG development / test split by subject, before any tuning.

Subjects (first three digits of the record ID) are shuffled with a fixed seed and split in half.
Development subjects may be looked at freely while improving the algorithm; test subjects are
evaluated once, at the end. Writes data/processed/butppg_split.csv (subject, split).
Usage:  python scripts/butppg_split.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/butppg"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/butppg_split.csv"))
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()
    root = next(p.parent for p in args.raw.rglob("quality-hr-ann.csv"))
    ids = pd.read_csv(root / "quality-hr-ann.csv", encoding="utf-8-sig")["ID"].astype(str)
    subjects = np.array(sorted(ids.str[:3].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)
    half = len(subjects) // 2
    split = pd.DataFrame({"subject": subjects, "split": ["dev"] * half + ["test"] * (len(subjects) - half)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    split.sort_values("subject").to_csv(args.out, index=False)
    print(split.split.value_counts().to_dict(), "->", args.out)


if __name__ == "__main__":
    main()
