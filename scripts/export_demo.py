"""Export per-window reference vs. estimates for a few recordings to web/data/demo.json.

BIDMC is ODC-BY licensed, so the excerpt may be redistributed with attribution.
Usage:  python scripts/export_demo.py [--subjects 01 05 12]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clean(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=Path("results/predictions.csv"))
    ap.add_argument("--out", type=Path, default=Path("web/data/demo.json"))
    ap.add_argument("--subjects", nargs="*")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions, dtype={"subject": str})
    subjects = args.subjects or sorted(df.subject.unique())[:6]
    records = []
    for s in subjects:
        g = df[df.subject == s].sort_values("start_s")
        records.append({
            "subject": s,
            "duration": 480,
            "windows": [{"t": float(r.start_s), "ref": clean(r.ref_rr),
                         "fusion": clean(r.fusion_spec), "lgbm": clean(r.lightgbm)}
                        for r in g.itertuples()],
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_subjects": int(df.subject.nunique()), "records": records}),
                        encoding="utf-8")
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
