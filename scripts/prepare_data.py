"""Build the windowed feature table from the BIDMC recordings.

Usage:  python scripts/prepare_data.py [--raw data/raw] [--out data/processed/windows.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ppgrr.data import find_csv_dir, list_subjects, load_recording, reference_rr
from ppgrr.dsp import respiratory_modulations
from ppgrr.features import STEP_S, WINDOW_S, window_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/windows.csv"))
    args = ap.parse_args()

    csv_dir = find_csv_dir(args.raw)
    rows = []
    for subject in list_subjects(csv_dir):
        rec = load_recording(csv_dir, subject)
        mods = respiratory_modulations(rec.ppg, rec.fs)
        for start in np.arange(0.0, rec.duration - WINDOW_S + 1e-9, STEP_S):
            end = start + WINDOW_S
            refs = {k: reference_rr(v, start, end) for k, v in rec.breaths.items()}
            valid = [r for r in refs.values() if np.isfinite(r)]
            if not valid:
                continue
            row = {"subject": subject, "start_s": start,
                   "ref_rr": float(np.mean(valid)),
                   "ref_annotator_diff": float(abs(valid[0] - valid[1])) if len(valid) == 2 else np.nan}
            row.update({f"ref_{k}": v for k, v in refs.items()})
            row.update(window_features(mods, rec.fs, start, end))
            rows.append(row)
        print(f"subject {subject}: {sum(r['subject'] == subject for r in rows)} windows")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} windows from {df.subject.nunique()} subjects -> {args.out}")


if __name__ == "__main__":
    main()
