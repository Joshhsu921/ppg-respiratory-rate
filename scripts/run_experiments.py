"""Compare classical estimators with a LightGBM model using subject-wise cross-validation.

Usage:  python scripts/run_experiments.py [--windows data/processed/windows.csv] [--out results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from ppgrr.evaluate import agreement, bland_altman, per_subject_mae
from ppgrr.features import FEATURE_COLUMNS

CLASSICAL = {
    "RIIV (spectral)": "riiv_spec",
    "RIAV (spectral)": "riav_spec",
    "RIFV (spectral)": "rifv_spec",
    "Smart fusion (spectral)": "fusion_spec",
    "Smart fusion (counting)": "fusion_count",
}


def lightgbm_oof(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions; every subject is predicted by a model that never saw it."""
    X, y, groups = df[FEATURE_COLUMNS], df["ref_rr"].to_numpy(), df["subject"].to_numpy()
    pred = np.full(len(df), np.nan)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15,
                              min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed, verbose=-1)
        model.fit(X.iloc[train], y[train])
        pred[test] = model.predict(X.iloc[test])
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, default=Path("data/processed/windows.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.windows, dtype={"subject": str})
    df["lightgbm"] = lightgbm_oof(df)
    methods = {**CLASSICAL, "LightGBM (all windows)": "lightgbm"}

    # Safety-first mode: trust the classical RIIV estimate when two RIIV readings agree on
    # slow breathing (see scripts/slow_breathing.py for how this rule was chosen).
    slow = (df["riiv_spec"] < 12) & (df["riiv_count"] < 12)
    df["hybrid_safety"] = np.where(slow, df["riiv_spec"], df["lightgbm"])
    methods["Safety-first hybrid"] = "hybrid_safety"

    # Fair comparison: LightGBM restricted to the windows smart fusion accepts.
    accepted = np.isfinite(df["fusion_spec"])
    df["lightgbm_accepted"] = np.where(accepted, df["lightgbm"], np.nan)
    methods["LightGBM (fusion-accepted windows)"] = "lightgbm_accepted"

    rows = []
    for name, col in methods.items():
        m = agreement(df["ref_rr"], df[col])
        m["subject_mae_median"] = float(per_subject_mae(df, col).median())
        rows.append({"method": name, **m})
    if {"ref_ann1", "ref_ann2"} <= set(df.columns):
        rows.append({"method": "Annotator 2 vs annotator 1 (ceiling)",
                     **agreement(df["ref_ann1"], df["ref_ann2"])})
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "metrics.csv", index=False)
    df[["subject", "start_s", "ref_rr", *methods.values()]].to_csv(args.out / "predictions.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    shown = ["RIIV (spectral)", "Smart fusion (spectral)", "Smart fusion (counting)",
             "LightGBM (all windows)", "Safety-first hybrid", "LightGBM (fusion-accepted windows)"]
    for ax, (name, col) in zip(axes.flat, [(n, methods[n]) for n in shown]):
        bland_altman(ax, df["ref_rr"], df[col], name)
    fig.tight_layout()
    fig.savefig(args.out / "bland_altman.png", dpi=130)

    # Error by reference-rate band
    bins = pd.cut(df["ref_rr"], [0, 12, 20, 30, 60], labels=["<12", "12–20", "20–30", ">30"])
    by_band = {name: df.groupby(bins, observed=True).apply(
        lambda g: float(np.nanmean(np.abs(g[col] - g["ref_rr"])))) for name, col in methods.items()}
    pd.DataFrame(by_band).to_csv(args.out / "mae_by_rr_band.csv")

    with open(args.out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({"n_windows": int(len(df)), "n_subjects": int(df.subject.nunique())}, fh, indent=2)

    pd.set_option("display.width", 200)
    print(table[["method", "n", "coverage", "mae", "rmse", "bias", "loa_low", "loa_high", "within_2bpm"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
