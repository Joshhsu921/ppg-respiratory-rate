"""Can we fix LightGBM's regression to the mean at slow breathing (< 12 breaths/min)?

Compares, with the same subject-wise 5-fold CV as run_experiments.py:
  - baseline LightGBM
  - LightGBM with inverse-frequency sample weights (computed on each training fold only)
  - hybrid: use the RIIV spectral estimate when it reads below 12, else LightGBM
    (threshold fixed at the clinical cut-off in advance, not tuned)
  - safety-first hybrid: switch only when RIIV spectral AND RIIV counting both read below 12.
    Chosen after comparing five switching rules on this same data, so it is somewhat optimistic
    and needs confirming on an independent dataset.
  - LightGBM quantile models (10th / 90th percentile): does the interval cover the truth,
    and does abstaining on wide intervals remove the bad windows?
Beyond MAE per band, reports how well each method flags slow (< 12) and fast (> 20) breathing.
Usage:  python scripts/slow_breathing.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from ppgrr.features import FEATURE_COLUMNS

PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
BANDS = [0, 12, 20, 60]
BAND_LABELS = ["<12", "12–20", ">20"]


def inverse_frequency_weights(y: np.ndarray, width: float = 2.0, cap: float = 20.0) -> np.ndarray:
    bins = np.floor(y / width)
    _, inv, counts = np.unique(bins, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    w = w / w.mean()
    return np.minimum(w, cap)


def oof(df: pd.DataFrame, seed: int = 0) -> dict[str, np.ndarray]:
    X, y, groups = df[FEATURE_COLUMNS], df["ref_rr"].to_numpy(), df["subject"].to_numpy()
    out = {k: np.full(len(df), np.nan) for k in ("base", "weighted", "q10", "q90")}
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        Xtr, ytr, Xte = X.iloc[train], y[train], X.iloc[test]
        out["base"][test] = LGBMRegressor(random_state=seed, **PARAMS).fit(Xtr, ytr).predict(Xte)
        out["weighted"][test] = LGBMRegressor(random_state=seed, **PARAMS).fit(
            Xtr, ytr, sample_weight=inverse_frequency_weights(ytr)).predict(Xte)
        for q in (10, 90):
            out[f"q{q}"][test] = LGBMRegressor(objective="quantile", alpha=q / 100, random_state=seed,
                                               **PARAMS).fit(Xtr, ytr).predict(Xte)
    return out


def safety_first(df: pd.DataFrame, model_pred) -> np.ndarray:
    """Use the classical RIIV estimate when two independent RIIV readings agree on slow breathing."""
    slow = ((df["riiv_spec"] < 12) & (df["riiv_count"] < 12)).to_numpy()
    return np.where(slow, df["riiv_spec"], model_pred)


def detection(ref: np.ndarray, est: np.ndarray, low: bool, cut: float) -> tuple[float, float]:
    ok = np.isfinite(est)
    truth = (ref < cut) if low else (ref > cut)
    flag = (est < cut) if low else (est > cut)
    sens = float((flag & truth & ok).sum() / max(1, (truth & ok).sum()))
    spec = float((~flag & ~truth & ok).sum() / max(1, (~truth & ok).sum()))
    return sens, spec


def summarise(df: pd.DataFrame, col: str) -> dict:
    ref, est = df["ref_rr"].to_numpy(), df[col].to_numpy()
    ok = np.isfinite(est)
    err = np.abs(est - ref)
    row = {"coverage": float(ok.mean()), "mae": float(np.nanmean(err))}
    band = pd.cut(df["ref_rr"], BANDS, labels=BAND_LABELS)
    for b in BAND_LABELS:
        row[f"mae {b}"] = float(np.nanmean(err[(band == b).to_numpy()]))
    row["slow sens"], row["slow spec"] = detection(ref, est, low=True, cut=12)
    row["fast sens"], row["fast spec"] = detection(ref, est, low=False, cut=20)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, default=Path("data/processed/windows.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/slow_breathing.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.windows, dtype={"subject": str})
    slow_subjects = df[df.ref_rr < 12].subject.nunique()
    print(f"{len(df)} windows; {int((df.ref_rr < 12).sum())} slow windows from {slow_subjects} subjects")

    p = oof(df)
    df["base"], df["weighted"] = p["base"], p["weighted"]
    df["hybrid"] = np.where(df["riiv_spec"] < 12, df["riiv_spec"], df["base"])
    df["hybrid_weighted"] = np.where(df["riiv_spec"] < 12, df["riiv_spec"], df["weighted"])
    df["hybrid_safety"] = safety_first(df, df["base"])
    width = p["q90"] - p["q10"]
    inside = (df["ref_rr"] >= p["q10"]) & (df["ref_rr"] <= p["q90"])
    df["base_abstain"] = np.where(width <= 6, df["base"], np.nan)

    methods = {
        "RIIV spectral (classical)": "riiv_spec",
        "LightGBM baseline": "base",
        "LightGBM, inverse-frequency weights": "weighted",
        "Hybrid: RIIV if < 12, else LightGBM": "hybrid",
        "Hybrid with weighted LightGBM": "hybrid_weighted",
        "Safety-first: RIIV if spectral and counting both < 12": "hybrid_safety",
        "LightGBM, abstain if 80% interval > 6": "base_abstain",
    }
    table = pd.DataFrame([{"method": n, **summarise(df, c)} for n, c in methods.items()])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    band = pd.cut(df["ref_rr"], BANDS, labels=BAND_LABELS)
    print("80% quantile interval: coverage overall {:.0%}, median width {:.1f};".format(inside.mean(), np.median(width)),
          ", ".join(f"{b}: coverage {inside[band == b].mean():.0%}, width {np.median(width[(band == b).to_numpy()]):.1f}"
                    for b in BAND_LABELS))
    pd.set_option("display.width", 220)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
