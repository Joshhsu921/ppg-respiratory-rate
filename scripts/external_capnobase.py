"""External validation: models built on BIDMC, tested unchanged on CapnoBase.

Nothing is tuned on CapnoBase (its authors ask that it be used for testing only): LightGBM is
refit on all BIDMC windows with the same settings, and the safety-first rule keeps its threshold.
Reports agreement overall, by age group and ventilation mode, by breathing-rate band, and how
well slow (< 12) and fast (> 20) breathing are flagged.
Usage:  python scripts/external_capnobase.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ppgrr.capnobase import list_cases, load_case
from ppgrr.data import reference_rr
from ppgrr.dsp import respiratory_modulations
from ppgrr.evaluate import agreement, bland_altman
from ppgrr.features import FEATURE_COLUMNS, STEP_S, WINDOW_S, window_features

PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)


def build_windows(root: Path) -> pd.DataFrame:
    rows = []
    for case in list_cases(root):
        rec = load_case(root, case)
        mods = respiratory_modulations(rec.ppg, rec.fs)
        for start in np.arange(0.0, rec.duration - WINDOW_S + 1e-9, STEP_S):
            end = start + WINDOW_S
            ref = reference_rr(rec.breaths["co2"], start, end)
            if not np.isfinite(ref) or rec.overlaps_artifact(start, end):
                continue
            rows.append({"subject": case, "start_s": float(start), "ref_rr": ref, "age": rec.age,
                         "ventilation": rec.ventilation, **window_features(mods, rec.fs, start, end)})
    return pd.DataFrame(rows)


def detection(ref, est, low: bool, cut: float) -> tuple[float, float]:
    ok = np.isfinite(est)
    truth = (ref < cut) if low else (ref > cut)
    flag = (est < cut) if low else (est > cut)
    sens = (flag & truth & ok).sum() / max(1, (truth & ok).sum())
    spec = (~flag & ~truth & ok).sum() / max(1, (~truth & ok).sum())
    return float(sens), float(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capnobase", type=Path, default=Path("data/raw/capnobase"))
    ap.add_argument("--bidmc", type=Path, default=Path("data/processed/windows.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    cb = build_windows(args.capnobase)
    cb.to_csv(Path("data/processed/capnobase_windows.csv"), index=False)
    train = pd.read_csv(args.bidmc, dtype={"subject": str})
    model = LGBMRegressor(**PARAMS).fit(train[FEATURE_COLUMNS], train["ref_rr"])
    cb["lightgbm"] = model.predict(cb[FEATURE_COLUMNS])
    slow = (cb["riiv_spec"] < 12) & (cb["riiv_count"] < 12)
    cb["hybrid_safety"] = np.where(slow, cb["riiv_spec"], cb["lightgbm"])

    methods = {"RIIV (spectral)": "riiv_spec", "Smart fusion (spectral)": "fusion_spec",
               "Smart fusion (counting)": "fusion_count", "LightGBM (trained on BIDMC)": "lightgbm",
               "Safety-first hybrid": "hybrid_safety"}
    cb["group"] = np.where(cb["age"] < 18, "child", np.where(cb["age"] >= 18, "adult", "unknown age"))
    subsets = {"all": cb, "children (< 18)": cb[cb.group == "child"], "adults": cb[cb.group == "adult"],
               "spontaneous breathing": cb[cb.ventilation == "spontaneous"],
               "controlled ventilation": cb[cb.ventilation == "controlled"]}
    rows = []
    for sname, d in subsets.items():
        for name, col in methods.items():
            m = agreement(d["ref_rr"], d[col])
            ref, est = d["ref_rr"].to_numpy(), d[col].to_numpy()
            m["slow_sens"], m["slow_spec"] = detection(ref, est, True, 12)
            m["fast_sens"], m["fast_spec"] = detection(ref, est, False, 20)
            rows.append({"subset": sname, "method": name, "subjects": int(d.subject.nunique()), **m})
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "capnobase_metrics.csv", index=False)

    bands = pd.cut(cb["ref_rr"], [0, 12, 20, 30, 80], labels=["<12", "12–20", "20–30", ">30"])
    by_band = pd.DataFrame({name: cb.groupby(bands, observed=True).apply(
        lambda g: float(np.nanmean(np.abs(g[col] - g["ref_rr"])))) for name, col in methods.items()})
    by_band["windows"] = cb.groupby(bands, observed=True).size()
    by_band.to_csv(args.out / "capnobase_mae_by_rr_band.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, name in zip(axes, ["RIIV (spectral)", "LightGBM (trained on BIDMC)", "Safety-first hybrid"]):
        bland_altman(ax, cb["ref_rr"], cb[methods[name]], f"CapnoBase · {name}")
    fig.tight_layout()
    fig.savefig(args.out / "capnobase_bland_altman.png", dpi=130)

    print(f"{len(cb)} windows from {cb.subject.nunique()} cases; reference RR "
          f"{cb.ref_rr.min():.1f}–{cb.ref_rr.max():.1f} (median {cb.ref_rr.median():.1f})")
    print(by_band.round(2).to_string())
    pd.set_option("display.width", 240)
    print(table[["subset", "method", "subjects", "n", "coverage", "mae", "bias", "loa_low", "loa_high",
                 "within_2bpm", "slow_sens", "slow_spec", "fast_sens", "fast_spec"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
