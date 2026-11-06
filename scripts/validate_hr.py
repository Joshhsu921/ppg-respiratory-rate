"""Validate pulse-rate estimation against the ECG heart rate recorded by the BIDMC monitor.

Compares, on 32 s and 10 s windows:
  - peak detection (Python, 125 Hz)      60 / median beat interval
  - spectral peak (Python, 125 Hz)
  - the deployed web algorithm (ppg-dsp.js on 30 Hz PPG; run scripts/hr_web_eval.js first)
  - the monitor's own pulse rate (PULSE), as a commercial reference point
Usage:  python scripts/validate_hr.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppgrr.data import find_csv_dir, list_subjects, load_numerics, load_recording, window_mean
from ppgrr.dsp import peak_hr, respiratory_modulations, spectral_hr
from ppgrr.evaluate import agreement, bland_altman

WINDOWS = [(32.0, 8.0), (10.0, 4.0)]


def build(csv_dir: Path) -> pd.DataFrame:
    rows = []
    for s in list_subjects(csv_dir):
        rec, num = load_recording(csv_dir, s), load_numerics(csv_dir, s)
        mods = respiratory_modulations(rec.ppg, rec.fs)
        for win, step in WINDOWS:
            for start in np.arange(0.0, rec.duration - win + 1e-9, step):
                end = start + win
                rows.append({
                    "subject": s, "start_s": float(start), "win_s": win,
                    "ref_hr": window_mean(num, "HR", start, end),
                    "monitor_pulse": window_mean(num, "PULSE", start, end),
                    "peak_hr": peak_hr(mods["beat_times"], start, end),
                    "spectral_hr": spectral_hr(mods["band"], rec.fs, start, end),
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--web", type=Path, default=Path("results/hr_web_predictions.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    df = build(find_csv_dir(args.raw))
    if args.web.exists():
        web = pd.read_csv(args.web, dtype={"subject": str})
        df = df.merge(web[["subject", "start_s", "win_s", "web_hr", "web_quality"]],
                      on=["subject", "start_s", "win_s"], how="left")
    df = df[np.isfinite(df["ref_hr"])]

    methods = {"Peak detection (125 Hz)": "peak_hr", "Spectral peak (125 Hz)": "spectral_hr",
               "Web algorithm (30 Hz)": "web_hr", "Monitor pulse rate (commercial)": "monitor_pulse"}
    rows = []
    for win, _ in WINDOWS:
        d = df[df.win_s == win]
        for name, col in methods.items():
            if col not in d:
                continue
            m = agreement(d["ref_hr"], d[col])
            ok = np.isfinite(d[col])
            err = (d[col] - d["ref_hr"])[ok]
            m["mape_pct"] = float((err.abs() / d["ref_hr"][ok]).mean() * 100)
            m["arms"] = float(np.sqrt((err ** 2).mean()))
            m["within_5bpm"] = float((err.abs() <= 5).mean())
            rows.append({"window_s": int(win), "method": name, **m})
        if "web_quality" in d:   # does the quality flag single out bad windows?
            for q in ("good", "fair", "poor"):
                dq = d[d.web_quality == q]
                if len(dq):
                    rows.append({"window_s": int(win), "method": f"Web algorithm, quality={q}",
                                 **agreement(dq["ref_hr"], dq["web_hr"])})
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "hr_metrics.csv", index=False)

    d32 = df[df.win_s == 32]
    cols = [c for c in methods.values() if c in d32]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.5), sharey=True)
    for ax, (name, col) in zip(np.atleast_1d(axes), [(n, c) for n, c in methods.items() if c in d32]):
        bland_altman(ax, d32["ref_hr"], d32[col], name)
        ax.set_xlabel("Mean of ECG HR and estimate (beats/min)")
        ax.set_ylabel("Estimate − ECG HR (beats/min)")
    fig.tight_layout()
    fig.savefig(args.out / "hr_bland_altman.png", dpi=130)

    # per-subject view: where do large errors come from?
    if "web_hr" in d32:
        per = d32.assign(err=(d32.web_hr - d32.ref_hr).abs()).groupby("subject")["err"].mean().sort_values(ascending=False)
        per.round(2).to_csv(args.out / "hr_web_error_by_subject.csv")
        print("worst subjects (web, 32 s MAE):", per.head(5).round(1).to_dict())

    pd.set_option("display.width", 220)
    print(table[["window_s", "method", "n", "coverage", "mae", "bias", "loa_low", "loa_high",
                 "within_5bpm", "mape_pct", "arms"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
