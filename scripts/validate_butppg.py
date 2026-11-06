"""Heart rate from smartphone-camera PPG in adults: the web algorithm vs ECG on BUT PPG.

Run scripts/butppg_export.py and `node scripts/butppg_eval.js` first. Reports agreement overall
and for the intended use (fingertip on the rear camera, at rest), by motion type, against the
dataset's expert quality label, and whether the algorithm's own quality flag separates good from
bad recordings. The flipped-polarity run is a sensitivity check only.
Usage:  python scripts/validate_butppg.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppgrr.evaluate import agreement, bland_altman

MOTION = {0: "resting", 1: "pressing harder", 2: "sliding finger", 3: "walking", 4: "coughing",
          5: "laughing", 6: "changing light", 7: "talking"}


def summary(d: pd.DataFrame, col: str = "web_hr") -> dict:
    m = agreement(d["HR"], d[col])
    err = (d[col] - d["HR"])[np.isfinite(d[col])]
    m["within_5bpm"] = float((err.abs() <= 5).mean()) if len(err) else np.nan
    m["within_5bpm_of_all"] = float((err.abs() <= 5).sum() / len(d)) if len(d) else np.nan
    return {"recordings": len(d), "subjects": int(d.subject.nunique()), **m}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/butppg"))
    ap.add_argument("--web", type=Path, default=Path("results/butppg_web_predictions.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    root = next(p.parent for p in args.raw.rglob("quality-hr-ann.csv"))

    info = pd.read_csv(root / "subject-info.csv", encoding="utf-8-sig")
    ann = pd.read_csv(root / "quality-hr-ann.csv", encoding="utf-8-sig")
    web = pd.read_csv(args.web)
    d = ann.merge(info, on="ID").merge(web, left_on="ID", right_on="id")
    d["subject"] = d["ID"].astype(str).str[:3]
    d["site"] = np.where(d["Ear/finger"] == 1, "finger", "ear")
    d["motion"] = d["Motion"].astype(str).map(lambda s: MOTION.get(int(s.split(";")[0]), s) if s[0].isdigit() else s)
    d.loc[d["Motion"].astype(str).str.contains(";"), "motion"] = "combined"

    intended = d[(d.site == "finger") & (d["Motion"].astype(str) == "0")]
    subsets = {
        "all recordings": d,
        "finger, at rest (intended use)": intended,
        "finger, at rest, expert quality good": intended[intended.Quality == 1],
        "finger, at rest, expert quality poor": intended[intended.Quality == 0],
        "finger, at rest, web quality good": intended[intended.web_quality == "good"],
        "finger, at rest, web quality fair or poor": intended[intended.web_quality != "good"],
        "ear, at rest": d[(d.site == "ear") & (d["Motion"].astype(str) == "0")],
    }
    for code, name in MOTION.items():
        if code:
            subsets[f"finger, {name}"] = d[(d.site == "finger") & (d["Motion"].astype(str) == str(code))]
    rows = [{"subset": k, **summary(v)} for k, v in subsets.items() if len(v)]
    rows.append({"subset": "finger, at rest — polarity flipped (sensitivity check)", **summary(intended, "web_hr_flipped")})

    # development / test split by subject (scripts/butppg_split.py); the algorithm was tuned on dev only
    split_file = Path("data/processed/butppg_split.csv")
    if split_file.exists():
        split = pd.read_csv(split_file, dtype={"subject": str})
        intended = intended.merge(split, on="subject", how="left")
        before = Path("data/processed/butppg_web_before.csv")
        if before.exists():
            b = pd.read_csv(before)[["id", "web_hr"]].rename(columns={"web_hr": "web_hr_before"})
            intended = intended.merge(b, on="id", how="left")
        for s in ("dev", "test"):
            part = intended[intended.split == s]
            if "web_hr_before" in part:
                rows.append({"subset": f"finger, at rest, {s} subjects — before tuning", **summary(part, "web_hr_before")})
            rows.append({"subset": f"finger, at rest, {s} subjects — current", **summary(part)})
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "butppg_metrics.csv", index=False)

    # does the web quality flag agree with the experts' judgement?
    agree = pd.crosstab(intended["web_quality"], intended["Quality"].map({1: "expert good", 0: "expert poor"}))
    agree.to_csv(args.out / "butppg_quality_crosstab.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    bland_altman(axes[0], intended["HR"], intended["web_hr"], "Smartphone finger, at rest (all)")
    g = intended[intended.web_quality == "good"]
    bland_altman(axes[1], g["HR"], g["web_hr"], "Smartphone finger, at rest, web quality good")
    for ax in axes:
        ax.set_xlabel("Mean of ECG HR and estimate (beats/min)")
        ax.set_ylabel("Estimate − ECG HR (beats/min)")
    fig.tight_layout()
    fig.savefig(args.out / "butppg_bland_altman.png", dpi=130)

    pd.set_option("display.width", 230)
    print(table[["subset", "recordings", "subjects", "coverage", "mae", "bias", "loa_low", "loa_high",
                 "within_5bpm", "within_5bpm_of_all"]].round(2).to_string(index=False))
    print(agree.to_string())


if __name__ == "__main__":
    main()
