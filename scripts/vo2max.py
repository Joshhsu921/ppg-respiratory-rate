"""Can VO2 max be estimated without an exercise test? Validation on measured treadmill VO2 max.

Data: Treadmill Maximal Exercise Tests, University of Malaga (PhysioNet, 992 tests, 857 people,
mostly amateur and professional athletes). Measured VO2 max = highest 30 s rolling mean of
breath-by-breath VO2, per kg. The "start heart rate" (median of the first 10 s of the test, standing
or starting to walk) stands in for the resting heart rate a phone would measure; it is not a true
resting value, which makes the formula look worse than it would with a proper resting reading.

Compared, with cross-validation grouped by person (some people did several tests):
  - Uth et al. (2004): 15.3 x HRmax / HRrest, with HRmax = 208 - 0.7 x age (Tanaka) -> phone-only
  - Uth with the measured HRmax (needs a maximal test; an upper reference, not usable at home)
  - linear regression and LightGBM on age, sex, BMI (+ start heart rate)
Then error propagation: how much worse does the best model get with the heart-rate error measured
for the web algorithm on BIDMC (bias -0.6, SD about 3.2 beats/min)?
Usage:  python scripts/vo2max.py
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from ppgrr.evaluate import agreement, bland_altman

HR_ERR_BIAS, HR_ERR_SD = -0.6, 3.2   # web algorithm vs ECG, 32 s windows (results/hr_metrics.csv)


def per_test(root: Path) -> pd.DataFrame:
    info = pd.read_csv(root / "subject-info.csv")
    m = pd.read_csv(root / "test_measure.csv").dropna(subset=["VO2", "HR"])
    rows = []
    for tid, g in m.groupby("ID_test"):
        g = g.sort_values("time").set_index(pd.to_timedelta(g.sort_values("time")["time"], unit="s"))
        vo2 = g["VO2"].rolling("30s").mean()
        hr = g["HR"].rolling("10s").mean()
        start = g[g["time"] <= g["time"].iloc[0] + 10]
        rows.append({"ID_test": tid, "vo2max_ml_min": float(vo2.max()), "hrmax": float(hr.max()),
                     "hr_start": float(start["HR"].median()), "start_speed": float(start["Speed"].iloc[0]),
                     "duration_s": float(g["time"].iloc[-1])})
    d = pd.DataFrame(rows).merge(info, on="ID_test")
    d["vo2max"] = d["vo2max_ml_min"] / d["Weight"]
    d["bmi"] = d["Weight"] / (d["Height"] / 100) ** 2
    d["female"] = d["Sex"].astype(int)
    keep = d["vo2max"].between(15, 90) & d["hr_start"].between(35, 140) & (d["duration_s"] > 300)
    return d[keep].reset_index(drop=True)


def uth(age, hr_rest, hrmax=None):
    hrmax = 208 - 0.7 * age if hrmax is None else hrmax
    return 15.3 * hrmax / hr_rest


def grouped_oof(d: pd.DataFrame, cols: list[str], make, hr_noise: np.ndarray | None = None) -> np.ndarray:
    pred = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(d, groups=d["ID"]):
        model = make().fit(d.iloc[tr][cols], d.iloc[tr]["vo2max"])
        Xte = d.iloc[te][cols].copy()
        if hr_noise is not None and "hr_start" in cols:
            Xte["hr_start"] = Xte["hr_start"] + hr_noise[te]
        pred[te] = model.predict(Xte)
    return pred


def summary(ref, est) -> dict:
    m = agreement(ref, est)
    ok = np.isfinite(est)
    m["mape_pct"] = float(np.mean(np.abs(est[ok] - ref[ok]) / ref[ok]) * 100)
    m["within_10pct"] = float(np.mean(np.abs(est[ok] - ref[ok]) / ref[ok] <= 0.10))
    m["r"] = float(np.corrcoef(ref[ok], est[ok])[0, 1]) if np.std(est[ok]) > 0 else np.nan
    return m


def export_web_model(d: pd.DataFrame, cv_residuals: np.ndarray, out: Path) -> None:
    """Linear model on age, sex, BMI for the web card, with its cross-validated 95 % error range."""
    cols = ["Age", "female", "bmi"]
    lr = LinearRegression().fit(d[cols], d["vo2max"])
    lo, hi = np.percentile(cv_residuals, [2.5, 97.5])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "intercept": round(float(lr.intercept_), 4),
        "coef": {c: round(float(v), 4) for c, v in zip(["age", "female", "bmi"], lr.coef_)},
        "residual_p2_5": round(float(lo), 2), "residual_p97_5": round(float(hi), 2),
        "cv_mae": round(float(np.mean(np.abs(cv_residuals))), 2),
        "train": {"tests": int(len(d)), "people": int(d.ID.nunique()),
                  "age_min": float(d.Age.min()), "age_max": float(d.Age.max()),
                  "bmi_min": round(float(d.bmi.min()), 1), "bmi_max": round(float(d.bmi.max()), 1),
                  "female_share": round(float(d.female.mean()), 3),
                  "population": "amateur and professional athletes (University of Malaga, PhysioNet)"},
    }, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/treadmill"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    root = next(p.parent for p in args.raw.rglob("subject-info.csv"))
    d = per_test(root)
    ref = d["vo2max"].to_numpy()

    lgbm = lambda: LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=30,
                                 random_state=0, verbose=-1)
    base_cols, hr_cols = ["Age", "female", "bmi"], ["Age", "female", "bmi", "hr_start"]
    est = {
        "Uth formula (age + start HR), phone-only": uth(d["Age"], d["hr_start"]).to_numpy(),
        "Uth formula with measured HRmax (needs max test)": uth(d["Age"], d["hr_start"], d["hrmax"]).to_numpy(),
        "Linear regression: age, sex, BMI": grouped_oof(d, base_cols, LinearRegression),
        "Linear regression: age, sex, BMI, start HR": grouped_oof(d, hr_cols, LinearRegression),
        "LightGBM: age, sex, BMI": grouped_oof(d, base_cols, lgbm),
        "LightGBM: age, sex, BMI, start HR": grouped_oof(d, hr_cols, lgbm),
        "Predict the training mean (no information)": np.full(len(d), ref.mean()),
    }
    # error propagation: phone heart-rate error added to the test-set heart rate
    rng = np.random.default_rng(0)
    noisy = np.zeros(len(d))
    for rep in range(20):
        noise = rng.normal(HR_ERR_BIAS, HR_ERR_SD, len(d))
        noisy += grouped_oof(d, hr_cols, LinearRegression, hr_noise=noise) / 20
    est["Linear regression with start HR + simulated phone HR error"] = noisy
    noise = rng.normal(HR_ERR_BIAS, HR_ERR_SD, len(d))
    est["Uth formula + simulated phone HR error"] = uth(d["Age"], d["hr_start"] + noise).to_numpy()

    table = pd.DataFrame([{"method": k, **summary(ref, v)} for k, v in est.items()])
    table.to_csv(args.out / "vo2max_metrics.csv", index=False)

    sub = d["start_speed"] == 0
    stand = pd.DataFrame([{"subset": "tests starting from standing still", "n": int(sub.sum()),
                           **summary(ref[sub], est["Uth formula (age + start HR), phone-only"][sub])}])
    stand.to_csv(args.out / "vo2max_standing_start.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, k in zip(axes, ["Uth formula (age + start HR), phone-only", "Linear regression: age, sex, BMI, start HR"]):
        bland_altman(ax, ref, est[k], k)
        ax.set_xlabel("Mean of measured and estimated VO2 max (mL/kg/min)")
        ax.set_ylabel("Estimated − measured (mL/kg/min)")
    fig.tight_layout()
    fig.savefig(args.out / "vo2max_bland_altman.png", dpi=130)

    export_web_model(d, ref - est["Linear regression: age, sex, BMI"], Path("web/data/vo2max_model.json"))

    print(f"{len(d)} tests from {d.ID.nunique()} people; measured VO2 max {ref.mean():.1f} ± {ref.std():.1f} mL/kg/min; "
          f"start HR median {d.hr_start.median():.0f}; female {d.female.mean():.0%}")
    pd.set_option("display.width", 220)
    print(table[["method", "mae", "bias", "loa_low", "loa_high", "mape_pct", "within_10pct", "r"]].round(2).to_string(index=False))
    print(stand[["subset", "n", "mae", "bias", "mape_pct", "r"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
