"""Agreement metrics and Bland-Altman plots."""
from __future__ import annotations

import numpy as np
import pandas as pd


def agreement(ref, est) -> dict:
    """MAE, RMSE, bias and 95 % limits of agreement; NaN estimates count against coverage."""
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    err = est[ok] - ref[ok]
    if not len(err):
        return {"n": 0, "coverage": 0.0}
    sd = err.std(ddof=1) if len(err) > 1 else 0.0
    return {
        "n": int(ok.sum()),
        "coverage": float(ok.mean()),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "loa_low": float(err.mean() - 1.96 * sd),
        "loa_high": float(err.mean() + 1.96 * sd),
        "within_2bpm": float((np.abs(err) <= 2).mean()),
    }


def per_subject_mae(df: pd.DataFrame, est_col: str, ref_col: str = "ref_rr") -> pd.Series:
    d = df[np.isfinite(df[est_col])]
    return d.groupby("subject").apply(lambda g: float(np.abs(g[est_col] - g[ref_col]).mean()))


def bland_altman(ax, ref, est, title: str) -> None:
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    mean, diff = (ref[ok] + est[ok]) / 2, est[ok] - ref[ok]
    m = agreement(ref, est)
    ax.scatter(mean, diff, s=6, alpha=0.35, color="#1d5a86", linewidths=0)
    ax.axhline(m["bias"], color="#14242c", lw=1.2, label=f"bias {m['bias']:+.2f}")
    for k in ("loa_low", "loa_high"):
        ax.axhline(m[k], color="#c3291f", lw=1, ls="--")
    ax.set_title(f"{title}\nMAE {m['mae']:.2f} bpm · coverage {m['coverage']:.0%}", fontsize=10)
    ax.set_xlabel("Mean of reference and estimate (breaths/min)")
    ax.set_ylabel("Estimate − reference (breaths/min)")
    ax.grid(alpha=0.25)
