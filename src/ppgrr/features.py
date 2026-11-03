"""Per-window features: classical estimates plus signal-quality descriptors for the ML model."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .dsp import counting_rr, resample_window, smart_fusion, spectral_rr

MODULATIONS = ("riiv", "riav", "rifv")
WINDOW_S = 32.0
STEP_S = 4.0


def window_features(mods: dict, fs: float, start: float, end: float) -> dict:
    f: dict[str, float] = {}
    for m in MODULATIONS:
        t, v = mods[m]
        y = resample_window(t, v, start, end, reject_outliers=(m == "rifv"))
        if y is None:
            f[f"{m}_spec"], f[f"{m}_conc"], f[f"{m}_count"] = np.nan, np.nan, np.nan
        else:
            f[f"{m}_spec"], f[f"{m}_conc"] = spectral_rr(y)
            f[f"{m}_count"] = counting_rr(y)

    specs = [f[f"{m}_spec"] for m in MODULATIONS]
    counts = [f[f"{m}_count"] for m in MODULATIONS]
    f["fusion_spec"] = smart_fusion(specs)
    f["fusion_count"] = smart_fusion(counts)
    f["spec_std"] = float(np.nanstd(specs)) if np.isfinite(specs).any() else np.nan
    f["spec_median"] = float(np.nanmedian(specs)) if np.isfinite(specs).any() else np.nan

    bt = mods["beat_times"]
    bt = bt[(bt >= start) & (bt < end)]
    ibi = np.diff(bt)
    f["n_beats"] = float(len(bt))
    f["hr"] = 60.0 / float(np.mean(ibi)) if len(ibi) else np.nan
    f["ibi_cv"] = float(np.std(ibi) / np.mean(ibi)) if len(ibi) > 1 else np.nan

    seg = mods["band"][int(start * fs):int(end * fs)]
    f["ppg_skew"] = float(stats.skew(seg))      # skewness SQI (Elgendi, 2016)
    f["ppg_kurt"] = float(stats.kurtosis(seg))
    return f


FEATURE_COLUMNS = (
    [f"{m}_{k}" for m in MODULATIONS for k in ("spec", "conc", "count")]
    + ["fusion_spec", "fusion_count", "spec_std", "spec_median",
       "n_beats", "hr", "ibi_cv", "ppg_skew", "ppg_kurt"]
)
