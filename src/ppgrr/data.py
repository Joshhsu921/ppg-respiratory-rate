"""Load the BIDMC PPG and Respiration Dataset (PhysioNet, ODC-BY 1.0).

Pimentel et al., "Towards a robust estimation of respiratory rate from pulse
oximeters", IEEE TBME 2017. 53 ICU recordings, 8 min each, PPG at 125 Hz, with
breaths annotated by two annotators on the impedance respiratory signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Recording:
    subject: str
    fs: float
    ppg: np.ndarray
    resp: np.ndarray
    breaths: dict[str, np.ndarray] = field(default_factory=dict)  # annotator -> breath times (s)

    @property
    def duration(self) -> float:
        return len(self.ppg) / self.fs


def _col(df: pd.DataFrame, key: str) -> str:
    for c in df.columns:
        if key.lower() in c.lower():
            return c
    raise KeyError(f"no column containing {key!r} in {list(df.columns)}")


def find_csv_dir(root: Path) -> Path:
    hits = sorted(Path(root).rglob("bidmc_01_Signals.csv"))
    if not hits:
        raise FileNotFoundError(f"BIDMC CSV files not found under {root}")
    return hits[0].parent


def list_subjects(csv_dir: Path) -> list[str]:
    return sorted(p.name.split("_")[1] for p in Path(csv_dir).glob("bidmc_*_Signals.csv"))


def load_recording(csv_dir: Path, subject: str) -> Recording:
    csv_dir = Path(csv_dir)
    sig = pd.read_csv(csv_dir / f"bidmc_{subject}_Signals.csv")
    sig.columns = [c.strip() for c in sig.columns]
    t = sig[_col(sig, "Time")].to_numpy(dtype=float)
    # The CSV rounds timestamps (steps of 0.008 and 0.01 s), so the median step gives 100 Hz;
    # derive the rate from sample count over the recording span instead (125 Hz).
    fs = float(round((len(t) - 1) / (t[-1] - t[0])))

    br = pd.read_csv(csv_dir / f"bidmc_{subject}_Breaths.csv")
    br.columns = [c.strip() for c in br.columns]
    breaths = {}
    for c in br.columns:
        if "ann" in c.lower():
            samples = br[c].dropna().to_numpy(dtype=float)
            breaths[c.split()[1] if len(c.split()) > 1 else c] = (samples - 1) / fs  # 1-based sample numbers
    return Recording(
        subject=subject,
        fs=fs,
        ppg=sig[_col(sig, "PLETH")].to_numpy(dtype=float),
        resp=sig[_col(sig, "RESP")].to_numpy(dtype=float),
        breaths=breaths,
    )


def load_numerics(csv_dir: Path, subject: str) -> pd.DataFrame:
    """1 Hz monitor values: HR (from ECG), PULSE (monitor's own PPG pulse rate), RESP, SpO2."""
    num = pd.read_csv(Path(csv_dir) / f"bidmc_{subject}_Numerics.csv")
    num.columns = [c.strip() for c in num.columns]
    return num.rename(columns={_col(num, "Time"): "time"})


def window_mean(num: pd.DataFrame, col: str, start: float, end: float) -> float:
    v = num.loc[(num["time"] >= start) & (num["time"] < end), col].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) >= 0.8 * (end - start) else np.nan


def reference_rr(breath_times: np.ndarray, start: float, end: float) -> float:
    """Reference respiratory rate in [start, end): 60 / mean breath interval (needs >= 3 breaths)."""
    b = breath_times[(breath_times >= start) & (breath_times < end)]
    if len(b) < 3:
        return np.nan
    return 60.0 / float(np.mean(np.diff(b)))
