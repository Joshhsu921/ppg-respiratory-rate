"""Load the CapnoBase IEEE TBME Respiratory Rate Benchmark (Karlen et al., IEEE TBME 2013).

42 eight-minute recordings during elective surgery and anaesthesia (children and adults,
spontaneous or controlled ventilation), sampled at 300 Hz. The breath reference is the start of
each expiration marked on the capnogram (CO2 waveform), which is more direct than BIDMC's
impedance signal. Labels store sample numbers (1-based) as space-separated lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Recording


@dataclass
class CapnoCase(Recording):
    age: float = np.nan
    ventilation: str = ""
    artifacts: np.ndarray | None = None      # (n, 2) PPG artifact intervals in seconds

    def overlaps_artifact(self, start: float, end: float) -> bool:
        if self.artifacts is None or not len(self.artifacts):
            return False
        return bool(((self.artifacts[:, 0] < end) & (self.artifacts[:, 1] > start)).any())


def _samples(cell) -> np.ndarray:
    if not isinstance(cell, str):
        return np.array([], dtype=float)
    return np.array([float(x) for x in cell.split()], dtype=float)


def list_cases(root: Path) -> list[str]:
    return sorted(p.name.replace("_signal.csv", "") for p in Path(root).glob("*_signal.csv"))


def load_case(root: Path, case: str) -> CapnoCase:
    root = Path(root)
    sig = pd.read_csv(root / f"{case}_signal.csv")
    lab = pd.read_csv(root / f"{case}_labels.csv", dtype=str)
    par = pd.read_csv(root / f"{case}_param.csv")
    meta = pd.read_csv(root / f"{case}_meta.csv")
    fs = float(par["samplingrate_pleth"][0])
    to_s = lambda x: (x - 1) / fs
    art = _samples(lab["pleth_artif_x"][0])
    art = to_s(art[: len(art) // 2 * 2].reshape(-1, 2)) if len(art) >= 2 else np.empty((0, 2))
    return CapnoCase(
        subject=case, fs=fs,
        ppg=sig["pleth_y"].to_numpy(dtype=float), resp=sig["co2_y"].to_numpy(dtype=float),
        breaths={"co2": to_s(_samples(lab["co2_startexp_x"][0]))},
        age=float(meta["subject_age"][0]) if pd.notna(meta["subject_age"][0]) else np.nan,
        ventilation=str(par["case_ventilation"][0]),
        artifacts=art,
    )


def ecg_hr(root: Path, case: str, fs: float, start: float, end: float) -> float:
    """Reference heart rate from the annotated ECG R peaks in [start, end)."""
    lab = pd.read_csv(Path(root) / f"{case}_labels.csv", dtype=str)
    t = (_samples(lab["ecg_peak_x"][0]) - 1) / fs
    t = t[(t >= start) & (t < end)]
    return 60.0 / float(np.median(np.diff(t))) if len(t) >= 4 else np.nan
