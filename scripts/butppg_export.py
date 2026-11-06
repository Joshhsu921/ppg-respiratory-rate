"""Export BUT PPG smartphone recordings for the browser algorithm (scripts/butppg_eval.js).

BUT PPG (Brno University of Technology, PhysioNet, CC BY 4.0): 3,888 ten-second smartphone PPG
recordings from 50 adults at 30 Hz, with ECG-derived reference heart rate and quality labels.
Two storage layouts exist:
  - early records: 300 one-sample "signals" holding a single channel stored as negative intensity;
  - later records: three channels (PPG_R, PPG_G, PPG_B), 300 samples each, as raw intensity.
The web demo feeds blood-volume orientation (more blood -> higher) and picks the better of red and
green, so this export writes red and green in that orientation. Orientation is checked from the
waveform shape (a blood-volume pulse rises fast and falls slowly, so its derivative is positively
skewed), not from heart-rate accuracy.
Usage:  python scripts/butppg_export.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import wfdb
from scipy import signal as ss
from scipy import stats

SOS = ss.butter(2, [0.5, 4.0], btype="bandpass", fs=30, output="sos")


def derivative_skew(x: np.ndarray) -> float:
    return float(stats.skew(np.diff(ss.sosfiltfilt(SOS, x))))


def channels(rec_dir: Path) -> tuple[float, np.ndarray, np.ndarray | None, str]:
    r = wfdb.rdrecord(str(rec_dir / f"{rec_dir.name}_PPG"))
    if r.sig_len == 1:                       # single channel spread over one-sample signals
        return float(r.fs), r.p_signal.ravel(), None, "single (-intensity)"
    names = [n.split()[-1] for n in r.sig_name]
    red = r.p_signal[:, names.index("PPG_R")]
    green = r.p_signal[:, names.index("PPG_G")]
    return float(r.fs), -red, -green, "rgb (raw intensity, inverted)"   # invert raw intensity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/butppg"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/butppg_ppg.csv"))
    args = ap.parse_args()
    root = next(p.parent for p in args.raw.rglob("quality-hr-ann.csv"))

    skews: dict[str, list[float]] = {}
    lines = []
    for rec in sorted(p for p in root.iterdir() if p.is_dir()):
        fs, red, green, layout = channels(rec)
        skews.setdefault(layout, []).append(derivative_skew(red))
        g = green if green is not None else np.full_like(red, np.nan)
        lines.append(",".join([rec.name, f"{fs:g}", str(len(red))]
                              + [f"{v:.5f}" for v in red] + [f"{v:.5f}" if np.isfinite(v) else "" for v in g]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    for layout, s in skews.items():
        s = np.array(s)
        ok = np.isfinite(s)   # a flat channel (e.g. saturated) has no defined skew
        print(f"{layout}: {len(s)} records ({(~ok).sum()} flat), derivative skew median {np.median(s[ok]):+.2f}, "
              f"positive in {np.mean(s[ok] > 0):.0%}")
    print(f"wrote {len(lines)} recordings -> {args.out}")


if __name__ == "__main__":
    main()
