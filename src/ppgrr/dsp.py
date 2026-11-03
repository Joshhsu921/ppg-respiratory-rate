"""Signal processing: extract respiratory modulations from PPG and estimate respiratory rate.

Breathing modulates the PPG in three ways (Karlen et al., IEEE TBME 2013):
  - RIIV  respiratory-induced intensity variation (baseline wander)
  - RIAV  respiratory-induced amplitude variation (pulse amplitude)
  - RIFV  respiratory-induced frequency variation (beat intervals, i.e. RSA)
Each modulation is sampled once per beat, resampled to a uniform grid, and
its dominant frequency in the breathing band gives one respiratory-rate estimate.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as ss

RR_MIN_BPM = 4.0
RR_MAX_BPM = 42.0
RESAMPLE_HZ = 4.0


def _sos(kind: str, cutoff, fs: float, order: int = 4):
    return ss.butter(order, cutoff, btype=kind, fs=fs, output="sos")


def fill_nan(x) -> np.ndarray:
    """Linearly interpolate over NaN samples."""
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.any() and (~bad).any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def preprocess(ppg, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (low, band): low keeps the baseline (< 8 Hz), band is 0.5–8 Hz for beat detection."""
    x = fill_nan(ppg)
    low = ss.sosfiltfilt(_sos("lowpass", 8.0, fs), x)
    band = ss.sosfiltfilt(_sos("bandpass", [0.5, 8.0], fs), x)
    return low, band


def detect_beats(band: np.ndarray, fs: float, max_hr: float = 180.0) -> tuple[np.ndarray, np.ndarray]:
    """Detect systolic peaks; pair each peak with the trough just before it."""
    peaks, _ = ss.find_peaks(
        band, distance=max(1, int(fs * 60.0 / max_hr)), prominence=0.3 * np.std(band)
    )
    if len(peaks) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    troughs = np.array([a + int(np.argmin(band[a:b])) for a, b in zip(peaks[:-1], peaks[1:])])
    return peaks[1:], troughs


def respiratory_modulations(ppg, fs: float) -> dict:
    """Per-beat respiratory modulation series for a whole recording.

    Returns a dict with keys riiv / riav / rifv -> (times_s, values),
    plus "beat_times" and the filtered "band" signal for quality features.
    """
    low, band = preprocess(ppg, fs)
    pk, tr = detect_beats(band, fs)
    t = pk / fs
    return {
        "riiv": (t, low[pk]),
        "riav": (t, low[pk] - low[tr]),
        "rifv": ((t[1:] + t[:-1]) / 2.0, np.diff(t)),
        "beat_times": t,
        "band": band,
    }


def resample_window(times, values, start: float, end: float, fs: float = RESAMPLE_HZ,
                    reject_outliers: bool = False) -> np.ndarray | None:
    """Resample the beat-by-beat series inside [start, end) to a uniform grid."""
    m = (times >= start) & (times < end)
    t, v = times[m], values[m]
    if reject_outliers and len(v):
        # drop ectopic / missed beats: > 30 % away from the window median
        med = np.median(v)
        keep = np.abs(v - med) <= 0.3 * med
        t, v = t[keep], v[keep]
    if len(t) < 8:
        return None
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    if len(grid) < fs * 10:
        return None
    return np.interp(grid, t, v)


def _breathing_band(y: np.ndarray, fs: float) -> np.ndarray:
    y = ss.detrend(np.asarray(y, dtype=float))
    return ss.sosfiltfilt(_sos("bandpass", [RR_MIN_BPM / 60, RR_MAX_BPM / 60], fs, order=2), y)


def spectral_rr(y: np.ndarray, fs: float = RESAMPLE_HZ, nfft: int = 2048) -> tuple[float, float]:
    """Respiratory rate (breaths/min) from the spectral peak, and the peak's share of band power."""
    y = _breathing_band(y, fs) * np.hanning(len(y))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    p = np.abs(np.fft.rfft(y, nfft)) ** 2
    band = (f >= RR_MIN_BPM / 60) & (f <= RR_MAX_BPM / 60)
    total = p[band].sum()
    if total <= 0:
        return np.nan, np.nan
    i = int(np.argmax(np.where(band, p, -1.0)))
    near = band & (np.abs(f - f[i]) <= 0.05)
    return 60.0 * f[i], float(p[near].sum() / total)


def counting_rr(y: np.ndarray, fs: float = RESAMPLE_HZ) -> float:
    """Respiratory rate by counting breaths (after Schäfer & Kratky, 2008)."""
    y = _breathing_band(y, fs)
    peaks, props = ss.find_peaks(y, height=0)
    if len(peaks) < 3:
        return np.nan
    heights = props["peak_heights"]
    peaks = peaks[heights >= 0.2 * np.percentile(heights, 75)]
    keep = [peaks[0]] if len(peaks) else []
    for p in peaks[1:]:
        # a valid breath needs a trough below zero between two peaks
        if y[keep[-1]:p].min() < 0:
            keep.append(p)
    if len(keep) < 3:
        return np.nan
    return 60.0 * fs / float(np.mean(np.diff(keep)))


def smart_fusion(estimates, max_std: float = 4.0) -> float:
    """Average the modulation estimates only when they agree (Karlen et al., 2013); else NaN."""
    arr = np.asarray(estimates, dtype=float)
    if np.isnan(arr).any() or arr.std() > max_std:
        return np.nan
    return float(arr.mean())


def synthetic_ppg(duration: float = 64.0, fs: float = 125.0, hr: float = 72.0, rr: float = 15.0,
                  noise: float = 0.02, seed: int = 0) -> np.ndarray:
    """PPG-like test signal carrying all three respiratory modulations at a known rate."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs)
    resp = np.sin(2 * np.pi * rr / 60.0 * t)
    phase = 2 * np.pi * np.cumsum(hr / 60.0 * (1 + 0.05 * resp)) / fs   # RIFV
    pulse = np.sin(phase) + 0.4 * np.sin(2 * phase - 1.0)
    return (1 + 0.15 * resp) * pulse + 0.3 * resp + noise * rng.standard_normal(len(t))  # RIAV + RIIV
