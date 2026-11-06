import numpy as np
import pytest

from ppgrr.dsp import (counting_rr, peak_hr, resample_window, respiratory_modulations, smart_fusion,
                       spectral_hr, spectral_rr, synthetic_ppg)
from ppgrr.evaluate import agreement


@pytest.mark.parametrize("rr", [8.0, 15.0, 24.0])
def test_each_modulation_recovers_known_rate(rr):
    fs = 125.0
    mods = respiratory_modulations(synthetic_ppg(duration=64, fs=fs, rr=rr), fs)
    for m in ("riiv", "riav", "rifv"):
        t, v = mods[m]
        y = resample_window(t, v, 0, 64)
        est, conc = spectral_rr(y)
        assert abs(est - rr) < 1.0, (m, est)
        assert 0 < conc <= 1


@pytest.mark.parametrize("hr", [50.0, 72.0, 120.0])
def test_heart_rate_estimators(hr):
    fs = 125.0
    mods = respiratory_modulations(synthetic_ppg(duration=40, fs=fs, hr=hr), fs)
    assert abs(peak_hr(mods["beat_times"], 0, 40) - hr) < 2.0
    assert abs(spectral_hr(mods["band"], fs, 0, 32) - hr) < 2.0


def test_counting_matches_rate():
    t = np.arange(0, 60, 0.25)
    assert abs(counting_rr(np.sin(2 * np.pi * 12 / 60 * t)) - 12) < 1.0


def test_smart_fusion_rejects_disagreement():
    assert smart_fusion([15, 15.5, 14.5]) == pytest.approx(15.0)
    assert np.isnan(smart_fusion([10, 15, 25]))
    assert np.isnan(smart_fusion([15, np.nan, 15]))


def test_loader_sampling_rate_ignores_rounded_timestamps(tmp_path):
    from ppgrr.data import load_recording
    n = 1001
    t = np.round(np.arange(n) / 125.0, 2)          # rounded like the BIDMC CSVs
    import pandas as pd
    pd.DataFrame({"Time [s]": t, " RESP": 0.0, " PLETH": 0.0}).to_csv(tmp_path / "bidmc_01_Signals.csv", index=False)
    pd.DataFrame({"breaths ann1 [signal sample no]": [1, 126]}).to_csv(tmp_path / "bidmc_01_Breaths.csv", index=False)
    rec = load_recording(tmp_path, "01")
    assert rec.fs == 125.0
    assert rec.breaths["ann1"][1] == pytest.approx(1.0)


def test_agreement_metrics():
    m = agreement([10, 12, 14, 16], [11, 12, np.nan, 15])
    assert m["n"] == 3 and m["coverage"] == pytest.approx(0.75)
    assert m["mae"] == pytest.approx(2 / 3)
