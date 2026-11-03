import numpy as np
import pytest

from ppgrr.dsp import (counting_rr, resample_window, respiratory_modulations, smart_fusion,
                       spectral_rr, synthetic_ppg)
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


def test_counting_matches_rate():
    t = np.arange(0, 60, 0.25)
    assert abs(counting_rr(np.sin(2 * np.pi * 12 / 60 * t)) - 12) < 1.0


def test_smart_fusion_rejects_disagreement():
    assert smart_fusion([15, 15.5, 14.5]) == pytest.approx(15.0)
    assert np.isnan(smart_fusion([10, 15, 25]))
    assert np.isnan(smart_fusion([15, np.nan, 15]))


def test_agreement_metrics():
    m = agreement([10, 12, 14, 16], [11, 12, np.nan, 15])
    assert m["n"] == 3 and m["coverage"] == pytest.approx(0.75)
    assert m["mae"] == pytest.approx(2 / 3)
