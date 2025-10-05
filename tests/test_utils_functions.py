import numpy as np
import pytest

import tatpulsar.utils.functions as fn


@pytest.mark.parametrize("telescope", [
    "fermi",
    "hxmt",
    "nicer",
    "gecam",
    "nustar",
    "ixpe",
    "swift",
    "rxte",
    "catch",
])
def test_met_mjd_round_trip(telescope):
    met = 123456.789
    mjd = fn.met2mjd(met, telescope=telescope)
    recon = fn.mjd2met(mjd, telescope=telescope)
    assert recon == pytest.approx(met)


def test_cal_chisquare_prefers_true_frequency():
    rng = np.random.default_rng(42)
    true_freq = 2.0
    time = np.sort(rng.uniform(0, 10, size=200))

    freqs = np.linspace(true_freq - 0.1, true_freq + 0.1, 5)
    chi = fn.cal_chisquare(time, freqs, pepoch=0.0, nbins=16)

    best_idx = int(np.argmin(chi))
    assert np.isfinite(chi).all()
    assert freqs[best_idx] == pytest.approx(true_freq, rel=0.1)


def test_cal_2dchisquare_returns_expected_shape():
    time = np.linspace(0, 5, num=150)
    freqs = np.linspace(1.0, 1.2, 3)
    fdots = np.linspace(-0.02, 0.02, 4)

    chi_surface = fn.cal_2dchisquare(time, freqs, fdots, pepoch=0.0, nbins=10)
    assert chi_surface.shape == (fdots.size, freqs.size)
    assert np.isfinite(chi_surface).all()


def test_ccf_identifies_known_shift():
    base = np.array([0.0, 1.0, 0.5, 0.2])
    shifted = np.roll(base, 2)
    values, delay = fn.ccf(base, shifted)
    assert delay == 2
    assert values[delay] == pytest.approx(values.max())


def test_cal_event_gti_splits_on_gap():
    times = np.array([0.0, 0.5, 1.0, 10.0, 10.5, 11.0])
    gtis = fn.cal_event_gti(times, tgap=2.0)
    expected = np.array([[0.0, 1.0], [10.0, 11.0]])
    np.testing.assert_allclose(gtis, expected)

    with pytest.raises(TypeError):
        fn.cal_event_gti([0.0, 1.0, 2.0])
