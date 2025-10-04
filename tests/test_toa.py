import numpy as np
import pytest

from tatpulsar.data.profile import Profile
from tatpulsar.pulse import toa as toa_module


def test_cal_toa_max_basic():
    counts = np.array([1.0, 3.0, 5.0, 2.0])
    profile = Profile(counts)
    profile.ref_time = 100.0
    nsteps = 8
    rng_seed = 42

    rng_call = np.random.default_rng(rng_seed)
    rng_expected = np.random.default_rng(rng_seed)

    toa, err = toa_module.cal_toa(2.0, profile, method="max", nsteps=nsteps, rng=rng_call)

    phase = profile.phase
    delta_phi = phase[np.argmax(counts)]

    samples = toa_module._draw_poisson(counts, nsteps, rng_expected)
    sample_indices = np.argmax(samples, axis=1)
    delta_phis = phase[sample_indices].astype(float)
    if (delta_phi < 0.25) or (delta_phi > 0.75):
        delta_phis = np.where(delta_phis < 0.5, delta_phis + 1.0, delta_phis)

    expected_toa = profile.ref_time + delta_phi / 2.0
    expected_err = np.std(delta_phis) / 2.0

    assert toa == pytest.approx(expected_toa)
    assert err == pytest.approx(expected_err)


def test_cal_toa_requires_profile():
    with pytest.raises(TypeError):
        toa_module.cal_toa(1.0, [1, 2, 3])


def test_cal_toa_phi_range_validation():
    profile = Profile(np.ones(4))
    profile.ref_time = 0.0

    with pytest.raises(ValueError):
        toa_module.cal_toa(1.0, profile, phi_range=(0.9, 0.2))

    with pytest.raises(ValueError):
        toa_module.cal_toa(1.0, profile, phi_range=(1.5, 1.6))


def test_cal_toa_ref_time_required():
    profile = Profile(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(IOError):
        toa_module.cal_toa(1.0, profile)


def test_cal_toa_ccf(monkeypatch):
    profile = Profile(np.array([1.0, 2.0, 3.0, 4.0]))
    profile.ref_time = 10.0
    std_profile = Profile(np.array([4.0, 1.0, 0.0, 2.0]))

    def fake_ccf(data, template):
        return np.array([]), 1

    monkeypatch.setattr(toa_module, "ccf", fake_ccf)

    toa, err = toa_module.cal_toa(
        fbest=2.0,
        profile=profile,
        method="ccf",
        std_pro=std_profile,
        nsteps=4,
        rng=np.random.default_rng(0),
    )

    expected_phase = np.mod(std_profile.phase[np.argmax(np.roll(std_profile.counts, 1))], 1.0)
    expected_toa = profile.ref_time + expected_phase / 2.0

    assert toa == pytest.approx(expected_toa)
    assert err == pytest.approx(0.0)


def test_cal_toa_requires_std_profile_for_ccf():
    profile = Profile(np.array([1.0, 2.0, 3.0]))
    profile.ref_time = 0.0
    with pytest.raises(TypeError):
        toa_module.cal_toa(1.0, profile, method="ccf")


def test_cal_toa_invalid_method():
    profile = Profile(np.array([1.0, 2.0, 3.0]))
    profile.ref_time = 0.0
    with pytest.raises(ValueError):
        toa_module.cal_toa(1.0, profile, method="unknown")

