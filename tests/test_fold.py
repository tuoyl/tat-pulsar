import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from tatpulsar.data.profile import Profile

fold_module = importlib.import_module("tatpulsar.pulse.fold")


def test_cal_phase_met_and_mjd():
    time = np.array([0.0, 1.0, 2.0])
    pepoch = 0.0
    f0 = 0.25

    met_phase = fold_module.cal_phase(time, pepoch, f0, to_1=True)
    expected_met = np.mod(f0 * time, 1.0)
    np.testing.assert_allclose(met_phase, expected_met)

    mjd_time = time / 86400.0
    mjd_pepoch = pepoch / 86400.0
    mjd_phase = fold_module.cal_phase(mjd_time, mjd_pepoch, f0, format="mjd", to_1=False)
    np.testing.assert_allclose(mjd_phase, f0 * time)


def test_phase_exposure_wraps_across_pulses():
    gti = np.array([[0.1, 2.6]])
    expo = fold_module.phase_exposure(gti, nbins=8, f0=1.0)
    assert expo.shape == (8,)
    assert expo.max() == pytest.approx(1.0)
    assert np.all(expo >= 0)


def test_get_phase_index_handles_edge_and_midpoint():
    edges = np.linspace(0, 1, 5)
    assert fold_module._get_phase_index(0.25, edges) == 1
    assert fold_module._get_phase_index(0.3, edges) == 1
    assert fold_module._get_phase_index(0.99, edges) == 3


def test_fold_lightcurve_basic_and_with_errors():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    counts = np.array([10.0, 20.0, 30.0, 40.0])
    counts_err = np.array([1.0, 2.0, 3.0, 4.0])

    profile = fold_module.fold_lightcurve(time, counts, pepoch=0.0, f0=0.25, nbins=2, dt=1.0)
    np.testing.assert_allclose(profile.counts, np.array([15.0, 35.0]))
    assert profile.ref_time == 0.0

    profile_with_err = fold_module.fold_lightcurve(
        time,
        counts,
        pepoch=0.0,
        f0=0.25,
        nbins=2,
        dt=1.0,
        counts_err=counts_err,
    )
    np.testing.assert_allclose(profile_with_err.counts, np.array([15.0, 35.0]))
    np.testing.assert_allclose(profile_with_err.error, np.array([1.11803399, 2.5]))

    profile_mjd = fold_module.fold_lightcurve(
        time,
        counts,
        pepoch=0.0,
        f0=0.25,
        nbins=2,
        dt=1.0,
        format="mjd",
        telescope="fermi",
    )
    assert profile_mjd.counts.shape == (2,)

    with pytest.raises(IOError):
        fold_module.fold_lightcurve(time, counts, pepoch=0.0, f0=0.25, nbins=2, dt=1.0, format="mjd")


def test_fold_requires_parameters_and_handles_empty_time():
    time = np.array([0.0, 1.0])
    with pytest.raises(IOError):
        fold_module.fold(time)
    with pytest.raises(IOError):
        fold_module.fold(np.array([]), pepoch=0.0, f0=1.0)


def test_fold_with_gti_and_use_data_gti(monkeypatch):
    time = np.array([0.0, 1.0, 2.0, 3.0])
    pepoch = 0.0
    f0 = 0.25

    def fake_phihist(phi, nbins):
        return Profile(counts=np.array([10.0, 20.0]), error=np.array([4.0, 9.0]))

    def fake_phase_exposure(gti, nbins, *args, **kwargs):
        return np.array([2.0, 1.0])

    captured = {}

    def fake_cal_event_gti(times, tgap):
        captured["times"] = times
        captured["tgap"] = tgap
        return np.array([[0.0, 4.0]])

    monkeypatch.setattr(fold_module, "phihist", fake_phihist)
    monkeypatch.setattr(fold_module, "phase_exposure", fake_phase_exposure)
    monkeypatch.setattr(fold_module, "cal_event_gti", fake_cal_event_gti)

    profile = fold_module.fold(
        time,
        pepoch=pepoch,
        f0=f0,
        nbins=2,
        use_data_gti=True,
    )

    np.testing.assert_allclose(profile.counts, np.array([5.0, 20.0]))
    np.testing.assert_allclose(profile.error, np.array([2.0, 9.0]))
    assert profile.ref_time == pytest.approx(pepoch)
    np.testing.assert_allclose(captured["times"], time)
    assert captured["tgap"] == 1


def test_fold2d_basic_and_with_gti(monkeypatch):
    time = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.1, 0.2, 0.8, 0.9])

    baseline_profiles = fold_module.fold2d(time, y, nseg=2, pepoch=0.0, f0=0.25, nbins=2)
    assert len(baseline_profiles) == 2
    for prof in baseline_profiles:
        assert isinstance(prof, Profile)
    baseline_counts = [prof.counts.copy() for prof in baseline_profiles]

    def fake_phase_exposure(gti, nbins, *args, **kwargs):
        return np.array([1.0, 0.5])

    monkeypatch.setattr(fold_module, "phase_exposure", fake_phase_exposure)

    gti_profiles = fold_module.fold2d(
        time,
        y,
        nseg=2,
        pepoch=0.0,
        f0=0.25,
        nbins=2,
        gti=np.array([[0.0, 4.0]]),
    )
    assert len(gti_profiles) == 2
    for base, prof in zip(baseline_counts, gti_profiles):
        np.testing.assert_allclose(prof.counts, base / np.array([1.0, 0.5]))

    with pytest.raises(IOError):
        fold_module.fold2d(np.array([]), y, nseg=2, pepoch=0.0, f0=0.25)


def test_align_profile_and_merge(monkeypatch):
    template = np.array([0.0, 1.0, 0.0])
    profile_list = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])]

    def fake_ccf(ref, prof):
        if np.array_equal(prof, profile_list[0]):
            return np.array([]), 2
        return np.array([]), 1

    monkeypatch.setattr(fold_module, "ccf", fake_ccf)
    monkeypatch.setattr(fold_module, "tqdm", lambda iterable, **_: iterable)

    aligned = fold_module.align_profile(profile_list, template)
    expected_first = np.roll(profile_list[0], 2)
    expected_second = np.roll(profile_list[1], 1)
    np.testing.assert_allclose(aligned[0], expected_first)
    np.testing.assert_allclose(aligned[1], expected_second)

    merged = fold_module.merge_aligned_profile(profile_list, template)
    np.testing.assert_allclose(merged, expected_first + expected_second)

    with pytest.raises(TypeError):
        fold_module.align_profile(np.array(profile_list), template)
    with pytest.raises(TypeError):
        fold_module.align_profile(profile_list, template.tolist())


def test_fold_lightcurve_counts_err_branch(monkeypatch):
    time = np.array([0.0, 1.0])
    counts = np.array([5.0, 7.0])
    counts_err = np.array([0.1, 0.2])

    profile = fold_module.fold_lightcurve(
        time,
        counts,
        pepoch=0.0,
        f0=0.5,
        nbins=2,
        dt=1.0,
        counts_err=counts_err,
    )
    assert profile.error.shape == (2,)
    assert np.all(profile.error >= 0)
