import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

search_module = importlib.import_module("tatpulsar.pulse.search")


class DummyProfile:
    def __init__(self, counts):
        self.counts = np.asarray(counts)


def test_parameters_legal_accepts_known_keys():
    search_module._parameters_legal({"check_par": True, "f0step": 0.1, "nbins": 16})


def test_parameters_legal_rejects_unknown_key():
    with pytest.raises(IOError):
        search_module._parameters_legal({"check_par": True, "unsupported": 1})


def test_get_parameters_with_frequency_and_derivative_grid():
    kwargs = {
        "pepoch": 0.0,
        "f0": 10.0,
        "f0_step": 0.25,
        "f0_nstep": 2,
        "f1": 0.0,
        "f1_step": 0.5,
        "f1_nstep": 1,
        "f2": 0.01,
        "f3": 0.0,
        "f4": 0.0,
    }
    pepoch, F0, F1, F2, F3, F4, flag = search_module._get_parameters(kwargs)

    np.testing.assert_allclose(F0, np.arange(9.5, 10.5, 0.25))
    np.testing.assert_allclose(F1, np.arange(-0.5, 0.5, 0.5))
    assert pepoch == 0.0
    assert F2 == 0.01
    assert F3 == 0.0
    assert F4 == 0.0
    assert flag is True


def test_get_parameters_with_zero_f1_step():
    kwargs = {
        "pepoch": 1.0,
        "f0": 15.0,
        "f0_step": 0.2,
        "f0_nstep": 1,
        "f1": 0.1,
        "f1_step": 0.0,
        "f1_nstep": 10,
    }
    pepoch, F0, F1, F2, F3, F4, flag = search_module._get_parameters(kwargs)
    assert pepoch == 1.0
    np.testing.assert_allclose(F0, np.arange(14.8, 15.2, 0.2))
    assert F1 == 0.1
    assert F2 == 0
    assert F3 == 0
    assert F4 == 0
    assert flag is False


def test_get_parameters_without_f1_step_information():
    kwargs = {
        "pepoch": 2.0,
        "f0": 12.0,
        "f0_step": 0.5,
        "f0_nstep": 1,
        "f1": -0.05,
    }
    pepoch, F0, F1, F2, F3, F4, flag = search_module._get_parameters(kwargs)
    assert pepoch == 2.0
    np.testing.assert_allclose(F0, np.array([11.5, 12.0]))
    assert F1 == -0.05
    assert F2 == 0
    assert F3 == 0
    assert F4 == 0
    assert flag is False


def test_get_parameters_from_parfile(monkeypatch):
    class DummyParam:
        def __init__(self, value):
            self.value = value

    class DummyTimingModel:
        def __init__(self, path):
            self.F0 = DummyParam(11.0)
            self.F1 = DummyParam(0.1)
            self.F2 = DummyParam(1e-3)
            self.F3 = DummyParam(0.0)
            self.F4 = DummyParam(0.0)
            self.reftime = 123.456

    fake_module = SimpleNamespace(TimingModel=DummyTimingModel)
    monkeypatch.setitem(sys.modules, "tatpulsar.utils.timingmodel", fake_module)

    pepoch, F0, F1, F2, F3, F4, flag = search_module._get_parameters({"parfile": "ignored"})
    assert pepoch == 123.456
    assert F0 == 11.0
    assert F1 == 0.1
    assert F2 == 1e-3
    assert F3 == 0.0
    assert F4 == 0.0
    assert flag is False


def test_search_raises_for_empty_data():
    with pytest.raises(IOError):
        search_module.search(np.array([]), pepoch=0.0, f0=10.0, f0_step=0.1, f0_nstep=1)


def test_search_1d_frequency_scan(monkeypatch):
    data = np.linspace(0.0, 1.0, 5)
    kwargs = {"pepoch": 0.0, "f0": 10.0, "f0_step": 0.5, "f0_nstep": 2, "nbins": 8}

    expected_f0 = np.arange(9.0, 11.0, 0.5)
    fake_chi_square = np.array([0.0, 1.0, 3.0, 7.0])

    def fake_cal_chisquare(data_arg, F0, t0, nbins, F1, F2, F3, F4):
        np.testing.assert_allclose(data_arg, data.astype(np.float64))
        np.testing.assert_allclose(F0, expected_f0)
        assert t0 == 0.0
        assert nbins == 8
        assert F1 == 0
        assert F2 == 0
        assert F3 == 0
        assert F4 == 0
        return fake_chi_square

    captured_phi = {}

    def fake_phihist(phi, nbins):
        captured_phi["phi"] = phi
        captured_phi["nbins"] = nbins
        return DummyProfile(np.arange(nbins))

    monkeypatch.setattr(search_module, "cal_chisquare", fake_cal_chisquare)
    monkeypatch.setattr(search_module, "phihist", fake_phihist)

    result = search_module.search(data, **kwargs)

    assert np.all(result.freq == expected_f0)
    assert np.all(result.chisquare == fake_chi_square)
    assert result.freqderiv == 0
    assert result.profile.size == kwargs["nbins"]
    np.testing.assert_allclose(captured_phi["phi"], np.mod((data - 0.0) * expected_f0[3], 1))
    assert captured_phi["nbins"] == kwargs["nbins"]


def test_search_2d_frequency_and_derivative_scan(monkeypatch):
    data = np.linspace(0.0, 2.0, 6)
    kwargs = {
        "pepoch": 0.0,
        "f0": 20.0,
        "f0_step": 1.0,
        "f0_nstep": 2,
        "f1": 0.0,
        "f1_step": 0.5,
        "f1_nstep": 1,
        "nbins": 4,
        "f2": 0.01,
    }

    expected_f0 = np.arange(18.0, 22.0, 1.0)
    expected_f1 = np.arange(-0.5, 0.5, 0.5)
    fake_surface = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    def fake_cal_2dchisquare(data_arg, F0, F1, t0, nbins, F2=0, F3=0, F4=0):
        np.testing.assert_allclose(data_arg, data.astype(np.float64))
        np.testing.assert_allclose(F0, expected_f0)
        np.testing.assert_allclose(F1, expected_f1)
        assert t0 == 0.0
        assert nbins == 4
        assert F2 == kwargs["f2"]
        assert F3 == 0
        assert F4 == 0
        return fake_surface

    def fake_phihist(phi, nbins):
        return DummyProfile(np.arange(nbins) + 1)

    monkeypatch.setattr(search_module, "cal_2dchisquare", fake_cal_2dchisquare)
    monkeypatch.setattr(search_module, "phihist", fake_phihist)

    result = search_module.search(data, **kwargs)

    assert np.all(result.chisquare == fake_surface)
    np.testing.assert_allclose(result.freq, expected_f0)
    np.testing.assert_allclose(result.freqderiv, expected_f1)
    assert result.f2 == kwargs["f2"]
    assert result.f3 == 0
    assert result.f4 == 0
    assert result.profile.size == kwargs["nbins"]


def test_search_uses_default_nbins(monkeypatch):
    data = np.array([0.0, 0.5])
    kwargs = {"pepoch": 0.0, "f0": 5.0, "f0_step": 0.5, "f0_nstep": 1}

    def fake_cal_chisquare(data_arg, F0, t0, nbins, *args):
        assert nbins == 20
        return np.array([1.0, 2.0])

    def fake_phihist(phi, nbins):
        assert nbins == 20
        return DummyProfile(np.ones(nbins))

    monkeypatch.setattr(search_module, "cal_chisquare", fake_cal_chisquare)
    monkeypatch.setattr(search_module, "phihist", fake_phihist)

    result = search_module.search(data, **kwargs)
    assert result.profile.size == 20
