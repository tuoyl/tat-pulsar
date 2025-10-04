from tatpulsar.data.profile import Profile, phihist

import io
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import unittest
import matplotlib.pyplot as plt
import matplotlib as mpl
import pytest
#mpl.rcParams['figure.figsize'] = (15,10)
# sns.set_context('talk')
# sns.set_style("whitegrid")
# sns.set_palette("colorblind")

# Test Profile cycle function

class TestCycles(unittest.TestCase):

    def test_init(self):
        cnt = np.random.rand(100)
        err = np.sqrt(cnt)

        pro = Profile(cnt, cycles=1)
        np.testing.assert_array_equal(cnt, pro.counts)
        np.testing.assert_array_equal(err, pro.error)

        pro = Profile(cnt, cycles=2)
        self.assertEqual(cnt.size*2, pro.counts.size)
        self.assertEqual(err.size*2, pro.error.size)

        pro = Profile(cnt, error=np.sqrt(cnt), cycles=2)
        self.assertEqual(cnt.size*2, pro.counts.size)
        self.assertEqual(err.size*2, pro.error.size)

        self.assertEqual(cnt.size, pro.size)

    def test_raises(self):
        cnt = np.random.rand(100)
        with self.assertRaises(IOError):
            pro = Profile(cnt, cycles=3)
        with self.assertRaises(TypeError):
            pro = Profile(cnt, cycles=1.5)

        pro = Profile(cnt, cycles=1)
        with self.assertRaises(IOError):
            pro.cycles=3.5

    def test_operator(self):
        cnt = np.random.rand(100)
        pro1 = Profile(cnt, cycles=1)
        pro2 = Profile(cnt, cycles=1)
        pro = pro1 + pro2
        np.testing.assert_array_equal(pro.counts, pro1.counts + pro2.counts)
        np.testing.assert_array_equal(pro.error, np.sqrt(pro1.error**2 + pro2.error**2))
        pro = pro1 - pro2
        np.testing.assert_array_equal(pro.counts, pro1.counts - pro2.counts)
        np.testing.assert_array_equal(pro.error, np.sqrt(pro1.error**2 + pro2.error**2))

        with self.assertRaises(IOError):
            _ = pro1 + 1.0
        with self.assertRaises(IOError):
            _ = pro1 - 1.0
        pro_cycle2 = Profile(cnt, cycles=2)
        with self.assertRaises(ValueError):
            _ = pro1 + pro_cycle2
        with self.assertRaises(ValueError):
            _ = pro_cycle2 - pro1


    def test_setter(self):
        cnt = np.random.rand(100)
        pro = Profile(cnt, cycles=1)
        pro_phase_cycle1 = pro.phase
        pro_counts_cycle1 = pro.counts
        pro_error_cycle1 = pro.error

        # set cycle
        pro.cycles = 2
        pro_phase_cycle2  = pro.phase
        pro_counts_cycle2 = pro.counts
        pro_error_cycle2  = pro.error

        np.testing.assert_array_almost_equal(pro_counts_cycle2,
                                      np.tile(pro_counts_cycle1, reps=2))
        np.testing.assert_array_almost_equal(pro_phase_cycle2,
                                      np.append(pro_phase_cycle1, pro_phase_cycle1+1))

        # set cycle
        pro.cycles = 1
        pro_phase_cycle1  = pro.phase
        pro_counts_cycle1 = pro.counts
        pro_error_cycle1  = pro.error

        np.testing.assert_array_almost_equal(pro_counts_cycle2,
                                      np.tile(pro_counts_cycle1, reps=2))
        np.testing.assert_array_almost_equal(pro_phase_cycle2,
                                      np.append(pro_phase_cycle1, pro_phase_cycle1+1))

        buf = io.StringIO()
        with redirect_stdout(buf):
            pro.cycles = 1
        self.assertIn("Cycle is already 1", buf.getvalue())

    @pytest.mark.filterwarnings("ignore:divide by zero encountered")
    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_norm(self):
        from copy import deepcopy
        from unittest import mock
        dir = "./tests/data/Crab_profile.dat"
        cnt = np.loadtxt(dir)
        pro = Profile(cnt, cycles=1)
        pro_norm0 = deepcopy(pro)
        pro_norm1 = deepcopy(pro)

        with mock.patch("matplotlib.pyplot.savefig"), mock.patch("matplotlib.pyplot.figure"), mock.patch("matplotlib.pyplot.close"):
            plt.figure()
            plt.subplot(3,1,1)
            plt.errorbar(pro.phase, pro.counts, pro.error, label='norm method=0')

            with np.errstate(divide="ignore", invalid="ignore"):
                plt.subplot(3,1,2)
                pro_norm0.norm(method=0)
                plt.errorbar(pro_norm0.phase, pro_norm0.counts, pro_norm0.error, label='norm method=0')

                plt.subplot(3,1,3)
                pro_norm1.norm(method=1)
                plt.errorbar(pro_norm1.phase, pro_norm1.counts, pro_norm1.error, label='norm method=1')
            plt.legend()
            plt.savefig("ignored.png")
            plt.clf()

            bkg_range = [0.6, 0.8]
            plt.figure()
            with np.errstate(divide="ignore", invalid="ignore"):
                pro_norm0.norm(method=0, bkg_range=bkg_range)
            plt.errorbar(pro_norm0.phase, pro_norm0.counts, pro_norm0.error, label='norm method=0')
            plt.axvline(bkg_range[0])
            plt.axvline(bkg_range[1])
            plt.axhline(y=0, ls='--')
            plt.savefig("ignored.png")
            plt.close("all")

    def test_hist(self):
        phi = np.random.rand(1000)
        pro = phihist(phi, nbins=20)
        self.assertEqual(pro.size, 20)

    def test_rebin_nbins(self):
        cnt = np.random.rand(100)
        pro = Profile(cnt, cycles=1)

        nbins = 5
        pro.rebin(nbins=nbins)
        self.assertEqual(pro.counts.size, nbins, "Failed: counts size mismatch")
        self.assertEqual(pro.error.size,  nbins, "Failed: error size mismatch")
        self.assertEqual(pro.phase.size,  nbins, "Failed: phase size mismatch")

    def test_rebin_factor(self):
        cnt = np.random.rand(100)
        pro = Profile(cnt, cycles=1)

        factor = 2
        pro.rebin(factor=factor)
        self.assertEqual(pro.counts.size, cnt.size//factor, "Failed: counts size mismatch")
        self.assertEqual(pro.error.size,  cnt.size//factor, "Failed: error size mismatch")
        self.assertEqual(pro.phase.size,  cnt.size//factor, "Failed: phase size mismatch")

        # ---
        cnt = np.random.rand(100)
        pro = Profile(cnt, cycles=1)

        new_pro = pro.rebin(factor=factor, return_profile=True)
        self.assertEqual(new_pro.counts.size, cnt.size//factor, "Failed: counts size mismatch")
        self.assertEqual(new_pro.error.size,  cnt.size//factor, "Failed: error size mismatch")
        self.assertEqual(new_pro.phase.size,  cnt.size//factor, "Failed: phase size mismatch")
        self.assertEqual(new_pro.counts.size, pro.size//factor, "Failed: counts size mismatch")
        self.assertEqual(new_pro.error.size,  pro.size//factor, "Failed: error size mismatch")
        self.assertEqual(new_pro.phase.size,  pro.size//factor, "Failed: phase size mismatch")

    def test_draw_random_pulse(self):
        from tatpulsar.data.profile import draw_random_pulse
        np.random.seed(19930727)
        pro = draw_random_pulse(nbins=100, baseline=1000, pulsefrac=0.2)
        self.assertTrue(isinstance(pro, Profile), "The output of pulse drawing function is not Profile object")
        self.assertIsInstance(pro.significance, float, "Failed: significance calculated is not a float")
        self.assertIsInstance(pro.chisq, float, "Failed: chisquare of profile calculated is not a float")
        self.assertEqual(pro.dof, pro.size - 1, "Failed: d.o.f. of profile calculated is not (binsize - 1)")

    def test_chisq_two_cycles_branch(self):
        counts = np.array([5.0, 7.0, 9.0, 11.0])
        pro = Profile(counts, cycles=2)
        chisq_value = pro.chisq
        self.assertGreater(chisq_value, 0)

    def test_pulsefrac_and_rms(self):
        counts = np.array([10.0, 30.0, 20.0, 40.0])
        pro = Profile(counts)
        pf, pf_err = pro.pulsefrac
        self.assertGreater(pf, 0)
        self.assertGreaterEqual(pf_err, 0)
        self.assertGreater(pro.rms, 0)

    def test_sampling_helpers(self):
        counts = np.array([10.0, 20.0, 15.0, 25.0])
        pro = Profile(counts)

        with mock.patch("tatpulsar.data.profile.poisson_rejection_sampling", return_value=np.array([0.1, 0.2])) as mock_poisson:
            phases = pro.sampling_phase(2)
        np.testing.assert_allclose(phases, np.array([0.1, 0.2]))
        args, kwargs = mock_poisson.call_args
        np.testing.assert_allclose(args[0], pro.phase)
        np.testing.assert_allclose(args[1], pro.counts)
        self.assertEqual(args[2], 2)

        with mock.patch("tatpulsar.data.profile.poisson_rejection_sampling", return_value=np.array([0.3, 0.4])) as mock_poisson, \
             mock.patch("tatpulsar.data.profile.draw_event_from_phase", return_value=np.array([42.0, 43.0])) as mock_draw:
            events = pro.sampling_event(2, 0.0, 1.0, f0=10.0, f1=0.1)
        np.testing.assert_allclose(events, np.array([42.0, 43.0]))
        mock_poisson.assert_called_once()
        draw_args, draw_kwargs = mock_draw.call_args
        np.testing.assert_allclose(draw_args[0], np.array([0.3, 0.4]))
        self.assertEqual(draw_args[1:], (0.0, 1.0))
        self.assertEqual(draw_kwargs, dict(f0=10.0, f1=0.1, f2=0, f3=0, pepoch=0))

    def test_resample_variants(self):
        counts = np.array([12, 18, 24, 30])
        pro = Profile(counts)
        resampled = pro.resample(sample_num=2)
        self.assertEqual(resampled.shape, (2, pro.size))
        np.random.seed(123)
        gaussian_resampled = pro.resample(sample_num=3, kind="gaussian")
        self.assertEqual(gaussian_resampled.shape, (3, pro.size))
        self.assertTrue(np.all(gaussian_resampled >= 0))
        with self.assertRaises(IOError):
            pro.resample(sample_num=0)
        with self.assertRaises(ValueError):
            pro.resample(kind="invalid")

    @pytest.mark.filterwarnings("ignore:divide by zero encountered")
    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_norm_return_profile_and_background(self):
        counts = np.array([50.0, 80.0, 60.0, 90.0])
        pro = Profile(counts)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = pro.norm(method=0, return_profile=True)
        self.assertIsInstance(normalized, Profile)

        bkg_range = [0.0, pro.phase[1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_bkg = pro.norm(method=0, bkg_range=bkg_range, return_profile=True)
        self.assertIsInstance(normalized_bkg, Profile)

        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_mean = pro.norm(method=1, return_profile=True)
        self.assertIsInstance(normalized_mean, Profile)

    def test_rebin_requires_argument(self):
        counts = np.array([10.0, 20.0, 30.0, 40.0])
        pro = Profile(counts)
        with self.assertRaises(IOError):
            pro.rebin()

    def test_rebin_with_nbins_and_factor(self):
        counts = np.arange(1.0, 9.0)
        pro = Profile(counts)
        pro.rebin(nbins=4, factor=2)
        self.assertEqual(pro.counts.size, 4)


if __name__ == "__main__":
    unittest.main()
