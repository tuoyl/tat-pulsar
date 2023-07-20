from tatpulsar.data.profile import Profile, phihist

import numpy as np
import unittest
import matplotlib.pyplot as plt
import matplotlib as mpl
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

    def test_norm(self):
        from copy import deepcopy
        dir = "./tests/data/Crab_profile.dat"
        cnt = np.loadtxt(dir)
        pro = Profile(cnt, cycles=1)
        pro_norm0 = deepcopy(pro)
        pro_norm1 = deepcopy(pro)

        plt.figure()
        plt.subplot(3,1,1)
        plt.errorbar(pro.phase, pro.counts, pro.error, label='norm method=0')

        plt.subplot(3,1,2)
        pro_norm0.norm(method=0)
        plt.errorbar(pro_norm0.phase, pro_norm0.counts, pro_norm0.error, label='norm method=0')

        plt.subplot(3,1,3)
        pro_norm1.norm(method=1)
        plt.errorbar(pro_norm1.phase, pro_norm1.counts, pro_norm1.error, label='norm method=1')
        plt.legend()
        plt.savefig("test_profile_norm.png")
        plt.clf()

        bkg_range = [0.6, 0.8]
        plt.figure()
        pro_norm0.norm(method=0, bkg_range=bkg_range)
        plt.errorbar(pro_norm0.phase, pro_norm0.counts, pro_norm0.error, label='norm method=0')
        plt.axvline(bkg_range[0])
        plt.axvline(bkg_range[1])
        plt.axhline(y=0, ls='--')
        plt.savefig("test_profile_norm1.png")

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

if __name__ == "__main__":
    unittest.main()
