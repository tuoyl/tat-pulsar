from tatpulsar.data.profile import Profile, phihist
from tatpulsar.data.profile import draw_random_pulse

import numpy as np
import unittest


class TestSim(unittest.TestCase):

    def test_draw_event(self):
        np.random.seed(19930727)
        from tatpulsar.simulation.profile_sim import poisson_rejection_sampling
        pro = draw_random_pulse(nbins=128, baseline=100, pulsefrac=0.3)
        sample_y = poisson_rejection_sampling(pro.phase, pro.counts, nphot=100)

        from tatpulsar.simulation.profile_sim import draw_event_from_phase
        f0 = 29.6366215666433
        f1 = -3.69981e-10
        f2 = -8.1e-18
        pepoch = 58068
        event = draw_event_from_phase(sample_y, tstart=58000,
                                      tstop=59000,
                                      f0=f0,
                                      f1=f1,
                                      pepoch=pepoch)
