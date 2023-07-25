import unittest
import numpy as np
import os

from tatpulsar.pulse.Crab.retrive_eph import retrieve_ephemeris, get_par
from tatpulsar.utils.functions import met2mjd
from tatpulsar.data.profile import Profile

class TestCrabModule(unittest.TestCase):
    def test_retrieve_eph(self):
        cnt = np.random.rand(100)
        err = np.sqrt(cnt)

        filename = 'Crab.gro'
        eph = retrieve_ephemeris(write_to_file=True, ephfile=filename)
        self.assertTrue(os.path.exists(filename), "Failed: File was not created")
        if os.path.exists(filename):
            os.remove(filename)

    def test_get_par(self):
        eph = retrieve_ephemeris(write_to_file=True, ephfile='Crab.gro')
        par = get_par(59000, eph)
        self.assertIsInstance(par.f0, float, 'Failed: F0 is not a float')
        self.assertIsInstance(par.f1, float, 'Failed: F1 is not a float')
        self.assertIsInstance(par.f2, float, 'Failed: F2 is not a float')

if __name__ == "__main__":
    unittest.main()
