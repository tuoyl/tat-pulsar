import os
import unittest
from urllib.error import URLError

import numpy as np

from tatpulsar.pulse.Crab.retrive_eph import retrieve_ephemeris, get_par
from tatpulsar.utils.functions import met2mjd
from tatpulsar.data.profile import Profile

class TestCrabModule(unittest.TestCase):
    def test_retrieve_eph(self):
        cnt = np.random.rand(100)
        err = np.sqrt(cnt)

        filename = 'Crab.gro'
        try:
            eph = retrieve_ephemeris(write_to_file=True, ephfile=filename)
        except URLError as error:
            self.skipTest(f"Network unavailable for ephemeris download: {error}")
        self.assertTrue(os.path.exists(filename), "Failed: File was not created")
        os.remove(filename)

    def test_get_par(self):
        try:
            eph = retrieve_ephemeris(write_to_file=True, ephfile='Crab.gro')
        except URLError as error:
            self.skipTest(f"Network unavailable for ephemeris download: {error}")
        par = get_par(59000, eph)
        self.assertIsInstance(par.f0, float, 'Failed: F0 is not a float')
        self.assertIsInstance(par.f1, float, 'Failed: F1 is not a float')
        self.assertIsInstance(par.f2, float, 'Failed: F2 is not a float')

if __name__ == "__main__":
    unittest.main()
