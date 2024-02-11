import unittest
import numpy as np
from tatpulsar.utils import readpar

class TestReadpar(unittest.TestCase):
    def test_readpar(self):
        eph = readpar("./tests/data/Vela.par")
        par_to_test = ['F0', "F1", "F2", "PEPOCH"]
        value_to_varify = [11.183908501919459511, -1.5607051596431583974e-11, 1.2164543201957172106e-21,
                           59650]

        for par, val in zip(par_to_test, value_to_varify):
            print(par, getattr(eph, par).value, val)
            try:
                np.testing.assert_array_almost_equal(getattr(eph, par).value, val)
            except Exception as e:
                self.fail(f"{e}: The Ephemeris reading is incorrect")

    # Test some properties
    def test_properties(self):
        eph = readpar("./tests/data/Vela.par")
        freq_list = np.array([11.183908501919459511, -1.5607051596431583974e-11, 1.2164543201957172106e-21])
        pepoch = 59650

        try:
            np.testing.assert_array_almost_equal(
                    eph.frequency,
                    freq_list)
        except Exception as e:
            self.fail(f"{e}: The Ephemeris reading is incorrect")
        self.assertEqual(eph.reftime, pepoch, "Failed: pepoch could not properly obtained")
        self.assertEqual(eph.freq_order, 3)

