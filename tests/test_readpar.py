import unittest
import numpy as np
from tatpulsar.utils import TimingModel as tmodel

class TestReadpar(unittest.TestCase):
    def test_readpar(self):
        eph = tmodel("./tests/data/Vela.par")
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
        eph = tmodel("./tests/data/Vela.par")
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

    def test_update_feature(self):
        eph = tmodel("./tests/data/Vela.par")
        freq_list = np.array([11.183908501919459511, -1.5607051596431583974e-11, 1.2164543201957172106e-21])
        pepoch = 59650
        new_pepoch = pepoch + 1
        eph.update(new_pepoch)
        new_freq_list = np.array([11.183907153474742, -1.560694649477832e-11, 1.2164543201957172106e-21])
        try:
            np.testing.assert_array_almost_equal(
                    eph.frequency,
                    new_freq_list)
        except Exception as e:
            self.fail(f"{e}: The Ephemeris update is incorrect")
        self.assertEqual(eph.reftime, new_pepoch, "Failed: pepoch not properly updated")


#from tatpulsar.utils import TimingModel as tmodel
#eph = tmodel(".//data/Vela.par")
#freq_list = np.array([11.183908501919459511, -1.5607051596431583974e-11, 1.2164543201957172106e-21])
#pepoch = 59650
#new_pepoch = pepoch + 1
#eph.update(new_pepoch)
#new_freq_list = np.array([11.183907153474742, -1.560694649477832e-11, 1.2164543201957172106e-21])
#print("OLD = ", freq_list)
#print("NEW=", new_freq_list)
#print(eph.frequency)
