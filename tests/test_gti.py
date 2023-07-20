import unittest
import numpy as np
from tatpulsar.utils.gti import gti_intersection, gti_union

class TestGtiMethods(unittest.TestCase):
    def test_gti_intersection(self):
        gti1 = [[1, 3], [4, 6]]
        gti2 = [[2, 5]]
        result = gti_intersection(gti1, gti2)
        expected_result = [[2, 3], [4, 5]]
        self.assertEqual(result, expected_result, "Failed: gti_intersection is not working as expected")

        # Test if the GTIs are array
        gti1 = np.array([[1, 3], [4, 6]])
        gti2 = np.array([[2, 5]])
        result = gti_intersection(gti1, gti2)
        expected_result = [[2, 3], [4, 5]]
        self.assertEqual(result, expected_result, "Failed: gti_intersection is not working as expected")

        # Test if the GTIs is 1d list
        gti1 = [1, 3]
        gti2 = [2, 5]
        result = gti_intersection(gti1, gti2)
        expected_result = [[2, 3]]
        self.assertEqual(result, expected_result, "Failed: gti_intersection is not working as expected")

    def test_gti_union(self):
        gti1 = [[1, 3], [4, 6]]
        gti2 = [[2, 5]]
        result = gti_union(gti1, gti2)
        expected_result = [[1, 6]]
        self.assertEqual(result, expected_result, "Failed: gti_union is not working as expected")

        # Test if the GTIs are array
        gti1 = np.array([[1, 3], [4, 6]])
        gti2 = np.array([[2, 5]])
        result = gti_union(gti1, gti2)
        expected_result = [[1, 6]]
        self.assertEqual(result, expected_result, "Failed: gti_union is not working as expected")

        # Test if the GTIs is 1d list
        gti1 = [1, 3]
        gti2 = [2, 5]
        result = gti_union(gti1, gti2)
        expected_result = [[1, 5]]
        self.assertEqual(result, expected_result, "Failed: gti_intersection is not working as expected")

if __name__ == '__main__':
    unittest.main()

