import numpy as np

from tatpulsar.utils import gti as gti_utils


def test_gti_intersection_and_union():
    g1 = [[0.0, 5.0], [10.0, 12.0]]
    g2 = [[3.0, 8.0], [11.0, 15.0]]

    inter = gti_utils.gti_intersection(g1, g2)
    assert inter == [[3.0, 5.0], [11.0, 12.0]]

    union = gti_utils.gti_union(g1, g2)
    assert union == [[0.0, 8.0], [10.0, 15.0]]


def test_sort_and_conversion_helpers():
    g = [[5.0, 8.0], [1.0, 2.0]]
    sorted_g = gti_utils.sort_gti(g)
    assert sorted_g == [[1.0, 2.0], [5.0, 8.0]]

    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert gti_utils._to_2d_list(arr) == [[0.0, 1.0], [2.0, 3.0]]

    assert gti_utils._is_2d_list([[0.0, 1.0]]) is True
    assert gti_utils._is_2d_list([0.0, 1.0]) is False
