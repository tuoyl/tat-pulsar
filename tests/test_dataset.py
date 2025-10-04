import numpy as np
import tatpulsar.data.dataset as dataset_module

# Ensure the dataset module has access to numpy inside the class methods
if not hasattr(dataset_module, "np"):
    dataset_module.np = np

Dataset = dataset_module.Dataset


def test_dataset_add_and_get_by_label():
    ds = Dataset()
    ds.add(3.14, 42.0, label="first", xerr=0.05, yerr=0.1)

    np.testing.assert_allclose(ds.get_x("first"), np.array([3.14]))
    np.testing.assert_allclose(ds.get_y("first"), np.array([42.0]))
    np.testing.assert_allclose(ds.get_xerr("first"), np.array([0.05]))
    np.testing.assert_allclose(ds.get_yerr("first"), np.array([0.1]))


def test_dataset_get_all_concatenated_arrays():
    ds = Dataset()
    ds.add([2.0, 1.0], [20.0, 10.0], label="alpha", xerr=[0.2, 0.1], yerr=[0.4, 0.3])
    ds.add(1.5, 15.0, label="beta")

    np.testing.assert_allclose(ds.get_x(), np.array([2.0, 1.0, 1.5]))
    np.testing.assert_allclose(ds.get_y(), np.array([20.0, 10.0, 15.0]))

    # xerr and yerr include None values for entries without explicit uncertainties
    assert ds.get_xerr().tolist() == [0.2, 0.1, None]
    assert ds.get_yerr().tolist() == [0.4, 0.3, None]


def test_dataset_concatenate_and_sort_with_reference_x():
    ds = Dataset()
    ds.add([2.0, 1.0], [200.0, 100.0], label="a")
    ds.add([1.5], [150.0], label="b")

    sorted_y = ds._concatenate_and_sort(ds.data["y"], ds.data["x"])
    np.testing.assert_allclose(sorted_y, np.array([100.0, 150.0, 200.0]))
