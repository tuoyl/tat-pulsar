
__all__ = ["Dataset"]

class Dataset:
    def __init__(self):
        self.data = {'x': {}, 'y': {}, 'yerr': {}, 'xerr': {}}

    def add(self, x, y, label, xerr=None, yerr=None):
        self.data['x'][label] = np.atleast_1d(x)
        self.data['y'][label] = np.atleast_1d(y)
        self.data['yerr'][label] = np.atleast_1d(yerr)
        self.data['xerr'][label] = np.atleast_1d(xerr)

    def get_x(self, label=None):
        return self.data['x'][label] if label else self._concatenate_and_sort(self.data['x'])

    def get_y(self, label=None):
        return self.data['y'][label] if label else self._concatenate_and_sort(self.data['y'])

    def get_yerr(self, label=None):
        return self.data['yerr'][label] if label else self._concatenate_and_sort(self.data['yerr'])

    def get_xerr(self, label=None):
        return self.data['xerr'][label] if label else self._concatenate_and_sort(self.data['xerr'])

    def _concatenate_and_sort(self, data_dict, x_dict=None):
        # Concatenate all data
        concatenated = np.concatenate(list(data_dict.values()))
        if x_dict is not None:
            # Sort based on x values if x_dict is provided
            x_concatenated = np.concatenate(list(x_dict.values()))
            sorted_indices = x_concatenated.argsort()
            return concatenated[sorted_indices]
        return concatenated
