"""
Class to read the TEMPO2 parameter file (.par)

"""
import numpy as np
import math

__all__ = ['TimingParameter', 'TimingModel']

freq_pars = [f'F{int(i)}' for i in range(20)]
pars_we_care = ['PEPOCH', 'START', 'FINISH', 'PHI0'] +\
               freq_pars

class TimingParameter:
    """
    The parameter in Timing Model. One can retrieve the value and/or error (if valid)
    of the parameter.

    Examples
    --------
    >>> f0 = TimingParameter(value=27, error=1e-3, name='F0')
    >>> print(f'{f0.name} = {f0.value} +- {f0.error}')
    >>> F0 = 27 +- 0.001
    """
    def __init__(self, value=None, error=None, parname=None):
        self.value = value
        self.error = error
        self.parname = parname

class TimingModel:
    """
    Class of Pulsar timing model

    Examples
    --------
    >>> eph = TimingModel()
    >>> eph.readpar("test.par")
    >>> print("F0 = ", eph.F0.value, eph.f0.value)
    >>> F0 =  29.636679699921209437
    >>> print("F0 error = ", eph.F0.error)
    >>> F0 error =  1.7247236495e-09
    >>> print("PEPOCH = ", eph.PEPOCH.value)
    >>> PEPOCH =  58066.18087539147382
    >>> print("PEPOCH error = ", eph.PEPOCH.error)
    >>> PEPOCH error =  None
    """

    def __init__(self, parfile=None):
        self.parfile = parfile
        if parfile:
            self.readpar(parfile)

    def readpar(self, filename):
        """
        Read the TEMPO2 parameter file (.par)
        """
        with open(filename, 'r') as file:
            lines = file.readlines()

        for line in lines:
            parts = line.split()
            if parts:
                key = parts[0]
                values = parts[1:]
                if len(values) == 3:
                    # | value | flag | error | case
                    value = self._smart_convert(values[0])
                    error = self._smart_convert(values[2])
                else:
                    value = self._smart_convert(values[0])
                    error = None
                setattr(self, key,
                        TimingParameter(value=value, error=error, parname=key))
                setattr(self, key.lower(),
                        TimingParameter(value=value, error=error, parname=key))

    def update(self, new_pepoch):
        """Update the timing model based on a new given pepoch value
        """
        dt = (new_pepoch - self.PEPOCH.value)*86400
        old_frequency = self.frequency
        new_frequency = np.zeros(self.freq_order)
        for i in range(self.freq_order):
            frequency_incorporated = old_frequency[i:]
            vectorized_factorial = np.vectorize(math.factorial)
            indices = np.arange(frequency_incorporated.size)
            new_frequency[i] = np.sum(
                    (1 / vectorized_factorial(indices)) * frequency_incorporated * (dt ** indices))

        ## Update attributes
        for i in range(self.freq_order):
            setattr(self, f"F{int(i)}", TimingParameter(value=new_frequency[i],
                                        error=getattr(self, f"F{int(i)}").error))
            setattr(self, f"f{int(i)}", TimingParameter(value=new_frequency[i],
                                        error=getattr(self, f"f{int(i)}").error))
        setattr(self, "PEPOCH", TimingParameter(value=new_pepoch,
                                                error=getattr(self, "PEPOCH").error))
        setattr(self, "pepoch", TimingParameter(value=new_pepoch,
                                                error=getattr(self, "pepoch").error))

    @property
    def reftime(self):
        """return the reference time in the timingpar
        """
        return getattr(self, 'PEPOCH').value

    @property
    def frequency(self):
        """return the list of frequency and its high order derivatives
        """
        freq_list = []
        for par in freq_pars:
            if hasattr(self, par):
                freq_list.append(getattr(self, par).value)
        return np.asarray(freq_list)

    @property
    def freq_order(self):
        """return the order of polynomial function of timing model
        """
        return self.frequency.size

    @staticmethod
    def _smart_convert(number_str):
        """
        Attempts to intelligently convert a numerical string to either an int or a float,
        depending on its format.

        Parameters
        ----------
        number_str: A string representing a number.

        Returns
        -------
        The number converted to either int or float, or the original string if conversion fails.
        """
        try:
            # First, try converting to int
            return int(number_str)
        except ValueError:
            # If int conversion fails, try converting to float
            try:
                return float(number_str)
            except ValueError:
                # If both conversions fail, return the original string
                return number_str
