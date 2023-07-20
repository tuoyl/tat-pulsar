"""
Class to read the TEMPO2 parameter file (.par)

"""
import numpy as np

freq_pars = [f'F{int(i)}' for i in range(20)]
pars_we_care = ['PEPOCH', 'START', 'FINISH', 'PHI0'] +\
               freq_pars

class readpar:
    """
    Class to parse the TEMPO2 parameter file (.par)
    The parameters in the '.par' file might be capitalized but those parameters
    stored in this object are case INSENSITIVE (see examples below).

    Example
    -------
    >>> eph = readpar('test.par')
    >>> print("F0 = ", eph.F0.value, eph.f0.value)
    >>> F0 =  29.636679699921209437
    >>> print("F0 error = ", eph.F0.error)
    >>> F0 error =  1.7247236495e-09
    >>> print("PEPOCH = ", eph.PEPOCH.value)
    >>> PEPOCH =  58066.18087539147382
    >>> print("PEPOCH error = ", eph.PEPOCH.error)
    >>> PEPOCH error =  None

    """
    def __init__(self, filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()

        for line in lines:
            parts = line.split()
            if parts:
                key = parts[0]
                values = parts[1:]
                setattr(self, key, values)
                setattr(self, key.lower(), values)

        for par in pars_we_care:
            if hasattr(self, par):
                value_list = getattr(self, par)

                setattr(self, par, type("timingpar", (object,), {})())
                setattr(self, par.lower(), type("timingpar", (object,), {})()) # Lower case

                setattr(getattr(self, par), 'value', np.float128(value_list[0]))
                setattr(getattr(self, par.lower()), 'value', np.float128(value_list[0])) # Lower case

                if len(value_list) == 3:
                    setattr(getattr(self, par), 'error', np.float128(value_list[2]))
                    setattr(getattr(self, par.lower()), 'error', np.float128(value_list[2]))
                else:
                    setattr(getattr(self, par), 'error', None)
                    setattr(getattr(self, par.lower()), 'error', None)

if __name__ == "__main__":
    eph = readpar('../../tests/test.par')
    print("F0 = ", eph.F0.value)
    print("F0 error = ", eph.F0.error)
    print("PEPOCH = ", eph.PEPOCH.value)
    print("PEPOCH error = ", eph.PEPOCH.error)

    print("F0 = ", eph.f0.value)
    print("F0 error = ", eph.f0.error)
    print("PEPOCH = ", eph.pepoch.value)
    print("PEPOCH error = ", eph.pepoch.error)
