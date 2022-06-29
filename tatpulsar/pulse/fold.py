#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
from numba import float64
from tatpulsar.utils.functions import met2mjd, get_parameters
from tatpulsar.data.profile import phihist

__all__ = ['fold']

def fold(data, **kwargs):
    """
    Epoch folding the photon array into profile

    Parameters :
    ---------------
    data : array-like
        the time series of TDB data

    parfile : str ,optional
        read the parameters from parfile

    pepoch : float, optional
        time for input frequecy values.
        NOTE: the output frequecy is the frequency at which
        the time of middle of the time interval

    f0 : float, optional


    f1 : float, optional
        fdot.

    f2 : float, optional
        the second derivative of frequency

    f3 : float, optional
        the third derivative of frequency

    f4 : float, optional
        the fourth derivative of frequency

    nbins : int, optional, default is 20
        the bin number of profile. default value is 20

    telescope : str, optional
        The name of the mission, support mission are
        {'fermi', 'hxmt', 'nicer', 'gecam', 'nustar', 'ixpe'}
        parameter determine the MET refenrence time.
        default is "fermi".

    pepochformat : str, optional
        the format of pepoch, "mjd" or "met".
        The default if "mjd"

    Returns
    -------
    results : dictionary
        return a dictionary for relative results. The key contains
        {"T0", "Profile", "Pars"}, in which results['Pars'] is a dictionary
        as well. It contains the keys as ``F0``, ``F1``, ``F2``, ...
        ``T0`` is the reference time of timing parameters (in MJD).
        ``Profile`` is the counts of each bin for folded events.

    chi_square : array-like
        The Chi Square distribution of Epoch folding
    """

    data = float64(data) # transfer the data to Numba float 64 bites

    # read the input parameters
    if "telescope" in kwargs:
        if kwargs['telescope'].lower() == "fermi":
            telescope = 'fermi'
        elif kwargs['telescope'].lower() == "hxmt":
            telescope = 'hxmt'
        elif kwargs['telescope'].lower() == "nicer":
            telescope = 'nicer'
        elif kwargs['telescope'].lower() == 'gecam':
            telescope = 'gecam'
        else:
            telescope='hxmt'
    else:
        telescope='hxmt'

    #read parfile and parameters
    pepoch, F_set_array, F1_searchflag = get_parameters(kwargs)

    t0 = pepoch

    if data.size==0:
        raise IOError("Error: Data is empty")
    if 'bin' in kwargs:
        bin_profile = kwargs['bin']
    else:
        print("Warning: number of bins not assigned, use default value 20")
        bin_profile = 20

    if 'phi0' in kwargs:
        phi0 = kwargs['phi0']
    else:
        phi0 = 0

    ## Taylor Series
    phi = np.sum(
            np.array([ (1/math.factorial(i+1))*((data-t0)**(i+1))*F_set_array[i] for i in range(len(F_set_array))]),
            axis=0) - phi0
    phi = phi - np.floor(phi)


    ## Use phihist to do histogram
    profile = phihist(phi, bin_profile).counts

    return {"T0": met2mjd(t0, telescope=telescope), "Profile" : profile,
            "Pars" : {"F{}".format(i) : F_set_array[i] for i in range(len(F_set_array))}}

