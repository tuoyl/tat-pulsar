#!/usr/bin/env python
import numpy as np
import numba 
import matplotlib.pyplot as plt
from pulsar_timing.utils import *

__all__ = ['ffold']



def ffold(data, **kwargs):
    """
    search the best frequency

    Parameters :
    ---------------
    data : array-like
        the time series of TDB data

    parfile : string (optional)
        read the parameters from parfile

    pepoch : float (optional)
        time for input frequecy values. 
        NOTE: the output frequecy is the frequency at which 
        the time of middle of the time interval

    f0 : float
        the init f0 value to search

    f0step : float 
        

    f1 : float(optional)
        fdot.

    f2 : float (optional)
        the second derivative of frequency

    f3 : float (optional)
        the third derivative of frequency

    f4 : float (optional)
        the fourth derivative of frequency

    bin : int (optional)
        the bin number of profile. default value is 20

    f1step : float (optional)
        the fdot search step length

    f1range : float (optional)
        the fdot search step range

    telescope : string (optional)
        the name of telescope NAME, "hxmt" or "fermi". this
        parameter determine the MET refenrence time.
        default is "fermi".

    pepochformat : string (optional)
        the format of pepoch, "mjd" or "met". 
        The default if "mjd"

    Returns :
    ---------------

    chi_square : array-like
        The Chi Square distribution of Epoch folding 
    """
    
    data = numba.float64(data) # transfer the data to Numba float 64 bites
    
    # read the input parameters 
    if "telescope" in kwargs:
        if kwargs['telescope'].lower() == "fermi":
            telescope = 'fermi'
        elif kwargs['telescope'].lower() == "hxmt":
            telescope = 'hxmt'
        elif kwargs['telescope'].lower() == "nicer":
            telescope = 'nicer'
    else:
        telescope='hxmt'

    #read parfile and parameters
    pepoch, F0, F1, F2, F3, F4, F1_searchflag = get_parameters(kwargs)
    
    t0 = pepoch
    f0 = F0
    f1 = F1
    f2 = F2
    f3 = F3
    f4 = F4
    
    if len(data)==0:
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

    phi = (data-t0)*f0 + (1.0/2.0)*((data-t0)**2)*f1+ (1.0/6.0)*((data-t0)**3)*f2 +\
            (1.0/24.0)*((data-t0)**4)*f3 + (1.0/120.0)*((data-t0)**5)*f4 
    phi = phi - np.floor(phi)
    profile, phase  = numba_histogram(phi, bin_profile)

    return {"T0": met2mjd(t0, telescope=telescope), "Profile" : profile, 
            "Pars" : {"F0":f0, "F1":f1, "F2":F2, "F3":F3, "F4":F4}}



if __name__ == "__main__":
    pass
