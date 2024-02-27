#!/usr/bin/env python
import numpy as np
from numba import float64
import matplotlib.pyplot as plt
from tatpulsar.utils.functions import *
from tatpulsar.data.profile import phihist

__all__ = ['search']


def _parameters_legal(kwargs):
    """
    check whether the input parameters are legal

    return bool
    """
    init_bool = True

    legal_par_list = ['check_par', 'pepoch', 'f0', 'f0step', 'f0range', 'f1', 'f1step', 'f1range',
            'f2', 'f3', 'f4', 'pepochformat', 'telescope', 'nbins']


    if 'check_par' in kwargs:
        if kwargs['check_par']:
            for key in kwargs:
                if key in legal_par_list:
                    continue
                else:
                    raise IOError("parameter '{}' not surpportted".format(key))




def _get_parameters(kwargs):
    """
    get the parameters for searching

    The format of input could be a name of parfile, a dictionary, or standard python function arguments.

    """

    #check whether input parameters are surpportted
    _parameters_legal(kwargs)

    #read parfile and parameters
    if "parfile" in kwargs:
        # Read parameters in parfile instead of keyboard input
        parname = kwargs['parfile']
        from tatpulsar.utils.timingmodel import TimingModel
        eph = TimingModel(parname)
        f1search_flag = False
        F0 = eph.F0.value
        F1 = eph.F1.value
        F2 = eph.F2.value
        F3 = eph.F3.value
        F4 = eph.F4.value
        pepoch = eph.reftime
        return pepoch, F0, F1, F2, F3, F4, f1search_flag

    else:
        pepoch = kwargs['pepoch']
        F0_mid     = kwargs['f0']
        F0_step    = kwargs['f0_step']
        F0_nstep    = kwargs['f0_nstep']
        F0 = np.arange(F0_mid-F0_nstep*F0_step, F0_mid+F0_nstep*F0_step, F0_step)
        if 'f1' in kwargs:
            F1 = kwargs['f1']
            if "f1_step" in kwargs:
                F1_step = kwargs["f1_step"]
                F1_nstep= kwargs["f1_nstep"]
                if F1_step*F1_nstep == 0:
                    F1 = F1
                    f1search_flag = False
                else:
                    F1 = np.arange(F1-F1_step*F1_nstep, F1+F1_step*F1_nstep, F1_step)
                    f1search_flag = True
                    print(f"number of parameters to search is {len(F1)*len(F0)}")
            else:
                f1search_flag = False
        else:
            F1 = 0
            F1_step = 0
            F1_nstep= 0
            f1search_flag = False

        if 'f2' in kwargs:
            F2 = kwargs['f2']
        else:
            F2 = 0
        if 'f3' in kwargs:
            F3 = kwargs['f3']
        else:
            F3 = 0
        if 'f4' in kwargs:
            F4 = kwargs['f4']
        else:
            F4 = 0

    return pepoch, F0, F1, F2, F3, F4, f1search_flag

def search(data, **kwargs):
    """
    search the best frequency

    Parameters
    ----------
    data : array-like
        the time series of TDB data

    f0 : float
        the init f0 value to search

    f0_step : float
        The step length for frequencies to search

    f0_nstep : int
        The number of steps for frequencies to search

    f1 : float, optional
        The frequency derivative used to calculate the phase of each photon

    f2 : float, optional
        the second derivative of frequency

    f3 : float, optional
        the third derivative of frequency

    f4 : float, optional
        the fourth derivative of frequency

    nbins : int, optional, default is 20
        the bin number of profile. default value is 20

    f1_step : float, optional
        The step length to search the frequency derivate as well. Only works if ``f1_step``
        and ``f1_nstep`` are both given

    f1_nstep : int, optional
        the number of fdot search steps to search

    pepoch : float, optional
        time for input frequecy values. The default is the minimux of given data set.

        .. note::
            the output frequecy is the frequency at which
            the middle of each time interval

    parfile : str, optional
        read the parameters from parfile


    Returns
    -------

    """

    data = float64(data) # transfer the data to Numba float 64 bites

    #read parfile and parameters
    pepoch, F0, F1, F2, F3, F4, F1_searchflag = _get_parameters(kwargs)
    t0 = pepoch

    if len(data)==0:
        raise IOError("Error: Data is empty")
    if 'nbins' in kwargs:
        nbins = kwargs['nbins']
    else:
        nbins = 20

    if F1_searchflag:
        ## F0 and F1 2-D search
        chi_square = cal_2dchisquare(data, F0, F1, t0, nbins, F2=F2, F3=F3, F4=F4)
        f_f1_index = np.unravel_index(np.argmax(chi_square, axis=None), chi_square.shape)
        fbest = F0[f_f1_index[1]]
        f1best = F1[f_f1_index[0]]
    else:
        ## F0 1-D search
        chi_square = cal_chisquare(data, F0, t0, nbins, F1, F2, F3, F4)
        fbest = F0[np.argmax(chi_square)]
        f1best = F1

    phi = (data-t0)*fbest + (1.0/2.0)*((data-t0)**2)*f1best + (1.0/6.0)*((data-t0)**3)*F2 +\
            (1.0/24.0)*((data-t0)**4)*F3 + (1.0/120.0)*((data-t0)**5)*F4
    phi = phi - np.floor(phi)
    profile = phihist(phi, nbins).counts

    result = type("result", (object,), {})()
    result.pepoch = t0
    result.chisquare = chi_square
    result.profile = profile
    result.freq = F0
    result.freqderiv = F1
    result.f2 = F2
    result.f3 = F3
    result.f4 = F4
    return result
