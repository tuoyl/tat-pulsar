#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
from numba import float64
from tatpulsar.utils.functions import met2mjd, get_parameters
from tatpulsar.data.profile import phihist

__all__ = ['fold',
        'cal_phase']

def fold(time, parfile=None,
        pepoch=None, f0=None, f1=0, f2=0, f3=0, f4=0, nbins=20,
        phi0=0,
        format='met'):
    """
    Epoch folding the photon array and return the folded profile.

    Parameters
    ----------
    time : array-like
        the time series of TDB data

    parfile : str ,optional
        read the parameters from parfile

    pepoch : float, optional
        time for input frequecy values.
        NOTE: the output frequecy is the frequency at which
        the time of middle of the time interval

    f0 : float, optional
        frequency

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

    phi0 : float, optional, default is 0
        the reference phase of the profile, if given,
        the phase is calculated by:

        .. math::
            \phi = \phi - \phi_{0}

    format : str, optional
        the format of time and pepoch, "mjd" or "met".
        The default if "met".

        .. warning::
            The format of event array and the reference time should be
            the same time format (MJD or MET).

    Returns
    -------
    profile : :class:`tatpulsar.data.profile.Profile` object
        return the profile object define in :class:`tatpulsar.data.profile.Profile`
    """

    time = float64(time) # transfer the data to Numba float 64 bites

    if parfile is not None:
        ## TODO: get pars from Tempo2 parfile
        pass
    elif (pepoch is None) or (f0 is None):
        raise IOError("Parameters to fold not given, use TEMPO2 parfile or set 'pepoch' and 'f0'")

    if time.size==0:
        raise IOError("Error: Data is empty")

    phi = cal_phase(time, pepoch, f0, f1=f1, f2=f2, f3=f3, format=format, phi0=phi0)

    ## Use phihist to do histogram
    ## TODO:parse Profile keywords
    profile = phihist(phi, nbins)

    return profile


def cal_phase(time, pepoch, f0, f1=0, f2=0, f3=0, f4=0, format='met', phi0=0):
    """
    calculate the phase for given time or time series.

    Parameters
    ----------
    time : array-like
        the time series to calculate the phase. The default format of time is "MET",
        if your input time is in "MJD" format, set ``format`` to 'mjd'

    pepoch : float
        time for input frequecy values. The default format of ``pepoch`` is "MET",
        if your input pepoch is in "MJD" format, set ``format`` to 'mjd'.
        NOTE: the output frequecy is the frequency at which
        the time of middle of the time interval

    f0 : float
        frequency

    f1 : float, optional
        fdot.

    f2 : float, optional
        the second derivative of frequency

    f3 : float, optional
        the third derivative of frequency

    f4 : float, optional
        the fourth derivative of frequency

    format : str, optional, default is 'met'
        The time system of given time and pepoch. Optional input are
        {'met', 'mjd'}

    Returns
    -------
    phase : array-like
        The phase value for given time and timing parameters
    """

    if format.lower() == 'mjd':
        dt = (time - pepoch)*86400
    elif format.lower() == "met":
        dt= time - pepoch
    phase = f0*dt + \
            0.5*f1*dt**2 + \
            (1/6)*f2*dt**3 + \
            (1/24)*f3*dt**4 + \
            (1/120)*f4*dt**5 - phi0
    return phase - np.floor(phase)
