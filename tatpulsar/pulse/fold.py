#!/usr/bin/env python
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
from numba import float64
from tqdm import tqdm

from tatpulsar.utils.functions import met2mjd, mjd2met, get_parameters, cal_event_gti
from tatpulsar.utils.functions import ccf
from tatpulsar.data.profile import phihist, Profile
from tatpulsar.pulse.residuals import parse_pfiles

__all__ = ['fold', 'fold2d',
           'cal_phase', 'phase_exposure',
           'align_profile', 'merge_aligned_profile', 'fold_lightcurve']

def fold_lightcurve(time, cts, pepoch, f0, f1=0, f2=0, f3=0, f4=0, format='met', nbins=16, dt=1, cts_err=None):

    """
    Fold the pulse profiles from the net light curves.


    Parameters
    ----------
    time : array-like
        the time series of TDB data

    cts : array-like
        the counts of the light curve

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

    nbins : int
        the bin number of profile. default value is 16

    dt : float
        the bin size of the light curve, default value is 1 s

    cts_err : array-like
        the counts error of the light curve

    format : str, optional
        the format of time and pepoch, "mjd" or "met".
        The default if "met".

    Returns
    -------
    profile : :class:`tatpulsar.data.profile.Profile` object
        return the profile object define in :class:`tatpulsar.data.profile.Profile`
    """
    if format == 'mjd':
        if 'telescope' not in kwargs:
            raise IOError("'telescope' must be assigned, used to convert MJD to MET")
        else:
            telescope=kwargs['telescope']
        pepoch = mjd2met(pepoch,
                         telescope=telescope)

    phase = cal_phase(time, pepoch, f0, f1=f1, f2=f2, f3=f3, f4=f4, format='met', phi0=0,
              to_1=True)
    phase_ind = phase // (1. / nbins)
    mk = [phase_ind == j for j in range(nbins)]
    # ---
    bin_time = np.array([len(cts[mk[k]]) * dt for k in range(nbins)])
    # ---
    if cts_err is None:
        bin_cts = np.array([cts[mk[k]].sum() for k in range(nbins)])
        bin_cts_err = np.sqrt(bin_cts)

        bin_cts = bin_cts / bin_time
        bin_cts_err = bin_cts_err / bin_time
    else:
        bin_cts = np.array([cts[mk[k]].sum() for k in range(nbins)])
        bin_cts_err = np.array([np.sqrt((cts_err[mk[k]]**2).sum()) for k in range(nbins)])

        bin_cts = bin_cts / bin_time
        bin_cts_err = bin_cts_err / bin_time

    # ---
    return Profile(bin_cts, error=bin_cts_err, cycles=1)

def fold(time, parfile=None,
        pepoch=None, f0=None, f1=0, f2=0, f3=0, f4=0, nbins=20,
        phi0=0, gti=None, use_data_gti=False,
        format='met', **kwargs):
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

    gti : N-darray, optional
        The good time intervals of given data. default is None, if given,
        the exposure correction for each phase bin will be considered.
        You can also use `use_data_gti` parameter to calculate the GTI
        for given `time` array.

    use_data_gti: bool, optional, default is False
        whether calculate the GTI for given `time` (if True the `time`
        must be an array instead of a value).

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
        frequencies, pepoch, start_time, stop_time = \
                parse_pfiles(parfile)
        f0, f1, f2, f3, f4, *_ = frequencies

        if 'telescope' not in kwargs:
            raise IOError("'telescope' must be assigned, used to convert MJD to MET")
        else:
            telescope=kwargs['telescope']

        pepoch = mjd2met(pepoch,
                         telescope=telescope)
        format = 'met'

    elif (pepoch is None) or (f0 is None):
        raise IOError("Parameters to fold not given, use TEMPO2 parfile or set 'pepoch' and 'f0'")

    if time.size==0:
        raise IOError("Error: Data is empty")

    if use_data_gti:
        gti = cal_event_gti(time, tgap=1)
    if gti is not None:
        exp_cor = phase_exposure(gti,
                                 nbins,
                                 f0, f1, f2, f3, f4=f4,
                                 pepoch=pepoch,
                                 phi0=phi0,
                                 format=format)
    else:
        exp_cor = 1

    phi = cal_phase(time, pepoch, f0, f1=f1, f2=f2, f3=f3, f4=f4, format=format, phi0=phi0)

    ## Use phihist to do histogram
    ## TODO:parse Profile keywords
    profile = phihist(phi, nbins)
    profile_cor = Profile(counts=profile.counts/exp_cor,
                          error=profile.error/exp_cor)
    profile_cor.ref_time = pepoch # store reference time to Profile

    return profile_cor


def cal_phase(time, pepoch, f0, f1=0, f2=0, f3=0, f4=0, format='met', phi0=0,
              to_1=True):
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

    to_1 : boolean, optional, default is False
        normalize phase from 0 to 1

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
    if to_1:
        return phase - np.floor(phase)
    return phase

def fold2d(time, y, nseg,
        parfile=None,
        pepoch=None, f0=None, f1=0, f2=0, f3=0, f4=0, nbins=20,
        phi0=0, gti=None, use_data_gti=False,
        format='met'):

    """
    Epoch folding the two array into 2 dimensional histogram

    .. note::
        histogram2d may be implemented in `Profile` object.

    Parameters
    ----------
    time: array-like
        time array
    y: array-like
        another dimensional information of each photon, such as energy channel, or
        time
    nseg: int
        the segment number to uniform split y
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

    gti : N-darray, optional
        The good time intervals of given data. default is None, if given,
        the exposure correction for each phase bin will be considered.
        You can also use `use_data_gti` parameter to calculate the GTI
        for given `time` array. In fold2d case, gti will apply on each profile
        extended in y dimension.
    use_data_gti: bool, optional, default is False
        whether calculate the GTI for given `time` (if True the `time`
        must be an array instead of a value).
    format : str, optional
        the format of time and pepoch, "mjd" or "met".
        The default if "met".

        .. warning::
            The format of event array and the reference time should be
            the same time format (MJD or MET).

    Raises
    ------
        IOError
            If pepoch, f0, and parfile not given,
            or if data is empty.

    Returns
    -------
    profiles : list
        return a list of Profile object (:class:`tatpulsar.data.profile.Profile`)
    """
    if parfile is not None:
        frequencies, pepoch, start_time, stop_time = \
                parse_pfiles(parfile)
        f0, f1, f2, f3, f4, *_ = frequencies

        if 'telescope' not in kwargs:
            raise IOError("'telescope' must be assigned, used to convert MJD to MET")
        else:
            telescope=kwargs['telescope']

        pepoch = mjd2met(pepoch,
                         telescope=telescope)
        format = 'met'

    elif (pepoch is None) or (f0 is None):
        raise IOError("Parameters to fold not given, use TEMPO2 parfile or set 'pepoch' and 'f0'")

    if time.size==0:
        raise IOError("Error: Data is empty")

    if use_data_gti:
        gti = cal_event_gti(time, tgap=1)
    if gti is not None:
        exp_cor = phase_exposure(gti,
                                 nbins,
                                 f0, f1, f2, f3, f4=f4,
                                 pepoch=pepoch,
                                 phi0=phi0,
                                 format=format)
    else:
        exp_cor = 1

    phase = cal_phase(time, pepoch, f0, f1=f1, f2=f2, f3=f3, f4=f4, format=format, phi0=phi0)
    H, X, Y = np.histogram2d(y, phase, bins=(nseg, nbins))

    #exposure correction and convert to Profile object
    H_tmp = [Profile(x) for x in H]
    H_cor = [Profile(counts=x.counts/exp_cor,
                     error=x.error/exp_cor) for x in H_tmp]

    return H_cor


def phase_exposure(gti, nbins,
                   f0, f1=0, f2=0, f3=0, f4=0,
                   pepoch=0,
                   phi0=0,
                   format='met'):
    """
    calculate the exposure correction coefficients for each phase bin.

    Parameters
    ----------
    gti: ndarray or list
        the list of GTI array, example
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
    nbins : int
        the bin number of profile
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
    pepoch : float, optional
        time for input frequecy values.
        NOTE: the output frequecy is the frequency at which
        the time of middle of the time interval
    format : str, optional
        the format of time and pepoch, "mjd" or "met".
        The default if "met".

    Returns
    -------
    expo_corr: array-like
        the coeffients for each phase bin to calculate the exposure correction.
        to correction the count in each phase bin, the count (:math:`R(\phi)`) should divide the corresponding
        expo_corr value (:math:`c(\phi)`).
    """
    gti = np.asarray(gti)
    phase_gti = cal_phase(gti, pepoch, f0, f1=f1, f2=f2, f3=f3, f4=f4, format=format, phi0=phi0, to_1=False)

    N_pulse = 0 # the number of fully observed pulse
    phase_bin_edges = np.linspace(0, 1, nbins + 1)
    phase_bin_counts = np.zeros_like(phase_bin_edges[:-1], dtype=np.float64)
    phase_binsize = np.median(np.diff(phase_bin_edges))

    for gti_i in phase_gti:
        phase_start = gti_i[0]
        phase_stop  = gti_i[1]
        phase_start_to1 = phase_start - np.floor(phase_start)
        phase_stop_to1  = phase_stop   - np.floor(phase_stop)
        N_pulse += np.floor(phase_stop - phase_start)

        idx_start = _get_phase_index(phase_start_to1, phase_bin_edges)
        idx_stop  = _get_phase_index(phase_stop_to1, phase_bin_edges)
        if idx_start != (nbins - 1):
            phase_bin_counts[idx_start+1:] += 1
        phase_bin_counts[idx_start] += \
                (phase_bin_edges[idx_start+1] - phase_start_to1)/phase_binsize

        if idx_stop != 0:
            phase_bin_counts[:idx_stop] += 1
        phase_bin_counts[idx_stop] += \
                (phase_stop_to1 - phase_bin_edges[idx_stop])/phase_binsize

    phase_bin_counts += N_pulse


    return phase_bin_counts/phase_bin_counts.max()

def _get_phase_index(phi, phase_bin_edges):
    """
    return the index of given phi value in the phase bin.
    purpose is to find which bin does the phi value located in.
    if the phi value is exact number of phase bin edges, locate that
    phi into the right side bin.
    """
    if phi in phase_bin_edges:
        return np.searchsorted(phase_bin_edges, phi)
    else:
        return np.searchsorted(phase_bin_edges, phi) - 1

def align_profile(profile_list, template):
    """
    use ccf function to align each profile in the list to the given template

    Parameters
    ----------
    profile_list: list of array
        A list of profile. each element in that list is an array of profile
    template: array-like
        the array of template profile.

    Returns
    -------
    new_list: list
        The list of aligned profiles
    """
    if not isinstance(profile_list, list):
        raise TypeError("The profile_list is not a list type")
    if not isinstance(template, np.ndarray):
        raise TypeError("The template profile is not an array")

    new_list = []
    for i in tqdm(range(len(profile_list))):
        y, delay = ccf(template, profile_list[i])
        new_list.append(np.roll(profile_list[i], delay))
    return new_list

def merge_aligned_profile(profile_list, template):
    """
    align each profile in the `profile_list` and return the profile that sum up
    the counts of each profile

    Parameters
    ----------
    profile_list: list of array
        A list of profile. each element in that list is an array of profile
    template: array-like
        the array of template profile.

    Returns
    -------
    new_profile: array
        The array of merged aligned profiles
    """
    new_list = align_profile(profile_list, template)
    return np.sum(np.asarray(new_list), axis=0)
