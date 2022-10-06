#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

"""
Several useful functions used througout packages
"""
# If numba is installed, import jit. Otherwise, define an empty decorator with
# the same name. (This method was Copied from Stingray.utils)
HAS_NUMBA = False
try:
    from numba import jit

    HAS_NUMBA = True
    from numba import njit, prange, vectorize, float32, float64, int32, int64
except ImportError:
    warnings.warn("Numba not installed. Faking it")

    class jit(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            def wrapped_f(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_f

    class njit(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            def wrapped_f(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_f

    class vectorize(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            wrapped_f = np.vectorize(func)

            return wrapped_f

    def generic(x, y=None):
        return None

    float32 = float64 = int32 = int64 = generic

    def prange(x):
        return range(x)


__all__ = ["numba_histogram",
        "met2mjd",
        "mjd2met",
        "cal_chisquare",
        "cal_2dchisquare",
        "get_parameters",
        "ccf",
        "gauss", "lorentz",
        "rms",
        "print_loop_percentage",
        "cal_event_gti"]

#@njit
def met2mjd(data, telescope="fermi"):
    """
    Convert Mission Elapse Time (MET) to Modified Julian Date (MJD).

    .. math::
        T_{\mathrm{MJD}} = T_{\mathrm{MET}}/86400 + \mathrm{MJDREF},

    where MJDREF is the reference time for each mission.

    Parameters
    ----------
    data : float
        The MET time
    telescope : str, default 'fermi'
        The name of the mission, support mission are
        {'fermi', 'hxmt', 'nicer', 'gecam', 'nustar', 'ixpe'}

    Returns
    -------
    mjd : float
        The MJD time
    """
    if telescope.lower() == "fermi":
        MJDREFF = 0.00074287037037037
        MJDREFI = 51910
    elif telescope.lower() == "hxmt":
        MJDREFF = 0.00076601852000000
        MJDREFI = 55927
    elif telescope.lower() == 'nicer':
        MJDREFF = 0.000777592592592593
        MJDREFI = 56658
    elif telescope.lower() == 'gecam':
        MJDREFF = 0.00080074074
        MJDREFI = 58484
    elif telescope.lower() == 'nustar':
        MJDREFI = 55197
        MJDREFF = 7.660185200000000E-04
    elif telescope.lower() == "ixpe":
        MJDREFF = 0.00080074074074 
        MJDREFI = 57754
    return data/86400 + MJDREFI + MJDREFF

#@njit
def mjd2met(data, telescope="fermi"):
    """
    Convert Modified Julian Date (MJD) to Mission Elapse Time (MET)

    .. math::
        T_{\mathrm{MJD}} = T_{\mathrm{MET}}/86400 + \mathrm{MJDREF},

    where MJDREF is the reference time for each mission.

    Parameters
    ----------
    data : float
        The MJD time
    telescope : str, default 'fermi'
        The name of the mission, support mission are
        {'fermi', 'hxmt', 'nicer', 'gecam', 'nustar', 'ixpe'}

    Returns
    -------
    met : float
        The MET time
    """
    if telescope.lower() == "fermi":
        MJDREFF = 0.00074287037037037
        MJDREFI = 51910
    elif telescope.lower() == "hxmt":
        MJDREFF = 0.00076601852000000
        MJDREFI = 55927
    elif telescope.lower() == 'nicer':
        MJDREFF = 0.000777592592592593
        MJDREFI = 56658
    elif telescope.lower() == 'gecam':
        MJDREFF = 0.00080074074
        MJDREFI = 58484
    elif telescope.lower() == 'nustar':
        MJDREFI = 55197
        MJDREFF = 7.660185200000000E-04
    elif telescope.lower() == "ixpe":
        MJDREFF = 0.00080074074074 
        MJDREFI = 57754
    return (data - MJDREFI - MJDREFF)*86400



@njit(nopython=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@njit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@njit(nopython=True)
def numba_histogram(a, bins):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges


#@njit(parallel=True, nogil=True)
def cal_chisquare(data, f, pepoch, nbins, F1=0, F2=0, F3=0, F4=0, parallel=False):
    """
    Calculate the Pearson-Chisquare value for given spinning parameters at given epoch time.

    .. math::
        \chi^2 = f_{0} \cdot (t-T_{\mathrm{ref}}) + \\frac{1}{2} \cdot f_{1} \cdot (t-T_{\mathrm{ref}})^2 +
        \\frac{1}{6} \cdot f_{2} \cdot (t-T_{\mathrm{ref}})^3 + \cdots,

    where :math:`T_{\mathrm{ref}` is the reference time, :math:`f_{0}`, :math:`f_{1}`, :math:`f_{2}`, ...,
    are the parameters of pulsar.

    Parameters
    ----------
    data : array-like
        The time array of photons to calculate the chisquare

    f : array-like
        A set of frequencies to calculate the chisquare for event array

    pepoch : float
        The reference time of pulsar timing parameters

    nbins : int
        The number of bins to fold profile

    F1 : float, optional, default 0
        The frequency derivative

    F2 : float, optional, default 0
        The second frequency derivative

    F3 : float, optional, default 0
        The third frequency derivative

    F4 : float, optional, default 0
        The forth frequency derivative

    parallel : boolean, optional, default ``False``
        whether to use multi-core CPU to calculate the chisquare

    Returns
    -------
    chi_square : array-like
        The calculated :math:`\chi^2` array
    """
    chi_square = np.zeros(len(f), dtype=np.float64)

    t0 = pepoch

    if parallel:
        for i in prange(len(f)):
            phi = (data-t0)*f[i] + (1.0/2.0)*((data-t0)**2)*F1 + (1.0/6.0)*((data-t0)**3)*F2 +\
                    (1.0/24.0)*((data-t0)**4)*F3 + (1.0/120.0)*((data-t0)**5)*F4
            phi = phi - np.floor(phi)
            #counts  = numba_histogram(phi, nbins)[0]
            #NOTE: The histogram bin should give the edge of bin, instead of the bin number.
            #NOTE: For those pulse with narrow peak, it will be incorrect while calculate the chisquare
            counts  = np.histogram(phi,
                    np.linspace(0, 1, nbins+1)[:-1])[0]
            expectation = np.mean(counts)
            chi_square[i] = np.sum( (counts - expectation)**2 / counts )
    else:
        for i in range(len(f)):
            phi = (data-t0)*f[i] + (1.0/2.0)*((data-t0)**2)*F1 + (1.0/6.0)*((data-t0)**3)*F2 +\
                    (1.0/24.0)*((data-t0)**4)*F3 + (1.0/120.0)*((data-t0)**5)*F4
            phi = phi - np.floor(phi)
            #counts  = numba_histogram(phi, nbins)[0]
            #NOTE: The histogram bin should give the edge of bin, instead of the bin number.
            #NOTE: For those pulse with narrow peak, it will be incorrect while calculate the chisquare
            counts  = np.histogram(phi,
                    np.linspace(0, 1, nbins+1)[:-1])[0]
            expectation = np.mean(counts)
            chi_square[i] = np.sum( (counts - expectation)**2 / counts )

    return chi_square

@njit
def cal_2dchisquare(data, f, F1, pepoch, nbins, F2=0, F3=0, F4=0):
    """
    Calculate the chisquare distribution for 2-D frequency search on the pepoch time.
    For example, search in a two-dimensianal parameter space (:math:`M \\times N`, as :math:`M`-length frequency
    array, and :math:`N`-length frequency derivative array).

    .. math::
        \chi^2 = f_{0} \cdot (t-T_{\mathrm{ref}}) + \\frac{1}{2} \cdot f_{1} \cdot (t-T_{\mathrm{ref}})^2 +
        \\frac{1}{6} \cdot f_{2} \cdot (t-T_{\mathrm{ref}})^3 + \cdots,

    where :math:`T_{\mathrm{ref}` is the reference time, :math:`f_{0}`, :math:`f_{1}`, :math:`f_{2}`, ...,
    are the parameters of pulsar.

    Parameters
    ----------
    data : array-like
        The time array of photons to calculate the chisquare

    f : array-like
        A set of frequencies to calculate the chisquare for event array

    F1 : float, optional, default 0
        The frequency derivative

    pepoch : float
        The reference time of pulsar timing parameters

    nbins : int
        The number of bins to fold profile

    F2 : float, optional, default 0
        The second frequency derivative

    F3 : float, optional, default 0
        The third frequency derivative

    F4 : float, optional, default 0
        The forth frequency derivative

    Returns
    -------
    chi_square : array-like
        An :math:`M \\times N` array, as :math:`M` is the length of frequency `f`, :math:`N` is the
        length of frequency derivative `F1`
    """

    chi_square = np.zeros(len(f)*len(F1), dtype=np.float64).reshape(len(F1), len(f))
    # n X m matrix, with n F1 and m f for each F1
    print(chi_square.shape, len(F1))

    t0 = pepoch

    for i in range(len(F1)):
        for j in range(len(f)):
            phi = (data-t0)*f[j] + (1.0/2.0)*((data-t0)**2)*F1[i] + (1.0/6.0)*((data-t0)**3)*F2 +\
                    (1.0/24.0)*((data-t0)**4)*F3 + (1.0/120.0)*((data-t0)**5)*F4
            phi = phi - np.floor(phi)
            counts  = numba_histogram(phi, nbins)[0]
            chi_square[i][j] = np.sum( (counts - np.mean(counts))**2 / counts )
    return chi_square


def _parameters_legal(kwargs):
    """
    check whether the input parameters are legal

    return bool
    """
    init_bool = True

    legal_par_list = ['check_par', 'pepoch', 'f0', 'f0step', 'f0range', 'f1', 'f1step', 'f1range',
            'f2', 'f3', 'f4', 'pepochformat', 'telescope', 'bin']


    if 'check_par' in kwargs:
        if kwargs['check_par']:
            for key in kwargs:
                if key in legal_par_list:
                    continue
                else:
                    raise IOError("parameter '{}' not surpportted".format(key))


def get_parameters(kwargs):
    """
    get the parameters for searching
    The format of input could be a name of parfile, a dictionary, or standard python function arguments.
    """

    #check whether input parameters are surpportted
    _parameters_legal(kwargs)

    #read parfile and parameters
    if "parfile" in kwargs:
        # Read parameters in parfile instead of keyboard input
        pardata = open(kwargs['parfile'],'r')
        stdpar = []
        for par in pardata:
            par = par[0:(len(par)-1)]
            stdpar.append(par)
        pardata.close()
        F0 = 0; F1 = 0; F2 = 0; F3 = 0; F4 = 0; F5 = 0; F6 = 0; F7 = 0; F8 = 0; F9 = 0; F10 = F11 = F12 = 0;
        for i in range(len(stdpar)):
            if stdpar[i][:6]=='PEPOCH':
                PEPOCH_lst = stdpar[i].split(' ');PEPOCH = [x for x in PEPOCH_lst if x != ''][1]
                pepoch = mjd2met(np.float64(PEPOCH), telescope=kwargs['telescope'])
            if stdpar[i][:2]=='F0': 
                F0_lst = stdpar[i].split(' ');F0 = [x for x in F0_lst if x != ''][1]
                F0 = np.float64(F0) 
            if stdpar[i][:2]=='F1':
                F1_lst = stdpar[i].split(' ');F1 = [x for x in F1_lst if x != ''][1]
                F1 = np.float64(F1)
            if stdpar[i][:2]=='F2':
                F2_lst = stdpar[i].split(' ');F2 = [x for x in F2_lst if x != ''][1]
                F2 = np.float64(F2)
            if stdpar[i][:2]=='F3':
                F3_lst = stdpar[i].split(' ');F3 = [x for x in F3_lst if x != ''][1]
                F3 = np.float64(F3)
            if stdpar[i][:2]=='F4':
                F4_lst = stdpar[i].split(' ');F4 = [x for x in F4_lst if x != ''][1]
                F4 = np.float64(F4)
            if stdpar[i][:2]=='F5':
                F5_lst = stdpar[i].split(' ');F5 = [x for x in F5_lst if x != ''][1]
                F5 = np.float64(F5)
            if stdpar[i][:2]=='F6':
                F6_lst = stdpar[i].split(' ');F6 = [x for x in F6_lst if x != ''][1]
                F6 = np.float64(F6)
            if stdpar[i][:2]=='F7':
                F7_lst = stdpar[i].split(' ');F7 = [x for x in F7_lst if x != ''][1]
                F7 = np.float64(F7)
            if stdpar[i][:2]=='F8':
                F8_lst = stdpar[i].split(' ');F8 = [x for x in F8_lst if x != ''][1]
                F8 = np.float64(F8)
            if stdpar[i][:2]=='F9':
                F9_lst = stdpar[i].split(' ');F9 = [x for x in F9_lst if x != ''][1]
                F9 = np.float64(F9)
            if stdpar[i][:2]=='F10':
                F10_lst = stdpar[i].split(' ');F10 = [x for x in F10_lst if x != ''][1]
                F10 = np.float64(F10)
            if stdpar[i][:2]=='F11':
                F11_lst = stdpar[i].split(' ');F11 = [x for x in F11_lst if x != ''][1]
                F11 = np.float64(F11)
            if stdpar[i][:2]=='F12':
                F12_lst = stdpar[i].split(' ');F12 = [x for x in F12_lst if x != ''][1]
                F12 = np.float64(F12)
            if stdpar[i][:5]=='START':
                START_lst = stdpar[i].split(' ');START = [x for x in START_lst if x != ''][1]
                TSTART = np.float64(START) 
            if stdpar[i][:6]=='FINISH':
                FINISH_lst = stdpar[i].split(' ');FINISH = [x for x in FINISH_lst if x != ''][1]
                TFINISH = np.float64(FINISH) 
        f1search_flag = False
        return pepoch, np.array([F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12]), f1search_flag

    else:
        pepoch = kwargs['pepoch']
        F0_mid     = kwargs['f0']
        if 'f0step' in kwargs:
            F0_step    = kwargs['f0step']
        else:
            F0_step = 0
        if 'f0range' in kwargs:
            F0_range    = kwargs['f0range']
        else:
            F0_range = 0
        if F0_range * F0_step == 0:
            F0 = F0_mid
        else:
            F0 = np.arange(F0_mid-F0_range, F0_mid+F0_range, F0_step)

        if 'f1' in kwargs:
            F1 = kwargs['f1']
            if "f1step" in kwargs:
                F1step = kwargs["f1step"]
                F1range = kwargs["f1range"]
                if F1step*F1range == 0:
                    F1 = F1
                    f1search_flag = False
                else:
                    F1 = np.arange(F1-F1range, F1+F1range, F1step)
                    f1search_flag = True
                    print("number of parameters to search is {}".format(F1.size*F0.size))
            else:
                f1search_flag = False
        else:
            F1 = 0
            F1step = 0
            F1range = 0
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
        if 'f5' in kwargs:
            F5 = kwargs['f5']
        else:
            F5 = 0
        if 'f6' in kwargs:
            F6 = kwargs['f6']
        else:
            F6 = 0
        if 'f7' in kwargs:
            F7 = kwargs['f7']
        else:
            F7 = 0
        if 'f8' in kwargs:
            F8 = kwargs['f8']
        else:
            F8 = 0
        if 'f9' in kwargs:
            F9 = kwargs['f9']
        else:
            F9 = 0
        if 'f10' in kwargs:
            F10 = kwargs['f10']
        else:
            F10 = 0
        if 'f11' in kwargs:
            F11 = kwargs['f11']
        else:
            F11 = 0
        if 'f12' in kwargs:
            F12 = kwargs['f12']
        else:
            F12 = 0

        if "pepochformat" in kwargs:
            if kwargs['pepochformat'].lower() == "met":
                pepoch = pepoch
            elif kwargs['pepochformat'].lower() == "mjd":
                pepoch = mjd2met(pepoch)
            else:
                raise IOError("pepoch format {} not supported".format(kwargs['pepochformat']))
    return pepoch, np.array([F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12]), f1search_flag

@njit(parallel=True, nogil=True)
def ccf(f1,f2):
    '''
    Calculate the cross-correlation function for given data.
    f1 is the original signal
    f2 is probe signal(shift and test)

    Returns
    -------
    y : array-like
        the ccf function distribution
    delay : float
        the index of the delay between the input data `f2` and the `f1`
    '''

    #f1 = (f1 - np.min(f1))/(np.max(f1)-np.min(f1))
    #f2 = (f2 - np.min(f2))/(np.max(f2)-np.min(f2))
    y = np.zeros(len(f2))
    mean_f1 = np.mean(f1)
    mean_f2 = np.mean(f2)
    delta_f1 = f1 - mean_f1
    delta_f2 = f2 - mean_f2
    sigma_f1 = np.sqrt(np.sum(f1*f1))
    sigma_f2 = np.sqrt(np.sum(f2*f2))
    #y = np.correlate(f1, f2, "full")
    for i in prange(len(f2)):
        y[i] = np.sum(delta_f1 * np.roll(delta_f2, i))/(sigma_f1 * sigma_f2)
    #y = [ np.sum(delta_f1 * np.roll(delta_f2,x))/(sigma_f1 * sigma_f2) for x in range(len(f2)) ]
    #delay = np.where(y==max(y))[0]
    delay = np.argmax(y)
    return y,delay

def print_loop_percentage(iterator_i, total, printstr=''):
    """
    print the percentage in a loop
    """
    percent = iterator_i*100/total
    sys.stdout.write("{} complete: {:.2f}".format(printstr, percent))
    sys.stdout.write("%\r")
    sys.stdout.flush()

def gauss(x,a,x0,sigma):
    '''
    Gaussian function
    '''
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def lorentz(x, amp, cen, wid):
    '''
    Lorentz function
    '''
    return (amp*wid**2/((x-cen)**2+wid**2))

def rms(x):
    '''
    root-mean-square function
    '''
    return np.sqrt(x.dot(x)/x.size)

def cal_event_gti(data, tgap=1):
    """
    calculate the gti edges of given event data.
    if the time gap between two adjacent event is larger
    than `tgap`, it split the event into two intervals. Otherwise,
    we take the event as continous observation.

    Parameters
    ---------
    data: array-like
        the event array
    tgap: float
        the critical time gap to split GTI

    Returns
    -------
    gtis: ndarray
        the list of GTI array, example
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Apply the GTI on input data, the type of data should be an array")

    data = np.sort(data) #sort data

    tdiff = np.diff(data)
    gap_indx = np.greater(tdiff, tgap) #index for right edges of gti, apply for data[:-1] array
    right_edges = data[:-1][gap_indx]
    left_edges  = data[:-1][np.roll(gap_indx, 1)]

    right_edges = np.append(right_edges, data[-1])
    left_edges  = np.append(data[0], left_edges)

    return np.dstack((left_edges, right_edges))[0]





