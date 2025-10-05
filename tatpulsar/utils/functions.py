#!/usr/bin/env python
import numpy as np
import warnings
import sys

"""Core utility functions shared across the package."""
# If numba is installed, import jit. Otherwise, define an empty decorator with
# the same name. (This method was Copied from Stingray.utils)
HAS_NUMBA = False
try:
    from numba import jit

    HAS_NUMBA = True
    from numba import njit, prange, vectorize, float32, float64, int32, int64
except ImportError:
    warnings.warn("Numba not installed. Faking it")

    def _passthrough_decorator(func):
        def wrapped_f(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped_f

    class jit(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            return _passthrough_decorator(func)

    class njit(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            return _passthrough_decorator(func)

    class vectorize(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            return np.vectorize(func)

    def _identity(x, y=None):
        return None

    float32 = float64 = int32 = int64 = _identity

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
        T_{\\mathrm{MJD}} = T_{\\mathrm{MET}}/86400 + \\mathrm{MJDREF},

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
    elif telescope.lower() == "swift":
        MJDREFF = 7.4287037e-4
        MJDREFI = 51910
    elif telescope.lower() == "rxte":
        MJDREFF = 6.96574074e-04
        MJDREFI = 49353
    elif telescope.lower() == "catch":
        MJDREFF = 4.65093544e-04
        MJDREFI = 40587
    return data/86400 + MJDREFI + MJDREFF

#@njit
def mjd2met(data, telescope="fermi"):
    """
    Convert Modified Julian Date (MJD) to Mission Elapse Time (MET)

    .. math::
        T_{\\mathrm{MJD}} = T_{\\mathrm{MET}}/86400 + \\mathrm{MJDREF},

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
    elif telescope.lower() == "swift":
        MJDREFF = 7.4287037e-4
        MJDREFI = 51910
    elif telescope.lower() == "rxte":
        MJDREFF = 6.96574074e-04
        MJDREFI = 49353
    elif telescope.lower() == "catch":
        MJDREFF = 4.65093544e-04
        MJDREFI = 40587
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


@njit(parallel=True, nogil=True)
def cal_chisquare(data, f, pepoch, nbins, F1=0, F2=0, F3=0, F4=0, parallel=False):
    """
    Calculate the Pearson-Chisquare value for given spinning parameters at given epoch time.

    .. math::
        \\chi^2 = f_{0} \\cdot (t-T_{\\mathrm{ref}}) + \\\frac{1}{2} \\cdot f_{1} \\cdot (t-T_{\\mathrm{ref}})^2 +
        \\\frac{1}{6} \\cdot f_{2} \\cdot (t-T_{\\mathrm{ref}})^3 + \\cdots,

    where :math:`T_{\\mathrm{ref}` is the reference time, :math:`f_{0}`, :math:`f_{1}`, :math:`f_{2}`, ...,
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
        The calculated :math:`\\chi^2` array
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

@njit(parallel=True, nogil=True)
def cal_2dchisquare(data, f, F1, pepoch, nbins, F2=0, F3=0, F4=0):
    """
    Calculate the chisquare distribution for 2-D frequency search on the pepoch time.
    For example, search in a two-dimensianal parameter space (:math:`M \\\times N`, as :math:`M`-length frequency
    array, and :math:`N`-length frequency derivative array).

    .. math::
        \\chi^2 = f_{0} \\cdot (t-T_{\\mathrm{ref}}) + \\\frac{1}{2} \\cdot f_{1} \\cdot (t-T_{\\mathrm{ref}})^2 +
        \\\frac{1}{6} \\cdot f_{2} \\cdot (t-T_{\\mathrm{ref}})^3 + \\cdots,

    where :math:`T_{\\mathrm{ref}` is the reference time, :math:`f_{0}`, :math:`f_{1}`, :math:`f_{2}`, ...,
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
        An :math:`M \\\times N` array, as :math:`M` is the length of frequency `f`, :math:`N` is the
        length of frequency derivative `F1`
    """

    chi_square = np.zeros(len(f)*len(F1), dtype=np.float64).reshape(len(F1), len(f))
    # n X m matrix, with n F1 and m f for each F1
    print(chi_square.shape, len(F1))

    t0 = pepoch

    for i in prange(len(F1)):
        for j in prange(len(f)):
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
    """Parse search parameters provided either as a parfile or keyword args."""

    _parameters_legal(kwargs)

    if "parfile" in kwargs:
        return _get_parameters_from_parfile(kwargs)

    pepoch = kwargs["pepoch"]
    f0_mid = _safe_float(kwargs["f0"])
    f0 = _build_frequency_grid(f0_mid, kwargs)

    f1, f1_flag = _build_search_grid("f1", kwargs)

    higher_orders = [
        _safe_float(kwargs.get(key, 0.0))
        for key in ("f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12")
    ]

    pepoch = _normalize_pepoch(pepoch, kwargs.get("pepochformat"))

    return (
        pepoch,
        f0,
        f1,
        *higher_orders,
        f1_flag and isinstance(f1, np.ndarray) and f1.size > 1,
    )


def _safe_float(value):
    if isinstance(value, np.ndarray):
        arr = value.astype(float)
        if arr.size == 1:
            return float(arr)
        return arr
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float)
        if arr.size == 1:
            return float(arr)
        return arr
    return float(value)


def _build_frequency_grid(f0_mid, kwargs):
    step = kwargs.get("f0step") or kwargs.get("f0_step")
    nstep = kwargs.get("f0_nstep")
    span = kwargs.get("f0range") or kwargs.get("f0_range")

    if step and nstep:
        return np.arange(f0_mid - nstep * step, f0_mid + nstep * step, step, dtype=float)
    if step and span:
        return np.arange(f0_mid - span, f0_mid + span, step, dtype=float)
    return np.asarray([f0_mid], dtype=float)


def _build_search_grid(name, kwargs):
    base = kwargs.get(name)
    if base is None:
        return 0.0, False

    base_value = _safe_float(base)
    if isinstance(base_value, np.ndarray):
        return base_value, base_value.size > 1

    step = kwargs.get(f"{name}step") or kwargs.get(f"{name}_step")
    nstep = kwargs.get(f"{name}_nstep")
    span = kwargs.get(f"{name}range") or kwargs.get(f"{name}_range")

    if step and nstep:
        grid = np.arange(base_value - nstep * step, base_value + nstep * step, step, dtype=float)
        return grid, grid.size > 1
    if step and span:
        grid = np.arange(base_value - span, base_value + span, step, dtype=float)
        return grid, grid.size > 1

    return base_value, False


def _normalize_pepoch(pepoch, fmt):
    if fmt is None:
        return pepoch
    fmt = fmt.lower()
    if fmt == "met":
        return pepoch
    if fmt == "mjd":
        return mjd2met(pepoch)
    raise IOError(f"pepoch format {fmt} not supported")


def _get_parameters_from_parfile(kwargs):
    from tatpulsar.utils.timingmodel import TimingModel

    model = TimingModel(kwargs["parfile"])
    freq_names = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"]
    freq_values = []
    for name in freq_names:
        attr = getattr(model, name, None)
        if attr is None:
            freq_values.append(0.0)
        else:
            value = getattr(attr, "value", attr)
            freq_values.append(float(value))

    pepoch = getattr(model, "reftime", None)
    pepoch = _normalize_pepoch(pepoch, kwargs.get("pepochformat"))

    return (pepoch, *freq_values, False)

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
    sys.stdout.write("%\\r")
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
