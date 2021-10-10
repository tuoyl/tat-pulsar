#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings


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
        "Gauss", "Lorentz",
        "rms",
        "print_loop_percentage"]

#@njit
def met2mjd(data, telescope="fermi"):
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
    return data/86400 + MJDREFI + MJDREFF

#@njit
def mjd2met(data, telescope="fermi"):
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
def cal_chisquare(data, f, pepoch, bin_profile, F1, F2, F3, F4, parallel=False):
    """
    calculate the chisquare distribution for frequency search on the pepoch time.
    """
    chi_square = np.zeros(len(f), dtype=np.float64)

    t0 = pepoch

    if parallel:
        for i in prange(len(f)):
            phi = (data-t0)*f[i] + (1.0/2.0)*((data-t0)**2)*F1 + (1.0/6.0)*((data-t0)**3)*F2 +\
                    (1.0/24.0)*((data-t0)**4)*F3 + (1.0/120.0)*((data-t0)**5)*F4
            phi = phi - np.floor(phi)
            #counts  = numba_histogram(phi, bin_profile)[0]
            #NOTE: The histogram bin should give the edge of bin, instead of the bin number.
            #NOTE: For those pulse with narrow peak, it will be incorrect while calculate the chisquare
            counts  = np.histogram(phi,
                    np.linspace(0, 1, bin_profile+1)[:-1])[0]
            expectation = np.mean(counts)
            chi_square[i] = np.sum( (counts - expectation)**2 / expectation )
    else:
        for i in range(len(f)):
            phi = (data-t0)*f[i] + (1.0/2.0)*((data-t0)**2)*F1 + (1.0/6.0)*((data-t0)**3)*F2 +\
                    (1.0/24.0)*((data-t0)**4)*F3 + (1.0/120.0)*((data-t0)**5)*F4
            phi = phi - np.floor(phi)
            #counts  = numba_histogram(phi, bin_profile)[0]
            #NOTE: The histogram bin should give the edge of bin, instead of the bin number.
            #NOTE: For those pulse with narrow peak, it will be incorrect while calculate the chisquare
            counts  = np.histogram(phi,
                    np.linspace(0, 1, bin_profile+1)[:-1])[0]
            expectation = np.mean(counts)
            chi_square[i] = np.sum( (counts - expectation)**2 / expectation )

    return chi_square

@njit
def cal_2dchisquare(data, f, pepoch, bin_profile, F1, F2, F3, F4):
    """
    calculate the chisquare distribution for 2-D frequency search on the pepoch time.
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
            counts  = numba_histogram(phi, bin_profile)[0]
            chi_square[i][j] = np.sum( (counts - np.mean(counts))**2 / np.mean(counts) )
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
    f1 is the original signal
    f2 is probe signal(shift and test)
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

@njit
def print_loop_percentage(iterator_i, total, printstr=''):
    percent = iterator_i*100/total
    sys.stdout.write("{} complete: {:.2f}".format(printstr, percent))
    sys.stdout.write("%\r")
    sys.stdout.flush()

def Gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def Lorentz(x, amp, cen, wid):
    return (amp*wid**2/((x-cen)**2+wid**2))

def rms(x):
    return np.sqrt(x.dot(x)/x.size)
