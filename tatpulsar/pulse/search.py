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
        pardata = open(parname,'r')
        stdpar = []
        for par in pardata:
            par = par[0:(len(par)-1)]
            stdpar.append(par)
        pardata.close()
        for i in range(len(stdpar)):
            if stdpar[i][:6]=='PEPOCH':
                PEPOCH_lst = stdpar[i].split(' ');PEPOCH = [x for x in PEPOCH_lst if x is not ''][1]
                pepoch = mjd2met(np.longdouble(PEPOCH))
            if stdpar[i][:2]=='F0':
                F0_lst = stdpar[i].split(' ');F0 = [x for x in F0_lst if x is not ''][1]
                F0 = np.longdouble(F0)
            if stdpar[i][:2]=='F1':
                F1_lst = stdpar[i].split(' ');F1 = [x for x in F1_lst if x is not ''][1]
                F1 = np.longdouble(F1)
            if stdpar[i][:2]=='F2':
                F2_lst = stdpar[i].split(' ');F2 = [x for x in F2_lst if x is not ''][1]
                F2 = np.longdouble(F2)
            if stdpar[i][:2]=='F3':
                F3_lst = stdpar[i].split(' ');F3 = [x for x in F3_lst if x is not ''][1]
                F3 = np.longdouble(F3)
            if stdpar[i][:2]=='F4':
                F4_lst = stdpar[i].split(' ');F4 = [x for x in F4_lst if x is not ''][1]
                F4 = np.longdouble(F4)
            if stdpar[i][:2]=='F5':
                F5_lst = stdpar[i].split(' ');F5 = [x for x in F5_lst if x is not ''][1]
                F5 = np.longdouble(F5)
            if stdpar[i][:2]=='F6':
                F6_lst = stdpar[i].split(' ');F6 = [x for x in F6_lst if x is not ''][1]
                F6 = np.longdouble(F6)
            if stdpar[i][:2]=='F7':
                F7_lst = stdpar[i].split(' ');F7 = [x for x in F7_lst if x is not ''][1]
                F7 = np.longdouble(F7)
            if stdpar[i][:2]=='F8':
                F8_lst = stdpar[i].split(' ');F8 = [x for x in F8_lst if x is not ''][1]
                F8 = np.longdouble(F8)
            if stdpar[i][:2]=='F9':
                F9_lst = stdpar[i].split(' ');F9 = [x for x in F9_lst if x is not ''][1]
                F9 = np.longdouble(F9)
            if stdpar[i][:5]=='START':
                START_lst = stdpar[i].split(' ');START = [x for x in START_lst if x is not ''][1]
                TSTART = np.longdouble(START)
            if stdpar[i][:6]=='FINISH':
                FINISH_lst = stdpar[i].split(' ');FINISH = [x for x in FINISH_lst if x is not ''][1]
                TFINISH = np.longdouble(FINISH)
        f1search_flag = False
        return pepoch, F0, F1, F2, F3, F4, f1search_flag

    else:
        pepoch = kwargs['pepoch']
        F0_mid     = kwargs['f0']
        F0_step    = kwargs['f0step']
        F0_range    = kwargs['f0range']
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
                    print(f"number of parameters to search is {len(F1)*len(F0)}")
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

        if "pepochformat" in kwargs:
            if kwargs['pepochformat'].lower() == "met":
                pepoch = pepoch
            elif kwargs['pepochformat'].lower() == "mjd":
                pepoch = mjd2met(pepoch)
            else:
                raise IOError(f"pepoch format {kwargs['pepochformat']} not supported")
    return pepoch, F0, F1, F2, F3, F4, f1search_flag

def search(data, **kwargs):
    """
    search the best frequency

    Parameters
    ----------
    data : array-like
        the time series of TDB data

    parfile : str, optional
        read the parameters from parfile

    f0 : float
        the init f0 value to search

    pepoch : float, optional
        time for input frequecy values.

        .. note::
            the output frequecy is the frequency at which
            the middle of each time interval

    f0step : float
        The step length for frequencies to search

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

    f1step : float, optional
        The step length to search the frequency derivate as well. Only works if ``f1step``
        and ``f1range`` are both given

    f1range : float, optional
        the fdot search step range

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

    chi_square : dictionary
        The Chi Square distribution of Epoch folding. The valid keys are

        * ``T0``: the reference time in MJD
        * ``ChiSquare``: the calculated :math:`\chi^2` array (return a multi-dimensinal array if both f0 and f1 are searched)
        * ``Profile``: the counts of profile folded by best search frequency
        * ``Pars``: The dictionary of input and output frequency and frequency derivative {``F0``, ``F1``, ``F0_init``, ``F1_init``, ``F2_init``, ``F3_init``, ``F4_init``}
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
        elif kwargs['telescope'].lower() == 'ixpe':
            telescope = 'ixpe'
    else:
        telescope='hxmt'

    #read parfile and parameters
    pepoch, F0, F1, F2, F3, F4, F1_searchflag = _get_parameters(kwargs)
    F1_sample = (np.max(F1) + np.min(F1))/2

    # deduce the timing parameters at the middle of the time series
    #t0 = (np.min(data) + np.max(data))/2
    #t0 = np.min(data)
    #dt = t0 - pepoch
    #NOTE: t0 should be pepoch!!!
    t0 = pepoch
    dt = 0
    F0 = F0 + F1_sample*dt + (1/2)*F2*(dt**2) + (1/6)*F3*(dt**3) + (1/24)*F4*(dt**4)
    F1 = F1 + F2*dt + (1/2)*F3*(dt**2) + (1/6)*F4*(dt**3)
    F2 = F2 + F3*dt + (1/2)*F4*(dt**2)
    F3 = F3 + F4*dt

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
        #print(f"2D search is {F1_searchflag}")
        ## F0 1-D search
        chi_square = cal_chisquare(data, F0, t0, nbins, F1, F2, F3, F4)
        fbest = F0[np.argmax(chi_square)]
        f1best = F1

    phi = (data-t0)*fbest + (1.0/2.0)*((data-t0)**2)*f1best + (1.0/6.0)*((data-t0)**3)*F2 +\
            (1.0/24.0)*((data-t0)**4)*F3 + (1.0/120.0)*((data-t0)**5)*F4
    phi = phi - np.floor(phi)
    profile = phihist(phi, nbins).counts

    return {"T0": met2mjd(t0, telescope=telescope), "ChiSquare" : chi_square, "Profile" : profile,
            "Pars" : {"F0": fbest, "F1":f1best, "F0_init":F0, "F1_init":F1,
            "F2_init" : F2, "F3_init":F3, "F4_init":F4}}
