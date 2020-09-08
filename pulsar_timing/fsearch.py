#!/usr/bin/env python
import numpy as np
import numba 
import matplotlib.pyplot as plt
from pulsar_timing.utils import *

__all__ = ['fsearch']

def cal_toa(fbest, profile, data):
    delta_phi = np.argmax(profile)/len(profile)
    toa = (1/fbest)*delta_phi + np.min(data)
    toa = (toa / 86400.0) + MJDREFI + MJDREFF
    #TODO:ToA error
    return toa

def _get_parameters(kwargs):
    """
    get the parameters for searching
    
    The format of input could be a name of parfile, a dictionary, or standard python function arguments.

    """
    #read parfile and parameters
    if "parfile" in kwargs:
        pass
        #TODO read from parfile
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
                pass
            if kwargs['pepochformat'].lower() == "mjd":
                pepoch = mjd2met(pepoch)
            else:
                raise IOError(f"pepoch format {kwargs['pepochformat']} not supported")
    return pepoch, F0, F1, F2, F3, F4, f1search_flag


def fsearch(data, **kwargs):
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

    fstep : float 
        

    f1 : floatj(optional)
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
            MJDREFF = 0.00074287037037037
            MJDREFI = 51910
        elif kwargs['telescope'].lower() == "hxmt":
            MJDREFF = 0.00076601852000000
            MJDREFI = 55927
    else:
        MJDREFF = 0.00076601852000000
        MJDREFI = 55927

    #read parfile and parameters
    pepoch, F0, F1, F2, F3, F4, F1_searchflag = _get_parameters(kwargs)
    F1_sample = (np.max(F1) + np.min(F1))/2
    
    # deduce the timing parameters at the middle of the time series
    t0 = np.min(data) + np.max(data)/2
    dt = t0 - pepoch
    F0 = F0 + F1_sample*dt + (1/2)*F2*(dt**2) + (1/6)*F3*(dt**3) + (1/24)*F4*(dt**4)
    F1 = F1 + F2*dt + (1/2)*F3*(dt**2) + (1/6)*F4*(dt**3)
    F2 = F2 + F3*dt + (1/2)*F4*(dt**2)
    F3 = F3 + F4*dt
    
    if len(data)==0:
        raise IOError("Error: Data is empty")
    if 'bin' in kwargs:
        bin_profile = kwargs['bin']
    else:
        bin_profile = 20

    if F1_searchflag:
        ## F0 and F1 2-D search
        chi_square = cal_2dchisquare(data, F0, t0, bin_profile, F1, F2, F3, F4)
        f_f1_index = np.unravel_index(np.argmax(chi_square, axis=None), chi_square.shape)
        fbest = F0[f_f1_index[1]]
        f1best = F1[f_f1_index[0]]
    else:
        print(f"2D search is {F1_searchflag}")
        ## F0 1-D search
        chi_square = cal_chisquare(data, F0, t0, bin_profile, F1, F2, F3, F4)
        fbest = F0[np.argmax(chi_square)]
        f1best = F1

    phi = (data-t0)*fbest + (1.0/2.0)*((data-t0)**2)*f1best + (1.0/6.0)*((data-t0)**3)*F2 +\
            (1.0/24.0)*((data-t0)**4)*F3 + (1.0/120.0)*((data-t0)**5)*F4
    phi = phi - np.floor(phi)
    profile, phase  = numba_histogram(phi, bin_profile)

    return {"T0": met2mjd(t0), "ChiSquare" : chi_square, "Profile" : profile, "F0": fbest, "F1":f1best, "F0_init":F0, "F1_init":F1, 
            "F2_init" : F2, "F3_init":F3, "F4_init":F4}



if __name__ == "__main__":
    from astropy.io import fits
    #TESTING
    f0 = 29.946923
    f1 = -3.77535E-10
    f2 = 1.1147E-20
    f3 = -2.73E-30
    filename = "/Users/tuoyouli/Work/Fermi/script/fermi_pipeline/data/weekly_w009_p305_v001_gtbary.fits"
    hdulist =fits.open(filename)
    time = hdulist[1].data.field("TIME")
    mjd, chi2, pro, pars = fsearch(time, f0=f0, f0step=1e-8, f0range=1e-4, pepoch=48442.5, pepochformat='mjd', 
            f1=f1, f1step=1e-14, f1range=3e-14, f2=f2, f3=f3, bin=20)
    plt.imshow(chi2, aspect='auto', origin='lower', extent=[-1e-6, 1e-6, -1e-14, 1e-14])
    print(pars)
    plt.show()

