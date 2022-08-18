#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import argparse

__all__ = ["cal_residual",
        "cal_residual_from_parameters",
        "argparse_wildcat",
        "parse_pfiles",
        "read_toa",
        ]

def rms(data, err):
    mean = _weighted_phi_mean(data, err)
    return np.sqrt(np.sum( (data - mean)**2)/(data.size-1))

#read par file
def read_par(parname):
    pardata = open(parname,'r')
    stdpar = []
    parameters = np.zeros(13)
    for par in pardata:
        par = par[0:(len(par)-1)]
        stdpar.append(par)
    pardata.close()
    for i in range(len(stdpar)):
        if stdpar[i][:6]=='PEPOCH':
            PEPOCH_lst = stdpar[i].split(' ');PEPOCH = [x for x in PEPOCH_lst if x != ''][1]
            parameters[0] = np.longdouble(PEPOCH)
        if stdpar[i][:2]=='F0':
            F0_lst = stdpar[i].split(' ');F0 = [x for x in F0_lst if x != ''][1]
            parameters[1] = np.longdouble(F0)
        if stdpar[i][:2]=='F1':
            F1_lst = stdpar[i].split(' ');F1 = [x for x in F1_lst if x != ''][1]
            parameters[2] = np.longdouble(F1)
        if stdpar[i][:2]=='F2':
            F2_lst = stdpar[i].split(' ');F2 = [x for x in F2_lst if x != ''][1]
            parameters[3] = np.longdouble(F2)
        if stdpar[i][:2]=='F3':
            F3_lst = stdpar[i].split(' ');F3 = [x for x in F3_lst if x != ''][1]
            parameters[4] = np.longdouble(F3)
        if stdpar[i][:2]=='F4':
            F4_lst = stdpar[i].split(' ');F4 = [x for x in F4_lst if x != ''][1]
            parameters[5] = np.longdouble(F4)
        if stdpar[i][:2]=='F5':
            F5_lst = stdpar[i].split(' ');F5 = [x for x in F5_lst if x != ''][1]
            parameters[6] = np.longdouble(F5)
        if stdpar[i][:2]=='F6':
            F6_lst = stdpar[i].split(' ');F6 = [x for x in F6_lst if x != ''][1]
            parameters[7] = np.longdouble(F6)
        if stdpar[i][:2]=='F7':
            F7_lst = stdpar[i].split(' ');F7 = [x for x in F7_lst if x != ''][1]
            parameters[8] = np.longdouble(F7)
        if stdpar[i][:2]=='F8':
            F8_lst = stdpar[i].split(' ');F8 = [x for x in F8_lst if x != ''][1]
            parameters[9] = np.longdouble(F8)
        if stdpar[i][:2]=='F9':
            F9_lst = stdpar[i].split(' ');F9 = [x for x in F9_lst if x != ''][1]
            parameters[10] = np.longdouble(F9)
        if stdpar[i][:5]=='START':
            START_lst = stdpar[i].split(' ');START = [x for x in START_lst if x != ''][1]
            parameters[11] = np.longdouble(START)
        if stdpar[i][:6]=='FINISH':
            FINISH_lst = stdpar[i].split(' ');FINISH = [x for x in FINISH_lst if x != ''][1]
            parameters[12] = np.longdouble(FINISH)

    print("...finish reading ephemeris file... {}".format(parname))
    return parameters


def read_toa(timname):
    '''
    read the standard TEMPO2 format \*.tim
    ToA file and return the ToA and ToA errors
    '''
    f = open(timname)
    toa = np.array([])
    err = np.array([])
    for line in f.readlines():
        if line.startswith('C ') or line.startswith('FORMAT'):
            continue
        else:
            toa = np.append(toa, np.float128(line.split()[2]))
            err  = np.append(err,  np.float128(line.split()[3]))
    f.close()
    return toa, err

def _weighted_phi_mean(phi, errs, default=0):
    if phi.size == 0:return default
    return np.sum(phi/errs**2/np.sum(1/errs**2))


def cal_phi(toas, *F_all_set, PEPOCH=0, phi0=0):
    dt = (toas - PEPOCH)*86400

    phi = np.sum(
            np.array([ (1/math.factorial(i+1))*(dt**(i+1))*F_set_array[i] for i in range(len(F_set_array))]),
            axis=0) - phi0
    phi = phi - np.floor(phi)

    return phi


def get_index_in_eph(time, eph_start, eph_stop):
    """
    get the index in eph file
    if not match return none
    """
    for i in range(len(eph_start)):
        if (time >= eph_start[i]) & (time <= eph_stop[i]):
            return i
        else:
            continue
    return -1

def get_f(time, t0, f0, f1, f2, telescope='hxmt'):
    dt = (time - t0)*86400
    f = f0 + f1*dt + (1/2)*f2*(dt**2)
    return f

def argparse_wildcat(pfile_args):
    if ('*' in pfile_args) or ('[' in pfile_args) or (']' in pfile_args) or \
            ('?' in pfile_args):
        pfiles = glob.glob(pfile_args)
        return pfiles
    else:
        return pfile_args

def parse_pfiles(pfiles_args):
    if isinstance(pfiles_args, list):

        start_time = np.array([])
        stop_time  = np.array([])
        PEPOCH_all = np.array([])
        f0_all     = np.array([])
        f1_all     = np.array([])
        f2_all     = np.array([])
        f3_all     = np.array([])
        f4_all     = np.array([])
        f5_all     = np.array([])
        f6_all     = np.array([])
        f7_all     = np.array([])
        f8_all     = np.array([])
        for eph in pfiles_args:
            eph_data = read_par(eph)
            PEPOCH_all = np.append(PEPOCH_all, eph_data[0])
            f0_all     = np.append(f0_all,     eph_data[1])
            f1_all     = np.append(f1_all,     eph_data[2])
            f2_all     = np.append(f2_all,     eph_data[3])
            f3_all     = np.append(f3_all,     eph_data[4])
            f4_all     = np.append(f4_all,     eph_data[5])
            f5_all     = np.append(f5_all,     eph_data[6])
            f6_all     = np.append(f6_all,     eph_data[7])
            f7_all     = np.append(f7_all,     eph_data[8])
            f8_all     = np.append(f8_all,     eph_data[9])
            f_set_all  = np.array([f0_all, f1_all, f2_all, f3_all, f4_all, f5_all, f6_all, f7_all, f8_all])
            start_time = np.append(start_time, eph_data[-2])
            stop_time  = np.append(stop_time,  eph_data[-1])
    else:
        eph_data = read_par(pfiles_args)
        PEPOCH_all = eph_data[0]
        f0_all     = eph_data[1]
        f1_all     = eph_data[2]
        f2_all     = eph_data[3]
        f3_all     = eph_data[4]
        f4_all     = eph_data[5]
        f5_all     = eph_data[6]
        f6_all     = eph_data[7]
        f7_all     = eph_data[8]
        f8_all     = eph_data[9]
        f_set_all  = np.array([f0_all, f1_all, f2_all, f3_all, f4_all, f5_all, f6_all, f7_all, f8_all])
        start_time = eph_data[-2]
        stop_time  = eph_data[-1]
    return f_set_all, PEPOCH_all, start_time, stop_time

def cal_residual(toas, toa_errs, f_set_all, PEPOCH_all, start_time, stop_time, inperiod=False):
    """
    calculate the residuals for toas in each Ephemeride

    return the residuals as one
    """
    residuals = np.array([])
    residual_err = np.array([])
    residuals_toas = np.array([])

    for i in range(len(start_time)):
        toas_in_eph = toas    [(toas>=start_time[i]) & (toas<stop_time[i]) ]
        errs_in_eph = toa_errs[(toas>=start_time[i]) & (toas<stop_time[i]) ]
        f_set_in_eph = [x[i] for x in f_set_all]
        dt = (toas_in_eph - PEPOCH_all[i])*86400

        phi = np.sum(
                np.array([ (1/math.factorial(j+1))*(dt**(j+1))*f_set_in_eph[j] for j in range(len(f_set_in_eph))]),
                axis=0)
        phi -= np.floor(phi)
        f0_at_toa = np.sum(
                np.array([(1/math.factorial(j))*(dt**j)*f_set_in_eph[j] for j in range(len(f_set_in_eph))]),
                axis=0)
        phi0 = _weighted_phi_mean(phi, errs_in_eph/1e6*f0_at_toa)

        if inperiod:
            # residuals calculate in pulse periods
            residuals = np.append(
                    residuals, phi-phi0)
            residual_err = np.append(
                    residual_err, (errs_in_eph/1e6*f0_at_toa))
        else:
            # residuals calculate in time (sec)
            residuals = np.append(
                    residuals, (phi-phi0)/f0_at_toa)
            residual_err = np.append(
                    residual_err, errs_in_eph/1e6)

        residuals_toas = np.append(residuals_toas, toas_in_eph)

    ## Calculate RMS of total residuals
    if inperiod:
        residuals_rms = rms(residuals, residual_err)/f_set_in_eph[0]
        #NOTE: when the residuals are calculated in period, the rms (in sec) is not accurate.
    else:
        residuals_rms = rms(residuals, residual_err)

    return residuals_toas, residuals, residual_err, residuals_rms

def cal_residual_from_parameters(toas, toa_errs, F_set_array, PEPOCH, inperiod=False):
    """
    calculate the residuals for toas in one set of Ephemeris parameters
    return the residuals

    Parameters
    ----------
    toas : array-like
        ToAs in MJD unit

    toa_errs : array-float
        ToAs error in microsecond unit

    F_set_array : array-float
        An array of f0, f1, f2, ... etc.

    PEPOCH : float
        reference time in MJD unit

    inperiod : bool
        flag to calculate residuals in period unit

    Returns
    -------
    residuals : array-like
        The timing residuals in either period or time
    residual_err : array-like
        The error of timing residuals
    residuals_rms : array-like
        The root-mean-square of total ToA set from the weighted mean ToA values:

        .. math::
            \sqrt{\sum \\frac{(T - <T>)^2}{N-1}},

        where :math:`T` is the ToAs, and :math:`<T>` is the weighted mean value of ToAs
    """
    residuals = np.array([])
    residual_err = np.array([])

    dt = (toas - PEPOCH)*86400

    phi = np.sum(
            np.array([ (1/math.factorial(j+1))*(dt**(j+1))*F_set_array[j] for j in range(len(F_set_array))]),
            axis=0)
    phi -= np.floor(phi)
    f0_at_toa = np.sum(
            np.array([(1/math.factorial(j))*(dt**j)*F_set_array[j] for j in range(len(F_set_array))]),
            axis=0)
    phi0 = _weighted_phi_mean(phi, toa_errs/1e6*f0_at_toa)

    if inperiod:
        # residuals calculate in pulse periods
        residuals = np.append(
                residuals, phi-phi0)
        residual_err = np.append(
                residual_err, (toa_errs/1e6*f0_at_toa))
    else:
        # residuals calculate in time (sec)
        residuals = np.append(
                residuals, (phi-phi0)/f0_at_toa)
        residual_err = np.append(
                residual_err, toa_errs/1e6)

    ## Calculate RMS of total residuals
    if inperiod:
        residuals_rms = rms(residuals, residual_err)/F_set_array[0]
        #NOTE: when the residuals are calculated in period, the rms (in sec) is not accurate.
    else:
        residuals_rms = rms(residuals, residual_err)

    return residuals, residual_err, residuals_rms


