#!/usr/bin/env python
import numpy as np
import numba 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pulsar_timing.utils import *
from pulsar_timing.Profile import resampling_profile, norm_profile

import sys

__all__ = ['cal_toa']

def cal_toa(fbest, profile, data, method="max", error_method="default", 
        fig_flag=False, std_pro='', **kwargs):
    """
    Calculate the Time of Arrival (ToA) of profile

    Parameters
    ------------
    fbest : float
        The best frequency at the moment t0, where t0 is 
        the reference time for calculating the profile.

    profile : array
        The Profile to calculate the ToA

    data : array
        The time array of the data that generated the profile
        #TODO:Is this really necessary??
        
    method : str, optional
        The method to get to ToA value, default is "max".
        a list of method 
        {"max", "ccf", "gauss_fit", "lorentz_fit"} stands for
        1. get the maximum of profile as ToA phase
        2. cross-correlation to calculate the ToA phase. A standard profile 
            with high statistic should be assigned at the same time. see parameter
            "std_pro" for detail
        3. fit the profile with gaussian function to find the maximum
            position as the ToA phase. a phase range for fitting should
            be assigned. see parameter "fitting_range" for detail.
        4. fit the profile with lorentz function to find the maximum
            position as the ToA phase. a phase range for fitting should
            be assigned. see parameter "fitting_range" for detail.

    error_method : str, optional
        The method to estimate the ToA error, default is to calculate the error 
        based on different ToA method. 
        a list of available method
        {"default", "simulate"}
        1. default
        2. simulate the error by resample the profile. Parameter "sample_num" must be 
            assigned as well.

    std_pro : array, optional
        The standard profile with high statistic for CCF.


    Returns
    ------------
    toa : float
        The Time of Arrival of observed Profile

    toa_err : float
        The error of the obtained ToA, 

    """
    
    ###############################################
    ## ----------- Calculate ToA
    ###############################################

    ## Method is to carry out Cross-correlation Function with 
    ## high statistical Standar Profile to find the maximum phase
    ## as ToA.

    if method == "ccf":
        ## ccf shift
        print("Method to find ToA ... Cross-Correlation function")
        delta_phi = _calculate_delta_phi_by_ccf(profile, std_pro)

    ## plot normalized Observed Profile and normalized Standard Profile

        if fig_flag:
            plt.errorbar(np.linspace(0,1,len(profile)), norm_profile(profile))
            plt.errorbar(np.linspace(0,1,len(std_pro)), norm_profile(std_pro), color='green')
            plt.axvline(x=delta_phi, lw=1.2, color='red', label="CCF ToA Phi")
            plt.legend()
            plt.show()


    ## -----------------
    ## Method is to find the maximum phase as ToA Phase

    elif method == "max":
        print("Method to find ToA ... maximum")
        delta_phi = np.argmax(profile)/len(profile)
        if fig_flag:
            plt.errorbar(np.linspace(0,1,len(profile)), norm_profile(profile), 
                    drawstyle='steps-mid')
            plt.show()


    ## -----------------
    ## Method is to fit observed profile with Gaussian Function
    ## A fitting_range should be assigned (0-2 phase for two period)
    ## e.g. the peak phase is 0.99, a fitting range could be [0.9, 1.1]

    elif method == "gauss_fit":
        print("Method to find ToA ... Gaussian")
        if 'fitting_range' not in kwargs:
            raise IOError("parameter 'fitting_range' for gassian fitting not assigned")
        fitting_range = kwargs['fitting_range']
        x = np.linspace(0, 1, len(profile))
        if fitting_range[1] >= 1:  # if phase range larger than 1
            profile_to_fit = np.append(profile, profile) # duplicate the profile by two period
            x_to_fit       = np.append(x, x+1)
        else:
            profile_to_fit = profile
            x_to_fit       = x

        profile_to_fit = profile_to_fit[(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #profile at fitting range
        x_to_fit       = x_to_fit      [(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #phase   at fitting range

        # fit the profile using gaussian function
        gauss_popt,gauss_pcov = curve_fit(Gauss, x_to_fit, profile_to_fit, 
                p0=[np.max(profile_to_fit),
                    (np.max(x_to_fit)+np.min(x_to_fit))/2,
                    np.max(x_to_fit)-np.min(x_to_fit)],
                maxfev=9999999, sigma=np.sqrt(profile_to_fit), absolute_sigma=False)
        gauss_fitting_errs = np.sqrt(np.diag(gauss_pcov))
        delta_phi = gauss_popt[1]
        gauss_popt[2] = abs(gauss_popt[2]) # For some reason, the value of sigma is negative and forcibly fixed to a positive value

        if fig_flag:
            plt.errorbar(x, profile, yerr=np.sqrt(profile), color='black', label='Raw Data')
            plt.errorbar(x_to_fit, profile_to_fit, color='orange', label='data to fit')
            plt.errorbar(x_to_fit, Gauss(x_to_fit, *gauss_popt), color='red', lw=1.5, label="gaussian")
            plt.legend()
            plt.show()

    ## -----------------
    ## Method is to fit observed profile with Lorentz Function
    ## A fitting_range should be assigned (0-2 phase for two period)
    ## e.g. the peak phase is 0.99, a fitting range could be [0.9, 1.1]

    elif method == "lorentz_fit":
        print("Method to find ToA ... Lorentz")

        if 'fitting_range' not in kwargs:
            raise IOError("parameter 'fitting_range' for lorentzian fitting not assigned")
        fitting_range = kwargs['fitting_range']
        x = np.linspace(0, 1, len(profile))
        if fitting_range[1] >= 1:  # if phase range larger than 1
            profile_to_fit = np.append(profile, profile) # duplicate the profile by two period
            x_to_fit       = np.append(x, x+1)
            error_to_fit   = np.sqrt(profile_to_fit)
        else:
            profile_to_fit = profile
            error_to_fit   = np.sqrt(profile)
            x_to_fit       = x

        profile_to_fit = profile_to_fit[(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #profile at fitting range
        error_to_fit   = error_to_fit  [(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #profile at fitting range
        x_to_fit       = x_to_fit      [(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #phase   at fitting range

        # fit the profile using gaussian function
        popt,pcov = curve_fit(Lorentz, x_to_fit, profile_to_fit, 
                p0=[np.max(profile_to_fit),
                    (np.max(x_to_fit)+np.min(x_to_fit))/2,
                    0.1],
                sigma=error_to_fit, absolute_sigma=True, 
                maxfev=99999)
        lorentz_fitting_errs = np.sqrt(np.diag(pcov))
        delta_phi = popt[1]

        if fig_flag:
            plt.plot(x, profile)
            plt.plot(x_to_fit, profile_to_fit, color='black')
            plt.plot(x_to_fit, Gauss(x_to_fit, *popt), color='red')
            plt.show()

    else:
        raise IOError("the method {} does not supported".format(method))

    if 'fig_flag' in kwargs:
        if kwargs["fig_flag"]:
            plt.plot((profile-np.min(profile))/(np.max(profile)-np.min(profile)))
#            plt.plot((p_num_std-np.min(p_num_std))/(np.max(p_num_std)-np.min(p_num_std)), color='red')
            plt.show()

    ## Calculate the ToA by 
    ## ToA = t0 + P*\delta\Phi
    ## IMPORTANT: the profile should be folded by fbest, which is the 
    ## frequency at the PEPOCH (np.min(data)

    toa = (1/fbest)*delta_phi + np.min(data)



    ###############################################
    ## ----------- Calculate Error
    ###############################################

    ## -----------
    ## 
    ## -----------
    if error_method == "simulate":
        print("Bootstrap to calculate ToA error ... yes")
        if 'sample_num' not in kwargs:
            raise IOError("parameter 'sample_num' for profile resampling not given")
        if kwargs['sample_num'] == 1:
            print("Sampling number ... {}".format(kwargs['sample_num']))
            ## if sample_num == 1, only one profile generated
            resampled_profile  = resampling_profile(profile, kwargs['sample_num'])
            ## calculate ToAs for new profile 
            if method == "ccf":
                resampled_delta_phi = _calculate_delta_phi_by_ccf(resampling_profile,
                        std_pro)
            else:#TODO:method for other not finished
                resampled_delta_phi = np.argmax(resampled_profile)/len(resampled_profile)
        else:
            print("Sampling number ... {}".format(kwargs['sample_num']))
            resampled_profiles = resampling_profile(profile, kwargs['sample_num'])

            resampled_delta_phi = np.array([])
            ## calculate ToAs for new profile or profiles
            if method == "ccf":
                for resampled_profile in resampled_profiles:
                    resampled_delta_phi = np.append(resampled_delta_phi,
                            _calculate_delta_phi_by_ccf(resampled_profile,std_pro))
            else:#TODO:method for other not finished
                for resampled_profile in resampled_profiles:
                    resampled_delta_phi = np.append(resampled_delta_phi,
                            np.argmax(resampled_profile)/len(resampled_profile))

            resampled_toa = (1/fbest)*resampled_delta_phi + np.min(data)

        ## Calculate the 1sigma distribution of resampled ToA as 1 sigma error
        #toa_err = 1e6*_get_error_quantiles(resampled_delta_phi, delta_phi) ## Unit microsecond
        toa_err = 1e6*_get_error_rms(resampled_toa, toa) ## Unit microsecond

    elif error_method == "default":
        print("Bootstrap to calculate ToA error ... no")

        #TODO: method max and ccf Error are not accurate yet!
        if method == 'max':
            toa_err = (np.max(data)-np.min(data))/len(profile)/10

        elif method == "ccf":
            toa_err = (np.max(data)-np.min(data))/len(profile)/10

        elif method == 'gauss_fit':
            print("Calculate ToA error ... based on Profile shape (Lorimer & Kramer 2012)")
            period = 1/fbest
            peak_sigma = gauss_popt[2]
            
            N_background  =  np.min(profile) * len(profile_to_fit)
            N_source = np.sum( profile_to_fit ) - N_background
            toa_err = _get_error_by_profile_shape(period, peak_sigma, N_source, N_background)
            toa_err = 1e6* toa_err # in Unit of microsecond

    elif error_method == "gauss_fit":
        print("Calculate ToA error ... fitting")
        print(gauss_fitting_errs)
        toa_err = gauss_fitting_errs[1] ## Error is the error of phase x0
        toa_err = 1e6*toa_err/fbest ## Convert to time (Unit of microsec)

    else: raise IOError("The error_method {} not supported")

    toa = met2mjd(toa, telescope=kwargs['telescope'])
    return toa, toa_err

@numba.njit
def _get_error_quantiles(data, toa, low=0.16, mid=0.5, hig=0.84):
    """
    The error of 1sigma is obtained from 
    the posterior sampling distribution of ToA
    """

    ##TODO:pass parameter 
    data_low = sorted(data)[int(low*len(data))]
    data_mid = sorted(data)[int(mid*len(data))]
    data_hig = sorted(data)[int(hig*len(data))]
    if data_low == data_hig:
        return 0
    else:
        return max(abs(data_low-data_mid), abs(data_hig-data_mid))

def _get_error_rms(toas, toa):
    """
    The error of 1sigma is obtained from 
    the deviation between RMS of ToA and measured ToA
    """
    rms = np.sqrt(np.mean(toas**2))
    return abs(toa - rms)


def _get_error_by_profile_shape(period, peak_sigma, source_counts, background_counts):
    """
    Deneva et al 2019 equation 1

    (Lorimer & Kramer 2012)
    """
    a = period*peak_sigma
    b = source_counts/np.sqrt(source_counts+background_counts) 
    sigma_ToA = a/b
    return sigma_ToA


@numba.njit
def _calculate_delta_phi_by_ccf(profile, profile_std):
    #profile = np.append(profile, profile)
    #profile_std = np.append(profile_std, profile_std)
    y, delay = ccf(profile, profile_std)
    p_num_std = np.roll(profile_std, delay)
    p_num_x = np.linspace(0,1, len(profile))
    delta_phi = p_num_x[np.argmax(p_num_std)]
    return delta_phi
