#!/usr/bin/env python
import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
from copy import deepcopy
from tqdm import tqdm
from scipy.optimize import curve_fit
from tatpulsar.utils.functions import *
from tatpulsar.utils.functions import gauss as Gauss
from tatpulsar.data.profile import Profile

import sys

__all__ = ['cal_toa']

def cal_toa(fbest, profile, method="max",
            ref_time=None, debug=False, std_pro=None,
            nsteps=100, phi_range=None, **kwargs):
    """
    Calculate the Time of Arrival (ToA) of profile

    Parameters
    ----------
    fbest : float
        The best frequency at the moment t0, where t0 is
        the reference time for calculating the profile.

    profile : array
        The Profile object to calculate the ToA

    ref_time : float, optional
        The ref_time that folded the profile. The ref_time is provided by Profile object if you
        folded using `tatpulsar.pulse.fold` function. Otherwise you must assign.

    debug : bool, optional
        flag to plot the ToA fitting processes for debugging

    method : str, optional
        The method to get to ToA value, default is "max".
        a list of method
        {"max", "taylor_fft", "gauss_fit", "lorentz_fit"} stands for
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

    std_pro : array, optional
        The standard profile with high statistic for CCF.

    nsteps: int
        The steps number to calculate the error by simulation.

    phi_range: list
        The Phase range to find the maximun or to fitting. If the method is fitting,
        the phi_range could not be None.

    Returns
    -------
    toa : float
        The Time of Arrival of observed Profile

    toa_err : float
        The error of the obtained ToA,

    """
    if not isinstance(profile, Profile):
        raise TypeError("The input profile is not tatpulsar.data.Profile object")

    if phi_range is not None:
        mask = (profile.phase >= phi_range[0])&(profile.phase <= phi_range[1])
    else:
        mask = np.ones(profile.counts.size, dtype=bool)

    if method == 'max':
        ph = profile.phase[mask]
        pro = profile.counts[mask]
        delta_phi = ph[np.argmax(pro)]

        ## Calculate the error by profile simulation
        profile_duplicated = np.asarray([pro]*nsteps)
        profile_sim = np.random.poisson(profile_duplicated)
        delta_phis = ph[np.argmax(profile_sim, axis=1)]

        ## TODO: this is highly not safe
        if (delta_phi < 0.25) or (delta_phi >0.75):
            delta_phis[delta_phis<0.5]  = delta_phis[delta_phis<0.5]+1

    elif method == 'fft':
        # Taylor FFT ccf
        p0 = [1, 0]
        ph = profile.phase
        pro = profile.counts
        std_ph = std_pro.phase
        std_pro_masked = std_pro.counts

        res = basinhopping(obj_fun, p0,
                           minimizer_kwargs={'args':([pro, std_pro_masked])})
        amp, shift = res.x

        delta_phi = std_ph[np.argmax(std_pro_masked)] + shift
        delta_phi = delta_phi - np.floor(delta_phi)

        ## Caclualte the error by profile simulation
        profile_duplicated = np.asarray([pro]*nsteps)
        profile_sim = np.random.poisson(profile_duplicated)
        delta_phis = np.array([])
        for p_tmp in tqdm(profile_sim):
            res_tmp = basinhopping(obj_fun, p0,
                                  minimizer_kwargs={'args':([p_tmp, std_pro_masked])})
            amp_tmp, shift_tmp = res_tmp.x
            delta_phi_tmp = std_ph[np.argmax(std_pro_masked)] + shift_tmp
            delta_phi_tmp = delta_phi_tmp - np.floor(delta_phi_tmp)
            delta_phis = np.append(delta_phis, delta_phi_tmp)

    elif method == 'ccf':
        ph = profile.phase
        pro = profile.counts
        std_ph = std_pro.phase
        std_pro_masked = std_pro.counts
        y, shift = ccf(pro, std_pro_masked)

        delta_phi = std_ph[np.argmax(np.roll(std_pro_masked,shift))]

        ## Caclualte the error by profile simulation
        profile_duplicated = np.asarray([pro]*nsteps)
        profile_sim = np.random.poisson(profile_duplicated)
        delta_phis = np.array([])
        for p_tmp in tqdm(profile_sim):
            _, shift_tmp = ccf(p_tmp, std_pro_masked)

            delta_phi_tmp = std_ph[np.argmax(np.roll(std_pro_masked,shift_tmp))]
            delta_phis = np.append(delta_phis, delta_phi_tmp)


    if debug:
        profile_tmp = deepcopy(profile)
        profile_tmp.cycles = 2
        profile_tmp.norm()
        if method == 'fft':
            std_pro_tmp = deepcopy(std_pro)
            std_pro_tmp.cycles = 2
            std_pro_tmp.norm()
            new_ph = std_pro_tmp.phase + shift
            new_ph = new_ph - np.floor(new_ph)
            plt.errorbar(new_ph, std_pro_tmp.counts*amp, std_pro_tmp.error*amp,
                         c='r', ds='steps-mid', label='template profile')
        elif method == 'ccf':
            std_pro_tmp = deepcopy(std_pro)
            std_pro_tmp.cycles = 2
            std_pro_tmp.norm()
            new_ph = std_pro_tmp.phase
            plt.errorbar(new_ph, np.roll(std_pro_tmp.counts, shift),
                                 np.roll(std_pro_tmp.error, shift),
                         c='r', ds='steps-mid', label='template profile')

        plt.errorbar(profile_tmp.phase, profile_tmp.counts,
                     profile_tmp.error, ds='steps-mid', c='k')
        plt.axvline(delta_phi)
        for i in delta_phis:
            plt.axvline(i, lw=0.1)
        if phi_range is not None:
            plt.axvline(phi_range[0], ls='dotted')
            plt.axvline(phi_range[1], ls='dotted')
        plt.axvspan(delta_phi - np.std(delta_phis),
                    delta_phi + np.std(delta_phis), color='r',
                    alpha=0.3)

    if ref_time is None:
        ref_time = profile.ref_time
    if (profile.ref_time is None) & (ref_time is None):
        raise IOError("ref_time is required")


    TOA = (1/fbest)*delta_phi + ref_time
    TOA_err = (1/fbest)*np.std(delta_phis)
    return TOA, TOA_err

def fftfit_fun(profile, template, amplitude, phase):
    '''Objective function to be minimized - \'a la Taylor (1992).'''

    prof_ft = np.fft.fft(profile)
    temp_ft = np.fft.fft(template)
    freq = np.fft.fftfreq(len(profile))
    good = freq > 0
    idx = np.arange(0, prof_ft.size, dtype=int)
    sigma = np.std(prof_ft[good])
    return np.sum(np.absolute(prof_ft - temp_ft*amplitude*np.exp(-2*np.pi*1.0j*idx*phase))**2 / sigma)

def obj_fun(pars, data):
    '''
    Wrap parameters and input data up in order to be used with minimization
    algorithms.
    '''
    amplitude, phase = pars
    profile, template = data
    return fftfit_fun(profile, template, amplitude, phase)

#def cal_toa(fbest, profile, data, method="max", error_method="default",
#        fig_flag=False, std_pro='', t0=None, **kwargs):
#    """
#    Calculate the Time of Arrival (ToA) of profile
#
#    Parameters
#    ----------
#    fbest : float
#        The best frequency at the moment t0, where t0 is
#        the reference time for calculating the profile.
#
#    profile : array
#        The Profile to calculate the ToA
#
#    data : array
#        The time array of the data that generated the profile
#        #TODO:Is this really necessary??
#
#    fig_flag : bool
#        flag to plot the ToA fitting processes for debugging
#
#    t0: float
#        The reference time to calculate the ToA, default is the minimize of data set.
#
#    method : str, optional
#        The method to get to ToA value, default is "max".
#        a list of method
#        {"max", "ccf", "gauss_fit", "lorentz_fit"} stands for
#        1. get the maximum of profile as ToA phase
#        2. cross-correlation to calculate the ToA phase. A standard profile
#            with high statistic should be assigned at the same time. see parameter
#            "std_pro" for detail
#        3. fit the profile with gaussian function to find the maximum
#            position as the ToA phase. a phase range for fitting should
#            be assigned. see parameter "fitting_range" for detail.
#        4. fit the profile with lorentz function to find the maximum
#            position as the ToA phase. a phase range for fitting should
#            be assigned. see parameter "fitting_range" for detail.
#
#    error_method : str, optional
#        The method to estimate the ToA error, default is to calculate the error
#        based on different ToA method.
#        a list of available method
#        {"default", "simulate"}
#        1. default
#        2. simulate the error by resample the profile. Parameter "sample_num" must be
#            assigned as well.
#
#    std_pro : array, optional
#        The standard profile with high statistic for CCF.
#
#
#    Returns
#    -------
#    toa : float
#        The Time of Arrival of observed Profile
#
#    toa_err : float
#        The error of the obtained ToA,
#
#    """
#
#    # default t0 is minimize time of data set
#    if t0 is None:
#        t0 = np.min(data)
#
#
#    ###############################################
#    ## ----------- Calculate ToA
#    ###############################################
#
#    ## Method is to carry out Cross-correlation Function with
#    ## high statistical Standar Profile to find the maximum phase
#    ## as ToA.
#
#    if method == "ccf":
#        ## ccf shift
#        print("Method to find ToA ... Cross-Correlation function")
#        delta_phi = _calculate_delta_phi_by_ccf(profile, std_pro) + 0.5/len(profile) ## plus half binsize as mid bin
#
#    ## plot normalized Observed Profile and normalized Standard Profile
#
#        if fig_flag:
#            plt.figure()
#            profile_obj = Profile(profile)
#            profile_std_obj = Profile(std_pro)
#            plt.errorbar(profile_obj.phase, profile_obj.norm())
#            plt.errorbar(profile_std_obj.phase, profile_std_obj.norm(), color='green')
#            plt.axvline(x=delta_phi, lw=1.2, color='red', label="CCF ToA Phi")
#            plt.legend()
#
#
#    ## -----------------
#    ## Method is to find the maximum phase as ToA Phase
#
#    elif method == "max":
#        print("Method to find ToA ... maximum")
#        delta_phi = np.argmax(profile)/len(profile) + 0.5*len(profile)
#        if fig_flag:
#            plt.figure()
#            plt.errorbar(np.linspace(0,1,len(profile)), profile,
#                    drawstyle='steps-mid')
#
#
#    ## -----------------
#    ## Method is to fit observed profile with Gaussian Function
#    ## A fitting_range should be assigned (0-2 phase for two period)
#    ## e.g. the peak phase is 0.99, a fitting range could be [0.9, 1.1]
#
#    elif method == "gauss_fit":
#        print("Method to find ToA ... Gaussian")
#        if 'fitting_range' not in kwargs:
#            raise IOError("parameter 'fitting_range' for gassian fitting not assigned")
#        fitting_range = kwargs['fitting_range']
#        x = np.linspace(0, 1, len(profile))
#        if fitting_range[1] >= 1:  # if phase range larger than 1
#            profile_to_fit = np.append(profile, profile) # duplicate the profile by two period
#            x_to_fit       = np.append(x, x+1+1/profile.size)
#        elif fitting_range[0] < 0: # if phase range less than 0
#            profile_to_fit = np.append(profile, profile) # duplicate the profile by two period
#            x_to_fit       = np.append(x-1-1/profile.size, x)
#        else:
#            profile_to_fit = profile
#            x_to_fit       = x
#
#        profile_to_fit = profile_to_fit[(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #profile at fitting range
#        x_to_fit       = x_to_fit      [(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #phase   at fitting range
#
#        # fit the profile using gaussian function
#        gauss_popt,gauss_pcov = curve_fit(Gauss, x_to_fit, profile_to_fit,
#                p0=[np.max(profile_to_fit),
#                    (np.max(x_to_fit)+np.min(x_to_fit))/2,
#                    np.max(x_to_fit)-np.min(x_to_fit)],
#                maxfev=9999999, sigma=np.sqrt(profile_to_fit), absolute_sigma=False)
#        gauss_fitting_errs = np.sqrt(np.diag(gauss_pcov))
#        delta_phi = gauss_popt[1]
#        gauss_popt[2] = abs(gauss_popt[2]) # For some reason, the value of sigma is negative and forcibly fixed to a positive value
#
#        if fig_flag:
#            plt.figure()
#            plt.errorbar(x, profile, yerr=np.sqrt(profile), color='black', label='Raw Data')
#            plt.errorbar(x_to_fit, profile_to_fit, color='orange', label='data to fit')
#            x_tmp = np.linspace(np.min(x_to_fit), np.max(x_to_fit),100)
#            plt.errorbar(x_tmp, Gauss(x_tmp, *gauss_popt), color='red', lw=1.5, label="gaussian")
#            plt.axvline(x=delta_phi)
#            plt.legend()
#
#    ## -----------------
#    ## Method is to fit observed profile with Lorentz Function
#    ## A fitting_range should be assigned (0-2 phase for two period)
#    ## e.g. the peak phase is 0.99, a fitting range could be [0.9, 1.1]
#
#    elif method == "lorentz_fit":
#        print("Method to find ToA ... Lorentz")
#
#        if 'fitting_range' not in kwargs:
#            raise IOError("parameter 'fitting_range' for lorentzian fitting not assigned")
#        fitting_range = kwargs['fitting_range']
#        x = np.linspace(0, 1, len(profile))
#        if fitting_range[1] >= 1:  # if phase range larger than 1
#            profile_to_fit = np.append(profile, profile) # duplicate the profile by two period
#            x_to_fit       = np.append(x, x+1+1/profile.size)
#            error_to_fit   = np.sqrt(profile_to_fit)
#        else:
#            profile_to_fit = profile
#            error_to_fit   = np.sqrt(profile)
#            x_to_fit       = x
#
#        profile_to_fit = profile_to_fit[(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #profile at fitting range
#        error_to_fit   = error_to_fit  [(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #profile at fitting range
#        x_to_fit       = x_to_fit      [(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #phase   at fitting range
#
#        # fit the profile using gaussian function
#        popt,pcov = curve_fit(Lorentz, x_to_fit, profile_to_fit,
#                p0=[np.max(profile_to_fit),
#                    (np.max(x_to_fit)+np.min(x_to_fit))/2,
#                    0.1],
#                sigma=error_to_fit, absolute_sigma=True,
#                maxfev=99999)
#        lorentz_fitting_errs = np.sqrt(np.diag(pcov))
#        delta_phi = popt[1]
#
#        if fig_flag:
#            plt.figure()
#            plt.plot(x, profile)
#            plt.plot(x_to_fit, profile_to_fit, color='black')
#            plt.plot(x_to_fit, Gauss(x_to_fit, *popt), color='red')
#
#    else:
#        raise IOError("the method {} does not supported".format(method))
#
#    if 'fig_flag' in kwargs:
#        if kwargs["fig_flag"]:
#            plt.figure()
#            plt.plot((profile-np.min(profile))/(np.max(profile)-np.min(profile)))
#
#    ## Calculate the ToA by
#    ## ToA = t0 + P*\delta\Phi
#    ## IMPORTANT: the profile should be folded by fbest, which is the
#    ## frequency at the PEPOCH (np.min(data)
#
#    toa = (1/fbest)*delta_phi + t0
#
#
#
#    ###############################################
#    ## ----------- Calculate Error
#    ###############################################
#
#    ## -----------
#    ##
#    ## -----------
#    if error_method == "simulate":
#        print("Bootstrap to calculate ToA error ... yes")
#        if 'sample_num' not in kwargs:
#            raise IOError("parameter 'sample_num' for profile resampling not given")
#        if kwargs['sample_num'] == 1:
#            print("Sampling number ... {}".format(kwargs['sample_num']))
#            ## if sample_num == 1, only one profile generated
#            resampled_profile  = resampling_profile(profile, kwargs['sample_num'])
#            ## calculate ToAs for new profile
#            if method == "ccf":
#                resampled_delta_phi = _calculate_delta_phi_by_ccf(resampling_profile,
#                        std_pro)
#            else:#TODO:method for other not finished
#                resampled_delta_phi = np.argmax(resampled_profile)/len(resampled_profile)
#        else:
#            print("Sampling number ... {}".format(kwargs['sample_num']))
#            resampled_profiles = resampling_profile(profile, kwargs['sample_num'])
#
#            resampled_delta_phi = np.array([])
#            ## calculate ToAs for new profile or profiles
#            if method == "ccf":
#                for resampled_profile in resampled_profiles:
#                    resampled_delta_phi = np.append(resampled_delta_phi,
#                            _calculate_delta_phi_by_ccf(resampled_profile,std_pro))
#            else:#TODO:method for other not finished
#                for resampled_profile in resampled_profiles:
#                    resampled_delta_phi = np.append(resampled_delta_phi,
#                            np.argmax(resampled_profile)/len(resampled_profile))
#
#            resampled_toa = (1/fbest)*resampled_delta_phi + t0
#
#        ## Calculate the 1sigma distribution of resampled ToA as 1 sigma error
#        #toa_err = 1e6*_get_error_quantiles(resampled_delta_phi, delta_phi) ## Unit microsecond
#
#        ## Calculate the rms of resampled ToA as ToA error
#        toa_err = 1e6*_get_error_rms(resampled_toa, toa) ## Unit microsecond
#
#    elif error_method == "default":
#        print("Bootstrap to calculate ToA error ... no")
#
#        #TODO: method max and ccf Error are not accurate yet!
#        if method == 'max':
#            toa_err = (np.max(data)-np.min(data))/len(profile)/10
#
#        elif method == "ccf":
#            #toa_err = (np.max(data)-np.min(data))/len(profile)/10
#
#            if "pulse_range" in kwargs:
#                fitting_range = [delta_phi-kwargs['pulse_range'], delta_phi+kwargs['pulse_range']]
#            else:
#                raise IOError("""parameter 'pulse_range' of the pulse to calculate ToA must be assigned, the range of pulse
#                will be Phase_{ccf max} +- pulse_range
#                """)
#
#            x = np.linspace(0, 1, len(profile))
#            if fitting_range[1] >= 1:  # if phase range larger than 1
#                profile_to_fit = np.append(profile, profile) # duplicate the profile by two period
#                x_to_fit       = np.append(x, x+1)
#            elif fitting_range[0] < 0: # if phase range less than 0
#                profile_to_fit = np.append(profile, profile) # duplicate the profile by two period
#                x_to_fit       = np.append(x-1, x)
#            else:
#                profile_to_fit = profile
#                x_to_fit       = x
#
#            profile_to_fit = profile_to_fit[(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #profile at fitting range
#            x_to_fit       = x_to_fit      [(x_to_fit>=fitting_range[0])&(x_to_fit<=fitting_range[1])] #phase   at fitting range
#
#            # fit the profile using gaussian function
#            gauss_popt,gauss_pcov = curve_fit(Gauss, x_to_fit, profile_to_fit,
#                    p0=[np.max(profile_to_fit),
#                        (np.max(x_to_fit)+np.min(x_to_fit))/2,
#                        np.max(x_to_fit)-np.min(x_to_fit)],
#                    maxfev=9999999, sigma=np.sqrt(profile_to_fit), absolute_sigma=False)
#            gauss_popt[2] = abs(gauss_popt[2]) # For some reason, the value of sigma is negative and forcibly fixed to a positive value
#
#            if fig_flag:
#                plt.figure()
#                plt.errorbar(x, profile, yerr=np.sqrt(profile), color='black', label='Raw Data')
#                plt.errorbar(x_to_fit, profile_to_fit, color='orange', label='data to fit')
#                plt.errorbar(x_to_fit, Gauss(x_to_fit, *gauss_popt), color='red', lw=1.5, label="gaussian")
#                plt.legend()
#
#            print("Calculate ToA error ... based on pulse SNR (Lorimer & Kramer 2012)")
#            period = 1/fbest
#            peak_sigma = gauss_popt[2]
#
#            N_background  =  np.min(profile) * len(profile_to_fit)
#            N_source = np.sum( profile_to_fit ) - N_background
#            toa_err = _get_error_by_profile_shape(period, peak_sigma, N_source, N_background)
#            toa_err = 1e6* toa_err # in Unit of microsecond
#
#        # --------
#        elif method == 'gauss_fit':
#            print("Calculate ToA error ... based on pulse SNR (Lorimer & Kramer 2012)")
#            period = 1/fbest
#            peak_sigma = gauss_popt[2]
#
#            N_background  =  np.min(profile) * len(profile_to_fit)
#            N_source = np.sum( profile_to_fit ) - N_background
#            toa_err = _get_error_by_profile_shape(period, peak_sigma, N_source, N_background)
#            toa_err = 1e6* toa_err # in Unit of microsecond
#
#    elif error_method == "gauss_fit":
#        print("Calculate ToA error ... fitting")
#        print(gauss_fitting_errs)
#        toa_err = gauss_fitting_errs[1] ## Error is the error of phase x0
#        toa_err = 1e6*toa_err/fbest ## Convert to time (Unit of microsec)
#
#    else: raise IOError("The error_method {} not supported")
#
#    toa = met2mjd(toa, telescope=kwargs['telescope'])
#    return toa, toa_err

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


def _calculate_delta_phi_by_ccf(profile, profile_std):
    #profile = np.append(profile, profile)
    #profile_std = np.append(profile_std, profile_std)
    y, delay = ccf(profile, profile_std)
    p_num_std = np.roll(profile_std, delay)
    p_num_x = np.linspace(0,1, len(profile))
    delta_phi = p_num_x[np.argmax(p_num_std)]
    return delta_phi
