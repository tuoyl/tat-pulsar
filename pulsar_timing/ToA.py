#!/usr/bin/env python
import numpy as np
import numba 
import matplotlib.pyplot as plt
from pulsar_timing.utils import *
from pulsar_timing.Profile import resampling_profile

__all__ = ['cal_toa']

def cal_toa(fbest, profile, data, ccf_flag=True, std_pro='', **kwargs):
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

    ccf_flag : bool, optional
        The flag to do cross-correlation to calculate the ToA phase
        if True, A standard profile with high statistic should be assigned
        at the same time. see parameter "std_pro" for detail.

    std_pro : array, optional
        The standard profile with high statistic for CCF.


    Returns
    ------------
    toa : float
        The Time of Arrival of observed Profile

    toa_err : float
        The error of the obtained ToA, 

    """
    if ccf_flag:
        ## ccf shift
        delta_phi = _calculate_delta_phi_by_ccf(profile, std_pro)
    else:
        delta_phi = np.argmax(profile)/len(profile)

    if 'fig_flag' in kwargs:
        if kwargs["fig_flag"]:
            plt.plot((profile-np.min(profile))/(np.max(profile)-np.min(profile)))
            plt.plot((p_num_std-np.min(p_num_std))/(np.max(p_num_std)-np.min(p_num_std)), color='red')
            plt.show()

    toa = (1/fbest)*delta_phi + np.min(data)

    if 'simulate_error' in kwargs:
        ## simulate the pr
        if kwargs['simulate_error']:
            if 'sample_num' not in kwargs:
                raiseError("parameter 'sample_num' for profile resampling not given")
            if kwargs['sample_num'] == 1:
                ## if sample_num == 1, only one profile generated
                resampled_profile  = resampling_profile(profile, kwargs['sample_num'])
                ## calculate ToAs for new profile 
                if ccf_flag:
                    resampled_delta_phi = _calculate_delta_phi_by_ccf(resampling_profile,
                            std_pro)
                else:
                    resampled_delta_phi = np.argmax(resampled_profile)/len(resampled_profile)
            else:
                resampled_profiles = resampling_profile(profile, kwargs['sample_num'])

                resampled_delta_phi = np.array([])
                ## calculate ToAs for new profile or profiles
                if ccf_flag:
                    for resampled_profile in resampled_profiles:
                        resampled_delta_phi = np.append(resampled_delta_phi,
                                _calculate_delta_phi_by_ccf(resampled_profile,std_pro))
                else:
                    for resampled_profile in resampled_profiles:
                        resampled_delta_phi = np.append(resampled_delta_phi,
                                np.argmax(resampled_profile)/len(resampled_profile))

                resampled_toa = (1/fbest)*resampled_delta_phi + np.min(data)

            ## Calculate the 1sigma distribution of resampled ToA as 1 sigma error
            toa_low = sorted(resampled_toa)[int(0.16*len(resampled_toa))]
            toa_hig = sorted(resampled_toa)[int(0.84*len(resampled_toa))]
            toa_err = 1e6*max(toa_hig-toa, toa-toa_low)## Unit us
    else:
        toa_err = (np.max(data)-np.min(data))/len(profile)/10

    toa = met2mjd(toa, telescope=kwargs['telescope'])
    return toa, toa_err

def _calculate_delta_phi_by_ccf(profile, profile_std):
    y, delay = ccf(profile, profile_std)
    p_num_std = np.roll(profile_std, delay)
    p_num_x = np.linspace(0,1, len(profile))
    delta_phi = p_num_x[np.argmax(p_num_std)]
    return delta_phi
