import numpy as np
import matplotlib.pyplot as plt
from pulsar_timing.utils import *

__all__ = ['resampling_profile',
        "norm_profile"]

def resampling_profile(profile, sample_num=1, kind='poisson'):
    '''
    resampling the profile

    Parameters
    -----------
    profile : array
        The un-normalized profile

    sample_num : int, optional
        number of the resamplings for the profile, the default number is 1

    kind : str, optional
        The distribution of the profile, default is poisson.
        ('poisson', 'gaussian') are refering to the poisson and gauss distribution

    Returns
    -----------
    resampled_profile : array or ndarray
        if sample_num == 1, return a one dimensional array
        if sample_num >1 , return a multi-dimensional array
    '''
    raw_profile = np.array(profile.tolist()*sample_num)
    if sample_num <= 0:
        raiseError("The number of sampling must a positive integer")

    if kind == "poisson":
        resampled_profile = np.random.poisson(raw_profile)
    elif kind == "gaussian":
        pass #TODO

    resampled_profile = resampled_profile.reshape(int(len(resampled_profile)/len(profile)),
            int(len(profile)))
    return resampled_profile


def norm_profile(profile, yerr=None):
    return (profile-np.min(profile))/(np.max(profile)-np.min(profile))

