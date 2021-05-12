import numpy as np
import numba 
import matplotlib.pyplot as plt
from pulsar_timing.utils import *

__all__ = ['resampling_profile']

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
    '''
    resampled_profile = np.array([])
    if sample_num <= 0:
        raiseError("The number of sampling must a positive integer")
    for i in range(sample_num):
        if kind == "poisson":
            resampled_profile = np.append(resampled_profile,
                    np.random.poisson(profile))
        elif kind == "gaussian":
            pass #TODO
    resampled_profile = resampled_profile.reshape(int(len(resampled_profile)/len(profile)),
            int(len(profile)))
    return resampled_profile


