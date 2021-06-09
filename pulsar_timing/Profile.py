import numpy as np
import matplotlib.pyplot as plt
from pulsar_timing.utils import *

__all__ = ['resampling_profile',
        "norm_profile"]

class Profile():
    """
    Profile class
    """

    def __init__(self, counts, cycles=1):
        '''
        Initialize Parameter
            counts : array
                the counts in each phase bin of Profile

            cycles : int
                the period cycles of input Profile (default is 1).
                If cycles=2, the phase of profile would be np.linspace(0, 2, size_of_Profile)
        '''
        self.counts = counts
        if type(cycles) != int:
            raise TypeError("The cycles of profile should be int")
        self.phase  = np.linspace(0, cycles, self.size)

    @property
    def size(self):
        return self.counts.size

    def resample(self, sample_num=1, kind='poisson'):
        '''
        resampling the profile

        Parameters
        -----------

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
        raw_profile = np.array(self.counts.tolist()*sample_num)
        if sample_num <= 0:
            raise IOError("The number of sampling must a positive integer")

        if kind == "poisson":
            resampled_profile = np.random.poisson(raw_profile)
        elif kind == "gaussian":
            pass #TODO

        resampled_profile = resampled_profile.reshape(int(len(resampled_profile)/Profile.size),
                int(Profile.size))
        return resampled_profile

    def norm(self, yerr=None, method=0):
        '''
        normalize the profile

        method = 0 : normalization = (profile-min)/(max-min)
        method = 1 : normalization = (profile-min)/mean(profile)
        '''
        #TODO: normalize with error
        if method == 0:
            return (self.counts-np.min(self.counts))/(np.max(self.counts)-np.min(self.counts))
        elif method == 1:
            return (self.counts-np.min(self.counts))/(np.mean(self.counts))


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

