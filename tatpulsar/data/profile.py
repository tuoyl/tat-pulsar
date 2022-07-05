"""
The Class of Profile
"""
import numpy as np

# Transfer to OOP
#__all__ = ['resampling_profile',
#        "norm_profile",
#        "phihist"]
__all__ = ['Profile',
        "phihist"]

class Profile():
    """
    Profile class
    """

    def __init__(self, counts, cycles=1, error=None):
        '''
        Parameters
        ----------
        counts : array-like
            the counts in each phase bin of Profile

        cycles : int
            the period cycles of input Profile (default is 1).
            If cycles=2, the phase of profile would be ``np.linspace(0, 2, size_of_Profile+1)[:-1]``

        error : array-like
            the error of each phase bin, if not given the error will be the
            poisson error of counts (sqruare root of counts)

        Attributes
        ----------
        counts : array-like
            The counts in each phase bin of Profile
        phase : array-like
            The midpoints of phase bins
        phase_off : list
            The list of phase off interval, the two value are the left
            and right phase bin of off pulse phases.
            left_edge = phase_off[0]
            right_edge = phase_ff[1]
        '''
        if type(cycles) != int:
            raise TypeError("The cycles of profile should be int")
        if cycles > 2:
            raise IOError("Why do you have to setup so many cycles? 2 cycles is enough.")
        if cycles == 2:
            self.counts = np.append(counts, counts)
        else:
            self.counts = counts
        self.phase  = np.linspace(0, cycles, self.size+1)[:-1]
        if error is None:
            self.error = np.sqrt(counts)
        else:
            self.error = error

    @property
    def size(self):
        return self.counts.size

    def resample(self, sample_num=1, kind='poisson'):
        '''
        resampling the profile

        Parameters
        ----------
        sample_num : int, optional
            number of the resamplings for the profile, the default number is 1
        kind : str, optional
            The distribution of the profile, default is poisson.
            ('poisson', 'gaussian') are refering to the poisson and gauss distribution

        Returns
        -------
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

        resampled_profile = resampled_profile.reshape(int(resampled_profile.size/self.size),
                int(self.size))
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

def phihist(phi, nbins, **kwargs):
    '''
    Ensure that the input and output of the histogram are appropriate.
    The input variables are the pulse phi of events, and the nbins.
    The counts of each bin are calculated by dividing [0, 1] into number of nbins.

    Parameters
    ----------
    phi : array
        a set of phase value of events.

    nbins : int
        the number of bins of profile

    Returns
    -------
    Profile : object
        return the object of Profile
    '''

    x = np.linspace(0, 1, nbins + 1)
    counts, phase = np.histogram(phi, x)
    profile_object = Profile(counts, **kwargs)

    return profile_object


def resampling_profile(profile, sample_num=1, kind='poisson'):
    '''
    resampling the profile

    Parameters
    ----------
    profile : array
        The un-normalized profile

    sample_num : int, optional
        number of the resamplings for the profile, the default number is 1

    kind : str, optional
        The distribution of the profile, default is poisson.
        ('poisson', 'gaussian') are refering to the poisson and gauss distribution

    Returns
    -------
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

