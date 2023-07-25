"""
The Class of Profile
"""
import numpy as np
import scipy.stats
from tatpulsar.simulation.profile_sim import poisson_rejection_sampling, draw_event_from_phase

__all__ = ["Profile",
           "phihist",
           "draw_random_pulse"]

class Profile():
    """
    Profile class

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
    ref_time : float, default None
        The reference time (in second) used to fold the profile
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
        '''
        self._cycles = cycles
        if type(cycles) != int:
            raise TypeError("The cycles of profile should be int")
        if cycles > 2:
            raise IOError("Why do you have to setup so many cycles? 2 cycles is enough.")
        if cycles == 2:
            self.counts = np.tile(counts, reps=2)
            self._pickled = True # whether the profile has been duplicated or modified
        else:
            self.counts = counts
            self._pickled = False # whether the profile has been duplicated or modified
        self.phase  = np.linspace(0, cycles, self.counts.size + 1)[:-1]
        if error is None:
            self.error = np.sqrt(self.counts)
        elif cycles == 2:
            self.error = np.tile(error, reps=2)
        else:
            self.error = error
        self._pickled = False # whether the profile has been duplicated or modified
        self.ref_time = None

    def __add__(self, other):
        """
        merge two Profile.

        .. warning::

            When you add two Profile, we recommend that only add two Profile with ONE cycle.
            If you add two pickled Profile (e.g duplicated the phase into 2 cycles), the new
            returned Profile has a cycle of ONE, which is incorrect.

        """
        if not isinstance(other, Profile):
            raise IOError(f"{other} is not a Profile object")
        if self.cycles != other.cycles:
            raise ValueError("the cycles of two Profile object does not match")
        add_cnt = self.counts + other.counts
        add_err = np.sqrt(self.error**2 + other.error**2)
        return Profile(add_cnt, error=add_err)

    def __sub__(self, other):
        """
        subtract operator to subtract on Profile from another Profile
        """
        if not isinstance(other, Profile):
            raise IOError(f"{other} is not a Profile object")
        if self.cycles != other.cycles:
            raise ValueError("the cycles of two Profile object does not match")
        sub_cnt = self.counts - other.counts
        sub_err = np.sqrt(self.error**2 + other.error**2)
        return Profile(sub_cnt, error=sub_err)


    @property
    def cycles(self):
        return self._cycles
    @cycles.setter
    def cycles(self, value):
        """
        modify the cycles for profile
        """
        if value > 2:
            raise IOError("Why do you have to setup so many cycles? 2 cycles is enough.")

        if self._cycles == value:
            print(f"Cycle is already {value}, nothing to do")
        elif value == 2:
            self._cycles = value
            self.counts = np.tile(self.counts, reps=2)
            self.error  = np.tile(self.error, reps=2)
            self.phase  = np.linspace(0, value, self.counts.size+1)[:-1]
        elif value == 1:
            self._cycles = value
            idx = int(self.counts.size/2)
            self.counts = self.counts[:idx]
            self.error  = self.error[:idx]
            self.phase  = np.linspace(0, value, self.counts.size+1)[:-1]


    @property
    def size(self):
        """
        return the real bin size of the profile.
        If the profile is presented in 2 cycles, the size is still the
        size of the bin size of the profile in 1 cycle.
        """
        if self.cycles == 2:
            return int(self.counts.size/2)
        else:
            return int(self.counts.size)

    @property
    def dof(self):
        """
        degree of freedom is :math:`(n-1)` where :math:`n` is the number of bins.
        """
        return self.size - 1

    @property
    def chisq(self):
        """
        chisquare statistic value of profile
        """
        if self.cycles == 2:
            idx = int(self.counts.size/2)
            counts = self.counts[:idx]
            error  = self.error[:idx]
        else:
            counts = self.counts
            error = self.error
        return np.sum( (counts - np.mean(counts))**2 / counts )

    @property
    def significance(self):
        """
        Return the significance in unit of sigma of given profile.
        """
        p_value = scipy.stats.chi2.sf(self.chisq, self.dof)
        # we are dealing with one-tailed tests
        sigma = scipy.stats.norm.isf(p_value)
        return sigma

    @property
    def pulsefrac(self):
        """
        Calculate the pulse fraction of given profile. The algorithm of
        calculating pulse fraction is:

        .. math::
            PF = (p_{\mathrm{max}} - p_{\mathrm{min}})/(p_{\mathrm{max}} + p_{\mathrm{min}})

        where :math:`p` is the counts of profile, please note the pulse fraction has valid physical
        meaning only if the input profile is folded by the net lightcurve or the background level can
        be well estimated and subtracted from the observed pulse profile.

        Returns
        -------
        pf: float
            The pulse fraction of profile
        pf_err: float
            The error of pulse fraction
        """
        pf = (self.counts.max() - self.counts.min()) / (self.counts.max() + self.counts.min())

        indx_max = np.argmax(self.counts)
        indx_min = np.argmin(self.counts)

        term1 = -2 * self.counts.max() / (self.counts.max() + self.counts.min())**2 * self.error[indx_min]
        term2 =  2 * self.counts.min() / (self.counts.max() + self.counts.min())**2 * self.error[indx_max]
        pf_err = np.sqrt(term1**2 + term2**2)
        return pf, pf_err

    def sampling_phase(self, nphot):
        """
        use rejection sampling method to draw the phase list

        Parameters
        ----------
        nphot: int
            The output amount of sampled photons

        Returns
        -------
        phase: array-like
            The phase sample that satisfy the rejection rule
        """
        phase = poisson_rejection_sampling(
                self.phase,
                self.counts,
                nphot)
        return phase

    def sampling_event(self, nphot, tstart, tstop,
                       f0, f1=0, f2=0, f3=0, pepoch=0):
        """
        sampling the photon arrival time

        Parameters
        ----------
        nphot: int
            The output amount of sampled photons
        tstart: array-like
            The start time to generate arrival time (MJD)
        tstop: array-like
            The stop  time to generate arrival time (MJD)
        f0: float
            The frequency of the pulsar
        f1: float, optional
            The frequency derivative of the pulsar
        f2: float, optional
            The second derivative of the frequency
        f3: float, optional
            The third derivative of the frequency
        f4: float, optional
            The fourth derivative of the frequency
        pepoch: float
            The reference time of the timing parameters (MJD)

        Returns
        -------
        event_list: array-like
            The sampled arrival times of photons
        """
        phase = self.sampling_phase(nphot)
        event_list = draw_event_from_phase(
                phase,
                tstart,
                tstop,
                f0=f0, f1=f1, f2=f2, f3=f3,
                pepoch=pepoch)
        return event_list

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

    def norm(self,
            method=0,
            bkg_range=None,
            return_profile=False):
        '''
        normalize the profile, and return a normalized Profile object

        bkg_range is the background phase range selected to calculated the mean level
        of background, used in method=0.

        Parameters
        ----------
        method: int, optional
            The normalization method utilized, optional methods are {0, 1}
            method = 0 : :math:`N = (P - P_{min})/(P_{max} - P_{min})`
            if background range are selected (`bkg_range` is not None)
            :math:`N = (P - \\bar{B})/(P_{max} - \\bar{B})`
            where :math:`\\bar{B}` is the mean level in `bkg_range`
            method = 1 : :math:`N = (P-P_{min})/\\bar{P}`

        bkg_range: list, optional
            The background phase range for background estimation
        return_profile: bool, optional
            whether to return the profile, if False modify the attributes
        '''
        if method == 0:
            # without background section
            if bkg_range is None:
                norm_counts = (self.counts-np.min(self.counts))/\
                        (np.max(self.counts)-np.min(self.counts))

                X = self.counts
                m = self.counts.min()
                M = self.counts.max()
                delta_X = self.error
                delta_m = self.error[self.counts.argmin()]
                delta_M = self.error[self.counts.argmax()]
                delta_num = np.sqrt(delta_X**2 + delta_m**2)
                delta_den = np.sqrt(delta_M**2 + delta_m**2)

                expression = (X - m)/(M-m)
                norm_error = expression * np.sqrt((delta_num / (X - m))**2 + (delta_den / (M - m))**2)

            else:
                bkg_mask = (self.phase>=bkg_range[0]) & (self.phase<=bkg_range[1])
                bkg_counts = self.counts[bkg_mask]
                bkg_error  = self.error[bkg_mask]
                bkg_mean_error = np.sqrt(np.sum(bkg_error**2))/bkg_error.size
                norm_counts = (self.counts - np.mean(bkg_counts))/\
                        (np.max(self.counts) - np.mean(bkg_counts))

                X = self.counts
                m = np.mean(bkg_counts)
                M = self.counts.max()
                delta_X = self.error
                delta_m = bkg_mean_error
                delta_M = self.error[self.counts.argmax()]
                delta_num = np.sqrt(delta_X**2 + delta_m**2)
                delta_den = np.sqrt(delta_M**2 + delta_m**2)
                norm_error = np.sqrt(
                        self.error**2 + bkg_mean_error**2)/\
                                (np.max(self.counts) - np.mean(bkg_counts))
        elif method == 1:
            mean_error = np.sqrt(np.sum(self.error**2))/self.error.size
            norm_counts = (self.counts - np.min(self.counts))/\
                    np.mean(self.counts)

            X = self.counts
            m = self.counts.min()
            a = self.counts.mean()
            delta_X = self.error
            delta_m = self.error[self.counts.argmin()]
            delta_a = mean_error

            # calculate the uncertainty in (X - m)
            delta_Y = np.sqrt(delta_X**2 + delta_m**2)
            # calculate the expression and its uncertainty
            Y = X - m
            Z = Y / a
            norm_error = Z * np.sqrt((delta_Y / Y)**2 + (delta_a / a)**2)

        if return_profile:
            return Profile(norm_counts, error=norm_error)
        else:
            self.counts = norm_counts
            self.error  = norm_error
            self._pickled = True

    def rebin(self, nbins=None, factor=None, return_profile=False):
        """
        Rebin the profile into the given bin size or use the factor to
        split the intervals.

        Parameters
        ----------
        nbins: int
            the number of the bins of new profile, if 'nbins' is used,
            then the 'factor' parameter would not work
        factor: int
            rebin profile to a new shape, new shape must be a factor of the
            original shape
        return_profile: bool, optional
            whether to return the profile, if False modify the attributes

        Returns
        -------
        profile: `Profile` object
            return a new profile in one cycle
        """

        if (nbins is None) & (factor is None):
            raise IOError("Either 'nbins' or 'factor' should be given")
        if factor is None:
            assert nbins <= self.size, "The input 'nbins' value should be larger than the size of the profile"
            assert self.size % nbins  == 0, "The size of the profile should be divisible by new size"
        else:
            if nbins is not None:
                factor = self.size / nbins
            else:
                nbins = int(self.size / factor)
            assert self.size % factor == 0, "The size of the profile should be divisible by factor"
        new_counts = self.counts.reshape(nbins, -1).sum(axis=1)
        new_error  = np.sqrt(
                        (self.error.reshape(nbins, -1)**2).sum(axis=1))
        if return_profile:
            return Profile(new_counts, error=new_error)
        else:
            self.counts = new_counts
            self.error  = new_error
            self.phase  = np.linspace(0, self.cycles, self.counts.size + 1)[:-1]
            self._pickled = True

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

def draw_random_pulse(nbins=100, baseline=1000, pulsefrac=0.2):
    """
    Generate a random pulse profile that consists of multiple Gaussian-like pulse

    Parameters
    ----------
    nbins: int
        The number of the new profile bins
    baseline: int
        The baseline (background) level of the profile in unit of counts
    pulsefrac: float
        pulse frac

    Returns
    -------
    profile: `Profile` object
    """
    # Define the time array
    phase = np.linspace(0, 1, nbins)

    # Define a random number of pulses
    num_pulses = np.random.randint(4, 10)

    # Initialize the signal
    signal = np.zeros_like(phase)

    # Generate the signal by summing up the pulses
    for _ in range(num_pulses):
        # Randomly generate the Gaussian pulse parameters
        amp = np.random.uniform(0.1, 1.0)  # Amplitude of pulse
        mu = np.random.uniform(0.2, 0.8)  # Mean (peak location within 0 to 2)
        sigma = np.random.uniform(0.01, 0.1)  # Standard deviation (controls width of pulse)

        # Generate the Gaussian pulse
        pulse = amp * np.exp(-(phase - mu)**2 / (2 * sigma**2))

        # Add the pulse to the signal
        signal += pulse

    signal = signal/signal.max()

    pmax = baseline*(1+pulsefrac)/(1-pulsefrac)
    scale = pmax - baseline
    signal = signal*scale + baseline

    signal = np.random.poisson(signal) # poisson sampling

    return Profile(signal)

