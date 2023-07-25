"""
functions for the profile sampling and simulation
"""

__all__ ['rejection_sampling',
         'draw_event_from_phase']


def poisson_rejection_sampling(x, y, nphot):
    """
    rejection sampling the poisson like signal x-y

    Parameters
    ----------
    x: array-like
        Time or Phase array
    y: array-like
        counts, Poisson distributed
    nphot: int
        The output amount of sampled photons

    Returns
    -------
    xval: array-like
        The x sample that satisfy the rejection rule
    """
    binsize = np.median(np.diff(x))
    interp_fun = interp1d(x,
                      y,
                      kind='cubic')

    x_sample = np.random.uniform(x.min(), x.max(), nphot)
    yval_x_sample = interp_fun(x_sample) # The corresponding y value of sampled X, according to interplation function

    y_sample = np.random.uniform(0, y.max(), nphot) # Sampled y value for each x sampled value
    ## Reject the value that over the true y value
    mask = (y_sample <= yval_x_sample)

    return x_sample[mask]

def draw_event_from_phase(phase, tstart, tstop,
                          f0, f1=0, f2=0, f3=0, f4=0, pepoch=0):
    """
    sampling the photon arrival time given the phase of each photon

    Parameters
    ----------
    phase: array-like
        The phase of each photon, the phase is normalized to [0, 1]
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
    dt = (tstart - pepoch) * 86400
    tstart_phase = f0*dt + 0.5*f1*dt**2 + (1/6.)*f2*dt**3
    dt = (tstop  - pepoch) * 86400
    tstop_phase  = f0*dt + 0.5*f1*dt**2 + (1/6.)*f2*dt**3

    Npulse_sample = np.random.randint(tstart_phase, tstop_phase, phase.size)

    absphase_sample = Npulse_sample + phase

    def delta_phi(t, phi0):
        dt = (t - pepoch) * 86400
        return f0*dt + 0.5*f1*dt**2 + (1/6.)*f2*dt**3 - phi0

    def obj_fun(t):
        return delta_phi(t, phi)

    event_list = np.empty_like(phase)

    for idx, phi in enumerate(tqdm(absphase_sample)):
        event_list[idx] = brentq(obj_fun, tstart, tstop)

    return event_list
