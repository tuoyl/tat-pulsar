"""
Module for binary related analysis and correction
"""

import numpy as np
from astropy.io import fits

__all__ = ['orbit_cor_bt',
        'orbit_cor_deeter',
        'doppler_cor']

__method__ = ["BT", "Deeter"]


def orbit_cor_bt(t, Porb, axsini, e, omega, Tw, gamma):
    """
    use numerical method to solve Kepler equation and calculate delay
    BT model (Blandford & Teukolsky, 1976)

    Parameters
    ----------
    t : array-like
        The time series without binary correction
    Porb : float
        The period of binary motion (in units of second)
    axsini : float
        Projected semi-major orbital axis (in units of light-sec)
    e : float
        The orbital eccentricity
    omega : float
        Longitude of periastron
    Tw : float
        The epoch of periastron passage (in units of seconds, same time system with parameter t)
    gamma ; float
        the coefficient measures the combined effect of gravitational redshift and time dilation

    Returns
    -------
    new_t : array-like
        The time series after the binary correction

    Notes
    -----
    Reference: Blandford & Teukolsky, 1976, ApJ, 205:580-591
    """
    x = axsini
    gamma = 0

    if e == 0:
        E = 2*np.pi*(t-Tw)/Porb;
    else:
        E = np.array([])
        #solve Kepler Equation
        dE = 1e-3
        E_min = 2*np.pi*(t-Tw)/Porb - e
        E_max = 2*np.pi*(t-Tw)/Porb + e
        for i in range(len(E_min)): #TODO:the numerical method need to optimize!
            E_arr = np.arange(E_min[i], E_max[i], dE)

            equation_left = E_arr - e*np.sin(E_arr)
            equation_right= 2*np.pi*(t[i]-Tw)/Porb
            residual = np.abs(equation_left - equation_right)
            min_index = np.where(residual == min(residual))

            E = np.append(E, E_arr[min_index])

    #calculate time delay by orbit
    #factor1
    factor1 = x*np.sin(omega)*(np.cos(E)-e) + (x*np.cos(omega)* ((1-e**2)**0.5) + gamma )*np.sin(E)
    #factor2
    factor2 = 1- (2*np.pi/Porb)* (x*np.cos(omega)*((1-e**2)**0.5)-x*np.sin(omega)*np.sin(E)) * (1-e*np.cos(E))**(-1)
    factor = factor1 * factor2
    new_t = t + factor #NOTE:pulsar proper Time = time + facotr
    print(factor)
    return new_t


def orbit_cor_deeter(time, Porb, axsini, e, omega, Tnod):
    """
    Correct the photon arrival times to the photon emission time
    Deeter model.

    Parameters
    ----------
    time : array-like
        The time series before binary correction
    Porb : float
        The period of binary motion (in units of second)
    axsini : float
        Projected semi-major orbital axis (in units of light-sec)
    e : float
        The orbital eccentricity
    omega : float
        Longitude of periastron
    Tnod : float
        The epoch of ascending node passage (in units of seconds, same time system with parameter t)

    Returns
    -------
    t_em : array-like
        The photon emission time

    Notes
    -----
    Reference: Deeter et al. 1981, ApJ, 247:1003-1012

    """
    A = axsini
    mean_anomaly = 2*np.pi*(time-Tnod)/Porb

    term1 = np.sin(mean_anomaly + omega)
    term2 = (e/2)*np.sin(2*mean_anomaly + omega)
    term3 = (-3*e/2)*np.sin(omega)

    t_em = time - A * (term1 + term2 + term3)
    return t_em

def doppler_cor(time, f0, f1, f2, axsini, Porb, omega, e, T_halfpi):
    """
    correct the observed freqency of neutron star,
    convert the frequency moduled by the binary orbital Doppler effect to
    the intrinsic frequency of NS

    Parameters
    ----------
    time : array-like
        The epoch of observed time series
    f0 : float
        the observed frequency at reference epoch t0
    f1 : float
        the observed frequency derivative at reference epoch t0
    f2 : float
        the observed frequency second derivative at reference epoch t0
    axsini : float
        Projected semi-major orbital axis (in units of light-sec)

    Porb : float
        The period of binary motion (in units of second)
    omega : float
        Longitude of periastron
    e : float
        The orbital eccentricity
    T_halfpi : float
       The mean longitude, with T_halfpi the epoch at which the mean longitude is pi/2
       (in units of second)

    Returns
    -------
    f_intri : array-like
        The corrected intrinsic frequecy of neutron star

    Notes
    -----
    Reference: Galloway et. al, 2005, ApJ, 635, 1217
    """
    t0 = min(time) # set reference time as the start of time
    f_spin = f0 + f1*(time-t0) + 0.5*f2*(time-t0)**2
    f_dopp = _get_fdopp(time, f0, axsini, Porb, omega, e, T_halfpi)

    f_intri = f_spin - f_dopp
    return f_intri

def _get_fdopp(time, f0, axsini, Porb, omega, e, T_halfpi):
    """
    calculate the frequency modulated by Doppler effect
    """
    l = 2* np.pi * (time-T_halfpi)/Porb + np.pi/2
    g = e*np.sin(omega)
    h = e*np.cos(omega)
    f_dopp = (2 * np.pi * f0 * axsini / Porb) * (np.cos(l) + g*np.sin(2*l) + h*np.cos(2*l) )
    return f_dopp

