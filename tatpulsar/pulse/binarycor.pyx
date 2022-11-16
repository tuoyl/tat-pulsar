"""
Author: Ji Long (jilong@mail.sysu.edu.cn)
code are copied from:
https://code.ihep.ac.cn/jldirac/insight-hxmt-code-collection/-/blob/master/binaryCorr/binaryCorr.pyx

Thanks to Long for not claiming any copyright on the binary correction code.
"""

cimport cython
from libc.math cimport sin, cos, sqrt, fabs

'''
typedef struct {
  double pb ;           /* Orbital period (s) */
  double a1sini ;       /* Projected semi-major axis (light seconds) */
  double e ;            /* orbital eccentricity */
  MJDTime t0 ;          /* Barycentric time (in MJD(TDB)) of periastron */
  double omega ;        /* Longitude of periastron (degrees) */
  double omdot ;        /* First derivative of omdot (degrees per Julian year) */
  double gamma ;        /* Time-dilation and gravitational redshift parameter (s) */
  double pbdot ;        /* First derivative of pb */
} PsrBinPar ;
'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double Cor(double time, double pbp_t0, \
                 double pbp_e, double pbp_pb, double pbp_pbdot, double pbp_omega, double pbp_omdot, \
                 double pbp_a1sini, double pbp_gamma
                 ):
    '''
    Parameters
    ----------
    time: float
        The time of observed photon in MJD
    Tw: float
        The periastron time in MJD
    E: float
        Eccentricity
    Porb: float
        Orbital period in second
    PdotOrb: float
        Second derivative of Orbital period in sec/sec
    omega: float
        Long. of periastron in degree #TODO
    omegadot: float
        second derivative of Long. of periastron in deg/sec
    axsini: float
        projection of semi major axis in light-sec
    gamma: float
        0

    Returns
    -------
    torb: float
        The correction value
    '''
    cdef double torb
    cdef double orbph
    cdef double t
    cdef double dt
    cdef double dt2
    cdef double e
    cdef double ep, dep, denom ;
    cdef double alpha, beta, omega, sbe, cbe, q
    cdef double SECDAY = 86400.
    cdef double TWOPI = 6.283185307179586

    e = pbp_e
    t = SECDAY * (time - pbp_t0)
    dt = t / pbp_pb
    dt2 = 0.5 * dt * dt
    orbph = dt - dt2 * pbp_pbdot
    orbph -= int(orbph)
    if orbph < 0.0:
      orbph = orbph + 1.0
    orbph = TWOPI * orbph
    ep = orbph + e*sin(orbph)*(1 + e*cos(orbph))
    denom = 1.0 - e*cos(ep)

    dep = 1.0
    while ( fabs(dep) > 1.0e-12 ):
       dep = ( orbph - ( ep - e * sin(ep) ) ) / denom
       ep += dep

    omega = pbp_omega + pbp_omdot * t
    alpha = pbp_a1sini * sin(omega)
    beta = pbp_a1sini * cos(omega) * sqrt(1 - e * e)
    sbe = sin(ep)
    cbe = cos(ep)
    q = alpha * (cbe - e) + (beta + pbp_gamma) * sbe
    torb = -q + (TWOPI / pbp_pb) * q * (beta * cbe - alpha * sbe) / (1 - e * cbe)

    return torb

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double Doppler(double time, double pbp_t0, \
                 double pbp_e, double pbp_pb, double pbp_pbdot, double pbp_omega, double pbp_omdot, \
                 double pbp_a1sini, double pbp_gamma):
    '''
    The Astrophysical Journal, 584:996–1007, 2003 February 20
    nu_obs = nu_emit * Doppler_factor
    '''
    cdef double torb
    cdef double orbph
    cdef double t
    cdef double dt
    cdef double dt2
    cdef double e
    cdef double ep, dep, denom ;
    cdef double alpha, beta, omega, sbe, cbe, q
    cdef double SECDAY = 86400.
    cdef double TWOPI = 6.283185307179586

    e = pbp_e
    t = SECDAY * (time - pbp_t0)
    dt = t / pbp_pb
    dt2 = 0.5 * dt * dt
    orbph = dt - dt2 * pbp_pbdot
    orbph -= int(orbph)
    if orbph < 0.0:
      orbph = orbph + 1.0
    orbph = TWOPI * orbph
    ep = orbph + e*sin(orbph)*(1 + e*cos(orbph))
    denom = 1.0 - e*cos(ep)

    dep = 1.0
    while ( fabs(dep) > 1.0e-12 ):
       dep = ( orbph - ( ep - e * sin(ep) ) ) / denom
       ep += dep

    omega = pbp_omega + pbp_omdot * t
    alpha = pbp_a1sini * sin(omega)
    beta = pbp_a1sini * cos(omega) * sqrt(1 - e * e)
    sbe = sin(ep)
    cbe = cos(ep)

    return 1. - (TWOPI / pbp_pb) *  (beta * cbe - alpha * sbe) / (1 - e * cbe)

