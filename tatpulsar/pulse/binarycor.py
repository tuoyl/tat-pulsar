"""Pure Python fallback for binary orbital corrections.

This mirrors the algorithm implemented in :mod:`binarycor.pyx` so that the
package can run without compiling the Cython extension for every Python
version / platform combination.
"""

from __future__ import annotations

import math

__all__ = ["Cor", "Doppler"]

_SECDAY = 86400.0
_TWOPI = 2.0 * math.pi


def _orbital_phase(time: float, pbp_t0: float, pbp_pb: float, pbp_pbdot: float) -> float:
    """Return the orbital phase expressed in radians."""
    t = _SECDAY * (time - pbp_t0)
    dt = t / pbp_pb
    dt2 = 0.5 * dt * dt
    orbph = dt - dt2 * pbp_pbdot
    orbph -= math.floor(orbph)
    if orbph < 0.0:
        orbph += 1.0
    return _TWOPI * orbph, t


def _solve_eccentric_anomaly(orbph: float, eccentricity: float) -> float:
    """Solve Kepler's equation following the original Cython implementation."""
    ep = orbph + eccentricity * math.sin(orbph) * (1.0 + eccentricity * math.cos(orbph))
    denom = 1.0 - eccentricity * math.cos(ep)
    dep = 1.0
    while abs(dep) > 1.0e-12:
        dep = (orbph - (ep - eccentricity * math.sin(ep))) / denom
        ep += dep
    return ep


def _alpha_beta(eccentricity: float, pbp_a1sini: float, pbp_omega: float, t: float,
                pbp_omdot: float) -> tuple[float, float]:
    omega = pbp_omega + pbp_omdot * t
    alpha = pbp_a1sini * math.sin(omega)
    beta = pbp_a1sini * math.cos(omega) * math.sqrt(max(0.0, 1.0 - eccentricity * eccentricity))
    return alpha, beta


def _common_terms(time: float, pbp_t0: float, pbp_e: float, pbp_pb: float, pbp_pbdot: float,
                  pbp_omega: float, pbp_omdot: float, pbp_a1sini: float) -> tuple[float, ...]:
    orbph, t = _orbital_phase(time, pbp_t0, pbp_pb, pbp_pbdot)
    ep = _solve_eccentric_anomaly(orbph, pbp_e)
    alpha, beta = _alpha_beta(pbp_e, pbp_a1sini, pbp_omega, t, pbp_omdot)
    sbe = math.sin(ep)
    cbe = math.cos(ep)
    return alpha, beta, sbe, cbe


def Cor(time: float, pbp_t0: float, pbp_e: float, pbp_pb: float, pbp_pbdot: float,
        pbp_omega: float, pbp_omdot: float, pbp_a1sini: float, pbp_gamma: float) -> float:
    """Binary orbital delay correction."""
    alpha, beta, sbe, cbe = _common_terms(
        time, pbp_t0, pbp_e, pbp_pb, pbp_pbdot, pbp_omega, pbp_omdot, pbp_a1sini
    )
    q = alpha * (cbe - pbp_e) + (beta + pbp_gamma) * sbe
    torb = -q + (_TWOPI / pbp_pb) * q * (beta * cbe - alpha * sbe) / (1.0 - pbp_e * cbe)
    return torb


def Doppler(time: float, pbp_t0: float, pbp_e: float, pbp_pb: float, pbp_pbdot: float,
            pbp_omega: float, pbp_omdot: float, pbp_a1sini: float, pbp_gamma: float) -> float:
    """Return the Doppler factor converting emitted to observed frequency."""
    alpha, beta, sbe, cbe = _common_terms(
        time, pbp_t0, pbp_e, pbp_pb, pbp_pbdot, pbp_omega, pbp_omdot, pbp_a1sini
    )
    return 1.0 - (_TWOPI / pbp_pb) * (beta * cbe - alpha * sbe) / (1.0 - pbp_e * cbe)
