#!/usr/bin/env python
"""Utilities for computing pulse times of arrival (ToAs)."""

from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing import Dict, Iterable, Optional, Tuple

from scipy.optimize import basinhopping

from tatpulsar.data.profile import Profile
from tatpulsar.utils.functions import ccf

__all__ = ["cal_toa"]


def cal_toa(
    fbest: float,
    profile: Profile,
    method: str = "max",
    ref_time: Optional[float] = None,
    debug: bool = False,
    std_pro: Optional[Profile] = None,
    nsteps: int = 100,
    phi_range: Optional[Iterable[float]] = None,
    **kwargs,
) -> Tuple[float, float]:
    """Calculate the Time of Arrival (ToA) of a pulse profile.

    Parameters
    ----------
    fbest : float
        Best-frequency estimate (in Hz) for the reference epoch.
    profile : Profile
        The pulse profile to analyse.
    method : {"max", "ccf", "fft"}, optional
        Algorithm used for locating the pulse phase. Default is "max".
    ref_time : float, optional
        Reference time (seconds). If not provided, ``profile.ref_time`` is used.
    debug : bool, optional
        When ``True`` produce a diagnostic plot (matplotlib required).
    std_pro : Profile, optional
        Standard profile used when ``method`` is "ccf" or "fft".
    nsteps : int, optional
        Number of Monte-Carlo resamplings employed to estimate the phase error.
        Must be a positive integer. Default is 100.
    phi_range : (float, float), optional
        Restrict the phase search to this inclusive range. Only meaningful for
        the "max" method.
    **kwargs : dict
        Additional keyword arguments; ``rng`` can be supplied to pass a numpy
        random generator with a ``poisson`` method.

    Returns
    -------
    tuple (toa, toa_err)
        The ToA in seconds and its 1-sigma uncertainty.
    """

    profile = _ensure_profile(profile)
    method = method.lower()
    nsteps = _ensure_positive_int(nsteps, "nsteps")
    mask = _build_phase_mask(profile, phi_range)
    rng = kwargs.get("rng")

    handlers = {
        "max": _toa_via_max,
        "ccf": _toa_via_ccf,
        "fft": _toa_via_fft,
    }
    if method not in handlers:
        raise ValueError(f"Unsupported method '{method}'")

    delta_phi, delta_phis, debug_info = handlers[method](
        profile=profile,
        mask=mask,
        std_pro=std_pro,
        nsteps=nsteps,
        rng=rng,
    )

    if debug:
        _plot_debug(profile, method, delta_phi, delta_phis, debug_info, phi_range)

    if ref_time is None:
        ref_time = profile.ref_time
    if ref_time is None:
        raise IOError("ref_time is required")

    toa = ref_time + delta_phi / fbest
    toa_err = 0.0 if delta_phis.size == 0 else np.std(delta_phis) / fbest
    return toa, toa_err


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_profile(value: Optional[Profile], name: str = "profile") -> Profile:
    if not isinstance(value, Profile):
        raise TypeError(f"The input {name} is not tatpulsar.data.Profile object")
    return value


def _ensure_positive_int(value: int, name: str) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a positive integer") from exc
    if ivalue < 1:
        raise ValueError(f"{name} must be a positive integer")
    return ivalue


def _build_phase_mask(profile: Profile, phi_range: Optional[Iterable[float]]) -> np.ndarray:
    if phi_range is None:
        return np.ones(profile.counts.size, dtype=bool)

    try:
        start, stop = phi_range
    except (TypeError, ValueError) as exc:
        raise ValueError("phi_range must contain exactly two values") from exc

    if start >= stop:
        raise ValueError("phi_range must satisfy start < stop")

    mask = (profile.phase >= start) & (profile.phase <= stop)
    if not np.any(mask):
        raise ValueError("phi_range selects no phase bins")
    return mask


def _draw_poisson(lam: np.ndarray, nsteps: int, rng) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    size = (nsteps,) + lam.shape
    if rng is None:
        samples = np.random.poisson(lam, size=size)
    else:
        if not hasattr(rng, "poisson"):
            raise TypeError("rng must provide a 'poisson' method")
        samples = rng.poisson(lam, size=size)
    if nsteps == 1:
        samples = samples.reshape(1, *lam.shape)
    return samples


def _toa_via_max(
    profile: Profile,
    mask: np.ndarray,
    std_pro: Optional[Profile],
    nsteps: int,
    rng,
) -> Tuple[float, np.ndarray, Dict[str, object]]:
    phase = profile.phase[mask]
    counts = profile.counts[mask]
    idx = int(np.argmax(counts))
    delta_phi = float(phase[idx])

    samples = _draw_poisson(counts, nsteps, rng)
    sample_indices = np.argmax(samples, axis=1)
    delta_phis = phase[sample_indices].astype(float)

    if (delta_phi < 0.25) or (delta_phi > 0.75):
        delta_phis = np.where(delta_phis < 0.5, delta_phis + 1.0, delta_phis)

    return delta_phi, delta_phis, {}


def _toa_via_ccf(
    profile: Profile,
    mask: np.ndarray,
    std_pro: Optional[Profile],
    nsteps: int,
    rng,
) -> Tuple[float, np.ndarray, Dict[str, object]]:
    std_profile = _ensure_profile(std_pro, "std_pro")

    counts = profile.counts
    template_counts = std_profile.counts
    template_phase = std_profile.phase

    _, shift = ccf(counts, template_counts)
    aligned_template = np.roll(template_counts, shift)
    delta_phi = float(np.mod(template_phase[np.argmax(aligned_template)], 1.0))

    samples = _draw_poisson(counts, nsteps, rng)
    sample_phases = []
    for sample in samples:
        _, shift_tmp = ccf(sample, template_counts)
        aligned = np.roll(template_counts, shift_tmp)
        sample_phi = template_phase[np.argmax(aligned)]
        sample_phases.append(float(np.mod(sample_phi, 1.0)))

    return delta_phi, np.asarray(sample_phases), {"std_pro": std_profile, "shift": shift}


def _toa_via_fft(
    profile: Profile,
    mask: np.ndarray,
    std_pro: Optional[Profile],
    nsteps: int,
    rng,
) -> Tuple[float, np.ndarray, Dict[str, object]]:
    std_profile = _ensure_profile(std_pro, "std_pro")

    counts = profile.counts
    template_counts = std_profile.counts
    template_phase = std_profile.phase

    p0 = [1.0, 0.0]
    result = basinhopping(obj_fun, p0, minimizer_kwargs={"args": ([counts, template_counts],)})
    amp, shift = result.x

    peak_phase = template_phase[np.argmax(template_counts)]
    delta_phi = float(np.mod(peak_phase + shift, 1.0))

    samples = _draw_poisson(counts, nsteps, rng)
    sample_phases = []
    for sample in samples:
        res_tmp = basinhopping(obj_fun, p0, minimizer_kwargs={"args": ([sample, template_counts],)})
        _, shift_tmp = res_tmp.x
        sample_phi = peak_phase + shift_tmp
        sample_phases.append(float(np.mod(sample_phi, 1.0)))

    debug_info = {"std_pro": std_profile, "shift": shift, "amp": amp}
    return delta_phi, np.asarray(sample_phases), debug_info


def _plot_debug(
    profile: Profile,
    method: str,
    delta_phi: float,
    delta_phis: np.ndarray,
    debug_info: Dict[str, object],
    phi_range: Optional[Iterable[float]],
) -> None:
    import matplotlib.pyplot as plt

    profile_tmp = deepcopy(profile)
    profile_tmp.cycles = 2
    profile_tmp.norm()

    plt.errorbar(
        profile_tmp.phase,
        profile_tmp.counts,
        profile_tmp.error,
        ds="steps-mid",
        c="k",
        label="profile",
    )

    std_profile = debug_info.get("std_pro")
    shift = debug_info.get("shift")
    amp = debug_info.get("amp", 1.0)

    if std_profile is not None and shift is not None:
        std_tmp = deepcopy(std_profile)
        std_tmp.cycles = 2
        std_tmp.norm()
        if method == "fft":
            new_ph = (std_tmp.phase + shift) % 1.0
            plt.errorbar(
                new_ph,
                std_tmp.counts * amp,
                std_tmp.error * amp,
                c="r",
                ds="steps-mid",
                label="template",
            )
        else:
            plt.errorbar(
                std_tmp.phase,
                np.roll(std_tmp.counts, shift),
                np.roll(std_tmp.error, shift),
                c="r",
                ds="steps-mid",
                label="template",
            )

    plt.axvline(delta_phi, color="b", linestyle="--", label="delta_phi")
    for value in delta_phis:
        plt.axvline(value, lw=0.2, color="gray")

    if phi_range is not None:
        start, stop = phi_range
        plt.axvline(start, ls="dotted", color="gray")
        plt.axvline(stop, ls="dotted", color="gray")

    spread = np.std(delta_phis) if delta_phis.size else 0.0
    plt.axvspan(delta_phi - spread, delta_phi + spread, color="r", alpha=0.3)
    plt.legend()
    plt.title(f"ToA method: {method}")
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Legacy helpers retained for compatibility
# ---------------------------------------------------------------------------


def fftfit_fun(profile, template, amplitude, phase):
    """Objective function to be minimised - à la Taylor (1992)."""

    prof_ft = np.fft.fft(profile)
    temp_ft = np.fft.fft(template)
    freq = np.fft.fftfreq(len(profile))
    good = freq > 0
    idx = np.arange(0, prof_ft.size, dtype=int)
    sigma = np.std(prof_ft[good])
    return np.sum(np.absolute(prof_ft - temp_ft * amplitude * np.exp(-2 * np.pi * 1.0j * idx * phase)) ** 2 / sigma)


def obj_fun(pars, data):
    """Wrap parameters for minimisation algorithms."""

    amplitude, phase = pars
    profile, template = data
    return fftfit_fun(profile, template, amplitude, phase)


def _get_error_quantiles(data, toa, low=0.16, mid=0.5, hig=0.84):
    """Return symmetric quantile error estimates (legacy helper)."""

    data_low = sorted(data)[int(low * len(data))]
    data_mid = sorted(data)[int(mid * len(data))]
    data_hig = sorted(data)[int(hig * len(data))]
    if data_low == data_hig:
        return 0
    return max(abs(data_low - data_mid), abs(data_hig - data_mid))


def _get_error_rms(toas, toa):
    """Return RMS-based error estimate (legacy helper)."""

    rms = np.sqrt(np.mean(toas**2))
    return abs(toa - rms)


def _get_error_by_profile_shape(period, peak_sigma, source_counts, background_counts):
    """Deneva et al. (2019) equation 1."""

    a = period * peak_sigma
    b = source_counts / np.sqrt(source_counts + background_counts)
    sigma_toa = a / b
    return sigma_toa


def _calculate_delta_phi_by_ccf(profile, profile_std):
    """Legacy wrapper around ``ccf`` returning the maximum phase shift."""

    _, delay = ccf(profile, profile_std)
    p_num_std = np.roll(profile_std, delay)
    p_num_x = np.linspace(0, 1, len(profile))
    delta_phi = p_num_x[np.argmax(p_num_std)]
    return delta_phi

