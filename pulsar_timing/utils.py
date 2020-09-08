#!/usr/bin/env python
import numpy as np
import numba 

__all__ = ["numba_histogram",
        "met2mjd",
        "mjd2met",
        "cal_chisquare",
        "cal_2dchisquare"]

def met2mjd(data, telescope="fermi"):
    if telescope.lower() == "fermi":
        MJDREFF = 0.00074287037037037
        MJDREFI = 51910
    elif telescope.lower() == "hxmt":
        MJDREFF = 0.00076601852000000
        MJDREFI = 55927
    return data/86400 + MJDREFI + MJDREFF

def mjd2met(data, telescope="fermi"):
    if telescope.lower() == "fermi":
        MJDREFF = 0.00074287037037037
        MJDREFI = 51910
    elif telescope.lower() == "hxmt":
        MJDREFF = 0.00076601852000000
        MJDREFI = 55927
    return (data - MJDREFI - MJDREFF)*86400



@numba.njit(nopython=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@numba.njit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@numba.njit(nopython=True)
def numba_histogram(a, bins):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges


@numba.njit
def cal_chisquare(data, f, pepoch, bin_profile, F1, F2, F3, F4):
    """
    calculate the chisquare distribution for frequency search on the pepoch time.
    """
    chi_square = np.zeros(len(f), dtype=np.float64)

    t0 = pepoch

    for i in range(len(f)):
        phi = (data-t0)*f[i] + (1.0/2.0)*((data-t0)**2)*F1 + (1.0/6.0)*((data-t0)**3)*F2 +\
                (1.0/24.0)*((data-t0)**4)*F3 + (1.0/120.0)*((data-t0)**5)*F4
        phi = phi - np.floor(phi)
        counts  = numba_histogram(phi, bin_profile)[0]
        chi_square[i] = np.sum( (counts - np.mean(counts))**2 / np.mean(counts) )

    return chi_square

@numba.njit
def cal_2dchisquare(data, f, pepoch, bin_profile, F1, F2, F3, F4):
    """
    calculate the chisquare distribution for 2-D frequency search on the pepoch time.
    """

    chi_square = np.zeros(len(f)*len(F1), dtype=np.float64).reshape(len(F1), len(f))
    # n X m matrix, with n F1 and m f for each F1
    print(chi_square.shape, len(F1))

    t0 = pepoch

    for i in range(len(F1)):
        for j in range(len(f)):
            phi = (data-t0)*f[j] + (1.0/2.0)*((data-t0)**2)*F1[i] + (1.0/6.0)*((data-t0)**3)*F2 +\
                    (1.0/24.0)*((data-t0)**4)*F3 + (1.0/120.0)*((data-t0)**5)*F4
            phi = phi - np.floor(phi)
            counts  = numba_histogram(phi, bin_profile)[0]
            chi_square[i][j] = np.sum( (counts - np.mean(counts))**2 / np.mean(counts) )
    return chi_square
