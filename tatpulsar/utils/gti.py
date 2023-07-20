#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for GTI manipulation
"""
import numpy as np

__all__ = ['gti_intersection',
           'gti_union',
           'create_gti_txt',
           'create_gti_fits']

def gti_intersection(gti1, gti2):
    """
    find the intersection for given gti1 and gti2.

    Parameters
    ----------
    gti1: list
        the list of input GTI, the format of gti list is a 2D list:
        [[start, stop], [start, stop], ... ,[start, stop]]

    gti2: list
        the list of input GTI, the format of gti list is a 2D list:
        [[start, stop], [start, stop], ... ,[start, stop]]

    Returns
    -------
    gti: list
        the list of output GTI, the format of gti list is a 2D list:
        [[start, stop], [start, stop], ... ,[start, stop]]
    """
    if not _is_2d_list(gti1):
        gti1 = _to_2d_list(gti1)
    if not _is_2d_list(gti2):
        gti2 = _to_2d_list(gti2)

    # Sort gti
    git1 = sort_gti(gti1)
    gti2 = sort_gti(gti2)

    # Initialize index variables for the two lists and the results list
    i = j = 0
    intersections = []

    # Iterate while neither list is exhausted
    while i < len(gti1) and j < len(gti2):
        # Get the start and stop values for current pair in each list
        start1, stop1 = gti1[i]
        start2, stop2 = gti2[j]

        # Find the intersection between the two pairs
        start_max = max(start1, start2)
        stop_min = min(stop1, stop2)

        # If the intersection exists, append it to the results
        if start_max < stop_min:
            intersections.append([start_max, stop_min])

        # Move to next pair in gti1 or gti2
        if stop1 < stop2:
            i += 1
        else:
            j += 1

    return intersections

def gti_union(gti1, gti2):
    """
    find the intersection for given gti1 and gti2.

    Parameters
    ----------
    gti1: list
        the list of input GTI, the format of gti list is a 2D list:
        [[start, stop], [start, stop], ... ,[start, stop]]

    gti2: list
        the list of input GTI, the format of gti list is a 2D list:
        [[start, stop], [start, stop], ... ,[start, stop]]

    Returns
    -------
    gti: list
        the list of output GTI, the format of gti list is a 2D list:
        [[start, stop], [start, stop], ... ,[start, stop]]
    """

    if not _is_2d_list(gti1):
        gti1 = _to_2d_list(gti1)
    if not _is_2d_list(gti2):
        gti2 = _to_2d_list(gti2)

    # Sort gti
    git1 = sort_gti(gti1)
    gti2 = sort_gti(gti2)

    # Combine the two lists and sort them
    combined = sorted(gti1 + gti2)

    # Initialize the results list with the first interval
    union = [combined[0]]

    # Iterate over the rest of the intervals
    for current_start, current_stop in combined[1:]:
        # If the current interval overlaps or is adjacent to the last one, merge them
        last_start, last_stop = union[-1]
        if current_start <= last_stop:
            union[-1][1] = max(last_stop, current_stop)
        else:
            # Otherwise, add the current interval as is
            union.append([current_start, current_stop])

    return union

def _gti_gap(gti):
    return np.array([gti[i+1][0] - gti[i][1] for i in range(len(gti)-1)])

def sort_gti(gti):
    """
    sort a 2d gti list by the order of start value
    """
    start = np.array([x[0] for x in gti])
    stop  = np.array([x[1] for x in gti])
    mask  = np.argsort(start)
    start = start[mask]
    stop  = stop[mask]
    return np.column_stack([start, stop]).tolist()

def _is_2d_list(gti):
    return isinstance(gti[0], list)

def _to_2d_list(gti):
    if isinstance(gti, np.ndarray):
        return gti.tolist()
    elif isinstance(gti, list):
        return [gti]

def create_gti_txt(outfile, tstart, tstop):
    """
    create a text file storing the TSTART and TSTOP value,
    it's normally used for hxmtscreen

    Parameters
    ----------
        outfile : filename
            the name of the output FITS file

        tstart : array-like
            array of GTIs start time

        tstop : array-like
            array of GTIs stop time
    """
    np.savetxt(outfile, np.c_[tstart, tstop])

def create_gti_fits(gti_template, outfile, tstart, tstop):
    """
    write tstart and tstop array to a FITS file, based on the
    GTI FITS template

    -----------
    Parameters:

        gti_template : FITS file
            a GTI fits file as template

        outfile : filename
            the name of the output FITS file

        tstart : array-like
            array of GTIs start time

        tstop : array-like
            array of GTIs stop time

    """
    # Copy information from old gtifile
    hdulist_template = fits.open(gti_template)
    headers_template = []
    gti_table_old = []
    for i in range(len(hdulist_template)):
        headers_template.append(
                hdulist_template[i].header)
        gti_table_old.append(
                hdulist_template[i].data)

    # Primary HDU
    primary = fits.PrimaryHDU(header=headers_template[0])

    # create extension GTI
    c1 = fits.Column(name='START', array=tstart, format='1D')
    c2 = fits.Column(name='STOP', array=tstop, format='1D')
    gti_extension = fits.BinTableHDU.from_columns([c1,c2])

    # create empty extension besides GTI and Primary HDU
    empty_tables = []
    for i in range(len(hdulist_template))[2:]:
        c1_tmp = fits.Column(name='dummy_a', array=[0,0],format='1D')
        c2_tmp = fits.Column(name='dummy_b', array=[0,0],format='1D')
        data_tmp = fits.BinTableHDU.from_columns([c1_tmp, c2_tmp])
        empty_tables.append(data_tmp)

    # write new fits file
    new_hdulist= fits.HDUList([primary, gti_extension] + empty_tables)
    new_hdulist.writeto(outfile, overwrite=True)

    # update header
    new_hdulist = fits.open(outfile, mode='update')

    for i in range(len(headers_template)):
        new_hdulist[i].header = headers_template[i]

    new_hdulist[1].header['EXTNAME'] = 'GTI0'

    # update dumpy extension
    for i in range(len(empty_tables)):
        new_hdulist[i+2].data = gti_table_old[i+2]
    new_hdulist.writeto(outfile, overwrite=True)
