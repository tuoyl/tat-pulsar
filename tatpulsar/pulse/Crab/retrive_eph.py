from urllib.request import urlopen
import os
from astropy.io import ascii
import pandas as pd
import numpy as np

def retrieve_ephemeris(write_to_file=True, ephfile='Crab.gro'):
    """
    retrieve the Jordrell Bank Ephemeris of Crab pulsar
    return the Pandas DataFrame.

    Parameters
    ----------
    write_to_file : boolean
        Whether write the retrived ephemeris table into a TXT file

    ephfile : str
        If write the Table to a TXT file, then assign the TXT file name to
        write

    Returns
    -------
    df : Pandas DataFrame
        Return the Pandas DataFrame
    """
    url = "http://www.jb.man.ac.uk/pulsar/crab/all.gro"

    response = urlopen(url)
    data = response.read()
    if write_to_file:
        with open(ephfile, "wb") as out_file:
            out_file.write(
                b"PSR_B     RA(J2000)    DEC(J2000)   MJD1  MJD2    t0geo(MJD) "
                b"        f0(s^-1)      f1(s^-2)     f2(s^-3)  RMS O      B    "
                b"Name      Notes\n"
            )
            out_file.write(
                b"------- ------------- ------------ ----- ----- ---------------"
                b" ----------------- ------------ ---------- ---- -  ------ "
                b"------- -----------------\n"
            )

            out_file.write(data)

    df = pd.read_table(ephfile, delim_whitespace=True,
            names=['PSR_B', 'RA_hh', 'RA_mm', 'RA_ss',  'DEC_hh', 'DEC_mm', 'DEC_ss',\
                    'MJD1', 'MJD2', 't0geo', 'f0', 'f1', 'f2',\
                    'RMS', 'O', 'B', 'name', 'Notes'], skiprows=2)

    df.f1 = np.array([float(x.replace('D', 'e')) for x in df.f1.values])
    df.f2 = np.array([float(x.replace('D', 'e')) for x in df.f2.values])

    return df

def get_par(mjd, eph_df):
    """
    get the best JBL Crab ephemeris for a given time

    Parameters
    ----------
    mjd : float
        the time (MJD) to get the corresponding ephemeris parameters

    eph_df : DataFrame
        the pandas DataFrame (return from function :meth:`retrieve_ephemeris`)

    Returns
    -------
    par : list
        The best parameter list
    """

    mjd1 = eph_df.MJD1.values
    mjd2 = eph_df.MJD2.values
    t0geo = eph_df.t0geo.values

    index = np.min([
        np.searchsorted(mjd1, mjd, side='left'),
        np.searchsorted(mjd2, mjd, side='left')])
    return eph_df.loc[index]

