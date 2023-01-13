#!/usr/bin/env python

import numpy as np
import argparse
import warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import SpanSelector, Button, TextBox
import matplotlib as mpl
import pint
from astropy.io import fits

from tatpulsar.data import Profile
from tatpulsar.pulse.fold import fold2d
from tatpulsar.pulse.residuals import parse_pfiles
from tatpulsar.utils.functions import met2mjd, mjd2met

plt.style.use(['science', 'nature'])

def parse_args():
    #required
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e","--eventfile", nargs='+', help="The list of event file", required=True)
    parser.add_argument("-b","--nbins", type=int, help="number of bins of Profile", required=True)
    parser.add_argument("-n","--nseg", type=int, help="number of segments to divid", required=True)
    parser.add_argument("-c", "--colname", type=str,
                        help="The name of Column to be loaded, default is 'TDB'", required=False, default='TDB')
    parser.add_argument("-x", "--extnum", type=str,
                        help="The extension number to load event data (index start from 0)", required=False, default=1)

    #optinal
    parser.add_argument("-p","--parfile", nargs='+',
                        help="parameter file of pulsar", required=False, default=None)
    parser.add_argument("-r","--pepoch", type=float,
                        help="reference time (in MJD)", required=False)
    parser.add_argument("-t","--telescope", type=str,
                        help="the mission of data (e.g. 'hxmt', 'fermi', 'nicer'... see tatpulsar.utils.functions.mjd2met for details)"+\
                                " The default is using the 'TELESCOP' keyword in FITS header", required=False)
    parser.add_argument("-nu","--nu", type=float,
                        help="spinning frequency", required=False, default=0)
    parser.add_argument("-nudot","--nudot", type=float,
                        help="spinning frequency derivative", required=False, default=0)
    parser.add_argument("-nuddot","--nuddot", type=float,
                        help="spinning frequency second derivative", required=False, default=0)
    parser.add_argument("-nudddot","--nudddot", type=float,
                        help="spinning frequency third derivative", required=False, default=0)

    #optional
    parser.add_argument("--dpi", type=int, help='dpi value for output figure, default is 250', required=False,
                        default=250)
    parser.add_argument("--savefig", type=str, help="show residuals in period", required=False)
    parser.add_argument("--sourcename", type=str, help="assign the name of source", required=False)

    return parser.parse_args()

def _get_mission_name(hdulist):
    """
    Read the header of FITS file and return the
    corresponding mission name.

    "GLAST" --> 'fermi'
    "HXMT"  --> 'hxmt'
    "NICER" --> 'nicer'
    "NuSTAR"--> 'nustar'
    "IXPE"  --> "ixpe"
    """
    if 'TELESCOP' in hdulist[1].header:
        TELESCOP = hdulist[1].header['TELESCOP']
    else:
        return None
    if TELESCOP.lower() == "glast":
        return 'fermi'
    else:
        return TELESCOP.lower()

def _GUI_time_selection(fig, ax1, ax2, profiles, time):
    """GUI for time range selection
    profiles is a ndarray for a list of profiles
    y is the value of y axis.
    """

    def toggle_selector(event):
        #TODO press key to interact
        pass

    def onselect(ymin, ymax):
        indmin, indmax = np.searchsorted(time, (ymin, ymax))
        fig.canvas.mpl_connect('key_press_event', toggle_selector)

        new_cum_profile = Profile(np.sum(profiles[indmin:indmax], axis=0))

        ax1.clear()
        ax1.errorbar(new_cum_profile.phase + 1/new_cum_profile.phase.size/2,
                     new_cum_profile.counts,
                     new_cum_profile.error, ds='steps-mid')
        ax1.set_ylabel("Counts")
    span = SpanSelector(ax2, onselect, 'vertical', useblit=True,
            props=dict(alpha=0.5, facecolor='red'), interactive=True)
    return span

def main():
    """
    fold the FITS data using the giving pulsar parfile into 2D histogram Profile.
    And show the Plot with graphic user interface.
    """

    args = parse_args()

    data = np.array([])
    for file in args.eventfile:
        print(f'File {file} is now loaded')
        hdulist = fits.open(file)
        data = np.append(data,
                         hdulist[args.extnum].data[args.colname])

    # Get telescope parameter
    telescope = _get_mission_name(hdulist)
    if args.telescope:
        telescope = args.telescope
    if telescope is None:
        raise IOError("Could not find proper telescope value in FITS header, use --telescope to assign")

    # get pulsar spinning parameters
    if args.parfile:
        frequencies, pepoch, start_time, stop_time = \
                parse_pfiles(args.parfile)
        f0, f1, f2, f3, f4, *_ = frequencies

        pepoch = mjd2met(pepoch,
                         telescope=telescope)
    else:
        # pepoch is in MJD format
        pepoch = mjd2met(args.pepoch,
                         telescope=telescope)
        f0, f1, f2, f3, f4 = args.nu, args.nudot, args.nuddot, args.nudddot, 0

    profile = fold2d(data, y=data,
                     nbins=args.nbins,
                     nseg=args.nseg,
                     pepoch=pepoch, format='met',
                     f0=f0,
                     f1=f1,
                     f2=f2,
                     f3=f3,
                     f4=f4)

    profile_slices = np.asarray([x.counts for x in profile])
    cum_profile = Profile(np.sum(np.asarray([x.counts for x in profile]), axis=0))
    for x in profile:
        x.norm()

    # Plot
    mpl.rcParams['figure.dpi'] = args.dpi
    mpl.rcParams['figure.figsize'] = (4,4)

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    gs = gridspec.GridSpec(2,1,height_ratios=[1,4])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    fig.subplots_adjust(hspace=0.2)
    ax1.set_title("Cumulative Profile", fontsize=6, pad=1)
    ax1.set_ylabel("Counts")
    ax1.errorbar(cum_profile.phase+1/args.nbins/2, cum_profile.counts,
                 cum_profile.error, ds='steps-mid')

    heatmap = ax2.imshow([x.counts for x in profile], aspect='auto',
                         origin='lower',
                         extent=[0, 1, data.min(), data.max()],
                         cmap='jet')
    ax2.set_title("Time-resolved Profiles", fontsize=6, pad=1)
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Time")

    time_edges = np.linspace(data.min(), data.max(), args.nseg)
    span = _GUI_time_selection(fig, ax1, ax2, profile_slices, time_edges)
    plt.show()

if __name__ == "__main__":
    main()
