#!/usr/bin/env python
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tatpulsar.pulse.residuals import cal_residual,argparse_wildcat, parse_pfiles, read_toa

def parse_args():
    #required
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p","--parfile", nargs='+', help="parameter file of pulsar", required=True)
    parser.add_argument("-t","--timfile", help="parameter file of pulsar", required=True)

    #optional
    parser.add_argument("--period", action='store_true', help="show residuals in period", required=False)
    parser.add_argument("--saveresi", type=str, help="show residuals in period", required=False)
    parser.add_argument("--savefig", type=str, help="show residuals in period", required=False)
    parser.add_argument("--notitle", action='store_true', help="show rms image title", required=False)
    parser.add_argument("--sourcename", type=str, help="assign the name of source", required=False)

    return parser.parse_args()

def main():
    """
    calculate the residuals and show the plots
    """

    args = parse_args()

    parameters = parse_pfiles(argparse_wildcat(
        args.parfile))
    toas, toa_errs = read_toa(args.timfile)

    x, y, yerr, yrms = cal_residual(toas, toa_errs, *parameters, inperiod=args.period)

    if args.saveresi:
        np.savetxt(args.saveresi, np.c_[x, y, yerr])

    fig, ax1 = plt.subplots()

    ax1.errorbar(x, y, yerr=yerr, fmt='o')
    ax1.ticklabel_format(axis='y',
            style='sci')
    ax1.set_xlabel("MJD", fontsize=12)
    if args.period:
        ax1.set_ylabel("Postfit Residual in pulse period ", fontsize=12)
    else:
        ax1.set_ylabel("Postfit Residual (sec) ", fontsize=12)

    if not args.notitle:
        if not args.sourcename:
            args.sourcename = ''
        ax1.set_title("{} Wrms  =  {:.3f} $\mu$s post-fit".format(args.sourcename, yrms*1e6))

    if args.savefig:
        ax1.savefig(args.savefig)
    plt.show()




if __name__ == "__main__":
    main()
