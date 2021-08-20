"""
This is the master script to process the histograms returned by the cuda or cpp
programs and produce a plot of the point correlation function. It receives and
parse command line options.
"""

import argparse
from pathlib import Path
import os

import numpy as np

from pcf_pyplots.pcf_plots import plot_2ani, plot_2iso, plot_3iso, plot_3ani

# Build paths inside the project like this: RESULTS_DIR / 'subdir'.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

def estim_2p_ls(histo_dd, histo_rr, histo_dr):
    """
    Computes the Landley-Saley estimator for 2 point correlation.

    Receives three histograms which are numpy arrays and compute the Landy-Szalay
    estimator for later plotting. This function manages the zero-divisions when
    histo_rr[i] == 0.
    """
    # Sets the same value in the three histograms where the histo_rr == 0 in a form
    # that the ls == 0 where histo_rr == 0
    rr_zeros = histo_rr == 0
    histo_dd[rr_zeros] = 1
    histo_dr[rr_zeros] = 1
    histo_rr[rr_zeros] = 1
    return (histo_dd - 2*histo_dr + histo_rr)/histo_rr

def estim_3p_ls(histo_dd, histo_rr, histo_rd, histo_dr):
    """
    Computes the Landley-Saley estimator for 3 point correlation.

    Receives four histograms which are numpy arrays and compute the Landy-Szalay
    estimator for later plotting. This function manages the zero-divisions when
    histo_rr[i] == 0.
    """
    # Sets the same value in the four histograms where the histo_rr == 0 in a form
    # that the ls == 0 where histo_rr == 0
    rr_zeros = histo_rr == 0
    histo_dd[rr_zeros] = 1
    histo_rd[rr_zeros] = 1
    histo_dr[rr_zeros] = 1
    histo_rr[rr_zeros] = 1
    return (histo_dd - 3*histo_rd + 3*histo_dr - histo_rr)/histo_rr

def get_filtered_paths(histo_type, loc_dir):
    """
    Return a list with path to files terminated in .dat with the suffix histo_type.
    """
    result_files = list(filter(lambda file:file[-4:]==".dat",os.listdir(loc_dir)))
    filtered_files = list(filter(lambda file:file[:2]==histo_type,result_files))
    pathto_files = list(map(lambda r_file : loc_dir / r_file, filtered_files))

    return pathto_files

def get_histogram_mean(histogram_paths, save_as):
    """
    Receives an iterable of path to files and reads them with numpy, performs
    the sum of all the histograms and saves it. Returns None if histogram_paths
    is an empty list
    """
    if not histogram_paths:
        return None
    for i, f_name in enumerate(histogram_paths):
        if i == 0:
            histo = np.loadtxt(f_name)
            continue
        histo += np.loadtxt(f_name)
    histo = histo/len(histogram_paths)
    np.savetxt(save_as,histo)
    return histo

def compute_plot(compute_type, **kwargs):
    """
    This function reads and joins all the provided instagrams, computes the
    appropiate point correlation function and plots the result.
    """

    # Get the histograms
    if kwargs["loc_dir"]:
        # Reads all the files in loc_dir and sums up the histograms wit same prefix
        loc_dir = RESULTS_DIR / kwargs["loc_dir"]

        # Declare and initialize the histograms with None
        histo_dd = get_histogram_mean(get_filtered_paths("DD", loc_dir), loc_dir / "dd.dat")
        histo_rr = get_histogram_mean(get_filtered_paths("RR", loc_dir), loc_dir / "rr.dat")
        histo_dr = get_histogram_mean(get_filtered_paths("DR", loc_dir), loc_dir / "dr.dat")
        histo_rd = get_histogram_mean(get_filtered_paths("RD", loc_dir), loc_dir / "rd.dat")

    else:
        loc_dd = RESULTS_DIR / kwargs["loc_dd"]
        histo_dd = np.loadtxt(loc_dd)
        loc_rr = RESULTS_DIR / kwargs["loc_rr"]
        histo_rr = np.loadtxt(loc_rr)
        if kwargs["loc_dr"] is not None:
            loc_dr = RESULTS_DIR / kwargs["loc_dr"]
            histo_dr = np.loadtxt(loc_dr)
        else:
            histo_dr = None
        if kwargs["loc_rd"] is not None:
            loc_rd = RESULTS_DIR / kwargs["loc_rd"]
            histo_rd = np.loadtxt(loc_rd)
        else:
            histo_rd = None

    # Analytic functions may not have mixed histograms (histo_rd or histo_dr)
    # then it is assumed they would have the same values as histo_rr
    if histo_dr is None:
        histo_rd = histo_rr
        histo_dr = histo_rr

    # Compute the L-S estimator
    # Plot
    if compute_type == "2iso":
        eps_ls = estim_2p_ls(histo_dd,histo_rr,histo_dr)
        plot_2iso(eps_ls, kwargs["dmax"])
    elif compute_type == "2ani":
        eps_ls = estim_2p_ls(histo_dd,histo_rr,histo_dr)
        plot_2ani(eps_ls, kwargs["dmax"])
    elif compute_type == "3iso":
        bins = kwargs["bins"]
        histo_dd = np.reshape(histo_dd, (bins,bins,bins))
        histo_rr = np.reshape(histo_rr, (bins,bins,bins))
        histo_rd = np.reshape(histo_rd, (bins,bins,bins))
        histo_dr = np.reshape(histo_dr, (bins,bins,bins))
        eps_ls = estim_3p_ls(histo_dd, histo_rr, histo_rd, histo_dr)
        plot_3iso(eps_ls, kwargs["dmax"])
    elif compute_type == "3ani":
        bins = kwargs["bins"]
        histo_dd = np.reshape(histo_dd, (bins,bins,bins,bins,bins))
        histo_rr = np.reshape(histo_rr, (bins,bins,bins,bins,bins))
        histo_rd = np.reshape(histo_rd, (bins,bins,bins,bins,bins))
        histo_dr = np.reshape(histo_dr, (bins,bins,bins,bins,bins))
        eps_ls = estim_3p_ls(histo_dd, histo_rr, histo_rd, histo_dr)
        plot_3ani(eps_ls, kwargs["dmax"])

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        usage="""
        %(prog)s [-h] (--dir DIRECTORY | [-DD LOC_DD -RR LOC_RR -DR LOC_DR [-RD LOC_RD] )
        [-b BINS] DMAX {2iso,2ani,3iso,3ani}
        """,
        description="""
        Computes the Point Correlation Function and plots the results. You
        must specify the location of the histograms and the type of Correlation
        Function that the histograms represent.
        It is assumed a path relative to the ./results/ path.
        """
    )
    parser.add_argument("compute_type", choices=["2iso", "2ani", "3iso", "3ani"],
        help="The type of computation the the histograms represent."
    )
    parser.add_argument("dmax", type=float,
        help="Maximum distance used for compute the histograms"
    )
    groupDD = parser.add_mutually_exclusive_group(required=True)
    groupDD.add_argument("-dir", dest="loc_dir", metavar="DIRECTORY", help="""
        Joins all the *.dat ended files in the directory. Every files started with
        DR into one file, all the files starting with RR into another file,
        all the files started with DD into another file, and all DDR into one file
        if a 3 point correlation is specified.
        """
    )
    groupDD.add_argument("-DD", dest="loc_dd", help="""
        Location of the DD histogram for 2 point correlation functions or 
        DDD for 3 point correlation functions.
        """
    )
    parser.add_argument("-RR", dest="loc_rr", help="""
        Location of the RR histogram for 2 point correlation functions or 
        RRR for 3 point correlation functions
        """
    )
    parser.add_argument("-DR", dest="loc_dr", help="""
        Location of the DR histogram for 2 point correlation functions or 
        DRR for 3 point correlation functions
        """
    )
    RD_arg = parser.add_argument("-RD", dest="loc_rd", help="""
        Location of the RDD only for 3 point correlation functions
        """
    )
    bins_arg = parser.add_argument("-b", "--bins", dest="bins", help="""
        Number of bins per histogram dimension. Required in 3 point correlation
        functions to properly reshape the readed files.
        """
    )
    args = parser.parse_args()
    if args.loc_dd and (not args.loc_dr or not args.loc_rr):
        parser.error(
            message = """
                You must specify the location of all the histograms with
                the options -DR and -RR.
                """
        )
    if "3" in args.compute_type:
        if not args.loc_rd and not args.loc_dir:
            parser.error(
                message = "The argument -RD or -d is required for 3 point correlation functions"
            )
        if not args.bins:
            parser.error(
                message = "The argument -b is required for 3 point correlation functions"
            )

    # Create the plot
    compute_plot(**vars(args))
