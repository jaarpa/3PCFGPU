"""
This module contains the functions to plot each point correlation function type.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
from numpy.lib.npyio import save
from scipy import ndimage
import numpy as np
import os

# Build paths inside the project like this: RESULTS_DIR / 'subdir'.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def estim_2p_ls(histo_dd, histo_rr, histo_dr):
    """
    Computes the Landley-Saley estimator for 2 point correlation.

    Receives three histograms which are numpy arrays and compute the Landy-Szalay
    estimator for later plotting. This function manages the zero-divisions when
    histo_rr[i] == 0.
    """
    # Sets the same value in the three histograms where the histo_rr == 0 in a form
    # that the ls == 0 where histo_rr == 0
    # norm = np.sum(histo_dd)/np.sum(histo_rr)
    # norm2 = np.sum(histo_dr)/np.sum(histo_rr)

    rr_zeros = histo_rr == 0
    histo_dd[rr_zeros] = 1
    histo_dr[rr_zeros] = 1
    histo_rr[rr_zeros] = 1
    histo_dd = histo_dd/histo_dd.sum()
    histo_dr = histo_dr/histo_dr.sum()
    histo_rr = histo_rr/histo_rr.sum()

    # return ((histo_dd/norm) - 2*(histo_dr/norm2) + histo_rr)/histo_rr
    return (histo_dd - 2*(histo_dr) + histo_rr)/histo_rr

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
    result_files = list(filter(lambda file:file[-4:]==".dat", os.listdir(loc_dir)))
    filtered_files = list(filter(lambda file:file[:2]==histo_type,result_files))
    pathto_files = list(map(lambda r_file : loc_dir / r_file, filtered_files))

    return pathto_files

def get_histogram_mean(histogram_paths, save_as = None):
    """
    Receives an iterable of path to files and reads them with numpy, performs
    the average of all the histograms and saves it. Returns None if histogram_paths
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
    if save_as:
        np.savetxt(save_as,histo)
    return histo

def get_histogramfrom_dir(loc_dir, compute_type, bins=None):
    """
    This function receives one path to a directory where many histogram of the same
    computation are stored, obtains the average for each kind of histogram, reshape
    and returns them
    """
    loc_dir = RESULTS_DIR / loc_dir

    histo_dd = get_histogram_mean(get_filtered_paths("DD", loc_dir), loc_dir / "dd.dat")
    histo_rr = get_histogram_mean(get_filtered_paths("RR", loc_dir), loc_dir / "rr.dat")
    histo_dr = get_histogram_mean(get_filtered_paths("DR", loc_dir), loc_dir / "dr.dat")
    histo_rd = get_histogram_mean(get_filtered_paths("RD", loc_dir), loc_dir / "rd.dat")

    if compute_type == "3iso":
        if not bins:
            raise Exception('You must specify the number of bins')
        histo_dd = np.reshape(histo_dd, (bins,bins,bins))
        histo_rr = np.reshape(histo_rr, (bins,bins,bins))
        histo_rd = np.reshape(histo_rd, (bins,bins,bins))
        histo_dr = np.reshape(histo_dr, (bins,bins,bins))
    elif compute_type == "3ani":
        if not bins:
            raise Exception('You must specify the number of bins')
        histo_dd = np.reshape(histo_dd, (bins,bins,bins,bins,bins))
        histo_rr = np.reshape(histo_rr, (bins,bins,bins,bins,bins))
        histo_rd = np.reshape(histo_rd, (bins,bins,bins,bins,bins))
        histo_dr = np.reshape(histo_dr, (bins,bins,bins,bins,bins))

    return histo_dd, histo_rr, histo_dr, histo_rd

def plot_2ani(eps_ls, dmax):
    """
    Receives the Landy-Szalay estimator and plots for the 2 points anisotropic
    correlation function.

    The ls estimator is a 1-D array
    """

    limit = 0.002*eps_ls.max()
    cmap = 'RdBu'

    plt.figure(figsize=(6,6), dpi=100)
    plt.imshow(
        eps_ls, origin='lower', cmap=cmap, extent=[0,dmax,0,dmax],
        interpolation= 'bilinear', vmin=-limit, vmax=limit
    )
    cax=plt.colorbar()
    plt.ylabel('$r_{\\pi}$', fontsize = 16)
    plt.xlabel('$r_{p}$', fontsize = 16)
    plt.title('Correlation function', fontsize = 16)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax.set_label('$\\epsilon(r)$', labelpad = 15,fontsize = 15)
    plt.plot()

    plt.figure(figsize = (7,7))
    gs1 = gridspec.GridSpec(2, 2)
    # Set the spacing between axes
    gs1.update(wspace=0.0, hspace=0.0)

    plt.subplot(gs1[0])
    rotated_img = ndimage.rotate(np.rot90(eps_ls), 90)
    plt.imshow(
        rotated_img, cmap=cmap,interpolation= 'bilinear', vmin=-limit, vmax=limit
    )
    plt.contour(
        rotated_img, 10, cmap=plt.cm.get_cmap('gray'), linewidths=1,
        vmin=-limit, vmax=limit
    )
    plt.axis('off')

    plt.subplot(gs1[1])
    rotated_img = ndimage.rotate(eps_ls.T,90)
    plt.imshow(
        rotated_img, cmap=cmap,interpolation= 'bilinear', vmin=-limit, vmax=limit
    )
    plt.contour(
        rotated_img, 10, cmap=plt.cm.get_cmap('gray'), linewidths=1,
        vmin=-limit, vmax=limit
    )
    plt.axis('off')

    plt.subplot(gs1[2])
    rotated_img = ndimage.rotate(eps_ls.T,-90)
    plt.imshow(
        rotated_img, cmap=cmap,interpolation= 'bilinear', vmin=-limit, vmax=limit
    )
    plt.contour(
        rotated_img, 10, cmap=plt.cm.get_cmap('gray'), linewidths=1,
        vmin=-limit, vmax=limit
    )
    plt.axis('off')

    plt.subplot(gs1[3])
    rotated_img = ndimage.rotate(eps_ls, 0)
    plt.imshow(
        rotated_img, cmap=cmap,interpolation= 'bilinear', vmin=-limit, vmax=limit
    )
    plt.contour(
        rotated_img, 10, cmap=plt.cm.get_cmap('gray'), linewidths=1,
        vmin=-limit, vmax=limit
    )
    plt.axis('off')
    plt.show()

def plot_2iso(eps_ls, dmax):
    """
    Receives the Landy-Szalay estimator and plots for the 2 points isotropic
    correlation function.

    The ls estimator is a 2-D array
    """
    bins = eps_ls.size
    distance_r = np.linspace(0,dmax,bins)

    plt.figure(figsize=(14,8))
    plt.scatter(distance_r,eps_ls, s=50, c='g',label='Landley-Saley')
    plt.plot(distance_r,eps_ls,'k-')
    plt.xlabel('r',fontsize=18)
    plt.ylabel('$\\epsilon(r)$',fontsize=18)
    plt.legend(shadow=True, fontsize='x-large')
    plt.grid()
    plt.show()

    plt.figure(figsize=(14,8))
    plt.scatter(distance_r,distance_r**2*eps_ls, s=50, c='g',label='Landley-Saley')
    plt.plot(distance_r,distance_r**2*eps_ls,'k-')
    plt.xlabel('r',fontsize=18)
    plt.ylabel('$r^2 \\cdot \\epsilon(r)$',fontsize=18)
    plt.legend(shadow=True, fontsize='x-large')
    plt.grid()
    plt.show()

def plot_3iso(eps_ls, dmax):
    """
    Receives the Landy-Szalay estimator and plots for the 3 points isotropic
    correlation function.

    The ls estimator is a 3-D array
    """
    bins = eps_ls.shape[0]
    limt = 0.01
    for histo_dim in range(bins):
        plt.figure(figsize=(6,6), dpi=100)
        plt.imshow(
            eps_ls[histo_dim], origin='lower', cmap='RdBu', interpolation= 'bilinear',
            extent=[0,dmax,0,dmax], vmin=-limt, vmax=limt
        )
        cax=plt.colorbar()
        plt.contour(
            eps_ls[histo_dim], 10, cmap=plt.cm.get_cmap('gray'), linewidths=1,
            extent=[0,dmax,0,dmax], vmin=-limt, vmax=limt
        )
        plt.ylabel('$r_1$',fontsize = 16)
        plt.xlabel('$r_2$',fontsize = 16)
        plt.title('3PCF',fontsize = 16)
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax.set_label(
            '$\\epsilon(r_1,r_2,r_3={0})$'.format(histo_dim*(dmax)/bins),
            labelpad = 15, fontsize = 15
        )
        plt.show()

def plot_3ani(eps_ls, dmax):
    """
    Receives the Landy-Szalay estimator and plots for the 3 points anisotropic
    correlation function.

    The ls estimator is a 5-D array
    """
    print((eps_ls, dmax))
