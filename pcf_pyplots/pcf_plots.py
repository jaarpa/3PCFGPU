"""
This module contains the functions to plot each point correlation function type.
"""
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import ndimage
import numpy as np

def plot_2ani(eps_ls, dmax):
    """
    Receives the Landy-Szalay estimator and plots for the 2 points anisotropic
    correlation function.

    The ls estimator is a 1-D array
    """

    limit = 0.02
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
