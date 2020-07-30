import numpy as np
import matplotlib.pyplot as plt
#from itertools import combinations
import time


def pcf2_ani_z_histo(data_location='fake_DATA/DATOS/data_500.dat',rand_location='fake_DATA/DATOS/rand0_500.dat', d_max=180.0, bins_number=30):
    """
    Calculates the DD, RR, DR bidimensional histograms for the anisotropic points given in the data and random text files.  Both files must have the same number of points with three dimesions.
    
    args:
        -data_location: str. It is the file location of the data. Each row a different point with the coordinates (x,y,z) as the first three columns separated with blank spaces.
        -rand_location: str. It is the file location of the file with random points. Each row a different point with the coordinates (x,y,z) as the first three columns separated with blank spaces.
        -d_max: float. The maximum distance that will be considered in the histogram
        -bins_number: int. The number of bins to use to separate the data.
        
    return:
        -DD: np.array(bins_number,bins_number). Array with the frequencies of the distances between data points and data points
        -RR: np.array(bins_number,bins_number). Array with the frequencies of the distances between random points and random points
        -DR: np.array(bins_number,bins_number). Array with the frequencies of the distances between data points and random points
        -x_edges: np.array(bins_number). With the edges of the bins for later xaxis plot
        -y_edges: np.array(bins_number). With the edges of the bins for later yaxis plot
    
    """
    
    #d_max = d_max**2
    data = np.loadtxt(fname=data_location, delimiter=" ", usecols=(0,1,2))
    rand0 = np.loadtxt(fname=rand_location, delimiter=" ", usecols=(0,1,2))
        
    if not data.shape == rand0.shape:
        raise Exception("The data file and rand file do not have the same size")
        
    #start = time.perf_counter()
    DR = np.zeros((bins_number,bins_number))
    for point in data:
        distances = point-rand0
        d_ll = np.abs(distances[:,2])
        d_T = np.sqrt(np.sum(distances[:,:2]**2,1))
        DR_temp, x_edges, y_edges = np.histogram2d(d_ll, d_T, bins=bins_number, range=[[0, d_max],[0, d_max]])
        #DR_temp, bins_DR = np.histogram(np.sqrt(np.sum((point-rand0)**2,1)), bins=bins_number, range=(0, d_max))
        DR += DR_temp
    #end = time.perf_counter()
    #print(f'{end-start} for the DR histogram')

    #start = time.perf_counter()
    DD = np.zeros((bins_number,bins_number))
    RR = np.zeros((bins_number,bins_number))
    for i, points in enumerate(zip(data,rand0),1):
        d_distances = points[0]-data[i:]
        #d_distances = np.array([d[0]-d[1] for d in list(combinations(data[:n],2))])
        d_ll = np.abs(d_distances[:,2])
        d_T = np.sqrt(np.sum(d_distances[:,:2]**2,1))
        DD_temp, x_edges, y_edges = np.histogram2d(d_ll, d_T, bins=bins_number, range=[[0, d_max],[0, d_max]])
        DD += DD_temp
        
        r_distances = points[1]-rand0[i:]
        r_ll = np.abs(r_distances[:,2])
        r_T = np.sqrt(np.sum(r_distances[:,:2]**2,1))
        RR_temp, x_edges, y_edges = np.histogram2d(r_ll, r_T, bins=bins_number, range=[[0, d_max],[0, d_max]])
        RR += RR_temp
    
    DD *=2
    RR *=2
    
    #end = time.perf_counter()
    #print(f'{end-start} for the DD, RR histogram')
        
    return DD, RR, DR, x_edges, y_edges

#Landy-Szalay
def LS_cf(DD, RR, DR):
    """
    Calculates the two dimentional Landy-Szalay correlation function estimator from the DD, RR and DR histograms.
    args:
        -DD: numpy array. Histogram with the data-data distances
        -RR: numpy array. Histogram with the random-random distances
        -DR: numpy array. Histogram with the random-data distances
    return:
        -LS: numpy array. Correlation function estimator
    """
    return (DD - 2*DR + RR)/RR
    
#Hamilton
def HM_cf(DD, RR, DR):
    """
    Calculates the two dimentional Hamilton correlation function estimator from the DD, RR and DR histograms.
    args:
        -DD: numpy array. Histogram with the data-data distances
        -RR: numpy array. Histogram with the random-random distances
        -DR: numpy array. Histogram with the random-data distances
    return:
        -LS: numpy array. Correlation function estimator
    """
    return (DD*RR/DR**2) - 1


"""
start = time.perf_counter()

#d_max = 180
#bins_number = 30
DD, RR, DR, x_edges, y_edges = pcf2_ani_z_histo(data_location='fake_DATA/DATOS/data.dat',rand_location='fake_DATA/DATOS/rand0.dat')

end = time.perf_counter()
print(f'Took {end-start} seconds to calculate DD, RR, and DR histograms')

LS = LS_cf(DD, RR, DR)
HM = HM_cf(DD, RR, DR)

def imag(x,y,t,cmap):
    plt.figure(figsize=(6,6), dpi=100)
    plt.imshow(x,origin='lower',cmap=cmap)
    cax=plt.colorbar()
    plt.contour(x,10,cmap=plt.cm.gray,linewidths=0.5)
    plt.ylabel('$r_{\pi}$',fontsize = 16)
    plt.xlabel('$r_{p}$',fontsize = 16)
    plt.title(y,fontsize = 16)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax.set_label(t,labelpad = 15,fontsize = 15)
    line_colour1 = ('royalblue', 'blue', 'mediumblue', 'darkblue')
    plt.plot()
    plt.show()
    

top = np.max(LS[np.where(LS < 0.05)])
down = np.min(LS[np.where(LS > -0.05)])
LS[np.where(LS > 0.05)] = top
LS[np.where(LS < -0.05)] = down

import cv2
p = 1
sig = 1

blur_OH = cv2.blur(LS,(p,p))
imag(blur_OH,'Función de Correlación','$\epsilon(r)$',cmap='RdBu')
"""