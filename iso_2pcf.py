import numpy as np
#import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
import time


def pcf2_iso_histo(data_location='fake_DATA/DATOS/data.dat',rand_location='fake_DATA/DATOS/rand0.dat', d_max=180.0, bins_number=30):
    """
    Calculates the DD, RR, DR histograms for the points given in the data and random text files. Both files must have the same size.
    
    args:
        -data_location: str. It is the file location of the data. Each row a different point with the coordinates (x,y,z) as the first three columns separated with blank spaces.
        -rand_location: str. It is the file location of the file with random points. Each row a different point with the coordinates (x,y,z) as the first three columns separated with blank spaces.
        -d_max: float. The maximum distance that will be considered in the histogram
        -bins_number: int. The number of bins to use to separate the data.
        
    return:
        -DD: np.array. Array with the frequencies of the distances between data points and data points
        -RR: np.array. Array with the frequencies of the distances between random points and random points
        -DR: np.array. Array with the frequencies of the distances between data points and random points
        -bins: np.array. With the edges of the bins for later xaxis plot
    
    """
    
    data = np.loadtxt(fname=data_location, delimiter=" ", usecols=(0,1,2))
    rand0 = np.loadtxt(fname=rand_location, delimiter=" ", usecols=(0,1,2))

    DD, bins_DD = np.histogram(pdist(data), bins=bins_number, range=(0,d_max))
    DD *= 2

    RR, bins_RR = np.histogram(pdist(rand0), bins=bins_number, range=(0,d_max))
    RR *= 2
    
    DR, bins_DR = np.histogram(np.sqrt(np.sum((data[0]-rand0)**2,1)), bins=bins_number, range=(0, d_max))
    for point in data[1:]:
        DR_temp, bins_DR = np.histogram(np.sqrt(np.sum((point-rand0)**2,1)), bins=bins_number, range=(0, d_max))
        DR += DR_temp
        
    return DD, RR, DR, bins_DR

#Landy-Szalay
def estim_LS(DD, RR, DR):
    return (DD - 2*DR + RR)/RR
    
#Hamilton
def estim_HM(DD, RR, DR):
    return (DD*RR/DR**2) - 1

start = time.perf_counter()

#d_max = 180
#bins_number = 30
DD, RR, DR, bins = pcf2_iso_histo(data_location='fake_DATA/DATOS/data.dat',rand_location='fake_DATA/DATOS/rand0.dat')

end = time.perf_counter()
print(f'Took {end-start} seconds to calculate DD, RR, and DR histograms')

LS = estim_LS(DD, RR, DR)
HM = estim_HM(DD, RR, DR)
