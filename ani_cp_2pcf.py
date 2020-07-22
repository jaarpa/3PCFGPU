import numpy as np
import pandas as pd
import matplotlib as plt

def pcf2_iso_histo(data_location='fake_DATA/DATOS/data_500.dat',rand_location='fake_DATA/DATOS/rand0_500.dat', observation_point=np.array([0,0,0]), d_max=180.0, bins_number=30):
    """
    Calculates the DD, RR, DR bidimensional histograms for the anisotropic points given in the data and random text files and the point of observation.  Both files must have the same number of points with three dimesions.
    
    args:
        -data_location: str. It is the file location of the data. Each row a different point with the coordinates (x,y,z) as the first three columns separated with blank spaces.
        -rand_location: str. It is the file location of the file with random points. Each row a different point with the coordinates (x,y,z) as the first three columns separated with blank spaces.
        -observation_point: array-like. It is the point where the observation line is drawn.
        -d_max: float. The maximum distance that will be considered in the histogram
        -bins_number: int. The number of bins to use to separate the data.
        
    return:
        -DD: np.array(bins_number,bins_number). Array with the frequencies of the distances between data points and data points
        -RR: np.array(bins_number,bins_number). Array with the frequencies of the distances between random points and random points
        -DR: np.array(bins_number,bins_number). Array with the frequencies of the distances between data points and random points
        -x_edges: np.array(bins_number). With the edges of the bins for later xaxis plot
        -y_edges: np.array(bins_number). With the edges of the bins for later yaxis plot
    
    """
    
    d_max = d_max**2
    data = np.loadtxt(fname=data_location, delimiter=" ", usecols=(0,1,2))
    rand0 = np.loadtxt(fname=rand_location, delimiter=" ", usecols=(0,1,2))

    for data_point in data:
        distance_vectors = data_point - rand0
        medium_point_vectors = 0.5*(distance_vectors)-observation_point

    for i, points in enumerate(zip(data,rand0),1):
        distance_vectors_DD = points[0] - data[i:]
        medium_point_vectors_DD = 0.5*(distance_vectors_DD)-observation_point

        distance_vectors_RR = points[1] - rand0[i:]
        medium_point_vectors_RR = 0.5*(distance_vectors_RR)-observation_point


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

start = time.perf_counter()

#d_max = 180
#bins_number = 30
DD, RR, DR, x_edges, y_edges = pcf2_iso_histo(data_location='fake_DATA/DATOS/data_500.dat',rand_location='fake_DATA/DATOS/rand0_500.dat', observation_point=np.array([0,0,0]))

end = time.perf_counter()
print(f'Took {end-start} seconds to calculate DD, RR, and DR histograms')