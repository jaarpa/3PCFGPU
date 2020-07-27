import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from itertools import combinations
import time


def pcf2_iso_histo(data_location='fake_DATA/DATOS/data_500.dat',rand_location='fake_DATA/DATOS/rand0_500.dat', d_max=180.0, bins_number=30):
    """
    Calculates the DDD, RRR, RRD, DDR multidimesional histograms for the isotropic points given in the data and random text files. Both files must have the same number of points with three dimesions.
    
    args:
        -data_location: str. It is the file location of the data. Each row a different point with the coordinates (x,y,z) as the first three columns separated with blank spaces.
        -rand_location: str. It is the file location of the file with random points. Each row a different point with the coordinates (x,y,z) as the first three columns separated with blank spaces.
        -d_max: float. The maximum distance that will be considered in the histogram. The same max value is used in every direction
        -bins_number: int. The number of bins to use to separate the data. The same max number of bins is used in every direction
        
    return:
        -DDD: np.array. Array with the frequencies of the distances in the triangles formed within the data points
        -RRR: np.array. Array with the frequencies of the distances of the triangles formed within the random points
        -DRR: np.array. Array with the frequencies of the distances of the triangles formed with two random points and one data point
        -RDD: np.array. Array with the frequencies of the distances of the triangles formed with two data points and one random point
        -bins: np.array. With the edges of the bins for later xaxis plot
    
    """
    
    data = np.loadtxt(fname=data_location, delimiter=" ", usecols=(0,1,2))
    rand0 = np.loadtxt(fname=rand_location, delimiter=" ", usecols=(0,1,2))
        
    if not data.shape == rand0.shape:
        raise Exception("The data file and rand file do not have the same size")
    #351 s

    #Pure histograms
    start = time.perf_counter()
    print('start DDD distances')
    triangle_points = np.array(list(combinations(data,3)))
    r_12 = triangle_points[:,0,:]-triangle_points[:,1,:]
    r_12=r_12**2
    r_12 = r_12[:,0]+r_12[:,1]+r_12[:,2]
    r_12 = np.sqrt(r_12)

    r_23 = triangle_points[:,1,:]-triangle_points[:,2,:]
    r_23 = r_23**2
    r_23 = r_23[:,0]+r_23[:,1]+r_23[:,2]
    r_23 = np.sqrt(r_23)

    r_31 = triangle_points[:,2,:]-triangle_points[:,0,:]
    r_31 = r_31**2
    r_31 = r_31[:,0]+r_31[:,1]+r_31[:,2]
    r_31 = np.sqrt(r_31)

    DDD, edges = np.histogramdd(np.column_stack((r_12, r_23, r_31)), bins=(bins_number,bins_number,bins_number), range=[[0,d_max],[0,d_max],[0,d_max]])
    
    end = time.perf_counter()
    print(f'Finished creating the DDD histo in {end-start} s')

    start = time.perf_counter()
    print('start RRR distances')
    triangle_points = np.array(list(combinations(rand0,3)))
    r_12 = triangle_points[:,0,:]-triangle_points[:,1,:]
    r_12=r_12**2
    r_12 = r_12[:,0]+r_12[:,1]+r_12[:,2]
    r_12 = np.sqrt(r_12)

    r_23 = triangle_points[:,1,:]-triangle_points[:,2,:]
    r_23 = r_23**2
    r_23 = r_23[:,0]+r_23[:,1]+r_23[:,2]
    r_23 = np.sqrt(r_23)

    r_31 = triangle_points[:,2,:]-triangle_points[:,0,:]
    r_31 = r_31**2
    r_31 = r_31[:,0]+r_31[:,1]+r_31[:,2]
    r_31 = np.sqrt(r_31)

    RRR, edges = np.histogramdd(np.column_stack((r_12, r_23, r_31)), bins=(bins_number,bins_number,bins_number), range=[[0,d_max],[0,d_max],[0,d_max]])
    
    end = time.perf_counter()


    print(f'Finished creating the RRR histo in {end-start} s')

    #Mixed histogram
    start = time.perf_counter()
    print("Started gathering the data points pairs")
    DD_side_points = np.array(list(combinations(data,2)))
    print("Finished data points pairs")

    print("Started gathering the rand0 points pairs")
    RR_side_points = np.array(list(combinations(rand0,2)))
    print("Finished rand0 points pairs")

    print("Started loop for DDR and RRD histograms")
    
    DDR = np.zeros((bins_number,bins_number,bins_number))
    RRD = np.zeros((bins_number,bins_number,bins_number))

    for data_point, rand_point in zip(data, rand0):
        ##DDR
        r_12 = rand_point-DD_side_points[:,0,:]
        r_12=r_12**2
        r_12 = r_12[:,0]+r_12[:,1]+r_12[:,2]
        r_12 = np.sqrt(r_12)

        r_23 = DD_side_points[:,0,:]-DD_side_points[:,1,:]
        r_23 = r_23**2
        r_23 = r_23[:,0]+r_23[:,1]+r_23[:,2]
        r_23 = np.sqrt(r_23)

        r_31 = DD_side_points[:,1,:]-rand_point
        r_31 = r_31**2
        r_31 = r_31[:,0]+r_31[:,1]+r_31[:,2]
        r_31 = np.sqrt(r_31)
        H_DDR, edges = np.histogramdd(np.column_stack((r_12, r_23, r_31)), bins=(bins_number,bins_number,bins_number), range=[[0,d_max],[0,d_max],[0,d_max]])
        DDR += H_DDR

        #RRD
        r_12 = data_point-RR_side_points[:,0,:]
        r_12=r_12**2
        r_12 = r_12[:,0]+r_12[:,1]+r_12[:,2]
        r_12 = np.sqrt(r_12)

        r_23 = RR_side_points[:,0,:]-RR_side_points[:,1,:]
        r_23 = r_23**2
        r_23 = r_23[:,0]+r_23[:,1]+r_23[:,2]
        r_23 = np.sqrt(r_23)

        r_31 = RR_side_points[:,1,:]-data_point
        r_31 = r_31**2
        r_31 = r_31[:,0]+r_31[:,1]+r_31[:,2]
        r_31 = np.sqrt(r_31)
        H_RRD, edges = np.histogramdd(np.column_stack((r_12, r_23, r_31)), bins=(bins_number,bins_number,bins_number), range=[[0,d_max],[0,d_max],[0,d_max]])
        RRD += H_RRD
    
    DDR = DDR/3
    RRD = RRD/3
    end = time.perf_counter()
    print(f'Finished the mixed histograms DDR an RRD in {end-start} s')

    return RRR, DDD, DDR, RRD, edges

def simetrize_3dhistogram(histogram):
    """
        Returns the simetrized version of a 3d histogram
        Params:
            -histogram: numpy array. The histogram to be simetrized. It is assumed to be of the from NxNxN

        Returns:
            -simetrized: histogram. The same histogram but simetrized, where any combination of its indices has the very same value.
    """
    N =len(histogram)
    for i in range(N-2):
        for j in range(i+1,N-1):
            for k in range(j+1, N):
                S = histogram[i][j][k] + histogram[k][i][j] + histogram[j][k][i] + histogram[i][k][j] + histogram[j][i][k] + histogram[k][j][i]
                [histogram[i][j][k], histogram[k][i][j], histogram[j][k][i], histogram[i][k][j], histogram[j][i][k], histogram[k][j][i]] = [S,S,S,S,S,S]
                #a[i][j][k], a[k][i][j], a[j][k][i], a[i][k][j], a[j][i][k], a[k][j][i]
    return histogram

def write_into_file(filenme,data):
    # Write the array to disk
    with open(filenme, 'w+') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write(f'# Array shape: {data.shape}\n')

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for i,data_slice in enumerate(data):

            # Writing out a break to indicate different slices...
            outfile.write(f'# Slice {i}\n')

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.  
            np.savetxt(outfile, data_slice)



#Unsimetrized histograms. Combinations without repetition.
start = time.perf_counter()
RRR, DDD, DDR, RRD, edges = pcf2_iso_histo(data_location='fake_DATA/DATOS/data_500.dat',rand_location='fake_DATA/DATOS/rand0_500.dat', d_max=180.0, bins_number=30)
end = time.perf_counter()
print(f'Finished the all the unsimetrized histograms in {end-start} s')


# DDR an RRD have been already multplied by 1/3 inside the pcf2_iso_histo function
#Simetrize the histograms
start = time.perf_counter()
RRR = simetrize_3dhistogram(RRR)
DDD = simetrize_3dhistogram(DDD)
DDR = simetrize_3dhistogram(DDR)
RRD = simetrize_3dhistogram(RRD)
end = time.perf_counter()
print(f'Finished simetrization of all the histograms in {end-start} s')

write_into_file('500_points_RRD.dat', RRD)
write_into_file('500_points_RRR.dat', RRR)
write_into_file('500_points_DDD.dat', DDD)
write_into_file('500_points_DDR.dat', DDR)
