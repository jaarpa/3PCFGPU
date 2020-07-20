import numpy as np
#import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
import time

start_0 = time.perf_counter()
d_max = 180
bins_number = 30

data = np.loadtxt(fname='fake_DATA/DATOS/data.dat', delimiter=" ", usecols=(0,1,2))
rand0 = np.loadtxt(fname='fake_DATA/DATOS/rand0.dat', delimiter=" ", usecols=(0,1,2))

NDD = np.zeros(bins_number)
NRR = np.zeros(bins_number)
NDR = np.zeros(bins_number)

n = 0

print("Started calculating distances")
start = time.perf_counter()
for (ii, jj) in zip(data, rand0):
    n = n+1
    
    # Histogramas para DD
    s = ii-data[n:] # Diferencia entre el punto pivote y los demas puntos siguientes 
    dis, r = np.histogram(np.sqrt(s[:,0]**2+s[:,1]**2+s[:,2]**2), bins=bins_number, range=(0, d_max))
    NDD = NDD + 2*dis
    
    # Histogramas para RR
    s = jj-rand0[n:]
    dis, r = np.histogram(np.sqrt(s[:,0]**2+s[:,1]**2+s[:,2]**2), bins=bins_number, range=(0, d_max))
    NRR = NRR + 2*dis 
end = time.perf_counter()
print(f'Calculated the random_histogram in {end-start} s')

end = time.perf_counter()

start = time.perf_counter()
NDR = np.zeros(bins_number)
for ii in data:
    # Histogramas para DR
    s = ii-rand0
    dis, r = np.histogram(np.sqrt(s[:,0]**2+s[:,1]**2+s[:,2]**2), bins=bins_number, range=(0, d_max))
    NDR = NDR + dis
end = time.perf_counter()
print(f'Calculated the random_data_histogram in {end-start} s')
print(f'Calculated {sum(NDR)}distances for the RD histogram')

print(f'Took {end-start_0} seconds to calculate RR and DD histograms')
