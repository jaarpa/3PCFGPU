
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule

import time

mod = SourceModule("""
    #include <stdio.h>
    #include <math.h>
    __global__ void XY(float *dest, float *a, float *b, int *N){
        int p_id = threadIdx.x + blockDim.x*blockIdx.x;
        int id = threadIdx.y + blockDim.y*blockIdx.y;

        if (id < *N && p_id <*N){
            int x = id*3;
            int y = x+1;
            int z = y+1;

            int p_x = p_id*3;
            int p_y = p_x+1;
            int p_z = p_y+1;
            float d;
            //float histo[30];
            int bin;
            d = sqrt(pow(a[p_x] - b[x],2)+pow(a[p_y]-b[y],2) + pow(a[p_z]-b[z],2));
            if (d<=180){
                bin = (int) (d/6.0);
                atomicAdd(&dest[bin],1);
            }
        }
    }

    __global__ void XX(float *dest, float *a, int *N){
        int p_id = threadIdx.x + blockDim.x*blockIdx.x;
        int id = threadIdx.y + blockDim.y*blockIdx.y;

        if (p_id<*N && id<*N && p_id<id){

            int p_x = p_id*3;
            int p_y = p_x+1;
            int p_z = p_y+1;

            float d;
            int bin;

            int x = id*3;
            int y = x+1;
            int z = y+1;

            d = sqrt(pow(a[p_x] - a[x],2)+pow(a[p_y]-a[y],2) + pow(a[p_z]-a[z],2));
            if (d<=180){
                bin = (int) (d/6.0);
                atomicAdd(&dest[bin],2);
            }
        }
    }
""")

XY = mod.get_function("XY")
XX = mod.get_function("XX")

data = np.loadtxt(fname='../fake_DATA/DATOS/data.dat', delimiter=" ", usecols=(0,1,2)).astype(np.float32)
rand = np.loadtxt(fname='../fake_DATA/DATOS/rand0.dat', delimiter=" ", usecols=(0,1,2)).astype(np.float32)

N = len(data)

N_even = N+(N%2 != 0) #Converts N into an even number.
temp = []

for i in range(1,6):
    threads = 2**i
    blocks = int(N_even/threads)+1
    unused_cores = (blocks*threads)-N_even
    temp += [(threads,blocks,unused_cores, (blocks)**2+unused_cores**2)]

temp.sort(key=lambda tup: tup[3])
threads, blocks, unused_cores, score = temp[0] #Gets the arrange of threads and blocks with the minimum number of blocks and unused cores
block_dim = (threads,threads,1)
grid_dim = (blocks,blocks,1)

print(block_dim)
print(grid_dim)

#template_array = np.arange(30)
N = np.int32(N) #Number of points
DD = np.zeros_like(data[:30,0]) #teplate_array)
RR = np.zeros_like(data[:30,0]) #template_array)
DR = np.zeros_like(data[:30,0]) #template_array)

t=0
start = time.perf_counter()
XY(drv.Out(DR), drv.In(data), drv.In(rand), drv.In(N), block=block_dim, grid=grid_dim)
end = time.perf_counter()
print(f'DR {end-start}s in GPU')
t+=(end-start)
np.savetxt('DR.dat', DR)

start = time.perf_counter()
XX(drv.Out(DD), drv.In(data), drv.In(N), block=block_dim, grid = grid_dim)
end = time.perf_counter()
print(f'DD {end-start}s in GPU')
t+=(end-start)
np.savetxt('DD.dat', DD)

start = time.perf_counter()
XX(drv.Out(RR), drv.In(rand), drv.In(N), block=block_dim, grid = grid_dim)
end = time.perf_counter()
print(f'RR {end-start}s in GPU')
t+=(end-start)
np.savetxt('RR.dat', RR)

print(f'Total GPU time {t}')

#start = time.perf_counter()
#count = np.zeros(30)
#for i,point in enumerate(rand):
#    test_r = np.sqrt(np.sum((point-rand[i:])**2,axis=1))
#    t_count, edges = np.histogram(test_r, bins=30, range=(0,180))
#    count += t_count
#end = time.perf_counter()
#print(f'{end-start} s in CPU')

#Compare with the backup
#RR_bak = np.loadtxt('RR_bak.dat')
#print(np.abs(RR-RR_bak)<=1e-5)

#DD_bak = np.loadtxt('DD_bak.dat')
#print(np.abs(DD-DD_bak)<=1e-5)

#DR_bak = np.loadtxt('DR_bak.dat')
#print(np.abs(DR-DR_bak)<=1e-5)
