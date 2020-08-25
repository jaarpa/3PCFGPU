
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule

import time

mod = SourceModule("""
#include <stdio.h>
#include <math.h>
__global__ void XY(float *dest, float *a, float *b, int *N)
{
   //threadIdx.x + blockIdx.x * blockDim.x;
   int id = threadIdx.x + blockDim.x*blockIdx.x;
   if (id < *N){
   int x = id*3;
   int y = x+1;
   int z = y+1;

   int p_id = blockIdx.y;
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
   //printf("I am %d.%d\\n", blockIdx.x, blockIdx.y);
   }
}
__global__ void XX(float *dest, float *a){
   int p_id = blockIdx.y;
   int p_x = p_id*3;
   int p_y = p_x+1;
   int p_z = p_y+1;

   float d;
   int bin;

   int id = threadIdx.x;
   int x = id*3;
   int y = x+1;
   int z = y+1;

   if (p_id<=id){
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

data = np.loadtxt(fname='../fake_DATA/DATOS/data_500.dat', delimiter=" ", usecols=(0,1,2)).astype(np.float32)
rand = np.loadtxt(fname='../fake_DATA/DATOS/rand0_500.dat', delimiter=" ", usecols=(0,1,2)).astype(np.float32)

N = len(data)

if N<1024: #Use simpler parallelization
    grid_dim = (1,N,1)
    block_dim = (N,1,1)
    N_even = N
else:
    N_even = N+(N%2 != 0) #Converts N into an even number.
    temp = []
    for i in range(1,11):
        threads = 2**i
        blocks = int(N_even/threads)+1
        unused_cores = (blocks*threads)-N_even
        temp += [(threads,blocks,unused_cores, blocks*unused_cores)]
    temp.sort(key=lambda tup: tup[3])
    threads, blocks, unused_cores, score = temp[0] #Gets the arrange of threads and blocks with the minimum number of blocks and unused cores
    block_dim = (threads,1,1)
    grid_dim = (blocks,blocks,1)
#block dim has the number  of threads in the block. Threads*grid_dim[1] > number of points. Threads*grid_dim[0] = Pivot point id

#template_array = np.arange(30)
N32 = np.int32(N)
DD = np.zeros_like(data[:30,0]) #teplate_array)
RR = np.zeros_like(data[:30,0]) #template_array)
DR = np.zeros_like(data[:30,0]) #template_array)

t=0
start = time.perf_counter()
XY(drv.Out(DR), drv.In(data), drv.In(rand), drv.In(N32), block=block_dim, grid=grid_dim)
end = time.perf_counter()
print(f'DR {end-start}s in GPU')
t+=(end-start)
np.savetxt('DR.dat', DR)

start = time.perf_counter()
XX(drv.Out(DD), drv.In(data),block=block_dim, grid = grid_dim)
end = time.perf_counter()
print(f'DD {end-start}s in GPU')
t+=(end-start)
np.savetxt('DD.dat', DD)

start = time.perf_counter()
XX(drv.Out(RR), drv.In(rand), block=block_dim, grid = grid_dim)
end = time.perf_counter()
print(f'RR {end-start}s in GPU')
t+=(end-start)
np.savetxt('RR.dat', RR)

print(f'Total GPU time {t}')

start = time.perf_counter()
count = np.zeros(30)
for i,point in enumerate(rand):
    test_r = np.sqrt(np.sum((point-rand[i:])**2,axis=1))
    t_count, edges = np.histogram(test_r, bins=30, range=(0,180))
    count += t_count
end = time.perf_counter()
print(f'{end-start} s in CPU')

#Compare with the backup
RR_bak = np.loadtxt('RR_bak.dat')
print(np.abs(RR-RR_bak)<=1e-5)

DD_bak = np.loadtxt('DD_bak.dat')
print(np.abs(DD-DD_bak)<=1e-5)

DR_bak = np.loadtxt('DR_bak.dat')
print(np.abs(DR-DR_bak)<=1e-5)
