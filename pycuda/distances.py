
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule

import time

mod = SourceModule("""
#include <stdio.h>
#include <math.h>
__global__ void XY(float *dest, float *a, float *b)
{
   int id = threadIdx.x;
   int x = id*3;
   int y = x+1;
   int z = y+1;

   int p_id = blockIdx.x;
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
__global__ void XX(float *dest, float *a){
   int p_id = blockIdx.x;
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
         atomicAdd(&dest[bin],1);
      }
   }

}
""")

XY = mod.get_function("XY")
XX = mod.get_function("XX")

data = np.loadtxt(fname='../fake_DATA/DATOS/data_500.dat', delimiter=" ", usecols=(0,1,2)).astype(np.float32)
rand = np.loadtxt(fname='../fake_DATA/DATOS/rand0_500.dat', delimiter=" ", usecols=(0,1,2)).astype(np.float32)

dest = np.zeros_like(data[:30,0])
start = time.perf_counter()
XY(drv.Out(dest), drv.In(data), drv.In(rand), block=(500,1,1), grid=(500,1))
end = time.perf_counter()
print(f'DR {end-start}s in GPU')
np.savetxt('DR.dat', dest)

dest = np.zeros_like(data[:30,0])
start = time.perf_counter()
XX(drv.Out(dest), drv.In(data), block=(500,1,1), grid = (500,1))
end = time.perf_counter()
print(f'DD {end-start}s in GPU')
np.savetxt('DD.dat', dest)

dest = np.zeros_like(data[:30,0])
start = time.perf_counter()
XX(drv.Out(dest), drv.In(rand), block=(500,1,1), grid = (500,1))
end = time.perf_counter()
print(f'RR {end-start}s in GPU')
np.savetxt('RR.dat', dest)


start = time.perf_counter()
count = np.zeros(30)
for i,point in enumerate(rand):
    test_r = np.sqrt(np.sum((point-rand[i:])**2,axis=1))
    t_count, edges = np.histogram(test_r, bins=30, range=(0,180))
    count += t_count
end = time.perf_counter()
print(f'{end-start} s in CPU')

print(test_r[:20])
print(count)
print(dest)
print(np.abs(dest-count)<=1e-5)
