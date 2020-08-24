
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule

import time

mod = SourceModule("""
#include <stdio.h>
#include <math.h>
__global__ void multiply_them(float *dest, float *a, float *b)
{
   int id = threadIdx.x;
   int x = id*3;
   int y = x+1;
   int z = y+1;

   dest[id] = sqrt(pow(a[x] - b[x],2)+pow(a[y]-b[y],2) + pow(a[z]-b[z],2));

   //dest[x] = sqrt(pow(a[x] - b[x],2)+pow(a[y]-b[y],2) + pow(a[z]-b[z],2));
   //printf("I am %d.%d\\n", blockIdx.x, blockIdx.y);
}
""")

multiply_them = mod.get_function("multiply_them")

data = np.loadtxt(fname='../fake_DATA/DATOS/data_500.dat', delimiter=" ", usecols=(0,1,2)).astype(np.float32)
rand = np.loadtxt(fname='../fake_DATA/DATOS/rand0_500.dat', delimiter=" ", usecols=(0,1,2)).astype(np.float32)

dest = np.zeros_like(data[:,0])

N = len(data)

start = time.perf_counter()
multiply_them(
        drv.Out(dest), drv.In(data), drv.In(rand),
        block=(500,1,1), grid = (500,1))
end = time.perf_counter()
print(f'{end-start}s in GPU')

np.savetxt('dest.dat', dest)

start = time.perf_counter()
test_r = np.sqrt(np.sum((data-rand)**2,axis=1))
end = time.perf_counter()
print(f'{end-start} s in CPU')

#print(dest)
#print(dest.shape)
print(np.abs(dest-test_r)<=1e-4)
#print(dest-np.linalg.norm((data-rand),axis=1))
