import numpy as np


import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule

import time


bins = 30
d_max = 100.0
size_box = 250.0

"""
temp=[]
for i in range(1,6):
    threads = 2**i
    blocks = int(N_even/threads)+1
    unused_cores = (blocks*threads)-N_even
    temp += [(threads,blocks,unused_cores, (blocks)**2+unused_cores**2)]

temp.sort(key=lambda tup: tup[3])
threads, blocks, unused_cores, score = temp[0] #Gets the arrange of threads and blocks with the minimum number of blocks and unused cores
block_dim = (threads,threads,1)
grid_dim = (blocks,blocks,1)
"""

class PCF3:
    def __init__(self, _bins = 30, _d_max = 100.0, _size_box = 250.0, _fdata = "../fake_DATA/DATOS/data.dat", _frand = "../fake_DATA/DATOS/rand0.dat"):
        self.bins = _bins
        self.d_max = _d_max
        self.size_box = _size_box
        self.fdata = _fdata
        self.frand = _frand
        self.data = np.loadtxt(fname='../fake_DATA/DATOS/data.dat', delimiter=" ").astype(np.float32)
        self.rand = np.loadtxt(fname='../fake_DATA/DATOS/rand0.dat', delimiter=" ").astype(np.float32)
        self.N = np.int32(len(self.data))
        self.N_even = self.N+(self.N%2 != 0) #Converts N into an even number.
        self.DDD = np.zeros((_bins,_bins), dtype=np.int32)
        self.RRR = np.zeros((_bins,_bins), dtype=np.int32)
        self.DDR = np.zeros((_bins,_bins), dtype=np.int32)
        self.DRR = np.zeros((_bins,_bins), dtype=np.int32)

    def create_grid(self):
        self.size_node = np.float32(2.17*self.size_box/self.bins)
        self.partitions = np.int32(np.ceil(self.size_box/self.size_node))
        kernel_classification = SourceModule("""
            #include<iostream>
            __global__ subhisto(float *dest, float *datos, int *Nnodes, float *Snode){
                cout<<"hola<<endl;
            }
        """)
        nodes_memsize = np.zeros((self.partitions,self.partitions,self.partitions), dtype=np.int32)
