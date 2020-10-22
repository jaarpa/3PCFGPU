import numpy as np

DDD_gpu = np.loadtxt("DDD_GPU.dat")
DDD_cpu = np.loadtxt("DDD_CPU.dat")

print((DDD_gpu==DDD_cpu).all())