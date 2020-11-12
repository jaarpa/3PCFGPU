import numpy as np

DDD_gpu = np.loadtxt("DDD_GPU.dat")
DDD_cpu = np.loadtxt("DDD_CPU.dat")

#DDD_gpu = np.loadtxt("DDD.res")
#DDD_cpu = np.loadtxt("DDDiso_mesh_3D_rand0_5K.dat.dat")

print((DDD_gpu==DDD_cpu).all())