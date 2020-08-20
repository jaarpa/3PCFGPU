import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  cout << thredIdy.y << endl;
  dest[i] = a[i] - b[i];
}
""")

multiply_them = mod.get_function("multiply_them")

a = np.random.randn(500,3).astype(np.float32)
b = np.random.randn(500,3).astype(np.float32)

dest = np.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(500,3,1), grid=(1,1))

print(dest-a*b)