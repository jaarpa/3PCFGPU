#include <iostream>

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__global__ void kernel(double *XX, float *data) {
    int bin = 0;
    double sum = 2.4;
    if (data[threadIdx.x]<15){
        atomicAdd(&XX[bin],sum);
    }
}

int main(){
    double *XX;
    float *data;
    cudaMallocManaged(&data, 50*sizeof(float));
    cudaMallocManaged(&XX, 10*sizeof(double));
    for (int i=0 ; i<50 ;i++){
        data[i] = i*0.5;
    }
    for (int i=0 ; i<10 ;i++){
        XX[i] = 0.0;
    }
    kernel<<<1,50>>>(XX, data);

    std::cout << XX[0] << std::endl;

    return 0;
}