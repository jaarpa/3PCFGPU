#include <iostream>
#include <stdlib.h>
#include <stdio.h>


__global__ void kernel(float *XX, float *data) {
    //int bin = 0;
    float sum = 2.3f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (data[idx]>0){
        atomicAdd(&XX[0],sum);
    }
}

int main(){
    float *XX;
    float *data;
    cudaMallocManaged(&data, 3300352*sizeof(float));
    cudaMallocManaged(&XX, 10*sizeof(float));
    for (int i=0 ; i<3300352 ;i++){
        data[i] = i*0.3;
    }

    for (int i=0 ; i<10 ;i++){
        XX[i] = 0.0;
    }

    kernel<<<3223,1024>>>(XX, data);
    //Waits for the GPU to finish
    cudaDeviceSynchronize();  

    //Check here for errors
    cudaError_t error = cudaGetLastError(); 
    std::cout << "The error code is " << error << std::endl;
    if(error != 0)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    std::cout << XX[0] << std::endl;


    cudaFree(&XX);
    cudaFree(&data);

    return 0;
}