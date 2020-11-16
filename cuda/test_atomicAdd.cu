#include <iostream>
#include <stdlib.h>
#include <stdio.h>


__global__ void kernel(float *XX, float *data) {
    //int bin = 0;
    float sum = 2.4f;
    if (data[threadIdx.x]<15){
        printf("%f \n", sum);
        atomicAdd(&XX[0],sum);
        printf("%f \n", XX);
    }
}

int main(){
    float *XX;
    float *data;
    cudaMallocManaged(&data, 50*sizeof(float));
    cudaMallocManaged(&XX, 10*sizeof(float));
    for (int i=0 ; i<50 ;i++){
        data[i] = i*0.3;
    }
    /*
    for (int i=0 ; i<10 ;i++){
        XX[i] = 0.0;
    }
    */
    kernel<<<1,50>>>(XX, data);

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