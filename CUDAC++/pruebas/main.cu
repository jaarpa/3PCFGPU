#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "/usr/local/cuda/include/cuda_runtime.h"

typedef std::chrono::high_resolution_clock Clock;
using namespace std;

 
/*
=================================================
        Prototipos de fuciones CPU
=================================================
*/

void suma_cpu(float*,float*,float*,int);


/*
=================================================
        Prototipos de fuciones GPU
=================================================
*/


__global__ void suma_gpu(float*,float*,float*);


int main(int argc, char *argv[]){

    int num_iteraciones = 65500,i;
    float *A,*B,*C;
    float *a,*b,*c;
    a = (float*)malloc(num_iteraciones*sizeof(float));
    b = (float*)malloc(num_iteraciones*sizeof(float));
    c = (float*)malloc(num_iteraciones*sizeof(float));

    cudaMallocManaged(&A,num_iteraciones*sizeof(float));
    cudaMallocManaged(&B,num_iteraciones*sizeof(float));
    cudaMallocManaged(&C,num_iteraciones*sizeof(float));

    for(int i = 0; i< num_iteraciones; i++){
        *(a+i) = (float)(i);
        *(b+i) = (float)(i);
        *(c+i) = (float)(i);
    }

    auto cpu_start = Clock::now();
    suma_cpu(a,b,c,num_iteraciones);
    auto cpu_end = Clock::now();


    cout << "vector_add_cpu: " << std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count() << " nanoseconds.\n";

    auto gpu_start = Clock::now();
    suma_gpu <<<1, num_iteraciones>>> (A, B, C);
    cudaDeviceSynchronize();
    auto gpu_end = Clock::now();
    cout << "vector_add_gpu: " << std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_end - gpu_start).count() << " nanoseconds.\n";

    for (i = 0; i < 10; i++)
    {
        cout << C[i] << "\t" << c[i] << endl;
    }
    
    //Liberamos memoria del CPU
    free(a);
    free(b);
    free(c);

    //Liberamos memoria del GPU
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}


__global__ void suma_gpu(float* A,float* B,float* C){
    int i = threadIdx.x;
    *(C+i) =  *(A + i) + *(B + i);
}

void suma_cpu(float* a, float *b, float* c, int n){
    int i;
    for(i = 0; i<n; i++){
        c[i] = a[i] + b[i]; 
    }
}
