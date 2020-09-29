#include <iostream>
#include <stdlib.h>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

using namespace std;

 
/*
=================================================
        Prototipos de fuciones CPU
=================================================
*/

void suma_cpu(float *, float *, float *, int);


/*
=================================================
        Prototipos de fuciones GPU
=================================================
*/

__global__
void suma_gpu(float *, float *, float*, int);

int main(int argc, char *argv[]){
    int N = 1e6, i;
    cout << N << endl;
    float *A_gpu, *B_gpu, *C_gpu, *a, *b, *c;
    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    c = (float*)malloc(N*sizeof(float));
    cudaMallocManaged(&A_gpu,N*sizeof(float));
    cudaMallocManaged(&B_gpu,N*sizeof(float));
    cudaMallocManaged(&C_gpu,N*sizeof(float));
    
    for(i = 0; i<N; i++){
        *(a+i) = (float)(i);
        *(b+i) = (float)(i);
        *(A_gpu + i) = (float)(i);
        *(B_gpu + i) = (float)(i);
    }

    auto cpu_start = Clock::now();
    suma_cpu(a,b,c,N);
    auto cpu_end = Clock::now();

    auto gpu_start = Clock::now();
    suma_gpu<<<1,256>>>(A_gpu,B_gpu,C_gpu,N);
    cudaDeviceSynchronize();
    auto gpu_end = Clock::now();

    for(i=0; i<10;i++){
        cout << *(c+i) << "\t" << *(C_gpu + i) << endl;
    }

    cout << "CPU time: " << chrono::duration_cast<chrono::nanoseconds>(cpu_end - cpu_start).count() << " nanoseconds.\n";
    cout << "GPU time: " << chrono::duration_cast<chrono::nanoseconds>(gpu_end - gpu_start).count() << " nanoseconds.\n";

    free(a);
    free(b);
    free(c);
    cudaFree(C_gpu);
    cudaFree(A_gpu);
    cudaFree(B_gpu);

    return 0;
}


void suma_cpu(float *a, float *b, float *c, int n){

    for(int i = 0; i<n ; i++){
        *(c+i) = *(a+i) + *(b+i);
    }

}



__global__
void suma_gpu(float *a, float *b, float* c, int n){

    int indice = threadIdx.x;
    int paso = blockDim.x;
    for(int i = indice; i < n; i+= paso){
        *(c+i) = *(a+i) + *(b+i);
    }

}