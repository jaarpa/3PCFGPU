#include <iostream>
#include <stdlib.h>
#include <math.h>
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
    int i;
    int N = 1e6; // cantidad de iteraciones
    int dim_bloque = 256;//cantidad de hilos en cada bloque (multiplo de 32, 32x8 = 256)
    int num_bloques = ceil(N/dim_bloque); // redondeamos al mayor numero, para asegurar que haya siempre los hilos justos
    cout << "Iteraciones: "<< N << "\n Numero de bloques: "<< num_bloques "\nHilos por bloque: " << dim_bloque << endl;
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
    suma_gpu<<<num_bloques,blockDim>>>(A_gpu,B_gpu,C_gpu,N);
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
    /*
        gridDim.x  --> regresa la cantidad de bloques (en este caso igual a la variable num_bloques)
        blockDim.x --> regresa la cantidad de hilos en cada bloque (en este caso igual a la variable dim_bloque)
        gridId.x   --> regresa el índice del grid actual (en este caso 0)
        blockIdx.x --> regresa el índice del bloque actual (de 0 a num_bloques-1 )
        threadId.x.--> regresa el índice del hilo actual (de 0 a dim_bloque -1 )

    */
    int indice = blockIdx.x*blockDim.x + threadIdx.x; // localiza un hilo 0-255 , 256 - 511 , etc
    int paso = blockDim.x * gridDim.x; // tamaño del paso, igual a la cantidad de hilos en todo el grid, como hay tantos hilos como iteraciones
    // el kernel se ejecutara al menos n veces, y los hilos cuyo indice i exceda n no entra al ciclo, y cada hilo se ejecuta una sola vez, de allí 
    // elegir el paso como eso.
    for(int i = indice; i < n; i+= paso){
        *(c+i) = *(a+i) + *(b+i);
    }

}