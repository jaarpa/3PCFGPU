#include <string.h>
#include <stdio.h>
#include <math.h>
#include "pcf3iso.cuh"
#include "device_functions.cuh"

/** CUDA check macro */
#define CUCHECK(call){\
    cudaError_t res = (call);\
    if(res != cudaSuccess) {\
        const char* err_str = cudaGetErrorString(res);\
        fprintf(stderr, "%s (%d): %s in %s \n", __FILE__, __LINE__, err_str, #call);\
        exit(-1);\
    }\
}\

#if !defined(__CUDA_ARCH__) || defined(DOUBLE_ATOMIC_ADD_ARCHLT600) || __CUDA_ARCH__ >= 600
#else
    #define DOUBLE_ATOMIC_ADD_ARCHLT600
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

/* Complains if it cannot allocate the array */
#define CHECKALLOC(p)  if(p == NULL) {\
    fprintf(stderr, "%s (line %d): Error - unable to allocate required memory \n", __FILE__, __LINE__);\
    exit(1);\
}\

/*
Kernel function to calculate the pure histograms for the 3 point isotropic correlation function. 
This version does NOT considers boudary periodic conditions. It stores the counts in the XXX histogram.

args:
XXX: (double*) The histogram where the distances are counted.
elements: (PointW3D*) Array of the points ordered coherently with the nodes.
nodeD: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node.
nonzero_nodes: (int) Number of nonzero nodes where the points have been classificated.
bn: (int) NUmber of bins in the XY histogram.
dmax: (float) The maximum distance of interest between points.
d_max_node: (float) The maximum internodal distance.
*/
__global__ void XXX3iso(double *XXX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset);
/*
Kernel function to calculate the mixed histograms for the 3 point isotropic correlation function. 
This version does NOT considers boudary periodic conditions. It stores the counts in the XXY histogram.

args:
XXY: (double*) The histogram where the distances are counted.
elementsX: (PointW3D*) Array of the points ordered coherently with the nodes. For the X points.
nodeX: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node. For the X points
nonzero_Xnodes: (int) Number of nonzero nodes where the points have been classificated. For the X points
elementsY: (PointW3D*) Array of the points ordered coherently with the nodes. For the Y points.
nodeY: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node. For the Y points
nonzero_Ynodes: (int) Number of nonzero nodes where the points have been classificated. For the Y points
bn: (int) NUmber of bins in the XY histogram.
dmax: (float) The maximum distance of interest between points.
d_max_node: (float) The maximum internodal distance.
*/
__global__ void XXY3iso(double *XXY, PointW3D *elementsX, DNode *nodeX, int nonzero_Xnodes, PointW3D *elementsY, DNode *nodeY, int nonzero_Ynodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset, int isDDR);

__global__ void XXX3iso_wpips(double *XXX, PointW3D *elements, int32_t *pipsD, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, int pips_width, int node_offset, int bn_offset);
__global__ void XXY3iso_wpips(double *XXY, PointW3D *elementsX, int32_t *pipsX, DNode *nodeX, int nonzero_Xnodes, PointW3D *elementsY, int32_t *pipsY, DNode *nodeY, int nonzero_Ynodes, int bn, float dmax, float d_max_node, int pips_width, int node_offset, int bn_offset, int isDDR);

void pcf_3iso(
        DNode *dnodeD, PointW3D *dataD, int nonzero_Dnodes,
        DNode *dnodeR, PointW3D *dataR, int *nonzero_Rnodes, int *acum_nonzero_Rnodes,
        char **histo_names, int n_randfiles, int bins, float size_node, float dmax
    )
{

    /* =======================================================================*/
    /* =====================   Var declaration ===============================*/
    /* =======================================================================*/

    const int PREFIX_LENGTH = 8;
    float time_spent, d_max_node;

    double *DDD, *RRR, *DRR, *DDR;
    double *d_DDD, *d_RRR, *d_DRR, *d_DDR;

    //n_kernel_calls should depend of the number of points, its density, and the number of bins
    int threads_perblock_dim = 8;
    int blocks_D, blocks_R;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    CUCHECK(cudaEventCreate(&start_timmer));
    CUCHECK(cudaEventCreate(&stop_timmer));

    cudaStream_t streamDDD, *streamRRR, *streamDDR, *streamDRR;
    streamRRR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamRRR);
    streamDDR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamDDR);
    streamDRR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamDRR);
    CUCHECK(cudaStreamCreate(&streamDDD));
    for (int i = 0; i < n_randfiles; i++){
        CUCHECK(cudaStreamCreate(&streamRRR[i]));
        CUCHECK(cudaStreamCreate(&streamDDR[i]));
        CUCHECK(cudaStreamCreate(&streamDRR[i]));
    }

    // Name of the files where the results are saved
    char *nameDDD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    char *nameRRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    char *nameDDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    char *nameDRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    strcpy(nameDDD,"DDDiso_");
    strcpy(nameRRR,"DRRiso_");
    strcpy(nameDDR,"DDRiso_");
    strcpy(nameDRR,"RRRiso_");

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    // Allocate memory for the histogram as double
    DDD = (double*)calloc(bins*bins*bins, sizeof(double));
    CHECKALLOC(DDD);
    RRR = (double*)calloc(n_randfiles*bins*bins*bins, sizeof(double));
    CHECKALLOC(RRR);
    DDR = (double*)calloc(n_randfiles*bins*bins*bins, sizeof(double));
    CHECKALLOC(DDR);
    DRR = (double*)calloc(n_randfiles*bins*bins*bins, sizeof(double));
    CHECKALLOC(DRR);

    CUCHECK(cudaMalloc(&d_DDD, bins*bins*bins*sizeof(double)));
    CUCHECK(cudaMalloc(&d_RRR, n_randfiles*bins*bins*bins*sizeof(double)));
    CUCHECK(cudaMalloc(&d_DRR, n_randfiles*bins*bins*bins*sizeof(double)));
    CUCHECK(cudaMalloc(&d_DDR, n_randfiles*bins*bins*bins*sizeof(double)));

    //Restarts the main histograms in host to zero
    CUCHECK(cudaMemsetAsync(d_DDD, 0, bins*bins*bins*sizeof(double), streamDDD));
    CUCHECK(cudaMemsetAsync(d_RRR, 0, n_randfiles*bins*bins*bins*sizeof(double), streamRRR[0]));
    CUCHECK(cudaMemsetAsync(d_DRR, 0, n_randfiles*bins*bins*bins*sizeof(double), streamDRR[0]));
    CUCHECK(cudaMemsetAsync(d_DDR, 0, n_randfiles*bins*bins*bins*sizeof(double), streamDDR[0]));
    CUCHECK(cudaStreamSynchronize(streamRRR[0]));
    CUCHECK(cudaStreamSynchronize(streamDRR[0]));
    CUCHECK(cudaStreamSynchronize(streamDDR[0]));

    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/


    //Compute the dimensions of the GPU grid
    //One thread for each node
    
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));
    dim3 gridDDD(blocks_D,blocks_D,blocks_D);
    dim3 threads_perblock(threads_perblock_dim,threads_perblock_dim,threads_perblock_dim);
    
    //Dummy declaration
    dim3 gridRRR(2,2,1);
    dim3 gridDDR(blocks_D,blocks_D,1);
    dim3 gridDRR(2,2,blocks_D);

    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    XXX3iso<<<gridDDD,threads_perblock,0,streamDDD>>>(d_DDD, dataD, dnodeD, nonzero_Dnodes, bins, dmax, d_max_node, 0, 0);
    for (int i=0; i<n_randfiles; i++){
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridRRR.x = blocks_R;
        gridRRR.y = blocks_R;
        gridRRR.z = blocks_R;
        XXX3iso<<<gridRRR,threads_perblock,0,streamRRR[i]>>>(d_RRR, dataR, dnodeR, nonzero_Rnodes[i], bins, dmax, d_max_node, acum_nonzero_Rnodes[i], i);
        gridDDR.z = blocks_R;
        XXY3iso<<<gridDDR,threads_perblock,0,streamDDR[i]>>>(d_DDR, dataD, dnodeD, nonzero_Dnodes, dataR, dnodeR, nonzero_Rnodes[i], bins, dmax, d_max_node, acum_nonzero_Rnodes[i], i, 1);
        gridDRR.x = blocks_R;
        gridDRR.y = blocks_R;
        XXY3iso<<<gridDRR,threads_perblock,0,streamDRR[i]>>>(d_DRR, dataR, dnodeR, nonzero_Rnodes[i], dataD, dnodeD, nonzero_Dnodes, bins, dmax, d_max_node, acum_nonzero_Rnodes[i], i, 0);
    }

    //Waits for all the kernels to complete
    CUCHECK(cudaDeviceSynchronize());

    //Save the results
    CUCHECK(cudaMemcpy(DDD, d_DDD, bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost));

    nameDDD = (char*)realloc(nameDDD,PREFIX_LENGTH + strlen(histo_names[0]));
    strcpy(&nameDDD[PREFIX_LENGTH-1],histo_names[0]);
    save_histogram3D(nameDDD, bins, DDD, 0);

    CUCHECK(cudaMemcpy(RRR, d_RRR, n_randfiles*bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameRRR = (char*)realloc(nameRRR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameRRR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram3D(nameRRR, bins, RRR, i);
    }
    CUCHECK(cudaMemcpy(DDR, d_DDR, n_randfiles*bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameDDR = (char*)realloc(nameDDR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameDDR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram3D(nameDDR, bins, DDR, i);
    }
    CUCHECK(cudaMemcpy(DRR, d_DRR, n_randfiles*bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameDRR = (char*)realloc(nameDRR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameDRR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram3D(nameDRR, bins, DRR, i);
    }

    CUCHECK(cudaEventRecord(stop_timmer));
    CUCHECK(cudaEventSynchronize(stop_timmer));
    CUCHECK(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    printf("Spent %f miliseconds to compute and save all the histograms. \n", time_spent);
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory
    CUCHECK(cudaStreamDestroy(streamDDD));
    for (int i = 0; i < n_randfiles; i++){
        CUCHECK(cudaStreamDestroy(streamDDR[i]));
        CUCHECK(cudaStreamDestroy(streamDRR[i]));
        CUCHECK(cudaStreamDestroy(streamRRR[i]));
    }
    free(streamDDR);
    free(streamDRR);
    free(streamRRR);
    
    CUCHECK(cudaEventDestroy(start_timmer));
    CUCHECK(cudaEventDestroy(stop_timmer));

    free(DDD);
    free(RRR);
    free(DRR);
    free(DDR);
    
    CUCHECK(cudaFree(d_DDD));
    CUCHECK(cudaFree(d_RRR));
    CUCHECK(cudaFree(d_DRR));
    CUCHECK(cudaFree(d_DDR));

}


void pcf_3iso_wpips(
        DNode *dnodeD, PointW3D *dataD, int32_t *pipsD, int nonzero_Dnodes,
        DNode *dnodeR, PointW3D *dataR, int32_t *pipsR, int *nonzero_Rnodes, int *acum_nonzero_Rnodes,
        char **histo_names, int n_randfiles, int bins, float size_node, float dmax, int pips_width
    )
{

    /* =======================================================================*/
    /* =====================   Var declaration ===============================*/
    /* =======================================================================*/

    const int PREFIX_LENGTH = 13;
    float time_spent, d_max_node;

    double *DDD, *RRR, *DRR, *DDR;
    double *d_DDD, *d_RRR, *d_DRR, *d_DDR;

    //n_kernel_calls should depend of the number of points, its density, and the number of bins
    int threads_perblock_dim = 8;
    int blocks_D, blocks_R;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    CUCHECK(cudaEventCreate(&start_timmer));
    CUCHECK(cudaEventCreate(&stop_timmer));

    cudaStream_t streamDDD, *streamRRR, *streamDDR, *streamDRR;
    streamRRR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamRRR);
    streamDDR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamDDR);
    streamDRR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamDRR);
    CUCHECK(cudaStreamCreate(&streamDDD));
    for (int i = 0; i < n_randfiles; i++){
        CUCHECK(cudaStreamCreate(&streamRRR[i]));
        CUCHECK(cudaStreamCreate(&streamDDR[i]));
        CUCHECK(cudaStreamCreate(&streamDRR[i]));
    }

    // Name of the files where the results are saved
    char *nameDDD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    char *nameRRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    char *nameDDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    char *nameDRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    strcpy(nameDDD,"DDDiso_pips_");
    strcpy(nameRRR,"DRRiso_pips_");
    strcpy(nameDDR,"DDRiso_pips_");
    strcpy(nameDRR,"RRRiso_pips_");

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    // Allocate memory for the histogram as double
    DDD = (double*)calloc(bins*bins*bins, sizeof(double));
    CHECKALLOC(DDD);
    RRR = (double*)calloc(n_randfiles*bins*bins*bins, sizeof(double));
    CHECKALLOC(RRR);
    DDR = (double*)calloc(n_randfiles*bins*bins*bins, sizeof(double));
    CHECKALLOC(DDR);
    DRR = (double*)calloc(n_randfiles*bins*bins*bins, sizeof(double));
    CHECKALLOC(DRR);

    CUCHECK(cudaMalloc(&d_DDD, bins*bins*bins*sizeof(double)));
    CUCHECK(cudaMalloc(&d_RRR, n_randfiles*bins*bins*bins*sizeof(double)));
    CUCHECK(cudaMalloc(&d_DRR, n_randfiles*bins*bins*bins*sizeof(double)));
    CUCHECK(cudaMalloc(&d_DDR, n_randfiles*bins*bins*bins*sizeof(double)));

    //Restarts the main histograms in host to zero
    CUCHECK(cudaMemsetAsync(d_DDD, 0, bins*bins*bins*sizeof(double), streamDDD));
    CUCHECK(cudaMemsetAsync(d_RRR, 0, n_randfiles*bins*bins*bins*sizeof(double), streamRRR[0]));
    CUCHECK(cudaMemsetAsync(d_DRR, 0, n_randfiles*bins*bins*bins*sizeof(double), streamDRR[0]));
    CUCHECK(cudaMemsetAsync(d_DDR, 0, n_randfiles*bins*bins*bins*sizeof(double), streamDDR[0]));
    CUCHECK(cudaStreamSynchronize(streamRRR[0]));
    CUCHECK(cudaStreamSynchronize(streamDRR[0]));
    CUCHECK(cudaStreamSynchronize(streamDDR[0]));

    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/


    //Compute the dimensions of the GPU grid
    //One thread for each node
    
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));
    dim3 gridDDD(blocks_D,blocks_D,blocks_D);
    dim3 threads_perblock(threads_perblock_dim,threads_perblock_dim,threads_perblock_dim);
    
    //Dummy declaration
    dim3 gridRRR(2,2,1);
    dim3 gridDDR(blocks_D,blocks_D,1);
    dim3 gridDRR(2,2,blocks_D);

    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    XXX3iso_wpips<<<gridDDD,threads_perblock,0,streamDDD>>>(d_DDD, dataD, pipsD, dnodeD, nonzero_Dnodes, bins, dmax, d_max_node, pips_width, 0, 0);
    for (int i=0; i<n_randfiles; i++){
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridRRR.x = blocks_R;
        gridRRR.y = blocks_R;
        gridRRR.z = blocks_R;
        XXX3iso_wpips<<<gridRRR,threads_perblock,0,streamRRR[i]>>>(d_RRR, dataR, pipsR, dnodeR, nonzero_Rnodes[i], bins, dmax, d_max_node, pips_width, acum_nonzero_Rnodes[i], i);
        gridDDR.z = blocks_R;
        XXY3iso_wpips<<<gridDDR,threads_perblock,0,streamDDR[i]>>>(d_DDR, dataD, pipsD, dnodeD, nonzero_Dnodes, dataR, pipsR, dnodeR, nonzero_Rnodes[i], bins, dmax, d_max_node, pips_width, acum_nonzero_Rnodes[i], i, true);
        gridDRR.x = blocks_R;
        gridDRR.y = blocks_R;
        XXY3iso_wpips<<<gridDRR,threads_perblock,0,streamDRR[i]>>>(d_DRR, dataR, pipsR, dnodeR, nonzero_Rnodes[i], dataD, pipsD, dnodeD, nonzero_Dnodes, bins, dmax, d_max_node, pips_width, acum_nonzero_Rnodes[i], i, false);
    }

    //Waits for all the kernels to complete
    CUCHECK(cudaDeviceSynchronize());

    //Save the results
    CUCHECK(cudaMemcpy(DDD, d_DDD, bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost));

    nameDDD = (char*)realloc(nameDDD,PREFIX_LENGTH + strlen(histo_names[0]));
    strcpy(&nameDDD[PREFIX_LENGTH-1],histo_names[0]);
    save_histogram3D(nameDDD, bins, DDD, 0);

    CUCHECK(cudaMemcpy(RRR, d_RRR, n_randfiles*bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameRRR = (char*)realloc(nameRRR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameRRR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram3D(nameRRR, bins, RRR, i);
    }
    CUCHECK(cudaMemcpy(DDR, d_DDR, n_randfiles*bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameDDR = (char*)realloc(nameDDR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameDDR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram3D(nameDDR, bins, DDR, i);
    }
    CUCHECK(cudaMemcpy(DRR, d_DRR, n_randfiles*bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameDRR = (char*)realloc(nameDRR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameDRR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram3D(nameDRR, bins, DRR, i);
    }

    CUCHECK(cudaEventRecord(stop_timmer));
    CUCHECK(cudaEventSynchronize(stop_timmer));
    CUCHECK(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    printf("Spent %f miliseconds to compute and save all the histograms. \n", time_spent);
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory
    CUCHECK(cudaStreamDestroy(streamDDD));
    for (int i = 0; i < n_randfiles; i++){
        CUCHECK(cudaStreamDestroy(streamDDR[i]));
        CUCHECK(cudaStreamDestroy(streamDRR[i]));
        CUCHECK(cudaStreamDestroy(streamRRR[i]));
    }
    free(streamDDR);
    free(streamDRR);
    free(streamRRR);
    
    CUCHECK(cudaEventDestroy(start_timmer));
    CUCHECK(cudaEventDestroy(stop_timmer));

    free(DDD);
    free(RRR);
    free(DRR);
    free(DDR);
    
    CUCHECK(cudaFree(d_DDD));
    CUCHECK(cudaFree(d_RRR));
    CUCHECK(cudaFree(d_DRR));
    CUCHECK(cudaFree(d_DDR));

}

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__global__ void XXX3iso(double *XXX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset){
    
    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = node_offset + blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<(nonzero_nodes+node_offset) && idx2<(nonzero_nodes+node_offset) && idx3<(nonzero_nodes+node_offset)){
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1)+(ny2-ny1)*(ny2-ny1)+(nz2-nz1)*(nz2-nz1);
        if (dd_nod12 <= d_max_node){
            float nx3=nodeD[idx3].nodepos.x, ny3=nodeD[idx3].nodepos.y, nz3=nodeD[idx3].nodepos.z;
            float dd_nod23 = (nx3-nx2)*(nx3-nx2)+(ny3-ny2)*(ny3-ny2)+(nz3-nz2)*(nz3-nz2);
            if (dd_nod23 <= d_max_node){
                float dd_nod31 = (nx1-nx3)*(nx1-nx3)+(ny1-ny3)*(ny1-ny3)+(nz1-nz3)*(nz1-nz3);
                if (dd_nod31 <= d_max_node){
                    int end1 = nodeD[idx1].end;
                    int end2 = nodeD[idx2].end;
                    int end3 = nodeD[idx3].end;
                    int bnx, bny, bnz, bin;
                    float dd_max=dmax*dmax;
                    float x1,y1,z1,w1,x2,y2,z2,w2,x3,y3,z3;
                    double d12, d23, d31, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;

                    for (int i=nodeD[idx1].start; i<end1; i++){
                        x1 = elements[i].x;
                        y1 = elements[i].y;
                        z1 = elements[i].z;
                        w1 = elements[i].w;
                        for (int j=nodeD[idx2].start; j<end2; j++){
                            x2 = elements[j].x;
                            y2 = elements[j].y;
                            z2 = elements[j].z;
                            w2 = elements[j].w;
                            v = w1*w2;
                            d12 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
                            if (d12 < dd_max && d12>0){
                                d12 = sqrt(d12);
                                for (int k=nodeD[idx3].start; k<end3; k++){
                                    x3 = elements[k].x;
                                    y3 = elements[k].y;
                                    z3 = elements[k].z;
                                    d23 = (x3-x2)*(x3-x2) + (y3-y2)*(y3-y2) + (z3-z2)*(z3-z2);
                                    if (d23 < dd_max && d23>0){
                                        d31 = (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1) + (z3-z1)*(z3-z1);
                                        if (d31 < dd_max && d31>0){
                                            d23 = sqrt(d23);
                                            d31 = sqrt(d31);

                                            bnx = (int)(d12*ds)*bn*bn;
                                            if (bnx>(bn*bn*(bn-1))) continue;
                                            bny = (int)(d23*ds)*bn;
                                            if (bny>(bn*(bn-1))) continue;
                                            bnz = (int)(d31*ds);
                                            if (bnz>(bn-1)) continue;
                                            bin = bnx + bny + bnz;
                                            bin += bn*bn*bn*bn_offset;
                                            v *= elements[k].w;

                                            atomicAdd(&XXX[bin],v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void XXY3iso(double *XXY, PointW3D *elementsX, DNode *nodeX, int nonzero_Xnodes, PointW3D *elementsY, DNode *nodeY, int nonzero_Ynodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset, int isDDR){

    int idx1 = (!isDDR)*node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = (!isDDR)*node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = isDDR*node_offset + blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<(nonzero_Xnodes + (!isDDR)*node_offset) && idx2<(nonzero_Xnodes + (!isDDR)*node_offset) && idx3<(nonzero_Ynodes + isDDR*node_offset)){
        float nx1=nodeX[idx1].nodepos.x, ny1=nodeX[idx1].nodepos.y, nz1=nodeX[idx1].nodepos.z;
        float nx2=nodeX[idx2].nodepos.x, ny2=nodeX[idx2].nodepos.y, nz2=nodeX[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1)+(ny2-ny1)*(ny2-ny1)+(nz2-nz1)*(nz2-nz1);
        if (dd_nod12 <= d_max_node){
            float nx3=nodeY[idx3].nodepos.x, ny3=nodeY[idx3].nodepos.y, nz3=nodeY[idx3].nodepos.z;
            float dd_nod23 = (nx3-nx2)*(nx3-nx2)+(ny3-ny2)*(ny3-ny2)+(nz3-nz2)*(nz3-nz2);
            if (dd_nod23 <= d_max_node){
                float dd_nod31 = (nx1-nx3)*(nx1-nx3)+(ny1-ny3)*(ny1-ny3)+(nz1-nz3)*(nz1-nz3);
                if (dd_nod31 <= d_max_node){
                    int end1 = nodeX[idx1].end;
                    int end2 = nodeX[idx2].end;
                    int end3 = nodeY[idx3].end;
                    int bnx, bny, bnz, bin;
                    float dd_max=dmax*dmax;
                    float x1,y1,z1,w1,x2,y2,z2,w2,x3,y3,z3;
                    double d12, d23, d31, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;

                    for (int i=nodeX[idx1].start; i<end1; i++){
                        x1 = elementsX[i].x;
                        y1 = elementsX[i].y;
                        z1 = elementsX[i].z;
                        w1 = elementsX[i].w;
                        for (int j=nodeX[idx2].start; j<end2; j++){
                            x2 = elementsX[j].x;
                            y2 = elementsX[j].y;
                            z2 = elementsX[j].z;
                            w2 = elementsX[j].w;
                            v = w1*w2;
                            d12 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
                            if (d12 < dd_max && d12>0){
                                d12 = sqrt(d12);
                                for (int k=nodeY[idx3].start; k<end3; k++){
                                    x3 = elementsY[k].x;
                                    y3 = elementsY[k].y;
                                    z3 = elementsY[k].z;
                                    d23 = (x3-x2)*(x3-x2) + (y3-y2)*(y3-y2) + (z3-z2)*(z3-z2);
                                    if (d23 < dd_max){
                                        d31 = (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1) + (z3-z1)*(z3-z1);
                                        if (d31 < dd_max){
                                            d23 = sqrt(d23);
                                            d31 = sqrt(d31);                                            
                                            
                                            bnx = (int)(d12*ds)*bn*bn;
                                            if (bnx>(bn*bn*(bn-1))) continue;
                                            bny = (int)(d23*ds)*bn;
                                            if (bny>(bn*(bn-1))) continue;
                                            bnz = (int)(d31*ds);
                                            if (bnz>(bn-1)) continue;
                                            bin = bnx + bny + bnz;
                                            bin += bn*bn*bn*bn_offset;

                                            v *= elementsY[k].w;
                                            atomicAdd(&XXY[bin],v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


__global__ void XXX3iso_wpips(double *XXX, PointW3D *elements, int32_t *pipsD, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, int pips_width, int node_offset, int bn_offset){

    
    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = node_offset + blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<(nonzero_nodes+node_offset) && idx2<(nonzero_nodes+node_offset) && idx3<(nonzero_nodes+node_offset)){
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1)+(ny2-ny1)*(ny2-ny1)+(nz2-nz1)*(nz2-nz1);
        if (dd_nod12 <= d_max_node){
            float nx3=nodeD[idx3].nodepos.x, ny3=nodeD[idx3].nodepos.y, nz3=nodeD[idx3].nodepos.z;
            float dd_nod23 = (nx3-nx2)*(nx3-nx2)+(ny3-ny2)*(ny3-ny2)+(nz3-nz2)*(nz3-nz2);
            if (dd_nod23 <= d_max_node){
                float dd_nod31 = (nx1-nx3)*(nx1-nx3)+(ny1-ny3)*(ny1-ny3)+(nz1-nz3)*(nz1-nz3);
                if (dd_nod31 <= d_max_node){
                    int end1 = nodeD[idx1].end;
                    int end2 = nodeD[idx2].end;
                    int end3 = nodeD[idx3].end;
                    int bnx, bny, bnz, bin;
                    float dd_max=dmax*dmax;
                    float x1,y1,z1,w1,x2,y2,z2,w2,x3,y3,z3;
                    double d12, d23, d31, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;

                    for (int i=nodeD[idx1].start; i<end1; i++){
                        x1 = elements[i].x;
                        y1 = elements[i].y;
                        z1 = elements[i].z;
                        for (int j=nodeD[idx2].start; j<end2; j++){
                            x2 = elements[j].x;
                            y2 = elements[j].y;
                            z2 = elements[j].z;
                            d12 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
                            if (d12 < dd_max && d12>0){
                                d12 = sqrt(d12);
                                for (int k=nodeD[idx3].start; k<end3; k++){
                                    x3 = elements[k].x;
                                    y3 = elements[k].y;
                                    z3 = elements[k].z;
                                    d23 = (x3-x2)*(x3-x2) + (y3-y2)*(y3-y2) + (z3-z2)*(z3-z2);
                                    if (d23 < dd_max && d23>0){
                                        d31 = (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1) + (z3-z1)*(z3-z1);
                                        if (d31 < dd_max && d31>0){
                                            d23 = sqrt(d23);
                                            d31 = sqrt(d31);

                                            bnx = (int)(d12*ds)*bn*bn;
                                            if (bnx>(bn*bn*(bn-1))) continue;
                                            bny = (int)(d23*ds)*bn;
                                            if (bny>(bn*(bn-1))) continue;
                                            bnz = (int)(d31*ds);
                                            if (bnz>(bn-1)) continue;
                                            bin = bnx + bny + bnz;
                                            bin += bn*bn*bn*bn_offset;

                                            v = get_3d_weight(pipsD, i, pipsD, j, pipsD, k, pips_width);

                                            atomicAdd(&XXX[bin],v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void XXY3iso_wpips(double *XXY, PointW3D *elementsX, int32_t *pipsX, DNode *nodeX, int nonzero_Xnodes, PointW3D *elementsY, int32_t *pipsY, DNode *nodeY, int nonzero_Ynodes, int bn, float dmax, float d_max_node, int pips_width, int node_offset, int bn_offset, int isDDR){


    int idx1 = (!isDDR)*node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = (!isDDR)*node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = isDDR*node_offset + blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<(nonzero_Xnodes + (!isDDR)*node_offset) && idx2<(nonzero_Xnodes + (!isDDR)*node_offset) && idx3<(nonzero_Ynodes + isDDR*node_offset)){
        float nx1=nodeX[idx1].nodepos.x, ny1=nodeX[idx1].nodepos.y, nz1=nodeX[idx1].nodepos.z;
        float nx2=nodeX[idx2].nodepos.x, ny2=nodeX[idx2].nodepos.y, nz2=nodeX[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1)+(ny2-ny1)*(ny2-ny1)+(nz2-nz1)*(nz2-nz1);
        if (dd_nod12 <= d_max_node){
            float nx3=nodeY[idx3].nodepos.x, ny3=nodeY[idx3].nodepos.y, nz3=nodeY[idx3].nodepos.z;
            float dd_nod23 = (nx3-nx2)*(nx3-nx2)+(ny3-ny2)*(ny3-ny2)+(nz3-nz2)*(nz3-nz2);
            if (dd_nod23 <= d_max_node){
                float dd_nod31 = (nx1-nx3)*(nx1-nx3)+(ny1-ny3)*(ny1-ny3)+(nz1-nz3)*(nz1-nz3);
                if (dd_nod31 <= d_max_node){
                    int end1 = nodeX[idx1].end;
                    int end2 = nodeX[idx2].end;
                    int end3 = nodeY[idx3].end;
                    int bnx, bny, bnz, bin;
                    float dd_max=dmax*dmax;
                    float x1,y1,z1,w1,x2,y2,z2,w2,x3,y3,z3;
                    double d12, d23, d31, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;

                    for (int i=nodeX[idx1].start; i<end1; i++){
                        x1 = elementsX[i].x;
                        y1 = elementsX[i].y;
                        z1 = elementsX[i].z;
                        for (int j=nodeX[idx2].start; j<end2; j++){
                            x2 = elementsX[j].x;
                            y2 = elementsX[j].y;
                            z2 = elementsX[j].z;
                            d12 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
                            if (d12 < dd_max && d12>0){
                                d12 = sqrt(d12);
                                for (int k=nodeY[idx3].start; k<end3; k++){
                                    x3 = elementsY[k].x;
                                    y3 = elementsY[k].y;
                                    z3 = elementsY[k].z;
                                    d23 = (x3-x2)*(x3-x2) + (y3-y2)*(y3-y2) + (z3-z2)*(z3-z2);
                                    if (d23 < dd_max){
                                        d31 = (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1) + (z3-z1)*(z3-z1);
                                        if (d31 < dd_max){
                                            d23 = sqrt(d23);
                                            d31 = sqrt(d31);                                            
                                            
                                            bnx = (int)(d12*ds)*bn*bn;
                                            if (bnx>(bn*bn*(bn-1))) continue;
                                            bny = (int)(d23*ds)*bn;
                                            if (bny>(bn*(bn-1))) continue;
                                            bnz = (int)(d31*ds);
                                            if (bnz>(bn-1)) continue;
                                            bin = bnx + bny + bnz;
                                            bin += bn*bn*bn*bn_offset;

                                            v = get_3d_weight(pipsX, i, pipsX, j, pipsY, k, pips_width);

                                            atomicAdd(&XXY[bin],v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
