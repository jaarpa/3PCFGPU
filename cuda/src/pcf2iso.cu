#include <string.h>
#include <math.h>
#include <stdio.h>
#include "device_functions.cuh"
#include "create_grid.cuh"
#include "pcf2iso.cuh"

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
Kernel function to calculate the pure histograms for the 2 point isotropic correlation function. 
This version does NOT considers boudary periodic conditions. It stores the counts in the XX histogram.

args:
XX: (double*) The histogram where the distances are counted.
elements: (PointW3D*) Array of the points ordered coherently with the nodes.
nodeD: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node.
nonzero_nodes: (int) Number of nonzero nodes where the points have been classificated.
bn: (int) NUmber of bins in the XY histogram.
dmax: (float) The maximum distance of interest between points.
d_max_node: (float) The maximum internodal distance.
*/
__global__ void XX2iso(double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset);

/*
Kernel function to calculate the mixed histograms for the 2 point isotropic correlation function. 
This version does NOT include boundary periodic conditions. It stores the counts in the XY histogram.

args:
XY: (double*) The histogram where the distances are counted.
elementsD: (PointW3D*) Array of the points ordered coherently with the nodes. For the data points.
nodeD: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node. For the data points
nonzero_Dnodes: (int) Number of nonzero nodes where the points have been classificated. For the data points
elementsR: (PointW3D*) Array of the points ordered coherently with the nodes. For the random points.
nodeR: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node. For the random points
nonzero_Rnodes: (int) Number of nonzero nodes where the points have been classificated. For the random points
bn: (int) NUmber of bins in the XY histogram.
dmax: (float) The maximum distance of interest between points.
d_max_node: (float) The maximum internodal distance.
*/
__global__ void XY2iso(double *XY, PointW3D *elementsD, DNode *nodeD, int nonzero_Dnodes, PointW3D *elementsR,  DNode *nodeR, int nonzero_Rnodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset);

__global__ void XX2iso_wpips(double *XX, PointW3D *elements, int32_t *pipsD, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, int pipis_width, int node_offset, int bn_offset);
__global__ void XY2iso_wpips(double *XY, PointW3D *elementsD, int32_t *pipsD, DNode *nodeD, int nonzero_Dnodes, PointW3D *elementsR, int32_t *pipsR,  DNode *nodeR, int nonzero_Rnodes, int bn, float dmax, float d_max_node, int pipis_width, int node_offset, int bn_offset);

void pcf_2iso(
    DNode *dnodeD, PointW3D *dataD, int nonzero_Dnodes,
    DNode *dnodeR, PointW3D *dataR, int *nonzero_Rnodes, int *acum_nonzero_Rnodes,
    char **histo_names, int n_randfiles, int bins, float size_node, float dmax
)
{
    /* =======================================================================*/
    /* ======================  Var declaration ===============================*/
    /* =======================================================================*/

    const int PREFIX_LENGTH = 7;
    float d_max_node, time_spent;
    double *DD, *RR, *DR, *d_DD, *d_RR, *d_DR;
    int  blocks_D, blocks_R, threads_perblock_dim = 32;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    CUCHECK(cudaEventCreate(&start_timmer));
    CUCHECK(cudaEventCreate(&stop_timmer));

    cudaStream_t streamDD, *streamRR, *streamDR;
    streamRR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamRR);
    streamDR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamDR);
    CUCHECK(cudaStreamCreate(&streamDD));
    for (int i = 0; i < n_randfiles; i++){
        CUCHECK(cudaStreamCreate(&streamDR[i]));
        CUCHECK(cudaStreamCreate(&streamRR[i]));
    }

    char *nameDD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    char *nameRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    char *nameDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    strcpy(nameDD,"DDiso_");
    strcpy(nameRR,"RRiso_");
    strcpy(nameDR,"DRiso_");

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    // Allocate memory for the histogram as double
    DD = (double*)malloc(bins*sizeof(double));
    CHECKALLOC(DD);
    RR = (double*)malloc(n_randfiles*bins*sizeof(double));
    CHECKALLOC(RR);
    DR = (double*)malloc(n_randfiles*bins*sizeof(double));
    CHECKALLOC(DR);

    CUCHECK(cudaMalloc(&d_DD, bins*sizeof(double)));
    CUCHECK(cudaMalloc(&d_RR, n_randfiles*bins*sizeof(double)));
    CUCHECK(cudaMalloc(&d_DR, n_randfiles*bins*sizeof(double)));

    //Restarts the main histograms in device to zero
    CUCHECK(cudaMemsetAsync(d_DD, 0, bins*sizeof(double), streamDD));
    CUCHECK(cudaMemsetAsync(d_RR, 0, n_randfiles*bins*sizeof(double), streamRR[0]));
    CUCHECK(cudaMemsetAsync(d_DR, 0, n_randfiles*bins*sizeof(double), streamDR[0]));
    CUCHECK(cudaStreamSynchronize(streamRR[0]));
    CUCHECK(cudaStreamSynchronize(streamDR[0]));

    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/


    //Compute the dimensions of the GPU grid
    //One thread for each node
    
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));
    dim3 threads_perblock(threads_perblock_dim,threads_perblock_dim,1);
    dim3 gridD(blocks_D,blocks_D,1);
    
    //Dummy declaration
    dim3 gridR(2,2,1);
    dim3 gridDR(blocks_D,2,1);

    //Launch the kernels
    time_spent=0; //Restarts timmer
    CUCHECK(cudaEventRecord(start_timmer));
    XX2iso<<<gridD,threads_perblock,0,streamDD>>>(d_DD, dataD, dnodeD, nonzero_Dnodes, bins, dmax, d_max_node, 0, 0);
    for (int i=0; i<n_randfiles; i++){
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridR.x = blocks_R;
        gridR.y = blocks_R;
        XX2iso<<<gridR,threads_perblock,0,streamRR[i]>>>(d_RR, dataR, dnodeR, nonzero_Rnodes[i], bins, dmax, d_max_node, acum_nonzero_Rnodes[i], i);
        gridDR.y = blocks_R;
        XY2iso<<<gridDR,threads_perblock,0,streamDR[i]>>>(d_DR, dataD, dnodeD, nonzero_Dnodes, dataR, dnodeR, nonzero_Rnodes[i], bins, dmax, d_max_node, acum_nonzero_Rnodes[i], i);
    }

    //Waits for all the kernels to complete
    CUCHECK(cudaDeviceSynchronize());

    //Save the results
    CUCHECK(cudaMemcpy(DD, d_DD, bins*sizeof(double), cudaMemcpyDeviceToHost));
    nameDD = (char*)realloc(nameDD,PREFIX_LENGTH + strlen(histo_names[0]));
    strcpy(&nameDD[PREFIX_LENGTH-1],histo_names[0]);
    save_histogram1D(nameDD, bins, DD, 0);
    CUCHECK(cudaMemcpy(RR, d_RR, n_randfiles*bins*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameRR = (char*)realloc(nameRR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameRR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram1D(nameRR, bins, RR, i);
    }
    CUCHECK(cudaMemcpy(DR, d_DR, n_randfiles*bins*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameDR = (char*)realloc(nameDR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameDR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram1D(nameDR, bins, DR, i);
    }

    CUCHECK(cudaEventRecord(stop_timmer));
    CUCHECK(cudaEventSynchronize(stop_timmer));
    CUCHECK(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    printf("Spent %f miliseconds to compute and save all the histograms. \n", time_spent);
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory
    CUCHECK(cudaStreamDestroy(streamDD));
    for (int i = 0; i < n_randfiles; i++){
        CUCHECK(cudaStreamDestroy(streamDR[i]));
        CUCHECK(cudaStreamDestroy(streamRR[i]));
    }
    free(streamDR);
    free(streamRR);

    CUCHECK(cudaEventDestroy(start_timmer));
    CUCHECK(cudaEventDestroy(stop_timmer));

    free(DD);
    free(RR);
    free(DR);
    
    CUCHECK(cudaFree(d_DD));
    CUCHECK(cudaFree(d_RR));
    CUCHECK(cudaFree(d_DR));

}

void pcf_2iso_wpips(
    DNode *dnodeD, PointW3D *dataD, int32_t *dpipsD, int nonzero_Dnodes,
    DNode *dnodeR, PointW3D *dataR, int32_t *dpipsR, int *nonzero_Rnodes,
    int *acum_nonzero_Rnodes, int n_pips,
    char **histo_names, int n_randfiles, int bins, float size_node, float dmax
)
{
    /* =======================================================================*/
    /* ======================  Var declaration ===============================*/
    /* =======================================================================*/

    const int PREFIX_LENGTH = 7;
    float d_max_node, time_spent;
    double *DD, *RR, *DR, *d_DD, *d_RR, *d_DR;
    int  blocks_D, blocks_R, threads_perblock_dim = 32;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    CUCHECK(cudaEventCreate(&start_timmer));
    CUCHECK(cudaEventCreate(&stop_timmer));

    cudaStream_t streamDD, *streamRR, *streamDR;
    streamRR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamRR);
    streamDR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamDR);
    CUCHECK(cudaStreamCreate(&streamDD));
    for (int i = 0; i < n_randfiles; i++){
        CUCHECK(cudaStreamCreate(&streamDR[i]));
        CUCHECK(cudaStreamCreate(&streamRR[i]));
    }

    char *nameDD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    CHECKALLOC(nameDD);
    char *nameRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    CHECKALLOC(nameRR);
    char *nameDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    CHECKALLOC(nameDR);
    strcpy(nameDD,"DDiso_");
    strcpy(nameRR,"RRiso_");
    strcpy(nameDR,"DRiso_");

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    // Allocate memory for the histogram as double
    DD = (double*)malloc(bins*sizeof(double));
    CHECKALLOC(DD);
    RR = (double*)malloc(n_randfiles*bins*sizeof(double));
    CHECKALLOC(RR);
    DR = (double*)malloc(n_randfiles*bins*sizeof(double));
    CHECKALLOC(DR);

    CUCHECK(cudaMalloc(&d_DD, bins*sizeof(double)));
    CUCHECK(cudaMalloc(&d_RR, n_randfiles*bins*sizeof(double)));
    CUCHECK(cudaMalloc(&d_DR, n_randfiles*bins*sizeof(double)));

    //Restarts the main histograms in device to zero
    CUCHECK(cudaMemsetAsync(d_DD, 0, bins*sizeof(double), streamDD));
    CUCHECK(cudaMemsetAsync(d_RR, 0, n_randfiles*bins*sizeof(double), streamRR[0]));
    CUCHECK(cudaMemsetAsync(d_DR, 0, n_randfiles*bins*sizeof(double), streamDR[0]));
    CUCHECK(cudaStreamSynchronize(streamRR[0]));
    CUCHECK(cudaStreamSynchronize(streamDR[0]));

    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/


    //Compute the dimensions of the GPU grid
    //One thread for each node
    
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));
    dim3 threads_perblock(threads_perblock_dim,threads_perblock_dim,1);
    dim3 gridD(blocks_D,blocks_D,1);
    
    //Dummy declaration
    dim3 gridR(2,2,1);
    dim3 gridDR(blocks_D,2,1);

    //Launch the kernels
    time_spent=0; //Restarts timmer
    CUCHECK(cudaEventRecord(start_timmer));
    XX2iso_wpips<<<gridD,threads_perblock,0,streamDD>>>(d_DD, dataD, dpipsD, dnodeD, nonzero_Dnodes, bins, dmax, d_max_node, n_pips, 0, 0);
    for (int i=0; i<n_randfiles; i++){
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridR.x = blocks_R;
        gridR.y = blocks_R;
        XX2iso_wpips<<<gridR,threads_perblock,0,streamRR[i]>>>(d_RR, dataR, dpipsR, dnodeR, nonzero_Rnodes[i], bins, dmax, d_max_node, n_pips, acum_nonzero_Rnodes[i], i);
        gridDR.y = blocks_R;
        XY2iso_wpips<<<gridDR,threads_perblock,0,streamDR[i]>>>(d_DR, dataD, dpipsD, dnodeD, nonzero_Dnodes, dataR, dpipsR, dnodeR, nonzero_Rnodes[i], bins, dmax, d_max_node, n_pips, acum_nonzero_Rnodes[i], i);
    }

    //Waits for all the kernels to complete
    CUCHECK(cudaDeviceSynchronize());

    //Save the results
    CUCHECK(cudaMemcpy(DD, d_DD, bins*sizeof(double), cudaMemcpyDeviceToHost));
    nameDD = (char*)realloc(nameDD,PREFIX_LENGTH + strlen(histo_names[0]));
    strcpy(&nameDD[PREFIX_LENGTH-1],histo_names[0]);
    save_histogram1D(nameDD, bins, DD, 0);
    CUCHECK(cudaMemcpy(RR, d_RR, n_randfiles*bins*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameRR = (char*)realloc(nameRR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameRR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram1D(nameRR, bins, RR, i);
    }
    CUCHECK(cudaMemcpy(DR, d_DR, n_randfiles*bins*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameDR = (char*)realloc(nameDR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameDR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram1D(nameDR, bins, DR, i);
    }

    CUCHECK(cudaEventRecord(stop_timmer));
    CUCHECK(cudaEventSynchronize(stop_timmer));
    CUCHECK(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    printf("Spent %f miliseconds to compute and save all the histograms. \n", time_spent);
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory
    CUCHECK(cudaStreamDestroy(streamDD));
    for (int i = 0; i < n_randfiles; i++){
        CUCHECK(cudaStreamDestroy(streamDR[i]));
        CUCHECK(cudaStreamDestroy(streamRR[i]));
    }
    free(streamDR);
    free(streamRR);

    CUCHECK(cudaEventDestroy(start_timmer));
    CUCHECK(cudaEventDestroy(stop_timmer));

    free(DD);
    free(RR);
    free(DR);
    
    CUCHECK(cudaFree(d_DD));
    CUCHECK(cudaFree(d_RR));
    CUCHECK(cudaFree(d_DR));
}

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__global__ void XX2iso(double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset)
{

    //Distributes all the indexes equitatively into the n_kernelc_calls.

    int idx1 = node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<(nonzero_nodes+node_offset) && idx2<(nonzero_nodes+node_offset)){
        
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1) + (ny2-ny1)*(ny2-ny1) + (nz2-nz1)*(nz2-nz1);

        if (dd_nod12 <= d_max_node){

            float x1,y1,z1,x2,y2,z2;
            float dd_max=dmax*dmax;
            double d, ds = floor(((double)(bn)/dmax)*1000000)/1000000;
            int bin, end1=nodeD[idx1].end, end2=nodeD[idx2].end;
            double v;

            for (int i=nodeD[idx1].start; i<end1; ++i){
                x1 = elements[i].x;
                y1 = elements[i].y;
                z1 = elements[i].z;
                for (int j=nodeD[idx2].start; j<end2; ++j){
                    x2 = elements[j].x;
                    y2 = elements[j].y;
                    z2 = elements[j].z;
                    d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                    if (d<dd_max && d>0){
                        bin = (int)(sqrt(d)*ds);
                        if (bin>(bn-1)) continue;
                        bin += bn_offset*bn;
                        v = elements[i].w*elements[j].w;
                        atomicAdd(&XX[bin],v);
                    }
                }
            }
        }
    }
}

__global__ void XY2iso(double *XY, PointW3D *elementsD, DNode *nodeD, int nonzero_Dnodes, PointW3D *elementsR,  DNode *nodeR, int nonzero_Rnodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset){

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1<nonzero_Dnodes && idx2<(nonzero_Rnodes+node_offset)){
        
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeR[idx2].nodepos.x, ny2=nodeR[idx2].nodepos.y, nz2=nodeR[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1) + (ny2-ny1)*(ny2-ny1) + (nz2-nz1)*(nz2-nz1);

        if (dd_nod12 <= d_max_node){

            float x1,y1,z1,x2,y2,z2;
            float dd_max=dmax*dmax;
            double d, ds = floor(((double)(bn)/dmax)*1000000)/1000000;
            int bin, end1=nodeD[idx1].end, end2=nodeR[idx2].end;
            double v;

            for (int i=nodeD[idx1].start; i<end1; ++i){
                x1 = elementsD[i].x;
                y1 = elementsD[i].y;
                z1 = elementsD[i].z;
                for (int j=nodeR[idx2].start; j<end2; ++j){
                    x2 = elementsR[j].x;
                    y2 = elementsR[j].y;
                    z2 = elementsR[j].z;
                    d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                    if (d<dd_max){
                        bin = (int)(sqrt(d)*ds);

                        if (bin>(bn-1)) continue;
                        bin += bn_offset*bn;

                        v = elementsD[i].w*elementsR[j].w;
                        atomicAdd(&XY[bin],v);
                    }
                }
            }
        }
    }
}


__global__ void XX2iso_wpips(double *XX, PointW3D *elements, int32_t *pipsD, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, int pipis_width, int node_offset, int bn_offset)
{

    //Distributes all the indexes equitatively into the n_kernelc_calls.

    int idx1 = node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<(nonzero_nodes+node_offset) && idx2<(nonzero_nodes+node_offset)){
        
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1) + (ny2-ny1)*(ny2-ny1) + (nz2-nz1)*(nz2-nz1);

        if (dd_nod12 <= d_max_node){

            float x1,y1,z1,x2,y2,z2;
            float dd_max=dmax*dmax;
            double d, ds = floor(((double)(bn)/dmax)*1000000)/1000000;
            int bin, end1=nodeD[idx1].end, end2=nodeD[idx2].end;
            double v;

            for (int i=nodeD[idx1].start; i<end1; ++i){
                x1 = elements[i].x;
                y1 = elements[i].y;
                z1 = elements[i].z;
                for (int j=nodeD[idx2].start; j<end2; ++j){
                    x2 = elements[j].x;
                    y2 = elements[j].y;
                    z2 = elements[j].z;
                    d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                    if (d<dd_max && d>0){
                        bin = (int)(sqrt(d)*ds);
                        if (bin>(bn-1)) continue;
                        bin += bn_offset*bn;
                        v = get_weight(pipsD, i, pipsD, j, pipis_width);
                        atomicAdd(&XX[bin],v);
                    }
                }
            }
        }
    }
}

__global__ void XY2iso_wpips(double *XY, PointW3D *elementsD, int32_t *pipsD, DNode *nodeD, int nonzero_Dnodes, PointW3D *elementsR, int32_t *pipsR,  DNode *nodeR, int nonzero_Rnodes, int bn, float dmax, float d_max_node, int pipis_width, int node_offset, int bn_offset)
{
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1<nonzero_Dnodes && idx2<(nonzero_Rnodes+node_offset)){
        
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeR[idx2].nodepos.x, ny2=nodeR[idx2].nodepos.y, nz2=nodeR[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1) + (ny2-ny1)*(ny2-ny1) + (nz2-nz1)*(nz2-nz1);

        if (dd_nod12 <= d_max_node){

            float x1,y1,z1,x2,y2,z2;
            float dd_max=dmax*dmax;
            double d, ds = floor(((double)(bn)/dmax)*1000000)/1000000;
            int bin, end1=nodeD[idx1].end, end2=nodeR[idx2].end;
            double v;

            for (int i=nodeD[idx1].start; i<end1; ++i){
                x1 = elementsD[i].x;
                y1 = elementsD[i].y;
                z1 = elementsD[i].z;
                for (int j=nodeR[idx2].start; j<end2; ++j){
                    x2 = elementsR[j].x;
                    y2 = elementsR[j].y;
                    z2 = elementsR[j].z;
                    d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                    if (d<dd_max){
                        bin = (int)(sqrt(d)*ds);

                        if (bin>(bn-1)) continue;
                        bin += bn_offset*bn;

                        v = get_weight(pipsD, i, pipsD, j, pipis_width);
                        atomicAdd(&XY[bin],v);
                    }
                }
            }
        }
    }
}
