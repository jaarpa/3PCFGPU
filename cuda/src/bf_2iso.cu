#include <string.h>
#include <math.h>
#include <stdio.h>

#include "cucheck_macros.cuh"
#include "device_functions.cuh"
#include "create_grid.cuh"
#include "bf_2iso.cuh"

/*
Kernel function to calculate the pure histograms for the 2 point isotropic correlation function. 
This version does NOT considers boudary periodic conditions. It stores the counts in the XX histogram.

args:
XX: (double*) The histogram where the distances are counted.
elements: (PointW3D*) Array of the points ordered coherently with the nodes.
nodeD: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node.
nonzero_nodes: (int) Number of nonzero nodes where the points have been classificated.
bins: (int) NUmber of bins in the XY histogram.
dmax: (float) The maximum distance of interest between points.
d_max_node: (float) The maximum internodal distance.
*/
__global__ void XX2iso(
    double *XX, PointW3D *elements, int np,
    double ds, int bins, float ddmax
);

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
bins: (int) NUmber of bins in the XY histogram.
dmax: (float) The maximum distance of interest between points.
d_max_node: (float) The maximum internodal distance.
*/
__global__ void XY2iso(
    double *XY,
    PointW3D *elementsD, int np,
    PointW3D *elementsR, int rnp,
    double ds, int bins, float ddmax
);

__global__ void XX2iso_wpips(
    double *XX,
    PointW3D *elements, int32_t *pipsD, int pips_width, int np,
    double ds, int bins, float ddmax
);

void pcf_bf_2iso(
    PointW3D *d_dataD, int32_t *d_pipsD, cudaStream_t streamDD, cudaEvent_t DDcopy_done, int np,
    PointW3D **d_dataR, cudaStream_t *streamRR, cudaEvent_t *RRcopy_done, int *rnp,
    char **histo_names, int n_randfiles, int bins, float dmax,
    int pips_width
)
{

    /* =======================================================================*/
    /* ======================  Var declaration ===============================*/
    /* =======================================================================*/

    float time_spent;
    float ddmax = dmax*dmax;
    double ds = (double)(bins)/dmax; // floor(((double)(bins)/dmax)*1000000)/1000000;
    double *DD=NULL, **RR=NULL, **DR=NULL, *d_DD=NULL, **d_RR=NULL, **d_DR=NULL;
    int threads_perblock_dim = 32;
    int  blocks_Dx, blocks_Rx;

    // GPU timmer
    cudaEvent_t start_timmer, stop_timmer;
    CUCHECK(cudaEventCreate(&start_timmer));
    CUCHECK(cudaEventCreate(&stop_timmer));

    //This may come from parameters
    cudaStream_t *streamDR;
    streamDR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamDR);
    for (int i = 0; i < n_randfiles; i++)
        CUCHECK(cudaStreamCreate(&streamDR[i]));

    //Prefix that will be used to save the histograms
    char *nameDD = NULL, *nameRR = NULL, *nameDR = NULL;
    int PREFIX_LENGTH;
    if (d_pipsD == NULL)
    {
        PREFIX_LENGTH = 10;
        nameDD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        nameRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        nameDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        strcpy(nameDD,"DDiso_BF_");
        strcpy(nameRR,"RRiso_BF_");
        strcpy(nameDR,"DRiso_BF_");
    }
    else
    {
        PREFIX_LENGTH = 15;
        nameDD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        nameRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        nameDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        strcpy(nameDD,"DDiso_BF_pips_");
        strcpy(nameRR,"RRiso_BF_pips_");
        strcpy(nameDR,"DRiso_BF_pips_");
    }

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    // Allocate memory for the histogram as double
    DD = (double*)malloc(bins*bins*sizeof(double));
    RR = (double**)malloc(n_randfiles*sizeof(double*));
    DR = (double**)malloc(n_randfiles*sizeof(double*));
    CHECKALLOC(DD);
    CHECKALLOC(RR);
    CHECKALLOC(DR);
    for (int i = 0; i < n_randfiles; i++)
    {
        RR[i] = (double*)malloc(bins*bins*sizeof(double));
        CHECKALLOC(RR[i]);
        DR[i] = (double*)malloc(bins*bins*sizeof(double));
        CHECKALLOC(DR[i]);
    }

    d_RR = (double**)malloc(n_randfiles*sizeof(double*));
    d_DR = (double**)malloc(n_randfiles*sizeof(double*));
    CHECKALLOC(d_DR);
    CHECKALLOC(d_RR);

    CUCHECK(cudaMalloc(&d_DD, bins*bins*sizeof(double)));
    CUCHECK(cudaMemsetAsync(d_DD, 0, bins*bins*sizeof(double), streamDD));
    for (int i = 0; i < n_randfiles; i++)
    {
        CUCHECK(cudaMalloc(&d_RR[i], bins*bins*sizeof(double)));
        CUCHECK(cudaMalloc(&d_DR[i], bins*bins*sizeof(double)));
        //Restarts the main histograms in device to zero
        CUCHECK(cudaMemsetAsync(d_RR[i], 0, bins*bins*sizeof(double), streamRR[i]));
        CUCHECK(cudaMemsetAsync(d_DR[i], 0, bins*bins*sizeof(double), streamDR[i]));
    }

    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/

    //Compute the dimensions of the GPU grid
    //One thread for each node
    
    blocks_Dx = (int)(ceil((float)((float)(np)/(float)(threads_perblock_dim))));
    dim3 threads_perblock(threads_perblock_dim,threads_perblock_dim,1);
    dim3 gridD(blocks_Dx,blocks_Dx,1);
    
    //Dummy declaration
    dim3 gridR(2,2,1);
    dim3 gridDR(blocks_Dx,2,1);

    //Launch the kernels
    time_spent = 0; //Restarts timmer
    CUCHECK(cudaEventRecord(start_timmer));
    if (d_pipsD == NULL)
        XX2iso<<<gridD,threads_perblock,0,streamDD>>>(d_DD, d_dataD, np, ds, bins, ddmax);
    else
        XX2iso_wpips<<<gridD,threads_perblock,0,streamDD>>>(d_DD, d_dataD, d_pipsD, pips_width, np, ds, bins, ddmax);
    CUCHECK(cudaMemcpyAsync(DD, d_DD, bins*bins*sizeof(double), cudaMemcpyDeviceToHost, streamDD));
    for (int i=0; i<n_randfiles; i++)
    {
        //Calculates grid dim for each file
        blocks_Rx = (int)(ceil((float)((float)(rnp[i])/(float)(threads_perblock_dim))));
        gridR.x = blocks_Rx;
        gridR.y = blocks_Rx;
        XX2iso<<<gridR,threads_perblock,0,streamRR[i]>>>(d_RR[i], d_dataR[i], rnp[i], ds, bins, ddmax);
        CUCHECK(cudaMemcpyAsync(RR[i], d_RR[i], bins*bins*sizeof(double), cudaMemcpyDeviceToHost, streamRR[i]));
        
        gridDR.y = blocks_Rx;
        cudaStreamWaitEvent(streamDR[i], DDcopy_done, 0);
        cudaStreamWaitEvent(streamDR[i], RRcopy_done[i], 0);
        XY2iso<<<gridDR,threads_perblock,0,streamDR[i]>>>(d_DR[i], d_dataD, np, d_dataR[i], rnp[i], ds, bins, ddmax);
        CUCHECK(cudaMemcpyAsync(DR[i], d_DR[i], bins*bins*sizeof(double), cudaMemcpyDeviceToHost, streamDR[i]));
    }


    //Waits for all the kernels to complete
    CUCHECK(cudaDeviceSynchronize());

    nameDD = (char*)realloc(nameDD,PREFIX_LENGTH + strlen(histo_names[0]));
    strcpy(&nameDD[PREFIX_LENGTH-1],histo_names[0]);
    save_histogram1D(nameDD, bins, DD);

    for (int i=0; i<n_randfiles; i++)
    {
        nameRR = (char*)realloc(nameRR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameRR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram1D(nameRR, bins, RR[i]);

        nameDR = (char*)realloc(nameDR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameDR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram1D(nameDR, bins, DR[i]);
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
    for (int i = 0; i < n_randfiles; i++)
    {
        CUCHECK(cudaStreamDestroy(streamDR[i]));
        CUCHECK(cudaStreamDestroy(streamRR[i]));
        
        free(RR[i]);
        RR[i] = NULL;
        free(DR[i]);
        DR[i] = NULL;
        CUCHECK(cudaFree(d_RR[i]));
        d_RR[i] = NULL;
        CUCHECK(cudaFree(d_DR[i]));
        d_DR[i] = NULL;
    }
    free(streamDR);
    streamDR = NULL;
    free(streamRR);
    streamRR = NULL;
    free(RR);
    RR = NULL;
    free(DR);
    DR = NULL;
    free(d_RR);
    d_RR = NULL;
    free(d_DR);
    d_DR = NULL;

    CUCHECK(cudaEventDestroy(start_timmer));
    CUCHECK(cudaEventDestroy(stop_timmer));

    free(DD);
    DD = NULL;
    CUCHECK(cudaFree(d_DD));
    d_DD = NULL;
}

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================
__global__ void XX2iso(
    double *XX, PointW3D *elements, int np, 
    double ds, int bins, float ddmax
)
{

    //Distributes all the indexes equitatively into the n_kernelc_calls.

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<np && idx2<np && idx1 < idx2)
    {
        int bin;
        float v;
        float dx = elements[idx1].x - elements[idx2].x;
        float dy = elements[idx1].y - elements[idx2].y;
        float dz = elements[idx1].z - elements[idx2].z;
        float dd = dx*dx + dy*dy + dz*dz;

        if (dd <= ddmax && dd > 0){

            bin = (int)(sqrt(dd)*ds);
            if (bin<bins)
            {
                v = elements[idx1].w*elements[idx2].w;
                atomicAdd(&XX[bin],v);
            }
        }
    }
}

__global__ void XY2iso(
    double *XY, 
    PointW3D *elementsD, int np,
    PointW3D *elementsR, int rnp,
    double ds, int bins, float ddmax
)
{
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1<np && idx2<rnp)
    {
        int bin;
        float v;
        float dx = elementsD[idx1].x - elementsR[idx2].x;
        float dy = elementsD[idx1].y - elementsR[idx2].y;
        float dz = elementsD[idx1].z - elementsR[idx2].z;
        float dd = dx*dx + dy*dy + dz*dz;

        if (dd <= ddmax)
        {
            bin = (int)(sqrt(dd)*ds);
            if (bin<bins)
            {
                v = elementsD[idx1].w*elementsR[idx2].w;
                atomicAdd(&XY[bin],v);
            }
        }
    }
}

__global__ void XX2iso_wpips(
    double *XX, PointW3D *elements, int32_t *pipsD, int pips_width, int np,
    double ds, int bins, float ddmax
)
{

    //Distributes all the indexes equitatively into the n_kernelc_calls.

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<np && idx2<np && idx1 < idx2)
    {
        int bin;
        float v;
        float dx = elements[idx1].x - elements[idx2].x;
        float dy = elements[idx1].y - elements[idx2].y;
        float dz = elements[idx1].z - elements[idx2].z;
        float dd = dx*dx + dy*dy + dz*dz;

        if (dd <= ddmax && dd > 0)
        {
            bin = (int)(sqrt(dd)*ds);
            if (bin<bins)
            {
                v = get_weight(pipsD, idx1, pipsD, idx2, pips_width);
                atomicAdd(&XX[bin],v);
            }
        }
    }
}
