#include <string.h>
#include <stdio.h>
#include <math.h>

#include "cucheck_macros.cuh"
#include "pcf3iso.cuh"
#include "device_functions.cuh"

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
__global__ void XXX3iso(
    double *XXX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn,
    float dmax, float d_max_node
);

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
__global__ void XXY3iso(
    double *XXY, PointW3D *elementsX, DNode *nodeX, int nonzero_Xnodes,
    PointW3D *elementsY, DNode *nodeY, int nonzero_Ynodes, int bn, float dmax,
    float d_max_node
);

__global__ void XXX3iso_wpips(
    double *XXX, PointW3D *elements, int32_t *pipsD, DNode *nodeD,
    int nonzero_nodes, int bn, float dmax, float d_max_node, int pips_width
);

void pcf_3iso(
    DNode *d_nodeD, PointW3D *d_dataD, int32_t *d_pipsD,
    int nonzero_Dnodes, cudaStream_t streamDD, cudaEvent_t DDcopy_done, 
    DNode **d_nodeR, PointW3D **d_dataR,
    int *nonzero_Rnodes, cudaStream_t *streamRR, cudaEvent_t *RRcopy_done,
    char **histo_names, int n_randfiles, int bins, float size_node, float dmax,
    int pips_width
)
{

    /* =======================================================================*/
    /* =====================   Var declaration ===============================*/
    /* =======================================================================*/

    float d_max_node, time_spent;

    double *DDD=NULL, **RRR=NULL, **RRD=NULL, **DDR=NULL;
    double *d_DDD=NULL, **d_RRR=NULL, **d_RRD=NULL, **d_DDR=NULL;

    //n_kernel_calls should depend of the number of points, its density, and the number of bins
    int threads_perblock_dim = 8;
    int blocks_D, blocks_R;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    CUCHECK(cudaEventCreate(&start_timmer));
    CUCHECK(cudaEventCreate(&stop_timmer));

    cudaStream_t *streamDR, *streamRD;
    streamDR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamDR);
    streamRD = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamRD);
    for (int i = 0; i < n_randfiles; i++){
        CUCHECK(cudaStreamCreate(&streamDR[i]));
        CUCHECK(cudaStreamCreate(&streamRD[i]));
    }

    // Name of the files where the results are saved
    char *nameDDD = NULL, *nameRRR = NULL, *nameDDR = NULL, *nameRRD = NULL;
    int PREFIX_LENGTH;
    if (d_pipsD == NULL)
    {
        PREFIX_LENGTH = 8;
        nameDDD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        nameRRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        nameDDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        nameRRD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        strcpy(nameDDD,"DDDiso_");
        strcpy(nameRRR,"RRRiso_");
        strcpy(nameDDR,"DDRiso_");
        strcpy(nameRRD,"RRDiso_");
    }
    else
    {
        PREFIX_LENGTH = 13;
        nameDDD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        nameRRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        nameDDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        nameRRD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
        strcpy(nameDDD,"DDDiso_pips_");
        strcpy(nameRRR,"RRRiso_pips_");
        strcpy(nameDDR,"DDRiso_pips_");
        strcpy(nameRRD,"RRDiso_pips_");
    }

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    // Allocate memory for the histogram as double
    DDD = (double*)calloc(bins*bins*bins, sizeof(double));
    RRR = (double**)malloc(n_randfiles*sizeof(double*));
    DDR = (double**)malloc(n_randfiles*sizeof(double*));
    RRD = (double**)malloc(n_randfiles*sizeof(double*));
    CHECKALLOC(DDD);
    CHECKALLOC(RRR);
    CHECKALLOC(DDR);
    CHECKALLOC(RRD);
    for (int i = 0; i < n_randfiles; i++)
    {
        RRR[i] = (double*)malloc(bins*bins*bins*sizeof(double));
        CHECKALLOC(RRR[i]);
        DDR[i] = (double*)malloc(bins*bins*bins*sizeof(double));
        CHECKALLOC(DDR[i]);
        RRD[i] = (double*)malloc(bins*bins*bins*sizeof(double));
        CHECKALLOC(RRD[i]);
    }

    d_RRR = (double**)malloc(n_randfiles*sizeof(double*));
    d_DDR = (double**)malloc(n_randfiles*sizeof(double*));
    d_RRD = (double**)malloc(n_randfiles*sizeof(double*));
    CHECKALLOC(d_RRR);
    CHECKALLOC(d_DDR);
    CHECKALLOC(d_RRD);

    CUCHECK(cudaMalloc(&d_DDD, bins*bins*bins*sizeof(double)));
    CUCHECK(cudaMemsetAsync(d_DDD, 0, bins*bins*bins*sizeof(double), streamDD));
    for (int i = 0; i < n_randfiles; i++)
    {
        CUCHECK(cudaMalloc(&d_RRR[i], bins*bins*bins*sizeof(double)));
        CUCHECK(cudaMalloc(&d_DDR[i], bins*bins*bins*sizeof(double)));
        CUCHECK(cudaMalloc(&d_RRD[i], bins*bins*bins*sizeof(double)));
        //Restarts the main histograms in device to zero
        CUCHECK(cudaMemsetAsync(d_RRR[i], 0, bins*bins*bins*sizeof(double), streamRR[i]));
        CUCHECK(cudaMemsetAsync(d_DDR[i], 0, bins*bins*bins*sizeof(double), streamDR[i]));
        CUCHECK(cudaMemsetAsync(d_RRD[i], 0, bins*bins*bins*sizeof(double), streamRD[i]));
    }

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
    dim3 gridRRD(2,2,blocks_D);

    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    if (d_pipsD == NULL)
        XXX3iso<<<gridDDD,threads_perblock,0,streamDD>>>(
            d_DDD, d_dataD, d_nodeD, nonzero_Dnodes, bins, dmax, d_max_node
        );
    else
        XXX3iso_wpips<<<gridDDD,threads_perblock,0,streamDD>>>(
            d_DDD, d_dataD, d_pipsD, d_nodeD, nonzero_Dnodes, bins, dmax, d_max_node, pips_width
        );
    CUCHECK(cudaMemcpyAsync(DDD, d_DDD, bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost, streamDD));
    for (int i=0; i<n_randfiles; i++){
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridRRR.x = blocks_R;
        gridRRR.y = blocks_R;
        gridRRR.z = blocks_R;
        XXX3iso<<<gridRRR,threads_perblock,0,streamRR[i]>>>(d_RRR[i], d_dataR[i], d_nodeR[i], nonzero_Rnodes[i], bins, dmax, d_max_node);
        CUCHECK(cudaMemcpyAsync(RRR[i], d_RRR[i], bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost, streamRR[i]));
        gridDDR.z = blocks_R;
        cudaStreamWaitEvent(streamDR[i], DDcopy_done, 0);
        cudaStreamWaitEvent(streamDR[i], RRcopy_done[i], 0);
        XXY3iso<<<gridDDR,threads_perblock,0,streamDR[i]>>>(d_DDR[i], d_dataD, d_nodeD, nonzero_Dnodes, d_dataR[i], d_nodeR[i], nonzero_Rnodes[i], bins, dmax, d_max_node);
        CUCHECK(cudaMemcpyAsync(DDR[i], d_DDR[i], bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost, streamDR[i]));
        gridRRD.x = blocks_R;
        gridRRD.y = blocks_R;
        cudaStreamWaitEvent(streamRD[i], DDcopy_done, 0);
        cudaStreamWaitEvent(streamRD[i], RRcopy_done[i], 0);
        XXY3iso<<<gridRRD,threads_perblock,0,streamRD[i]>>>(d_RRD[i], d_dataR[i], d_nodeR[i], nonzero_Rnodes[i], d_dataD, d_nodeD, nonzero_Dnodes, bins, dmax, d_max_node);
        CUCHECK(cudaMemcpyAsync(RRD[i], d_RRD[i], bins*bins*bins*sizeof(double), cudaMemcpyDeviceToHost, streamRD[i]));
    }

    //Waits for all the kernels to complete
    CUCHECK(cudaDeviceSynchronize());

    //Save the results
    nameDDD = (char*)realloc(nameDDD,PREFIX_LENGTH + strlen(histo_names[0]));
    strcpy(&nameDDD[PREFIX_LENGTH-1],histo_names[0]);
    save_histogram3D(nameDDD, bins, DDD);

    for (int i=0; i<n_randfiles; i++){
        nameRRR = (char*)realloc(nameRRR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameRRR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram3D(nameRRR, bins, RRR[i]);

        nameDDR = (char*)realloc(nameDDR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameDDR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram3D(nameDDR, bins, DDR[i]);

        nameRRD = (char*)realloc(nameRRD,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameRRD[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram3D(nameRRD, bins, RRD[i]);
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
        CUCHECK(cudaStreamDestroy(streamRD[i]));
        CUCHECK(cudaStreamDestroy(streamRR[i]));
        free(RRR[i]);
        RRR[i] = NULL;
        free(DDR[i]);
        DDR[i] = NULL;
        free(RRD[i]);
        RRD[i] = NULL;
        CUCHECK(cudaFree(d_RRR[i]));
        d_RRR[i] = NULL;
        CUCHECK(cudaFree(d_DDR[i]));
        d_DDR[i] = NULL;
        CUCHECK(cudaFree(d_RRD[i]));
        d_RRD[i] = NULL;
    }
    free(streamDR);
    streamDR = NULL;
    free(streamRD);
    streamRD = NULL;
    free(streamRR);
    streamRR = NULL;
    
    CUCHECK(cudaEventDestroy(start_timmer));
    CUCHECK(cudaEventDestroy(stop_timmer));

    CUCHECK(cudaFree(d_DDD));
    free(DDD);
    free(RRR);
    free(RRD);
    free(DDR);
    free(d_RRR);
    free(d_RRD);
    free(d_DDR);
}

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__global__ void XXX3iso(
    double *XXX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn,
    float dmax, float d_max_node
)
{
    
    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<nonzero_nodes && idx2<nonzero_nodes && idx3<nonzero_nodes)
    {
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1)+(ny2-ny1)*(ny2-ny1)+(nz2-nz1)*(nz2-nz1);

        if (dd_nod12 <= d_max_node)
        {
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

                    for (int i=nodeD[idx1].start; i<end1; i++)
                    {
                        x1 = elements[i].x;
                        y1 = elements[i].y;
                        z1 = elements[i].z;
                        w1 = elements[i].w;
                        for (int j=nodeD[idx2].start; j<end2; j++)
                        {
                            x2 = elements[j].x;
                            y2 = elements[j].y;
                            z2 = elements[j].z;
                            w2 = elements[j].w;
                            v = w1*w2;
                            d12 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
                            if (d12 < dd_max && d12>0)
                            {
                                d12 = sqrt(d12);
                                for (int k=nodeD[idx3].start; k<end3; k++)
                                {
                                    x3 = elements[k].x;
                                    y3 = elements[k].y;
                                    z3 = elements[k].z;
                                    d23 = (x3-x2)*(x3-x2) + (y3-y2)*(y3-y2) + (z3-z2)*(z3-z2);
                                    if (d23 < dd_max && d23>0)
                                    {
                                        d31 = (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1) + (z3-z1)*(z3-z1);
                                        if (d31 < dd_max && d31>0)
                                        {
                                            d23 = sqrt(d23);
                                            d31 = sqrt(d31);

                                            bnx = (int)(d12*ds)*bn*bn;
                                            if (bnx>(bn*bn*(bn-1))) continue;
                                            bny = (int)(d23*ds)*bn;
                                            if (bny>(bn*(bn-1))) continue;
                                            bnz = (int)(d31*ds);
                                            if (bnz>(bn-1)) continue;
                                            bin = bnx + bny + bnz;
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

__global__ void XXY3iso(
    double *XXY, PointW3D *elementsX, DNode *nodeX, int nonzero_Xnodes,
    PointW3D *elementsY, DNode *nodeY, int nonzero_Ynodes, int bn, float dmax,
    float d_max_node
)
{

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<nonzero_Xnodes && idx2<nonzero_Xnodes && idx3<nonzero_Ynodes)
    {
        float nx1=nodeX[idx1].nodepos.x, ny1=nodeX[idx1].nodepos.y, nz1=nodeX[idx1].nodepos.z;
        float nx2=nodeX[idx2].nodepos.x, ny2=nodeX[idx2].nodepos.y, nz2=nodeX[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1)+(ny2-ny1)*(ny2-ny1)+(nz2-nz1)*(nz2-nz1);
        if (dd_nod12 <= d_max_node)
        {
            float nx3=nodeY[idx3].nodepos.x, ny3=nodeY[idx3].nodepos.y, nz3=nodeY[idx3].nodepos.z;
            float dd_nod23 = (nx3-nx2)*(nx3-nx2)+(ny3-ny2)*(ny3-ny2)+(nz3-nz2)*(nz3-nz2);
            if (dd_nod23 <= d_max_node)
            {
                float dd_nod31 = (nx1-nx3)*(nx1-nx3)+(ny1-ny3)*(ny1-ny3)+(nz1-nz3)*(nz1-nz3);
                if (dd_nod31 <= d_max_node)
                {
                    int end1 = nodeX[idx1].end;
                    int end2 = nodeX[idx2].end;
                    int end3 = nodeY[idx3].end;
                    int bnx, bny, bnz, bin;
                    float dd_max=dmax*dmax;
                    float x1,y1,z1,w1,x2,y2,z2,w2,x3,y3,z3;
                    double d12, d23, d31, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;
                    for (int i=nodeX[idx1].start; i<end1; i++)
                    {
                        x1 = elementsX[i].x;
                        y1 = elementsX[i].y;
                        z1 = elementsX[i].z;
                        w1 = elementsX[i].w;
                        for (int j=nodeX[idx2].start; j<end2; j++)
                        {
                            x2 = elementsX[j].x;
                            y2 = elementsX[j].y;
                            z2 = elementsX[j].z;
                            w2 = elementsX[j].w;
                            v = w1*w2;
                            d12 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
                            if (d12 < dd_max && d12>0)
                            {
                                d12 = sqrt(d12);
                                for (int k=nodeY[idx3].start; k<end3; k++)
                                {
                                    x3 = elementsY[k].x;
                                    y3 = elementsY[k].y;
                                    z3 = elementsY[k].z;
                                    d23 = (x3-x2)*(x3-x2) + (y3-y2)*(y3-y2) + (z3-z2)*(z3-z2);
                                    if (d23 < dd_max)
                                    {
                                        d31 = (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1) + (z3-z1)*(z3-z1);
                                        if (d31 < dd_max)
                                        {
                                            d23 = sqrt(d23);
                                            d31 = sqrt(d31);                                            
                                            
                                            bnx = (int)(d12*ds)*bn*bn;
                                            if (bnx>(bn*bn*(bn-1))) continue;
                                            bny = (int)(d23*ds)*bn;
                                            if (bny>(bn*(bn-1))) continue;
                                            bnz = (int)(d31*ds);
                                            if (bnz>(bn-1)) continue;
                                            bin = bnx + bny + bnz;

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

__global__ void XXX3iso_wpips(
    double *XXX, PointW3D *elements, int32_t *pipsD, DNode *nodeD, 
    int nonzero_nodes, int bn, float dmax, float d_max_node, int pips_width
)
{

    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<nonzero_nodes && idx2<nonzero_nodes && idx3<nonzero_nodes)
    {
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
                    float x1,y1,z1,x2,y2,z2,x3,y3,z3;
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
