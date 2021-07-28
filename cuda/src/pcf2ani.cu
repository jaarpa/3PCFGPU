#include <string.h>
#include <math.h>
#include <stdio.h>
#include "device_functions.cuh"
#include "create_grid.cuh"
#include "pcf2ani.cuh"

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


void pcf_2ani(char **histo_names, DNode *dnodeD, PointW3D *dataD, int nonzero_Dnodes, DNode *dnodeR, PointW3D *dataR, int *nonzero_Rnodes, int *acum_nonzero_Rnodes, int n_randfiles, int bn, float size_node, float dmax)
{

    /* =======================================================================*/
    /* ======================  Var declaration ===============================*/
    /* =======================================================================*/
    const int PREFIX_LENGTH = 7;

    float d_max_node, time_spent;
    double *DD, *RR, *DR, *d_DD, *d_RR, *d_DR;
    int  blocks_D, blocks_R, threads_perblock_dim = 32;

    // GPU timmer
    cudaEvent_t start_timmer, stop_timmer;
    CUCHECK(cudaEventCreate(&start_timmer));
    CUCHECK(cudaEventCreate(&stop_timmer));

    cudaStream_t streamDD, *streamRR, *streamDR;
    streamRR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamRR);
    streamDR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamDR);
    CUCHECK(cudaStreamCreate(&streamDD));
    for (int i = 0; i < n_randfiles; i++)
    {
        CUCHECK(cudaStreamCreate(&streamDR[i]));
        CUCHECK(cudaStreamCreate(&streamRR[i]));
    }

    char *nameDD = (char*)malloc(PREFIX_LENGTH*sizeof(char)), *nameRR = (char*)malloc(PREFIX_LENGTH*sizeof(char)), *nameDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    strcpy(nameDD,"DDani_");
    strcpy(nameRR,"RRani_");
    strcpy(nameDR,"DRani_");

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    // Allocate memory for the histogram as double
    DD = (double*)malloc(bn*bn*sizeof(double));
    CHECKALLOC(DD);
    RR = (double*)malloc(n_randfiles*bn*bn*sizeof(double));
    CHECKALLOC(RR);
    DR = (double*)malloc(n_randfiles*bn*bn*sizeof(double));
    CHECKALLOC(DR);

    CUCHECK(cudaMalloc(&d_DD, bn*bn*sizeof(double)));
    CUCHECK(cudaMalloc(&d_RR, n_randfiles*bn*bn*sizeof(double)));
    CUCHECK(cudaMalloc(&d_DR, n_randfiles*bn*bn*sizeof(double)));

    //Restarts the main histograms in device to zero
    CUCHECK(cudaMemsetAsync(d_DD, 0, bn*bn*sizeof(double), streamDD));
    CUCHECK(cudaMemsetAsync(d_RR, 0, n_randfiles*bn*bn*sizeof(double), streamRR[0]));
    CUCHECK(cudaMemsetAsync(d_DR, 0, n_randfiles*bn*bn*sizeof(double), streamDR[0]));
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
    XX2ani<<<gridD,threads_perblock,0,streamDD>>>(d_DD, dataD, dnodeD, nonzero_Dnodes, bn, dmax, d_max_node, 0, 0);
    for (int i=0; i<n_randfiles; i++)
    {
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridR.x = blocks_R;
        gridR.y = blocks_R;
        XX2ani<<<gridR,threads_perblock,0,streamRR[i]>>>(d_RR, dataR, dnodeR, nonzero_Rnodes[i], bn, dmax, d_max_node, acum_nonzero_Rnodes[i], i);
        gridDR.y = blocks_R;
        XY2ani<<<gridDR,threads_perblock,0,streamDR[i]>>>(d_DR, dataD, dnodeD, nonzero_Dnodes, dataR, dnodeR, nonzero_Rnodes[i], bn, dmax, d_max_node, acum_nonzero_Rnodes[i], i);
    }

    //Waits for all the kernels to complete
    CUCHECK(cudaDeviceSynchronize());

    CUCHECK(cudaMemcpy(DD, d_DD, bn*bn*sizeof(double), cudaMemcpyDeviceToHost));

    nameDD = (char*)realloc(nameDD,PREFIX_LENGTH + strlen(histo_names[0]));
    strcpy(&nameDD[PREFIX_LENGTH-1],histo_names[0]);
    save_histogram2D(nameDD, bn, DD, 0);

    CUCHECK(cudaMemcpy(RR, d_RR, n_randfiles*bn*bn*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++)
    {
        nameRR = (char*)realloc(nameRR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameRR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram2D(nameRR, bn, RR, i);
    }
    CUCHECK(cudaMemcpy(DR, d_DR, n_randfiles*bn*bn*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++)
    {
        nameDR = (char*)realloc(nameDR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameDR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram2D(nameDR, bn, DR, i);
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

void pcf_2ani_wpips(char **histo_names, DNode *dnodeD, PointW3D *dataD, int32_t *dpipsD, int nonzero_Dnodes, DNode *dnodeR, PointW3D *dataR, int32_t *dpipsR, int *nonzero_Rnodes, int *acum_nonzero_Rnodes, int n_pips, int n_randfiles, int bn, float size_node, float dmax)
{

    /* =======================================================================*/
    /* ======================  Var declaration ===============================*/
    /* =======================================================================*/
    const int PREFIX_LENGTH = 12;
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
    for (int i = 0; i < n_randfiles; i++)
    {
        CUCHECK(cudaStreamCreate(&streamDR[i]));
        CUCHECK(cudaStreamCreate(&streamRR[i]));
    }

    char *nameDD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    char *nameRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    char *nameDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    strcpy(nameDD,"DDani_pips_");
    strcpy(nameRR,"RRani_pips_");
    strcpy(nameDR,"DRani_pips_");

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    // Allocate memory for the histogram as double
    DD = (double*)malloc(bn*bn*sizeof(double));
    CHECKALLOC(DD);
    RR = (double*)malloc(n_randfiles*bn*bn*sizeof(double));
    CHECKALLOC(RR);
    DR = (double*)malloc(n_randfiles*bn*bn*sizeof(double));
    CHECKALLOC(DR);

    CUCHECK(cudaMalloc(&d_DD, bn*bn*sizeof(double)));
    CUCHECK(cudaMalloc(&d_RR, n_randfiles*bn*bn*sizeof(double)));
    CUCHECK(cudaMalloc(&d_DR, n_randfiles*bn*bn*sizeof(double)));

    //Restarts the main histograms in device to zero
    CUCHECK(cudaMemsetAsync(d_DD, 0, bn*bn*sizeof(double), streamDD));
    CUCHECK(cudaMemsetAsync(d_RR, 0, n_randfiles*bn*bn*sizeof(double), streamRR[0]));
    CUCHECK(cudaMemsetAsync(d_DR, 0, n_randfiles*bn*bn*sizeof(double), streamDR[0]));
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
    XX2ani_wpips<<<gridD,threads_perblock,0,streamDD>>>(d_DD, dataD, dnodeD, dpipsD, n_pips, nonzero_Dnodes, bn, dmax, d_max_node, 0, 0);
    for (int i=0; i<n_randfiles; i++)
    {
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridR.x = blocks_R;
        gridR.y = blocks_R;
        XX2ani_wpips<<<gridR,threads_perblock,0,streamRR[i]>>>(d_RR, dataR, dnodeR, dpipsR, n_pips, nonzero_Rnodes[i], bn, dmax, d_max_node, acum_nonzero_Rnodes[i], i);
        gridDR.y = blocks_R;
        XY2ani_wpips<<<gridDR,threads_perblock,0,streamDR[i]>>>(d_DR, dataD, dnodeD, dpipsD, n_pips, nonzero_Dnodes, dataR, dnodeR, dpipsR, nonzero_Rnodes[i], bn, dmax, d_max_node, acum_nonzero_Rnodes[i], i);
    }

    //Waits for all the kernels to complete
    CUCHECK(cudaDeviceSynchronize());

    CUCHECK(cudaMemcpy(DD, d_DD, bn*bn*sizeof(double), cudaMemcpyDeviceToHost));
    nameDD = (char*)realloc(nameDD,PREFIX_LENGTH + strlen(histo_names[0]));
    strcpy(&nameDD[PREFIX_LENGTH-1],histo_names[0]);
    save_histogram2D(nameDD, bn, DD, 0);

    CUCHECK(cudaMemcpy(RR, d_RR, n_randfiles*bn*bn*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++)
    {
        nameRR = (char*)realloc(nameRR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameRR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram2D(nameRR, bn, RR, i);
    }
    CUCHECK(cudaMemcpy(DR, d_DR, n_randfiles*bn*bn*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++)
    {
        nameDR = (char*)realloc(nameDR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameDR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram2D(nameDR, bn, DR, i);
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

__global__ void XX2ani(double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset)
{

    int idx1 = node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<(nonzero_nodes+node_offset) && idx2<(nonzero_nodes+node_offset))
    {
        
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float dd_nod12_ort = (nx2-nx1)*(nx2-nx1) + (ny2-ny1)*(ny2-ny1);
        float dd_nod12_z = (nz2-nz1)*(nz2-nz1);

        if (dd_nod12_z <= d_max_node && dd_nod12_ort <= d_max_node)
        {

            float x1,y1,z1,w1,x2,y2,z2;
            float dd_max=dmax*dmax;
            int bnz, bnort, bin, end1=nodeD[idx1].end, end2=nodeD[idx2].end;
            double dd_z, dd_ort, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;

            for (int i=nodeD[idx1].start; i<end1; ++i)
            {
                x1 = elements[i].x;
                y1 = elements[i].y;
                z1 = elements[i].z;
                w1 = elements[i].w;
                for (int j=nodeD[idx2].start; j<end2; ++j)
                {
                    x2 = elements[j].x;
                    y2 = elements[j].y;
                    z2 = elements[j].z;

                    dd_z = (z2-z1)*(z2-z1);
                    dd_ort = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);

                    if (dd_z < dd_max && dd_z > 0 && dd_ort < dd_max && dd_ort > 0)
                    {

                        bnz = (int)(sqrt(dd_z)*ds)*bn;
                        if (bnz>(bn*(bn-1))) continue;
                        bnort = (int)(sqrt(dd_ort)*ds);
                        if (bnort>(bn-1)) continue;
                        bin = bnz + bnort;
                        bin += bn_offset*bn*bn;
                        v = w1*elements[j].w;

                        atomicAdd(&XX[bin],v);
                    }
                }
            }
        }
    }
}

__global__ void XY2ani(double *XY, PointW3D *elementsD, DNode *nodeD, int nonzero_Dnodes, PointW3D *elementsR,  DNode *nodeR, int nonzero_Rnodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset)
{

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<nonzero_Dnodes && idx2<(nonzero_Rnodes+node_offset))
    {
        
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeR[idx2].nodepos.x, ny2=nodeR[idx2].nodepos.y, nz2=nodeR[idx2].nodepos.z;
        float dd_nod12_ort = (nx2-nx1)*(nx2-nx1) + (ny2-ny1)*(ny2-ny1);
        float dd_nod12_z = (nz2-nz1)*(nz2-nz1);

        if (dd_nod12_z <= d_max_node && dd_nod12_ort <= d_max_node)
        {

            float x1,y1,z1,w1,x2,y2,z2;
            float dd_max=dmax*dmax;
            int bnz, bnort, bin, end1=nodeD[idx1].end, end2=nodeR[idx2].end;
            double dd_z, dd_ort, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;

            for (int i=nodeD[idx1].start; i<end1; ++i)
            {
                x1 = elementsD[i].x;
                y1 = elementsD[i].y;
                z1 = elementsD[i].z;
                w1 = elementsD[i].w;
                for (int j=nodeR[idx2].start; j<end2; ++j)
                {
                    x2 = elementsR[j].x;
                    y2 = elementsR[j].y;
                    z2 = elementsR[j].z;

                    dd_z = (z2-z1)*(z2-z1);
                    dd_ort = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);

                    if (dd_z < dd_max && dd_z > 0 && dd_ort < dd_max && dd_ort > 0)
                    {
                        
                        bnz = (int)(sqrt(dd_z)*ds)*bn;
                        if (bnz>(bn*(bn-1))) continue;
                        bnort = (int)(sqrt(dd_ort)*ds);
                        if (bnort>(bn-1)) continue;
                        bin = bnz + bnort;
                        bin += bn_offset*bn*bn;

                        v = w1*elementsD[j].w;
                        atomicAdd(&XY[bin],v);
                    }
                }
            }
        }
    }
}


__global__ void XX2ani_wpips(double *XX, PointW3D *elements, DNode *nodeD, int32_t *pipsD, int n_pips, int nonzero_nodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset)
{

    int idx1 = node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<(nonzero_nodes+node_offset) && idx2<(nonzero_nodes+node_offset))
    {
        
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float dd_nod12_ort = (nx2-nx1)*(nx2-nx1) + (ny2-ny1)*(ny2-ny1);
        float dd_nod12_z = (nz2-nz1)*(nz2-nz1);

        if (dd_nod12_z <= d_max_node && dd_nod12_ort <= d_max_node)
        {

            float x1,y1,z1,x2,y2,z2;
            float dd_max=dmax*dmax;
            int bnz, bnort, bin, end1=nodeD[idx1].end, end2=nodeD[idx2].end;
            double dd_z, dd_ort, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;

            for (int i=nodeD[idx1].start; i<end1; ++i)
            {
                x1 = elements[i].x;
                y1 = elements[i].y;
                z1 = elements[i].z;

                for (int j=nodeD[idx2].start; j<end2; ++j)
                {
                    x2 = elements[j].x;
                    y2 = elements[j].y;
                    z2 = elements[j].z;

                    dd_z = (z2-z1)*(z2-z1);
                    dd_ort = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);

                    if (dd_z < dd_max && dd_z > 0 && dd_ort < dd_max && dd_ort > 0)
                    {

                        bnz = (int)(sqrt(dd_z)*ds)*bn;
                        if (bnz>(bn*(bn-1))) continue;
                        bnort = (int)(sqrt(dd_ort)*ds);
                        if (bnort>(bn-1)) continue;
                        bin = bnz + bnort;
                        bin += bn_offset*bn*bn;

                        v = get_weight(pipsD, i, pipsD, j, n_pips);

                        atomicAdd(&XX[bin],v);
                    }
                }
            }
        }
    }
}

__global__ void XY2ani_wpips(double *XY, PointW3D *elementsD, DNode *nodeD, int32_t *pipsD, int n_pips, int nonzero_Dnodes, PointW3D *elementsR,  DNode *nodeR, int32_t *pipsR, int nonzero_Rnodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset)
{

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<nonzero_Dnodes && idx2<(nonzero_Rnodes+node_offset))
    {
        
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeR[idx2].nodepos.x, ny2=nodeR[idx2].nodepos.y, nz2=nodeR[idx2].nodepos.z;
        float dd_nod12_ort = (nx2-nx1)*(nx2-nx1) + (ny2-ny1)*(ny2-ny1);
        float dd_nod12_z = (nz2-nz1)*(nz2-nz1);

        if (dd_nod12_z <= d_max_node && dd_nod12_ort <= d_max_node)
        {

            float x1,y1,z1,x2,y2,z2;
            float dd_max=dmax*dmax;
            int bnz, bnort, bin, end1=nodeD[idx1].end, end2=nodeR[idx2].end;
            double dd_z, dd_ort, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;

            for (int i=nodeD[idx1].start; i<end1; ++i)
            {
                x1 = elementsD[i].x;
                y1 = elementsD[i].y;
                z1 = elementsD[i].z;

                for (int j=nodeR[idx2].start; j<end2; ++j)
                {
                    x2 = elementsR[j].x;
                    y2 = elementsR[j].y;
                    z2 = elementsR[j].z;

                    dd_z = (z2-z1)*(z2-z1);
                    dd_ort = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);

                    if (dd_z < dd_max && dd_z > 0 && dd_ort < dd_max && dd_ort > 0)
                    {
                        
                        bnz = (int)(sqrt(dd_z)*ds)*bn;
                        if (bnz>(bn*(bn-1))) continue;
                        bnort = (int)(sqrt(dd_ort)*ds);
                        if (bnort>(bn-1)) continue;
                        bin = bnz + bnort;
                        bin += bn_offset*bn*bn;

                        v = get_weight(pipsD, i, pipsR, j, n_pips);

                        atomicAdd(&XY[bin],v);
                    }
                }
            }
        }
    }
}