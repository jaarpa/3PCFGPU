#include <string.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>

#include "cucheck_macros.cuh"
#include "create_grid.cuh"
#include "pcf2aniBPC.cuh"


/*
Kernel function to calculate the pure histograms for the 2 point anisotropic correlation function WITH 
boundary periodic conditions. It stores the counts in the XX histogram.

args:
XX: (double*) The histogram where the distances are counted.
elements: (PointW3D*) Array of the points ordered coherently with the nodes.
nodeD: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node.
nonzero_nodes: (int) Number of nonzero nodes where the points have been classificated.
bn: (int) NUmber of bins in the XY histogram.
dmax: (float) The maximum distance of interest between points.
d_max_node: (float) The maximum internodal distance.
size_box: (float) The size of the box where the points were contained. It is used for the boundary periodic conditions
size_node: (float) Size of the nodes.
*/
__global__ void XX2ani_BPC(
        double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, 
        float dmax, float d_max_node, float size_box, float size_node
    );
    
/*
Kernel function to calculate the mixed histograms for the 2 point anisotropic correlation function with 
boundary periodic conditions. It stores the counts in the XY histogram.

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
size_box: (float) The size of the box where the points were contained. It is used for the boundary periodic conditions
size_node: (float) Size of the nodes.
*/
__global__ void XY2ani_BPC(
    double *XY, PointW3D *elementsD, DNode *nodeD, int nonzero_Dnodes, 
    PointW3D *elementsR,  DNode *nodeR, int nonzero_Rnodes, int bn, 
    float dmax, float d_max_node, float size_box, float size_node
);

void pcf_2aniBPC(
    DNode *d_nodeD, PointW3D *d_dataD,
    int nonzero_Dnodes, cudaStream_t streamDD, cudaEvent_t DDcopy_done, 
    DNode **d_nodeR, PointW3D **d_dataR,
    int *nonzero_Rnodes, cudaStream_t *streamRR, cudaEvent_t *RRcopy_done,
    char **histo_names, int n_randfiles, int bins, float size_node, float dmax,
    float size_box
)
{
    /* =======================================================================*/
    /* ======================  Var declaration ===============================*/
    /* =======================================================================*/

    float d_max_node, time_spent;
    double *DD=NULL, **RR=NULL, **DR=NULL, *d_DD=NULL, **d_RR=NULL, **d_DR=NULL;
    int  blocks_D, blocks_R, threads_perblock_dim = 32;

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
    int PREFIX_LENGTH = 11;

    nameDD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    nameRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    nameDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    strcpy(nameDD,"DDani_BPC_");
    strcpy(nameRR,"RRani_BPC_");
    strcpy(nameDR,"DRani_BPC_");

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/
    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node *= d_max_node;

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
    
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));
    dim3 threads_perblock(threads_perblock_dim,threads_perblock_dim,1);
    dim3 gridD(blocks_D,blocks_D,1);
    
    //Dummy declaration
    dim3 gridR(2,2,1);
    dim3 gridDR(blocks_D,2,1);

    //Launch the kernels
    time_spent=0; //Restarts timmer
    CUCHECK(cudaEventRecord(start_timmer));
    XX2ani_BPC<<<gridD,threads_perblock,0,streamDD>>>(d_DD, d_dataD, d_nodeD, nonzero_Dnodes, bins, dmax, d_max_node, size_box, size_node);
    CUCHECK(cudaMemcpyAsync(DD, d_DD, bins*bins*sizeof(double), cudaMemcpyDeviceToHost, streamDD));
    /*
    for (int i=0; i<n_randfiles; i++)
    {
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridR.x = blocks_R;
        gridR.y = blocks_R;
        XX2ani_BPC<<<gridR,threads_perblock,0,streamRR[i]>>>(d_RR[i], d_dataR[i], d_nodeR[i], nonzero_Rnodes[i], bins, dmax, d_max_node, size_box, size_node);
        CUCHECK(cudaMemcpyAsync(RR[i], d_RR[i], bins*bins*sizeof(double), cudaMemcpyDeviceToHost, streamRR[i]));
        gridDR.y = blocks_R;

        cudaStreamWaitEvent(streamDR[i], DDcopy_done, 0);
        cudaStreamWaitEvent(streamDR[i], RRcopy_done[i], 0);
        XY2ani_BPC<<<gridDR,threads_perblock,0,streamDR[i]>>>(d_DR[i], d_dataD, d_nodeD, nonzero_Dnodes, d_dataR[i], d_nodeR[i], nonzero_Rnodes[i], bins, dmax, d_max_node, size_box, size_node);
        CUCHECK(cudaMemcpyAsync(DR[i], d_DR[i], bins*bins*sizeof(double), cudaMemcpyDeviceToHost, streamDR[i]));
    }
    */

    //Waits for all the kernels to complete
    CUCHECK( cudaPeekAtLastError() );
    CUCHECK(cudaDeviceSynchronize());

    nameDD = (char*)realloc(nameDD,PREFIX_LENGTH + strlen(histo_names[0]));
    strcpy(&nameDD[PREFIX_LENGTH-1],histo_names[0]);
    save_histogram2D(nameDD, bins, DD);

    for (int i=0; i<n_randfiles; i++)
    {
        nameRR = (char*)realloc(nameRR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameRR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram2D(nameRR, bins, RR[i]);

        nameDR = (char*)realloc(nameDR,PREFIX_LENGTH + strlen(histo_names[i+1]));
        strcpy(&nameDR[PREFIX_LENGTH-1],histo_names[i+1]);
        save_histogram2D(nameDR, bins, DR[i]);
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

__global__ void XX2ani_BPC(
    double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn,
    float dmax, float d_max_node, float size_box, float size_node
)
{

    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1<nonzero_nodes && idx2<nonzero_nodes){
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float dd_max=dmax*dmax;
        float dxn12=fabsf(nx2-nx1), dyn12=fabsf(ny2-ny1), dzn12=fabsf(nz2-nz1);
        float dd_nod12_ort = dxn12*dxn12 + dyn12*dyn12;
        float dd_nod12_z = dzn12*dzn12;
        
        float x1, y1, z1, w1, x2, y2, z2;
        float dx, dy, dz;
        int bnz, bnort, bin, end1=nodeD[idx1].end, end2=nodeD[idx2].end;
        double dd_z, dd_ort, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;
        
        //Front vars
        float f_dmax = dmax+size_node;
        float _f_dmax = size_box - f_dmax;
        int boundx = ((nx1<=f_dmax)&&(nx2>=_f_dmax))||((nx2<=f_dmax)&&(nx1>=_f_dmax));
        int boundy = ((ny1<=f_dmax)&&(ny2>=_f_dmax))||((ny2<=f_dmax)&&(ny1>=_f_dmax));
        int boundz = ((nz1<=f_dmax)&&(nz2>=_f_dmax))||((nz2<=f_dmax)&&(nz1>=_f_dmax));
        float f_dxn12, f_dyn12, f_dzn12;
              
        //Regular histogram calculation
        if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){
            
            for (int i=nodeD[idx1].start; i<end1; ++i){
                x1 = elements[i].x;
                y1 = elements[i].y;
                z1 = elements[i].z;
                w1 = elements[i].w;
                for (int j=nodeD[idx2].start; j<end2; ++j){
                    x2 = elements[j].x;
                    y2 = elements[j].y;
                    z2 = elements[j].z;
                    dd_ort=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);
                    dd_z=(z2-z1)*(z2-z1);
                    if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                        
                        bnz = (int)(sqrt(dd_z)*ds)*bn;
                        if (bnz>(bn*(bn-1))) continue;
                        bnort = (int)(sqrt(dd_ort)*ds);
                        if (bnort>(bn-1)) continue;
                        bin = bnz + bnort;
                        
                        v = w1*elements[j].w;
                        atomicAdd(&XX[bin],v);
                    }
                }
            }
        }
        //Z front proyection
        if (boundz)
        {
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = dxn12*dxn12+dyn12*dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node)
            {
                for (int i=nodeD[idx1].start; i<end1; ++i)
                {
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j)
                    {
                        x2 = elements[j].x;
        /*
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
    */
                    }
                }
            }
        }

    /*
        //Y front proyection
        if (boundy){
            f_dyn12 = size_box-dyn12;
            dd_nod12_ort = dxn12*dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dy = size_box-fabsf(y2-y1);
                        dd_ort=(x2-x1)*(x2-x1)+dy*dy;
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }
        
        //X front proyection
        if (boundx){
            f_dxn12 = size_box-dxn12;
            dd_nod12_ort = f_dxn12*f_dxn12+dyn12*dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dd_ort=dx*dx+(y2-y1)*(y2-y1);
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }
        
        //XY front proyection
        if (boundx && boundy){
            f_dxn12 = size_box-dxn12;
            f_dyn12 = size_box-dyn12;
            dd_nod12_ort = f_dxn12*f_dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        dd_ort=dx*dx+dy*dy;
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }

        //XZ front proyection
        if (boundx && boundz){
            f_dxn12 = size_box-dxn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = f_dxn12*f_dxn12+dyn12*dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=dx*dx+(y2-y1)*(y2-y1);
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }
        
        //YZ front proyection
        if (boundy && boundz){
            f_dyn12 = size_box-dyn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = dxn12*dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=(x2-x1)*(x2-x1)+dy*dy;
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }
        
        //XYZ front proyection
        if (boundx && boundy && boundz){
            f_dxn12 = size_box-dxn12;
            f_dyn12 = size_box-dyn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = f_dxn12*f_dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=dx*dx+dy*dy;
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }
    */
    }
}

__global__ void XY2ani_BPC(double *XY, PointW3D *elementsD, DNode *nodeD, int nonzero_Dnodes, PointW3D *elementsR,  DNode *nodeR, int nonzero_Rnodes, int bn, float dmax, float d_max_node, float size_box, float size_node)
{
    
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<nonzero_Dnodes && idx2<nonzero_Rnodes){
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeR[idx2].nodepos.x, ny2=nodeR[idx2].nodepos.y, nz2=nodeR[idx2].nodepos.z;
        float dxn12=fabsf(nx2-nx1), dyn12=fabsf(ny2-ny1), dzn12=fabsf(nz2-nz1);
        float dd_nod12_ort = dxn12*dxn12+dyn12*dyn12;
        float dd_nod12_z = dzn12*dzn12;
        float dd_max=dmax*dmax;
        
        float x1,y1,z1,w1,x2,y2,z2;
        float dx,dy,dz;
        int bnz, bnort, bin, end1=nodeD[idx1].end, end2=nodeR[idx2].end;
        double dd_z, dd_ort, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;
        
        //Front vars
        float f_dmax = dmax+size_node;
        float _f_dmax = size_box - f_dmax;
        bool boundx = ((nx1<=f_dmax)&&(nx2>=_f_dmax))||((nx2<=f_dmax)&&(nx1>=_f_dmax));
        bool boundy = ((ny1<=f_dmax)&&(ny2>=_f_dmax))||((ny2<=f_dmax)&&(ny1>=_f_dmax));
        bool boundz = ((nz1<=f_dmax)&&(nz2>=_f_dmax))||((nz2<=f_dmax)&&(nz1>=_f_dmax));
        float f_dxn12, f_dyn12, f_dzn12;

        //Regular no BPC counting
        if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

            for (int i=nodeD[idx1].start; i<end1; ++i){
                x1 = elementsD[i].x;
                y1 = elementsD[i].y;
                z1 = elementsD[i].z;
                for (int j=nodeR[idx2].start; j<end2; ++j){
                    x2 = elementsR[j].x;
                    y2 = elementsR[j].y;
                    z2 = elementsR[j].z;
                    dd_ort=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);
                    dd_z=(z2-z1)*(z2-z1);
                    if (dd_z<=dd_max && dd_ort<=dd_max){
                        
                        bnz = (int)(sqrt(dd_z)*ds)*bn;
                        if (bnz>(bn*(bn-1))) continue;
                        bnort = (int)(sqrt(dd_ort)*ds);
                        if (bnort>(bn-1)) continue;
                        bin = bnz + bnort;

                        v = w1*elementsR[j].w;
                        atomicAdd(&XY[bin],v);
                    }
                }
            }
        }

                
        //Z front proyection
        if (boundz){
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = dxn12*dxn12+dyn12*dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }

        //Y front proyection
        if (boundy){
            f_dyn12 = size_box-dyn12;
            dd_nod12_ort = dxn12*dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dy = size_box-fabsf(y2-y1);
                        dd_ort=(x2-x1)*(x2-x1)+dy*dy;
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }
        
        //X front proyection
        if (boundx){
            f_dxn12 = size_box-dxn12;
            dd_nod12_ort = f_dxn12*f_dxn12+dyn12*dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dd_ort=dx*dx+(y2-y1)*(y2-y1);
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }
        
        //XY front proyection
        if (boundx && boundy){
            f_dxn12 = size_box-dxn12;
            f_dyn12 = size_box-dyn12;
            dd_nod12_ort = f_dxn12*f_dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        dd_ort=dx*dx+dy*dy;
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }

                
        //XZ front proyection
        if (boundx && boundz){
            f_dxn12 = size_box-dxn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = f_dxn12*f_dxn12+dyn12*dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=dx*dx+(y2-y1)*(y2-y1);
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }
        
        //YZ front proyection
        if (boundy && boundz){
            f_dyn12 = size_box-dyn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = dxn12*dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=(x2-x1)*(x2-x1)+dy*dy;
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }
        
        //XYZ front proyection
        if (boundx && boundy && boundz){
            f_dxn12 = size_box-dxn12;
            f_dyn12 = size_box-dyn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = f_dxn12*f_dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=dx*dx+dy*dy;
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrt(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }

    }
}