#include <string.h>
#include <stdio.h>
#include <math.h>

#include "cucheck_macros.cuh"
#include "create_grid.cuh"
#include "pcf2isoBPC.cuh"

/*
Kernel function to calculate the pure histograms for the 2 point isotropic correlation function WITH 
boundary periodic conditions. It stores the counts in the XX histogram.

args:
XX: (double*) The histogram where the distances are counted.
elements: (PointW3D*) Array of the points ordered coherently with the nodes.
nodeD: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node.
nonzero_nodes: (int) Number of nonzero nodes where the points have been classificated.
bins: (int) NUmber of bins in the XY histogram.
dmax: (float) The maximum distance of interest between points.
d_max_node: (float) The maximum internodal distance.
size_box: (float) The size of the box where the points were contained. It is used for the boundary periodic conditions
size_node: (float) Size of the nodes.
*/
__global__ void XX2iso_BPC(
    double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bins, 
    float dmax, float d_max_node, float size_box, float size_node
);

/*
Kernel function to calculate the mixed histograms for the 2 point isotropic correlation function WITH 
boundary periodic conditions. It stores the counts in the XY histogram.

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
size_box: (float) The size of the box where the points were contained. It is used for the boundary periodic conditions
size_node: (float) Size of the nodes.
*/
__global__ void XY2iso_BPC(
    double *XY, PointW3D *elementsD, DNode *nodeD, int nonzero_Dnodes,
    PointW3D *elementsR,  DNode *nodeR, int nonzero_Rnodes, int bins, float dmax,
    float d_max_node, float size_box, float size_node
);

/*
Analytic computation of the RR==DR histogram
*/
__global__ void RR2iso_BPC_analytic(
    double *RR, double alpha, int bn
);

void pcf_2iso_BPC(
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
    int  blocks_D, blocks_R, threads_perblock_dim = 16;

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
    const int PREFIX_LENGTH = 11;

    nameDD = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    nameRR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    nameDR = (char*)malloc(PREFIX_LENGTH*sizeof(char));
    strcpy(nameDD,"DDiso_BPC_");
    strcpy(nameRR,"RRiso_BPC_");
    strcpy(nameDR,"DRiso_BPC_");

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/
    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node *= d_max_node;

    // Allocate memory for the histogram as double
    DD = (double*)malloc(bins*sizeof(double));
    RR = (double**)malloc(n_randfiles*sizeof(double*));
    DR = (double**)malloc(n_randfiles*sizeof(double*));
    CHECKALLOC(DD);
    CHECKALLOC(RR);
    CHECKALLOC(DR);
    for (int i = 0; i < n_randfiles; i++)
    {
        RR[i] = (double*)malloc(bins*sizeof(double));
        CHECKALLOC(RR[i]);
        DR[i] = (double*)malloc(bins*sizeof(double));
        CHECKALLOC(DR[i]);
    }

    d_RR = (double**)malloc(n_randfiles*sizeof(double*));
    d_DR = (double**)malloc(n_randfiles*sizeof(double*));
    CHECKALLOC(d_DR);
    CHECKALLOC(d_RR);

    CUCHECK(cudaMalloc(&d_DD, bins*sizeof(double)));
    CUCHECK(cudaMemsetAsync(d_DD, 0, bins*sizeof(double), streamDD));
    for (int i = 0; i < n_randfiles; i++)
    {
        CUCHECK(cudaMalloc(&d_RR[i], bins*sizeof(double)));
        CUCHECK(cudaMalloc(&d_DR[i], bins*sizeof(double)));
        //Restarts the main histograms in device to zero
        CUCHECK(cudaMemsetAsync(d_RR[i], 0, bins*sizeof(double), streamRR[i]));
        CUCHECK(cudaMemsetAsync(d_DR[i], 0, bins*sizeof(double), streamDR[i]));
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
    XX2iso_BPC<<<gridD,threads_perblock,0,streamDD>>>(d_DD, d_dataD, d_nodeD, nonzero_Dnodes, bins, dmax, d_max_node, size_box, size_node);
    CUCHECK(cudaMemcpyAsync(DD, d_DD, bins*sizeof(double), cudaMemcpyDeviceToHost, streamDD));
    for (int i=0; i<n_randfiles; i++)
    {
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridR.x = blocks_R;
        gridR.y = blocks_R;
        XX2iso_BPC<<<gridR,threads_perblock,0,streamRR[i]>>>(d_RR[i], d_dataR[i], d_nodeR[i], nonzero_Rnodes[i], bins, dmax, d_max_node, size_box, size_node);
        CUCHECK(cudaMemcpyAsync(RR[i], d_RR[i], bins*sizeof(double), cudaMemcpyDeviceToHost, streamRR[i]));
        gridDR.y = blocks_R;

        cudaStreamWaitEvent(streamDR[i], DDcopy_done, 0);
        cudaStreamWaitEvent(streamDR[i], RRcopy_done[i], 0);
        XY2iso_BPC<<<gridDR,threads_perblock,0,streamDR[i]>>>(d_DR[i], d_dataD, d_nodeD, nonzero_Dnodes, d_dataR[i], d_nodeR[i], nonzero_Rnodes[i], bins, dmax, d_max_node, size_box, size_node);
        CUCHECK(cudaMemcpyAsync(DR[i], d_DR[i], bins*sizeof(double), cudaMemcpyDeviceToHost, streamDR[i]));
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

void pcf_2iso_BPCanalytic(
    DNode *d_nodeD, PointW3D *d_dataD,
    int nonzero_Dnodes, cudaStream_t streamDD,
    int bins, int np, float size_node, float size_box, float dmax,
    char *data_name
)
{

    /* =======================================================================*/
    /* ======================  Var declaration ===============================*/
    /* =======================================================================*/

    float d_max_node, time_spent;
    double *DD, *RR, *d_DD, *d_RR;
    int  blocks_D, blocks_analytic, threads_perblock_analytic=512, threads_perblock_dim = 16;
    
    //For analytic RR
    double alpha1, beta1, dr; 
    dr = (dmax/bins);
    beta1 = (np*np)/(size_box*size_box*size_box);
    alpha1 = 8*dr*dr*dr*(acos(0.0))*(beta1)/3;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    CUCHECK(cudaEventCreate(&start_timmer));
    CUCHECK(cudaEventCreate(&stop_timmer));

    cudaStream_t streamRR;
    CUCHECK(cudaStreamCreate(&streamRR));

    //Prefix that will be used to save the histograms
    char *nameDD = NULL, *nameRR = NULL, *nameDR = NULL;
    const int PREFIX_LENGTH = 14;

    nameDD = (char*)malloc((PREFIX_LENGTH + strlen(data_name))*sizeof(char));
    nameRR = (char*)malloc((PREFIX_LENGTH + strlen(data_name))*sizeof(char));
    nameDR = (char*)malloc((PREFIX_LENGTH + strlen(data_name))*sizeof(char));
    strcpy(nameDD,"DDiso_an_BPC_");
    strcat(nameDD,data_name);
    strcpy(nameRR,"RRiso_an_BPC_");
    strcat(nameRR,data_name);
    strcpy(nameDR,"DRiso_an_BPC_");
    strcat(nameDR,data_name);

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;
    // Allocate memory for the histogram as double
    DD = (double*)malloc(bins*sizeof(double));
    RR = (double*)malloc(bins*sizeof(double));

    CUCHECK(cudaMalloc(&d_DD, bins*sizeof(double)));
    CUCHECK(cudaMalloc(&d_RR, bins*sizeof(double)));

    //Restarts the main histograms in device to zero
    CUCHECK(cudaMemsetAsync(d_DD, 0, bins*sizeof(double), streamDD));
    CUCHECK(cudaMemsetAsync(d_RR, 0, bins*sizeof(double), streamRR));

    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/
    
    //Compute the dimensions of the GPU grid
    //One thread for each node
    
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));
    dim3 threads_perblock(threads_perblock_dim,threads_perblock_dim,1);
    dim3 gridD(blocks_D,blocks_D,1);
    
    if (bins<=512) threads_perblock_analytic = bins;
    blocks_analytic = (int)(ceil((float)((float)(bins)/(float)(threads_perblock_analytic))));
    if (bins<=512) blocks_analytic = 1;

    //Launch the kernels
    time_spent=0; //Restarts timmer
    CUCHECK(cudaEventRecord(start_timmer));
    XX2iso_BPC<<<gridD,threads_perblock,0,streamDD>>>(d_DD, d_dataD, d_nodeD, nonzero_Dnodes, bins, dmax, d_max_node, size_box, size_node);
    CUCHECK(cudaMemcpyAsync(DD, d_DD, bins*sizeof(double), cudaMemcpyDeviceToHost, streamDD));
    RR2iso_BPC_analytic<<<blocks_analytic,threads_perblock_analytic,0,streamRR>>>(d_RR, alpha1, bins);
    CUCHECK(cudaMemcpyAsync(RR, d_RR, bins*sizeof(double), cudaMemcpyDeviceToHost, streamRR));
    
    //Waits for all the kernels to complete
    CUCHECK(cudaStreamSynchronize(streamDD));
    save_histogram1D(nameDD, bins, DD);
    CUCHECK(cudaStreamSynchronize(streamRR));
    save_histogram1D(nameRR, bins, RR);
    save_histogram1D(nameDR, bins, RR);

    CUCHECK(cudaEventRecord(stop_timmer));
    CUCHECK(cudaEventSynchronize(stop_timmer));
    CUCHECK(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    printf("Spent %f miliseconds to compute and save all the histograms. \n", time_spent);
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/
    
    //Free the memory
    CUCHECK(cudaStreamDestroy(streamDD));
    CUCHECK(cudaStreamDestroy(streamRR));

    CUCHECK(cudaEventDestroy(start_timmer));
    CUCHECK(cudaEventDestroy(stop_timmer));

    free(DD);
    free(RR);
    
    CUCHECK(cudaFree(d_DD));
    CUCHECK(cudaFree(d_RR));

}


__global__ void RR2iso_BPC_analytic(
    double *RR, double alpha, int bn
)
{
    /*
    This function calculates analytically the RR histogram. I only requires the alpha value calculated in host.

    args:
    RR (*double): HIstogram where the values will be stored.
    alpha (double): Parameter calculated by the host.
    bn: (int) NUmber of bins in the RR histogram.

    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx<bn){
        int dr = 3*idx*idx + 3*idx +1;
        RR[idx] = alpha*((double)(dr));
    }

}

__global__ void XX2iso_BPC(double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bins, float dmax, float d_max_node, float size_box, float size_node){

    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<nonzero_nodes && idx2<nonzero_nodes){
        
        double v;
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float dd_max=dmax*dmax;
        float dxn12=fabsf(nx2-nx1), dyn12=fabsf(ny2-ny1), dzn12=fabsf(nz2-nz1);
        float dd_nod12 = dxn12*dxn12 + dyn12*dyn12 + dzn12*dzn12;
        double ds = ((double)(bins))/dmax, d;
        
        float x1,y1,z1,x2,y2,z2;
        float dx,dy,dz;
        int bin, end1=nodeD[idx1].end, end2=nodeD[idx2].end;
        
        //Front vars
        float f_dmax = dmax+size_node;
        float _f_dmax = size_box - f_dmax;
        bool boundx = ((nx1<=f_dmax)&&(nx2>=_f_dmax))||((nx2<=f_dmax)&&(nx1>=_f_dmax));
        bool boundy = ((ny1<=f_dmax)&&(ny2>=_f_dmax))||((ny2<=f_dmax)&&(ny1>=_f_dmax));
        bool boundz = ((nz1<=f_dmax)&&(nz2>=_f_dmax))||((nz2<=f_dmax)&&(nz1>=_f_dmax));
        float f_dxn12, f_dyn12, f_dzn12;

        //Regular histogram calculation
        if (dd_nod12 <= d_max_node){

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
                        if (bin>(bins-1)) continue;
                        v = elements[i].w*elements[j].w;
                        atomicAdd(&XX[bin],v);
                    }
                }
            }
        }
        
        //Z front proyection
        if (boundz){
            f_dzn12 = size_box-dzn12;
            dd_nod12 = dxn12*dxn12+dyn12*dyn12+f_dzn12*f_dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dz = size_box-fabsf(z2-z1);
                        d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+dz*dz;
                        if (d<dd_max && d>0){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elements[i].w*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }

        //Y front proyection
        if (boundy){
            f_dyn12 = size_box-dyn12;
            dd_nod12 = dxn12*dxn12+f_dyn12*f_dyn12+dzn12*dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dy = size_box-fabsf(y2-y1);
                        d = (x2-x1)*(x2-x1)+dy*dy+(z2-z1)*(z2-z1);
                        if (d<dd_max && d>0){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elements[i].w*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }
        
        //X front proyection
        if (boundx){
            f_dxn12 = size_box-dxn12;
            dd_nod12 = f_dxn12*f_dxn12+dyn12*dyn12+dzn12*dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        d = dx*dx+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                        if (d<dd_max && d>0){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elements[i].w*elements[j].w;
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
            dd_nod12 = f_dxn12*f_dxn12+f_dyn12*f_dyn12+dzn12*dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        d = dx*dx+dy*dy+(z2-z1)*(z2-z1);
                        if (d<dd_max && d>0){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elements[i].w*elements[j].w;
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
            dd_nod12 = f_dxn12*f_dxn12+dyn12*dyn12+f_dzn12*f_dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dz = size_box-fabsf(z2-z1);
                        d = dx*dx+(y2-y1)*(y2-y1)+dz*dz;
                        if (d<dd_max && d>0){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elements[i].w*elements[j].w;
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
            dd_nod12 = dxn12*dxn12+f_dyn12*f_dyn12+f_dzn12*f_dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        d = (x2-x1)*(x2-x1)+dy*dy+dz*dz;
                        if (d<dd_max && d>0){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elements[i].w*elements[j].w;
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
            dd_nod12 = f_dxn12*f_dxn12+f_dyn12*f_dyn12+f_dzn12*f_dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        d = dx*dx+dy*dy+dz*dz;
                        if (d<dd_max && d>0){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elements[i].w*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }

    }

}

__global__ void XY2iso_BPC(double *XY, PointW3D *elementsD, DNode *nodeD, int nonzero_Dnodes, PointW3D *elementsR,  DNode *nodeR, int nonzero_Rnodes, int bins, float dmax, float d_max_node, float size_box, float size_node){

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<nonzero_Dnodes && idx2<nonzero_Rnodes){
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeR[idx2].nodepos.x, ny2=nodeR[idx2].nodepos.y, nz2=nodeR[idx2].nodepos.z;
        float dxn12=fabsf(nx2-nx1), dyn12=fabsf(ny2-ny1), dzn12=fabsf(nz2-nz1);
        float dd_nod12 = dxn12*dxn12 + dyn12*dyn12 + dzn12*dzn12;
        float dd_max=dmax*dmax;
        double ds = ((double)(bins))/dmax, d;
        
        float x1,y1,z1,x2,y2,z2;
        float dx,dy,dz;
        int bin, end1=nodeD[idx1].end, end2=nodeR[idx2].end;
        double v;
        
        //Front vars
        float f_dmax = dmax+size_node;
        float _f_dmax = size_box - f_dmax;
        bool boundx = ((nx1<=f_dmax)&&(nx2>=_f_dmax))||((nx2<=f_dmax)&&(nx1>=_f_dmax));
        bool boundy = ((ny1<=f_dmax)&&(ny2>=_f_dmax))||((ny2<=f_dmax)&&(ny1>=_f_dmax));
        bool boundz = ((nz1<=f_dmax)&&(nz2>=_f_dmax))||((nz2<=f_dmax)&&(nz1>=_f_dmax));
        float f_dxn12, f_dyn12, f_dzn12;

        //Regular no BPC counting
        if (dd_nod12 <= d_max_node){

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
                        if (bin>(bins-1)) continue;
                        v = elementsD[i].w*elementsR[j].w;
                        atomicAdd(&XY[bin],v);
                    }
                }
            }
        }

        //Z front proyection
        if (boundz){
            f_dzn12 = size_box-dzn12;
            dd_nod12 = dxn12*dxn12+dyn12*dyn12+f_dzn12*f_dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    for (int j=nodeR[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dz = size_box-fabsf(z2-z1);
                        d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+dz*dz;
                        if (d<dd_max){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elementsD[i].w*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }
        
        //Y front proyection
        if (boundy){
            f_dyn12 = size_box-dyn12;
            dd_nod12 = dxn12*dxn12+f_dyn12*f_dyn12+dzn12*dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    for (int j=nodeR[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dy = size_box-fabsf(y2-y1);
                        d = (x2-x1)*(x2-x1)+dy*dy+(z2-z1)*(z2-z1);
                        if (d<dd_max){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elementsD[i].w*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }
        
        //X front proyection
        if (boundx){
            f_dxn12 = size_box-dxn12;
            dd_nod12 = f_dxn12*f_dxn12+dyn12*dyn12+dzn12*dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    for (int j=nodeR[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        d = dx*dx+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                        if (d<dd_max){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elementsD[i].w*elementsR[j].w;
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
            dd_nod12 = f_dxn12*f_dxn12+f_dyn12*f_dyn12+dzn12*dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    for (int j=nodeR[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        d = dx*dx+dy*dy+(z2-z1)*(z2-z1);
                        if (d<dd_max){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elementsD[i].w*elementsR[j].w;
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
            dd_nod12 = f_dxn12*f_dxn12+dyn12*dyn12+f_dzn12*f_dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    for (int j=nodeR[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dz = size_box-fabsf(z2-z1);
                        d = dx*dx+(y2-y1)*(y2-y1)+dz*dz;
                        if (d<dd_max){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elementsD[i].w*elementsR[j].w;
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
            dd_nod12 = dxn12*dxn12+f_dyn12*f_dyn12+f_dzn12*f_dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    for (int j=nodeR[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        d = (x2-x1)*(x2-x1)+dy*dy+dz*dz;
                        if (d<dd_max){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elementsD[i].w*elementsR[j].w;
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
            dd_nod12 = f_dxn12*f_dxn12+f_dyn12*f_dyn12+f_dzn12*f_dzn12;
            if (dd_nod12 <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    for (int j=nodeR[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        d = dx*dx+dy*dy+dz*dz;
                        if (d<dd_max){
                            bin = (int)(sqrt(d)*ds);
                            if (bin>(bins-1)) continue;
                            v = elementsD[i].w*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }

    }
}
