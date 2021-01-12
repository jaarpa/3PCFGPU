
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__global__ void XX2iso(double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, int node_offset=0, int bn_offset=0){
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

    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<nonzero_nodes && idx2<nonzero_nodes){
        
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
                        bin = (int)(sqrtf(d)*ds);
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

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1==0 && idx2==0){
        printf("Im in the thread 0,0, in the XY make \n");
        printf("Offset is %i \n", bn_offset)
    }
    idx2 += node_offset;

    if (idx1<nonzero_Dnodes && idx2<nonzero_Rnodes){
        
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
                        bin = (int)(sqrtf(d)*ds);

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
