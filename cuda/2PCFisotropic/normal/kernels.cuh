
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__device__ void count_distancesXY(double *XY, PointW3D *elements1, int start1, int end1, PointW3D *elements2, int start2, int end2, float ds, float dd_max, int sum){
    /*
    This device function counts the distances betweeen points between two different nodes from two different files. This function is used 
    to compute the XY histogram

    Args:
    XY: (double*) The histogram where the distances are counted in
    elements1: (PointW3D*)  Array of PointW3D points orderered coherently by the nodes
    start1: (int) index at which the nodeA starts to be defined by elements1. Inclusive.
    end1: (int) index at which the nodeA stops being defined by elements1. Non inclusive.
    elements2: (PointW3D*)  Array of PointW3D points orderered coherently by the nodes
    start2: (int) index at which the nodeB starts to be defined by elements2. Inclusive.
    end2: (int) index at which the nodeB stops being defined by elements1. Non inclusive.
    ds: (float) number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: (float) The maximum distance of interest.
    sum: (int) State if each distance is counted twice or once
    */

    int bin;
    double v;
    float d;
    float x1,y1,z1,w1,x2,y2,z2,w2;

    for (int i=start1; i<end1; ++i){
        x1 = elements1[i].x;
        y1 = elements1[i].y;
        z1 = elements1[i].z;
        w1 = elements1[i].w;
        for (int j=start2; j<end2; ++j){
            x2 = elements2[j].x;
            y2 = elements2[j].y;
            z2 = elements2[j].z;
            w2 = elements2[j].w;
            d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (d<=dd_max){
                bin = (int)(sqrtf(d)*ds);
                v = sum*w1*w2;
                atomicAdd(&XY[bin],v);
            }
        }
    }
}

__global__ void make_histoXX(double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int partitions, int bn, float dmax, float d_max_node, float size_node){
    /*
    Kernel function to calculate the pure histograms. It stores the counts in the XX histogram.

    args:
    XX: (double*) The histogram where the distances are counted.
    elements: (PointW3D*) Array of the points ordered coherently with the nodes.
    node: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node.
    partitions: (int) Number of partitions that are fitted by box side.
    bn: (int) NUmber of bins in the XY histogram.
    dmax: (dmax) The maximum distance of interest between points.
    size_node: (float) Size of the nodes
    */

    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.x * blockDim.x + threadIdx.y;
    if (idx1<nonzero_nodes && idx2<nonzero_nodes){
        
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1) + (ny2-ny1)*(ny2-ny1) + (nz2-nz1)*(nz2-nz1);

        if (dd_nod12 <= d_max_node){

            float x1,y1,z1,x2,y2,z2;
            float d, ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
            int bin, end1=nodeD[idx1].start+nodeD[idx1].len, end2=nodeD[idx2].start+nodeD[idx2].len;
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
                    if (d<=dd_max){
                        bin = (int)(sqrtf(d)*ds);
                        v = elements[i].w*elements[j].w;
                        atomicAdd(&XX[bin],v);
                    }
                }
            }
        }
    }
}

__global__ void make_histoXY(double *XY, PointW3D *elementsD, DNode *nodeD, PointW3D *elementsR,  DNode *nodeR, int partitions, int bn, float dmax, float size_node){
    /*
    Kernel function to calculate the mixed histogram. It stores the counts in the XY histogram.

    args:
    XY: (double*) The histogram where the distances are counted.
    elementsD: (PointW3D*) Array of the points ordered coherently with the nodes.
    nodeD: (DNode) Array of DNodes each of which define a node and the elements of elementD that correspond to that node.
    elementsR: (PointW3D*) Array of the points ordered coherently with the nodes.
    nodeR: (DNode) Array of RNodes each of which define a node and the elements of elementR that correspond to that node.
    partitions: (int) Number of partitions that are fitted by box side.
    bn: (int) NUmber of bins in the XY histogram.
    dmax: (dmax) The maximum distance of interest between points.
    size_node: (float) Size of the nodes
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon in this thread
        //int mom = (int) (idx/(partitions*partitions));
        //int col = (int) ((idx%(partitions*partitions))/partitions);
        //int row = idx%partitions;
        
        //idx = row + col*partitions + mom*partitions*partitions;

        if (nodeD[idx].len > 0){

            float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
            float nx1=nodeD[idx].nodepos.x, ny1=nodeD[idx].nodepos.y, nz1=nodeD[idx].nodepos.z;
            float d_max_node = dmax + size_node*sqrtf(3.0);
            d_max_node*=d_max_node;
            
            int idx2,u,v,w; //Position of the second node
            unsigned int dx_nod12, dy_nod12, dz_nod12, dd_nod12;

            //Second node mobil in XYZ
            for(u = 0; u < partitions; u++){
                dx_nod12 = nodeR[u].nodepos.x - nx1;
                for(v = 0; v < partitions; v++){
                    idx2 = u + v*partitions;
                    dy_nod12 = nodeR[idx2].nodepos.y - ny1;
                    for(w = 0; w < partitions; w++){
                        idx2 = u + v*partitions + w*partitions*partitions;
                        dz_nod12 = nodeR[idx2].nodepos.z - nz1;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=d_max_node){
                            count_distancesXY(XY, elementsD, nodeD[idx].start, nodeD[idx].start+nodeD[idx].len, elementsR, nodeR[idx2].start, nodeR[idx2].start + nodeR[idx2].len, ds, dd_max, 1);
                        }
                    }
                }
            }
            
        }
    }
}
