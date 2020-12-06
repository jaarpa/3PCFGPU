
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__device__ void count_distances11(double *XX, PointW3D *elements, int start, int end, int bns, float ds, float dd_max, int sum){
    /*
    This device function counts the distances betweeen points within the same node. This function is used 
    to compute the XX histogram

    Args:
    XX: (double*) The histogram where the distances are counted in
    elements: (PointW3D*)  Array of PointW3D points orderered coherently by the nodes
    start: (int) index at which the nodeA starts to be defined by elements1. Inclusive.
    end: (int) index at which the nodeA stops being defined by elements1. Non inclusive.
    bns: (int) number of bins per XX dimension
    ds: (float) number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: (float) The maximum distance of interest.
    sum: (int) State if each distance is counted twice or once
    */
    
    int bin;
    double v;
    float dd_z, dd_ort;
    float x1, y1, z1, w1;
    float x2, y2, z2;

    for (int i=start; i<end-1; ++i){
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        w1 = elements[i].w;
        for (int j=i+1; j<end; ++j){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;

            dd_z = (z2-z1)*(z2-z1);
            dd_ort = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);

            if (dd_z <= dd_max && dd_ort <= dd_max){
                bin = (int)(sqrt(dd_z)*ds)*bns + (int)(sqrtf(dd_ort)*ds);
                v = sum*w1*elements[j].w;
                atomicAdd(&XX[bin],v);
            }

        }
    }
}

__device__ void count_distances12(double *XX, PointW3D *elements, int start1, int end1, int start2, int end2, int bns, float ds, float dd_max, int sum){
    /*
    This device function counts the distances betweeen points between two different nodes from the same file. This function is used 
    to compute the XX histogram

    Args:
    XX: (double*) The histogram where the distances are counted in
    elements: (PointW3D*)  Array of PointW3D points orderered coherently by the nodes
    start1: (int) index at which the nodeA starts to be defined by elements1. Inclusive.
    end1: (int) index at which the nodeA stops being defined by elements1. Non inclusive.
    start2: (int) index at which the nodeB starts to be defined by elements1. Inclusive.
    end2: (int) index at which the nodeB stops being defined by elements1. Non inclusive.
    bns: (int) number of bins per XX dimension
    ds: (float) number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: (float) The maximum distance of interest.
    sum: (int) State if each distance is counted twice or once
    */

    int bin;
    double v;
    float dd_z, dd_ort;
    float x1,y1,z1,w1,x2,y2,z2;

    for (int i=start1; i<end1; ++i){
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        w1 = elements[i].w;
        for (int j=start2; j<end2; ++j){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;

            dd_z = (z2-z1)*(z2-z1);
            dd_ort = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);

            if (dd_z <= dd_max && dd_ort <= dd_max){
                bin = (int)(sqrt(dd_z)*ds)*bns + (int)(sqrtf(dd_ort)*ds);
                v = sum*w1*elements[j].w;
                atomicAdd(&XX[bin],v);
            }

        }
    }
}

__device__ void count_distancesXY(double *XY, PointW3D *elements1, int start1, int end1, PointW3D *elements2, int start2, int end2, int bns, float ds, float dd_max, int sum){
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
    bns: (int) number of bins per XY dimension
    ds: (float) number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: (float) The maximum distance of interest.
    sum: (int) State if each distance is counted twice or once
    */

    int bin;
    double v;
    float dd_z, dd_ort;
    float x1,y1,z1,w1,x2,y2,z2;

    for (int i=start1; i<end1; ++i){
        x1 = elements1[i].x;
        y1 = elements1[i].y;
        z1 = elements1[i].z;
        w1 = elements1[i].w;
        for (int j=start2; j<end2; ++j){
            x2 = elements2[j].x;
            y2 = elements2[j].y;
            z2 = elements2[j].z;

            dd_z = (z2-z1)*(z2-z1);
            dd_ort = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);

            if (dd_z <= dd_max && dd_ort <= dd_max){
                bin = (int)(sqrt(dd_z)*ds)*bns + (int)(sqrtf(dd_ort)*ds);
                v = sum*w1*elements2[j].w;
                atomicAdd(&XY[bin],v);
            }

        }
    }
}

__global__ void make_histoXX(double *XX, PointW3D *elements, DNode *nodeD, int partitions, int bn, float dmax, float size_node){
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon of this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;

        //idx = row + col*partitions + mom*partitions*partitions;

        if (nodeD[idx].len > 0){

            float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
            float nx1=nodeD[idx].nodepos.x, ny1=nodeD[idx].nodepos.y, nz1=nodeD[idx].nodepos.z;
            float d_max_node = dmax + size_node*sqrtf(3.0);
            d_max_node*=d_max_node;

            // Counts distances within the same node
            count_distances11(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, bn, ds, dd_max, 2);
            
            /*
            int idx2, u=row,v=col,w=mom; // Position index of the second node
            float dx_nod12, dy_nod12, dz_nod12, dd_nod12; //Internodal distance

            //Second node mobil in Z direction
            for(w = mom+1; w<partitions; w++){
                idx2 = row + col*partitions + w*partitions*partitions;
                dz_nod12 = nodeD[idx2].nodepos.z - nz1;
                dd_nod12 = dz_nod12*dz_nod12;
                if (dd_nod12 <= d_max_node){
                    count_distances12(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, bn, ds, dd_max, 2);
                }
            }

            //Second node mobil in YZ
            for(v=col+1; v<partitions; v++){
                idx2 = row + v*partitions;
                dy_nod12 = nodeD[idx2].nodepos.y - ny1;
                for(w=0; w<partitions; w++){
                    idx2 = row + v*partitions + w*partitions*partitions;
                    dz_nod12 = nodeD[idx2].nodepos.z - nz1;
                    dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12;
                    if (dz_nod12*dz_nod12<=d_max_node && dd_nod12<=d_max_node){
                        count_distances12(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, bn, ds, dd_max, 2);
                    }
                }
            }

            //Second node mobil in XYZ
            for(u = row+1; u < partitions; u++){
                dx_nod12 = nodeD[u].nodepos.x - nx1;
                for(v = 0; v < partitions; v++){
                    idx2 = u + v*partitions;
                    dy_nod12 = nodeD[idx2].nodepos.y - ny1;
                    for(w = 0; w < partitions; w++){
                        idx2 = u + v*partitions + w*partitions*partitions;
                        dz_nod12 = nodeD[idx2].nodepos.z - nz1;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dz_nod12*dz_nod12<=d_max_node && dd_nod12<=d_max_node){
                            count_distances12(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, bn, ds, dd_max, 2);
                        }
                    }
                }
            }
            */
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
                        if (dz_nod12*dz_nod12<=d_max_node && dd_nod12<=d_max_node){
                            count_distancesXY(XY, elementsD, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, elementsR, nodeR[idx2].prev_i, nodeR[idx2].prev_i + nodeR[idx2].len, bn, ds, dd_max, 1);
                        }
                    }
                }
            }
            
        }
    }
}
