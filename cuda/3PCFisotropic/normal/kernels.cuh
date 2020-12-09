
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__device__ void count_111_triangles(double *XXX, PointW3D *elements, int start, int end, int bns, float dmax){
    /*
    This device function counts the triangles betweeen points in three different nodes each. Between two different nodes from the same file. 
    This function is used to compute the XX histogram.

    Args:
    XXX: (double*) The histogram where the triangles are counted in
    elements: (PointW3D*)  Array of PointW3D points orderered coherently by the nodes
    start: (int) index at which the nodeA starts to be defined by elements1. Inclusive.
    end: (int) index at which the nodeA stops being defined by elements1. Non inclusive.
    bns: (int) number of bins per XXX dimension
    dmax: (float) The maximum distance of interest.
    */

    int bin, bx, by;
    double v;
    float x1,y1,z1,w1;
    float x2,y2,z2,w2;
    float x3,y3,z3;
    float dd12, dd23, dd31;
    float dd_max = dmax*dmax;
    float ds = ((float)(bns))/dmax;

    for (int i=start; i<end-2; i++){
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        w1 = elements[i].w;
        for (int j=i+1; j<end-1; j++){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            w2 = elements[j].w;
            dd12 = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (dd12<dd_max){
                bx = (int)(sqrtf(dd12)*ds)*bns*bns;
                v = w1*w2;
                for (int k=j+1; k<end; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2);
                    if (dd23<dd_max){
                        by = (int)(sqrtf(dd23)*ds)*bns;
                        dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1);
                        if (dd31<dd_max){
                            bin = bx + by + (int)(sqrtf(dd31)*ds);
                            v *= elements[k].w;
                            atomicAdd(&XXX[bin],v);
                        }
                    }
                }
            }
        }
    }
}

__device__ void count_112_triangles(double *XXX, PointW3D *elements, int start1, int end1, int start2, int end2, int bns, float dmax){
    /*
    This device function counts the triangles betweeen points in three different nodes each. Between two different nodes from the same file. 
    This function is used to compute the XX histogram.

    Args:
    XXX: (double*) The histogram where the triangles are counted in
    elements: (PointW3D*)  Array of PointW3D points orderered coherently by the nodes
    start1: (int) index at which the nodeA starts to be defined by elements1. Inclusive.
    end1: (int) index at which the nodeA stops being defined by elements1. Non inclusive.
    start2: (int) index at which the nodeB starts to be defined by elements1. Inclusive.
    end2: (int) index at which the nodeB stops being defined by elements1. Non inclusive.
    bns: (int) number of bins per XXX dimension
    dmax: (float) The maximum distance of interest.
    sum: (int) State if each distance is counted twice or once
    */

    int bin, bx, by;
    double v;
    float x1,y1,z1,w1;
    float x2,y2,z2,w2;
    float x3,y3,z3;
    float dd12, dd23, dd31;
    float dd_max = dmax*dmax;
    float ds = ((float)(bns))/dmax;

    for (int i=start1; i<end1; i++){
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        w1 = elements[i].w;
        for (int j=start2; j<end2; j++){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            w2 = elements[j].w;
            dd12 = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (dd12<dd_max){
                bx = (int)(sqrtf(dd12)*ds)*bns*bns;
                v = w1*w2;
                for (int k=i+1; k<end1; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2);
                    if (dd23<dd_max){
                        by = (int)(sqrtf(dd23)*ds)*bns;
                        dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1);
                        if (dd31<dd_max){
                            bin = bx + by + (int)(sqrtf(dd31)*ds);
                            v *= elements[k].w;
                            atomicAdd(&XXX[bin],v);
                        }
                    }
                }
                for (int k=j+1; k<end2; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2);
                    if (dd23<dd_max){
                        by = (int)(sqrtf(dd23)*ds)*bns;
                        dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1);
                        if (dd31<dd_max){
                            bin = bx + by + (int)(sqrtf(dd31)*ds);
                            v *= elements[k].w;
                            atomicAdd(&XXX[bin],v);
                        }
                    }
                }
            }
        }
    }
}

__device__ void count_123_triangles(double *XXX, PointW3D *elements, int start1, int end1, int start2, int end2, int start3, int end3, int bns, float dmax){
    /*
    This device function counts the triangles betweeen points in three different nodes each. Between two different nodes from the same file. 
    This function is used to compute the XX histogram.

    Args:
    XXX: (double*) The histogram where the triangles are counted in
    elements: (PointW3D*)  Array of PointW3D points orderered coherently by the nodes
    start1: (int) index at which the nodeA starts to be defined by elements1. Inclusive.
    end1: (int) index at which the nodeA stops being defined by elements1. Non inclusive.
    start2: (int) index at which the nodeB starts to be defined by elements1. Inclusive.
    end2: (int) index at which the nodeB stops being defined by elements1. Non inclusive.
    start3: (int) index at which the nodeC starts to be defined by elements1. Inclusive.
    end3: (int) index at which the nodeC stops being defined by elements1. Non inclusive.
    bns: (int) number of bins per XXX dimension
    ds: (float) number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: (float) The maximum distance of interest.
    sum: (int) State if each distance is counted twice or once
    */

    int bin, bx, by;
    double v;
    float x1,y1,z1,w1;
    float x2,y2,z2,w2;
    float x3,y3,z3;
    float dd12, dd23, dd31;
    float dd_max = dmax*dmax;
    float ds = ((float)(bns))/dmax;

    for (int i=start1; i<end1; i++){
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        w1 = elements[i].w;
        for (int j=start2; j<end2; j++){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            w2 = elements[j].w;
            dd12 = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (dd12<dd_max){
                bx = (int)(sqrtf(dd12)*ds)*bns*bns;
                v = w1*w2;
                for (int k=start3; k<end3; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2);
                    if (dd23<dd_max){
                        by = (int)(sqrtf(dd23)*ds)*bns;
                        dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1);
                        if (dd31<dd_max){
                            bin = bx + by + (int)(sqrtf(dd31)*ds);
                            v *= elements[k].w;
                            atomicAdd(&XXX[bin],v);
                        }
                    }
                }
            }
        }
    }
}

__global__ void make_histoXXX_child2(double *XXX, PointW3D *elements, DNode *nodeD, int idx1, int idx2, int partitions, int bn, float dmax, float size_node){
    int idx3 = blockIdx.x * blockDim.x + threadIdx.x + (idx2 + 1);
    if (idx3<(partitions*partitions*partitions)){
        if (nodeD[idx3].len > 0){
            float d_max_node = dmax + size_node*sqrtf(3.0);
            d_max_node*=d_max_node;

            float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
            float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
            float nx3=nodeD[idx3].nodepos.x, ny3=nodeD[idx3].nodepos.y, nz3=nodeD[idx3].nodepos.z;
            float dd_nod23 = (nx3-nx2)*(nx3-nx2)+(ny3-ny2)*(ny3-ny2)+(nz3-nz2)*(nz3-nz2);
            float dd_nod31 = (nx3-nx1)*(nx3-nx1)+(ny3-ny1)*(ny3-ny1)+(nz3-nz1)*(nz3-nz1);
            if (dd_nod23<=d_max_node && dd_nod31<=d_max_node){
                count_123_triangles(XXX, elements, nodeD[idx1].start, nodeD[idx1].end, nodeD[idx2].start, nodeD[idx2].end, nodeD[idx3].start, nodeD[idx3].end, bn, dmax);
            }
        }
    }
}

__global__ void make_histoXXX_child1(double *XXX, PointW3D *elements, DNode *nodeD, int idx1, int partitions, int bn, float dmax, float size_node){
    int idx2 = blockIdx.x * blockDim.x + threadIdx.x + (idx1 + 1);
    if (idx2<(partitions*partitions*partitions)){
        if (nodeD[idx2].len > 0){
            float d_max_node = dmax + size_node*sqrtf(3.0);
            d_max_node*=d_max_node;
            float dx21 = nodeD[idx2].nodepos.x - nodeD[idx1].nodepos.x;
            float dy21 = nodeD[idx2].nodepos.y - nodeD[idx1].nodepos.y;
            float dz21 = nodeD[idx2].nodepos.z - nodeD[idx1].nodepos.z;
            float dd_nod21 = dx21*dx21 + dy21*dy21 + dz21*dz21;

            if (dd_nod21<=d_max_node){
                count_112_triangles(XXX, elements, nodeD[idx1].start, nodeD[idx1].end, nodeD[idx2].start, nodeD[idx2].end, bn, dmax);
                int blocks, threads_perblock;
                if ((partitions*partitions*partitions)-idx2 < 512){
                    blocks = 1;
                    threads_perblock = (partitions*partitions*partitions)-idx;
                } else {
                    threads_perblock=512;
                    blocks = (int)((((partitions*partitions*partitions)-idx)/threads_perblock)+1)
                }
                make_histoXXX_child2<<<blocks,threads_perblock>>>(XXX, elements, nodeD, idx1, idx2, partitions, bn, dmax, size_node);
            }
            
        }
    }
}
__global__ void make_histoXXX(double *XXX, PointW3D *elements, DNode *nodeD, int partitions, int bn, float dmax, float size_node){
    /*
    Kernel function to calculate the pure histograms. It stores the counts in the XXX histogram.

    args:
    XXX: (double*) The histogram where the distances are counted.
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
        if (nodeD[idx].len > 0){
            count_111_triangles(XXX, elements, nodeD[idx].start, nodeD[idx].end, bn, dmax);
            int blocks, threads_perblock;
            if ((partitions*partitions*partitions)-idx < 512){
                blocks = 1;
                threads_perblock = (partitions*partitions*partitions)-idx;
            } else {
                threads_perblock=512;
                blocks = (int)((((partitions*partitions*partitions)-idx)/threads_perblock)+1)
            }
            make_histoXXX_child1<<<blocks,threads_perblock>>>(XXX, elements, nodeD, idx, partitions, bn, dmax, size_node);
        }
    }
}

__global__ void simmetrization(double *s_XXX,double *XXX , int bn){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<bn){
        s_XXX[i*bn*bn + i*bn + i] = XXX[i*bn*bn + i*bn + i];
        double v;
        for (int j=i; j<bn; j++){
            for (int k=j; k<bn; k++){
                v = XXX[i*bn*bn + j*bn + k] + XXX[i*bn*bn + k*bn + j] + XXX[j*bn*bn + k*bn + i] + XXX[j*bn*bn + i*bn + k] + XXX[k*bn*bn + i*bn + j] + XXX[k*bn*bn + j*bn + i];
                s_XXX[i*bn*bn + j*bn + k] = v;
                s_XXX[i*bn*bn + k*bn + j] = v;
                s_XXX[j*bn*bn + k*bn + i] = v;
                s_XXX[j*bn*bn + i*bn + k] = v;
                s_XXX[k*bn*bn + i*bn + j] = v;
                s_XXX[k*bn*bn + j*bn + i] = v;
            }
        }
    }
}