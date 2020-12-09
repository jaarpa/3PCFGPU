
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================
__device__ void addto_histo(double *XYZ, double v, float dz12, float d12, float dz31, float d31, float dz23, float d23, int bns, float dmax){
    
    int a, b, c, t, p, q, t_, p_, q_;
    int bin;
    float c_th_12, c_th_23, c_th_31, c_th_12_, c_th_23_, c_th_31_;
    float ds = ((float)(bns))/dmax;
    float ds_th = (float)(bns)*0.5;

    // angle between r and z
    c_th_12 = dz12/d12 + 1;
    c_th_31 = dz31/d31 + 1;
    c_th_23 = dz23/d23 + 1;

    c_th_12_ = 2 - c_th_12;
    c_th_31_ = 2 - c_th_31;
    c_th_23_ = 2 - c_th_23;

    // Indices 
    a = (int) (d12*ds);
    b = (int) (d31*ds);
    c = (int) (d23*ds);

    t = (int) (c_th_12*ds_th);
    p = (int) (c_th_31*ds_th);
    q = (int) (c_th_23*ds_th);

    t_ = (int) (c_th_12_*ds_th);
    p_ = (int) (c_th_31_*ds_th);
    q_ = (int) (c_th_23_*ds_th);

    // Atomic adds. Considers inner symmetrization.
    bin = a*bns*bns*bns*bns + b*bns*bns*bns + c*bns*bns + t*bns + p;
    atomicAdd(&XXX[bin],v);

    bin = a*bns*bns*bns*bns + c*bns*bns*bns + b*bns*bns + t_*bns + q;
    atomicAdd(&XXX[bin],v);

    bin = b*bns*bns*bns*bns + c*bns*bns*bns + a*bns*bns + q*bns + t_;
    atomicAdd(&XXX[bin],v);

    bin = b*bns*bns*bns*bns + a*bns*bns*bns + c*bns*bns + p*bns + t;
    atomicAdd(&XXX[bin],v);

    bin = c*bns*bns*bns*bns + b*bns*bns*bns + a*bns*bns + q_*bns + p_;
    atomicAdd(&XXX[bin],v);

    bin = c*bns*bns*bns*bns + a*bns*bns*bns + b*bns*bns + p_*bns + q_;
    atomicAdd(&XXX[bin],v);

}

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

    double v;
    float x1,y1,z1,w1;
    float x2,y2,z2,w2;
    float x3,y3,z3;
    float dz12, dz23, dz31;
    float dd12, dd23, dd31;
    float dd_max = dmax*dmax;

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
            dz12 = z2-z1;
            dd12 = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+dz12*dz12;
            if (dd12<dd_max){
                dd12 = sqrtf(dd12);
                v = w1*w2;
                for (int k=j+1; k<end; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dz31 = (z3-z1);
                    dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+dz31*dz31;
                    if (dd31<dd_max){
                        dd31 = sqrtf(dd31);
                        dz23 = (z3-z2);
                        dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+dz23*dz23;
                        if (dd23<dd_max){
                            dd23 = sqrtf(dd23);
                            v *= elements[k].w;

                            addto_histo(XXX, v, dz12, dd12, dz31, dd31, dz23, dd23, bns, dmax);

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
    
    double v;
    float x1,y1,z1,w1;
    float x2,y2,z2,w2;
    float x3,y3,z3;
    float dz12, dz23, dz31;
    float dd12, dd23, dd31;
    float dd_max = dmax*dmax;

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
            dz12 = (z2-z1);
            dd12 = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+dz12*dz12;
            if (dd12<dd_max){
                dd12 = sqrtf(dd12);
                v = w1*w2;
                for (int k=i+1; k<end1; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dz23 = z3-z2;
                    dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+dz23*dz23;
                    if (dd23<dd_max){
                        dd23 = sqrtf(dd23);
                        dz31 = z3-z1;
                        dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+dz31*dz31;
                        if (dd31<dd_max){
                            dd31 = sqrtf(dd31);
                            v *= elements[k].w;

                            addto_histo(XXX, v, dz12, dd12, dz31, dd31, dz23, dd23, bns, dmax);
                            
                        }
                    }
                }
                for (int k=j+1; k<end2; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dz23 = z3-z2;
                    dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+dz23*dz23;
                    if (dd23<dd_max){
                        dd23 = sqrtf(dd23);
                        dz31 = z3-z1;
                        dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+dz31*dz31;
                        if (dd31<dd_max){
                            dd31 = sqrtf(dd31);
                            v *= elements[k].w;

                            addto_histo(XXX, v, dz12, dd12, dz31, dd31, dz23, dd23, bns, dmax);
                            
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

    double v;
    float x1,y1,z1,w1;
    float x2,y2,z2,w2;
    float x3,y3,z3;
    float dz12, dz23, dz31;
    float dd12, dd23, dd31;
    float dd_max = dmax*dmax;

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
            dz12 = z2-z1;
            dd12 = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+dz12*dz12;
            if (dd12<dd_max){
                dd12 = sqrtf(dd12);
                v = w1*w2;
                for (int k=start3; k<end3; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dz23 = z3-z2;
                    dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+dz23*dz23;
                    if (dd23<dd_max){
                        dd23 = sqrtf(dd23);
                        dz31 = z3-z1;
                        dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+dz31*dz31;
                        if (dd31<dd_max){
                            dd31 = sqrtf(dd31);
                            v *= elements[k].w;
                            
                            addto_histo(XXX, v, dz12, dd12, dz31, dd31, dz23, dd23, bns, dmax);
                            
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
                    threads_perblock = (partitions*partitions*partitions)-idx2;
                } else {
                    threads_perblock=512;
                    blocks = (int)((((partitions*partitions*partitions)-idx2)/threads_perblock)+1);
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
                blocks = (int)((((partitions*partitions*partitions)-idx)/threads_perblock)+1);
            }
            make_histoXXX_child1<<<blocks,threads_perblock>>>(XXX, elements, nodeD, idx, partitions, bn, dmax, size_node);
        }
    }
}

__device__ void count_112_triangles_XXY(double *XXX, PointW3D *elements1, PointW3D *elements2, int start1, int end1, int start2, int end2, int bns, float dmax){
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
        x1 = elements1[i].x;
        y1 = elements1[i].y;
        z1 = elements1[i].z;
        w1 = elements1[i].w;
        for (int j=start2; j<end2; j++){
            x2 = elements2[j].x;
            y2 = elements2[j].y;
            z2 = elements2[j].z;
            w2 = elements2[j].w;
            dd12 = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (dd12<dd_max){
                bx = (int)(sqrtf(dd12)*ds)*bns*bns;
                v = w1*w2;
                for (int k=i+1; k<end1; k++){
                    x3 = elements1[k].x;
                    y3 = elements1[k].y;
                    z3 = elements1[k].z;
                    dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2);
                    if (dd23<dd_max){
                        by = (int)(sqrtf(dd23)*ds)*bns;
                        dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1);
                        if (dd31<dd_max){
                            bin = bx + by + (int)(sqrtf(dd31)*ds);
                            v *= elements1[k].w;
                            atomicAdd(&XXX[bin],v);
                        }
                    }
                }
            }
        }
    }
}

__device__ void count_123_triangles_XXY(double *XXX, PointW3D *elements1, PointW3D *elements2, int start1, int end1, int start2, int end2, int start3, int end3, int bns, float dmax){
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
        x1 = elements1[i].x;
        y1 = elements1[i].y;
        z1 = elements1[i].z;
        w1 = elements1[i].w;
        for (int j=start2; j<end2; j++){
            x2 = elements2[j].x;
            y2 = elements2[j].y;
            z2 = elements2[j].z;
            w2 = elements2[j].w;
            dd12 = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (dd12<dd_max){
                bx = (int)(sqrtf(dd12)*ds)*bns*bns;
                v = w1*w2;
                for (int k=start3; k<end3; k++){
                    x3 = elements1[k].x;
                    y3 = elements1[k].y;
                    z3 = elements1[k].z;
                    dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2);
                    if (dd23<dd_max){
                        by = (int)(sqrtf(dd23)*ds)*bns;
                        dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1);
                        if (dd31<dd_max){
                            bin = bx + by + (int)(sqrtf(dd31)*ds);
                            v *= elements1[k].w;
                            atomicAdd(&XXX[bin],v);
                        }
                    }
                }
            }
        }
    }
}


__global__ void make_histoXXY_child2(double *XXX, PointW3D *elementsX, DNode *nodeX, PointW3D *elementsY, DNode *nodeY, int idx1, int idx2, int partitions, int bn, float dmax, float size_node){
    int idx3 = blockIdx.x * blockDim.x + threadIdx.x + (idx1 + 1);
    if (idx3<(partitions*partitions*partitions)){
        if (nodeX[idx3].len > 0){
            float d_max_node = dmax + size_node*sqrtf(3.0);
            d_max_node*=d_max_node;

            float nx1=nodeX[idx1].nodepos.x, ny1=nodeX[idx1].nodepos.y, nz1=nodeX[idx1].nodepos.z;
            float nx2=nodeY[idx2].nodepos.x, ny2=nodeY[idx2].nodepos.y, nz2=nodeY[idx2].nodepos.z;
            float nx3=nodeX[idx3].nodepos.x, ny3=nodeX[idx3].nodepos.y, nz3=nodeX[idx3].nodepos.z;
            float dd_nod23 = (nx3-nx2)*(nx3-nx2)+(ny3-ny2)*(ny3-ny2)+(nz3-nz2)*(nz3-nz2);
            float dd_nod31 = (nx3-nx1)*(nx3-nx1)+(ny3-ny1)*(ny3-ny1)+(nz3-nz1)*(nz3-nz1);
            if (dd_nod23<=d_max_node && dd_nod31<=d_max_node){
                count_123_triangles_XXY(XXX, elementsX, elementsY, nodeX[idx1].start, nodeX[idx1].end, nodeY[idx2].start, nodeY[idx2].end, nodeX[idx3].start, nodeX[idx3].end, bn, dmax);
            }
        }
    }
}

__global__ void make_histoXXY_child1(double *XXX, PointW3D *elementsX, DNode *nodeX, PointW3D *elementsY, DNode *nodeY, int idx1, int partitions, int bn, float dmax, float size_node){
    int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx2<(partitions*partitions*partitions)){
        if (nodeY[idx2].len > 0){
            float d_max_node = dmax + size_node*sqrtf(3.0);
            d_max_node*=d_max_node;
            float dx21 = nodeY[idx2].nodepos.x - nodeX[idx1].nodepos.x;
            float dy21 = nodeY[idx2].nodepos.y - nodeX[idx1].nodepos.y;
            float dz21 = nodeY[idx2].nodepos.z - nodeX[idx1].nodepos.z;
            float dd_nod21 = dx21*dx21 + dy21*dy21 + dz21*dz21;

            if (dd_nod21<=d_max_node){
                count_112_triangles_XXY(XXX, elementsX, elementsY, nodeX[idx1].start, nodeX[idx1].end, nodeY[idx2].start, nodeY[idx2].end, bn, dmax);
                int blocks, threads_perblock;
                if ((partitions*partitions*partitions)-idx1 < 512){
                    blocks = 1;
                    threads_perblock = (partitions*partitions*partitions)-idx1;
                } else {
                    threads_perblock=512;
                    blocks = (int)((((partitions*partitions*partitions)-idx1)/threads_perblock)+1);
                }
                make_histoXXY_child2<<<blocks,threads_perblock>>>(XXX, elementsX, nodeX, elementsY, nodeY, idx1, idx2, partitions, bn, dmax, size_node);
            }
            
        }
    }
}

__global__ void make_histoXXY(double *XXY, PointW3D *elementsX, DNode *nodeX, PointW3D *elementsY, DNode *nodeY, int partitions, int bn, float dmax, float size_node){
    /*
    Kernel function to calculate the mixed histograms. It stores the counts in the XXX histogram.

    args:
    XXY: (double*) The histogram where the distances are counted.
    elementsD: (PointW3D*) Array of the points ordered coherently with the nodes. X data.
    nodeD: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node. X data
    elementsR: (PointW3D*) Array of the points ordered coherently with the nodes. Y data.
    nodeR: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node. Y data
    partitions: (int) Number of partitions that are fitted by box side.
    bn: (int) NUmber of bins in the XY histogram.
    dmax: (dmax) The maximum distance of interest between points.
    size_node: (float) Size of the nodes
    */

    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<(partitions*partitions*partitions)){
        if (nodeX[idx].len > 0){
            make_histoXXY_child1<<<gridDim.x,blockDim.x>>>(XXY, elementsX, nodeX, elementsY, nodeY, idx, partitions, bn, dmax, size_node);
        }
    }
}

__global__ void symmetrize(double *s_XXX,double *XXX , int bn){
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