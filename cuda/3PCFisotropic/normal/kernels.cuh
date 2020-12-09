
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__device__ void count_111_triangles(double *XXX, PointW3D *elements, int start, int end, int bns, float ds, float dd_max){
    /*
    This device function counts the triangles betweeen points within the same node. This function is used 
    to compute the XXX histogram

    Args:
    XXX: (double*) The histogram where the triangles are counted in
    elements: (PointW3D*)  Array of PointW3D points orderered coherently by the nodes
    start: (int) index at which the nodeA starts to be defined by elements1. Inclusive.
    end: (int) index at which the nodeA stops being defined by elements1. Non inclusive.
    bns: (int) number of bins per XXX dimension
    ds: (float) number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: (float) The maximum distance of interest.
    sum: (int) State if each distance is counted twice or once
    */
    
    int bin;
    double v;
    float x1,y1,z1,w1;
    float x2,y2,z2,w2;
    float x3,y3,z3;
    float dd12, dd23, dd31;

    for (int i=start; i<end-2; ++i){
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        w1 = elements[i].w;
        for (int j=i+1; j<end-1; ++j){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            w2 = elements[j].w;
            dd12 = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (dd12<dd_max){
                for (int k=j+1; k<end; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2);
                    if (dd23<dd_max){
                        dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1);
                        if (dd31<dd_max){
                            v = (double)(w1*w2*elements[k].w);
                            bin = (int)(sqrtf(dd12)*ds)*bns*bns + (int)(sqrtf(dd23)*ds)*bns + (int)(sqrtf(dd31)*ds);
                            atomicAdd(&XXX[bin],v);
                        }
                    }
                }
            }
        }
    }
}

__device__ void count_112_triangles(double *XXX, PointW3D *elements, int start1, int end1, int start2, int end2, int bns, float ds, float dd_max){
    /*
    This device function counts the triangles betweeen 2 points in one node and the third point in a second node. Between two different nodes from the same file. 
    This function is used to compute the XX histogram.

    Args:
    XXX: (double*) The histogram where the triangles are counted in
    elements: (PointW3D*)  Array of PointW3D points orderered coherently by the nodes
    start1: (int) index at which the nodeA starts to be defined by elements1. Inclusive.
    end1: (int) index at which the nodeA stops being defined by elements1. Non inclusive.
    start2: (int) index at which the nodeB starts to be defined by elements1. Inclusive.
    end2: (int) index at which the nodeB stops being defined by elements1. Non inclusive.
    bns: (int) number of bins per XXX dimension
    ds: (float) number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: (float) The maximum distance of interest.
    sum: (int) State if each distance is counted twice or once
    */

    int bin, bx, by, k;
    double v;
    float x1,y1,z1,w1;
    float x2,y2,z2,w2;
    float x3,y3,z3;
    float dd12, dd23, dd31;

    for (int i=start1; i<end1; ++i){
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        w1 = elements[i].w;
        for (int j=start2; j<end2; ++j){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            w2 = elements[j].w;
            dd12 = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (dd12<=dd_max){
                bx = (int)(sqrtf(dd12)*ds)*bns*bns;
                v = w1*w2;
                for (k=i+1; k<end1; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1);
                    if (dd31<dd_max){
                        by = (int)(sqrtf(dd31)*ds)*bns;
                        dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2);
                        if (dd23<dd_max){
                            bin = bx + by + (int)(sqrtf(dd23)*ds);
                            v *= elements[k].w;
                            atomicAdd(&XXX[bin],v);

                        }
                    }
                }
                for (k=j+1; k<end2; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dd31 = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1);
                    if (dd31<dd_max){
                        by = (int)(sqrtf(dd31)*ds)*bns;
                        dd23 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2);
                        if (dd23<dd_max){
                            bin = bx + by + (int)(sqrtf(dd23)*ds);
                            v *= elements[k].w;
                            atomicAdd(&XXX[bin],v);

                        }
                    }
                }
            }
        }
    }
}

__device__ void count_123_triangles(double *XXX, PointW3D *elements, int start1, int end1, int start2, int end2, int start3, int end3, int bns, float ds, float dd_max){
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

__device__ void inner_make_histoXXX(double *XXX, PointW3D *elements, DNode *nodeD, int idx, int idx2, int partitions, int bn, float ds, float d_max_node, float dd_max){

    //Get the node positon of this thread
    int col = (int) ((idx%(partitions*partitions))/partitions);
    int row = idx%partitions;

    //Get the node positon of this thread
    int u = (int) (idx2/(partitions*partitions));
    int v = (int) ((idx2%(partitions*partitions))/partitions);
    int w = idx2%partitions;
    
    int idx3, a=u, b=v, c=w; //Position index of the third node

    float nx1=nodeD[idx].nodepos.x, ny1=nodeD[idx].nodepos.y, nz1=nodeD[idx].nodepos.z;
    float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
    float nx3=nx2, ny3=ny2, nz3;
    float dx_nod31, dy_nod31, dz_nod31, dd_nod31, dd_nod23; //Internodal distance
    float ddx31, ddy31, ddx23, ddy23;

    ddx31 = (nx3 - nx1)*(nx3 - nx1);
    ddy31 = (ny3 - ny1)*(ny3 - ny1);
    ddx23 = (nx3 - nx2)*(nx3 - nx2);
    ddy23 = (ny3 - ny2)*(ny3 - ny2);

    //Third node mobil in Z direction
    for (c=w+1;  c<partitions; ++c){
        idx3 = row + col*partitions + c*partitions*partitions;
        nz3 = nodeD[idx3].nodepos.z;
        dz_nod31 = nz3-nz1;
        dd_nod31 = dz_nod31*dz_nod31 + ddx31 + ddy31;
        if (dd_nod31 <= d_max_node) {
            dd_nod23 = (nz3-nz2)*(nz3-nz2) + ddx23 + ddy23;
            if (dd_nod23 <= d_max_node) {
                count_123_triangles(XXX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, nodeD[idx2].prev_i, nodeD[idx3].prev_i + nodeD[idx3].len, bn, ds, dd_max);
            }
        }
    }

    //Third node mobil in YZ direction
    for (b=v+1; b<partitions; ++b){
        idx3 = row + b*partitions;
        ny3 = nodeD[idx3].nodepos.y;
        dy_nod31 = ny3-ny1;
        for (c=0;  c<partitions; ++c){
            idx3 = row + b*partitions + c*partitions*partitions;
            nz3 = nodeD[idx3].nodepos.z;
            dz_nod31 = nz3-nz1;
            dd_nod31 = dy_nod31*dy_nod31 + dz_nod31*dz_nod31 + ddx31;
            if (dd_nod31 <= d_max_node){
                dd_nod23 = (ny3-ny2)*(ny3-ny2) + (nz3-nz2)*(nz3-nz2) + ddx23;
                if (dd_nod23 <= d_max_node) {
                    count_123_triangles(XXX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, nodeD[idx2].prev_i, nodeD[idx3].prev_i + nodeD[idx3].len, bn, ds, dd_max);
                }
            }
        }
    }

    //Third node mobil in YZ direction
    for (a=u+1; a<partitions; ++a){
        nx3 = nodeD[a].nodepos.x;
        dx_nod31 = nx3 - nx1;
        for (b=0; b<partitions; ++b){
            idx3 = a + b*partitions;
            ny3 = nodeD[idx3].nodepos.y;
            dy_nod31 = ny3-ny1;
            for (c=0;  c<partitions; ++c){
                idx3 = a + b*partitions + c*partitions*partitions;
                nz3 = nodeD[idx3].nodepos.z;
                dz_nod31 = nz3-nz1;
                dd_nod31 = dx_nod31*dx_nod31 + dy_nod31*dy_nod31 + dz_nod31*dz_nod31;
                if (dd_nod31 <= d_max_node){
                    dd_nod23 = (nx3-nx2)*(nx3-nx2) + (ny3 - ny2)*(ny3 - ny2) + (nz3 - nz2)*(nz3 - nz2);
                    if (dd_nod23 <= d_max_node) {
                        count_123_triangles(XXX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, nodeD[idx2].prev_i, nodeD[idx3].prev_i + nodeD[idx3].len, bn, ds, dd_max);
                    }
                }
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
        //Get the node positon of this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;

        if (nodeD[idx].len > 0){

            float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
            float nx1=nodeD[idx].nodepos.x, ny1=nodeD[idx].nodepos.y, nz1=nodeD[idx].nodepos.z;
            float d_max_node = dmax + size_node*sqrtf(3.0);
            d_max_node*=d_max_node;

            // Counts distances within the same node
            count_111_triangles(XXX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, bn, ds, dd_max);
            
            
            int idx2, u=row,v=col,w=mom; // Position index of the second node
            float nx2, ny2, nz2;
            float dx_nod12, dy_nod12, dz_nod12, dd_nod12; //Internodal distance

            //Second node mobil in Z direction
            for(w = mom+1; w<partitions; w++){
                idx2 = row + col*partitions + w*partitions*partitions;
                nz2 = nodeD[idx2].nodepos.z;
                dz_nod12 = nz2 - nz1;
                dd_nod12 = dz_nod12*dz_nod12;

                if (dd_nod12 <= d_max_node){

                    count_112_triangles(XXX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, bn, ds, dd_max);
                    inner_make_histoXXX(XXX, elements, nodeD, idx, idx2, partitions, bn, ds, d_max_node, dd_max);

                }
            }

            /*
            //Second node mobil in YZ
            for(v=col+1; v<partitions; v++){
                idx2 = row + v*partitions;
                ny2 = nodeD[idx2].nodepos.y;
                dy_nod12 = ny2 - ny1;
                for(w=0; w<partitions; w++){
                    idx2 = row + v*partitions + w*partitions*partitions;
                    nz2 = nodeD[idx2].nodepos.z;
                    dz_nod12 = nz2 - nz1;
                    dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12;
                    if (dd_nod12<=d_max_node){

                        count_112_triangles(XXX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, bn, ds, dd_max);
                        //inner_make_histoXXX(XXX, elements, nodeD, idx, idx2, partitions, bn, ds, d_max_node, dd_max);

                    }
                }
            }

            //Second node mobil in XYZ
            for(u = row+1; u < partitions; u++){
                nx2 = nodeD[u].nodepos.x;
                dx_nod12 = nx2 - nx1;
                for(v = 0; v < partitions; v++){
                    idx2 = u + v*partitions;
                    ny2 = nodeD[idx2].nodepos.y;
                    dy_nod12 = ny2 - ny1;
                    for(w = 0; w < partitions; w++){
                        idx2 = u + v*partitions + w*partitions*partitions;
                        nz2 = nodeD[idx2].nodepos.z;
                        dz_nod12 = nz2 - nz1;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=d_max_node){
                            
                            count_112_triangles(XXX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, bn, ds, dd_max);
                            //inner_make_histoXXX(XXX, elements, nodeD, idx, idx2, partitions, bn, ds, d_max_node, dd_max);

                        }
                    }
                }
            }
            */
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