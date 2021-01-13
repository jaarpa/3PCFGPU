
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__global__ void XXX3iso(double *XXX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, int node_offset=0, int bn_offset=0){
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
    
    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = node_offset + blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<(nonzero_nodes+node_offset) && idx2<(nonzero_nodes+node_offset) && idx3<(nonzero_nodes+node_offset)){
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
                    float x1,y1,z1,w1,x2,y2,z2,w2,x3,y3,z3;
                    double d12, d23, d31, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;

                    for (int i=nodeD[idx1].start; i<end1; i++){
                        x1 = elements[i].x;
                        y1 = elements[i].y;
                        z1 = elements[i].z;
                        w1 = elements[i].w;
                        for (int j=nodeD[idx2].start; j<end2; j++){
                            x2 = elements[j].x;
                            y2 = elements[j].y;
                            z2 = elements[j].z;
                            w2 = elements[j].w;
                            v = w1*w2;
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
                                            bin += bn*bn*bn*bn_offset;
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

__global__ void XXY3iso(double *XXY, PointW3D *elementsX, DNode *nodeX, int nonzero_Xnodes, PointW3D *elementsY, DNode *nodeY, int nonzero_Ynodes, int bn, float dmax, float d_max_node, int node_offset, int bn_offset, bool isDDR){
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

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<nonzero_Xnodes && idx2<nonzero_Xnodes && idx3<nonzero_Ynodes){
        float nx1=nodeX[idx1].nodepos.x, ny1=nodeX[idx1].nodepos.y, nz1=nodeX[idx1].nodepos.z;
        float nx2=nodeX[idx2].nodepos.x, ny2=nodeX[idx2].nodepos.y, nz2=nodeX[idx2].nodepos.z;
        float dd_nod12 = (nx2-nx1)*(nx2-nx1)+(ny2-ny1)*(ny2-ny1)+(nz2-nz1)*(nz2-nz1);
        if (dd_nod12 <= d_max_node){
            float nx3=nodeY[idx3].nodepos.x, ny3=nodeY[idx3].nodepos.y, nz3=nodeY[idx3].nodepos.z;
            float dd_nod23 = (nx3-nx2)*(nx3-nx2)+(ny3-ny2)*(ny3-ny2)+(nz3-nz2)*(nz3-nz2);
            if (dd_nod23 <= d_max_node){
                float dd_nod31 = (nx1-nx3)*(nx1-nx3)+(ny1-ny3)*(ny1-ny3)+(nz1-nz3)*(nz1-nz3);
                if (dd_nod31 <= d_max_node){
                    int end1 = nodeX[idx1].end;
                    int end2 = nodeX[idx2].end;
                    int end3 = nodeY[idx3].end;
                    int bnx, bny, bnz, bin;
                    float dd_max=dmax*dmax;
                    float x1,y1,z1,w1,x2,y2,z2,w2,x3,y3,z3;
                    double d12, d23, d31, v, ds = floor(((double)(bn)/dmax)*1000000)/1000000;

                    for (int i=nodeX[idx1].start; i<end1; i++){
                        x1 = elementsX[i].x;
                        y1 = elementsX[i].y;
                        z1 = elementsX[i].z;
                        w1 = elementsX[i].w;
                        for (int j=nodeX[idx2].start; j<end2; j++){
                            x2 = elementsX[j].x;
                            y2 = elementsX[j].y;
                            z2 = elementsX[j].z;
                            w2 = elementsX[j].w;
                            v = w1*w2;
                            d12 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
                            if (d12 < dd_max && d12>0){
                                d12 = sqrt(d12);
                                for (int k=nodeY[idx3].start; k<end3; k++){
                                    x3 = elementsY[k].x;
                                    y3 = elementsY[k].y;
                                    z3 = elementsY[k].z;
                                    d23 = (x3-x2)*(x3-x2) + (y3-y2)*(y3-y2) + (z3-z2)*(z3-z2);
                                    if (d23 < dd_max){
                                        d31 = (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1) + (z3-z1)*(z3-z1);
                                        if (d31 < dd_max){
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