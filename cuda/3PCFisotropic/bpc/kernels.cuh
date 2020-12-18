
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================
__device__ void count123(double *XXX, PointW3D *elements, int start1, int end1, int start2, int end2, int start3, int end3, int bn, float ds, float dd_max, float size_box, bool fx_2, bool fy_2, bool fz_2, bool fx_3, bool fy_3, bool fz_3){
    int bin;
    float x1,y1,z1,w1,x2,y2,z2,w2,x3,y3,z3;
    float dx12, dy12, dz12, dx23, dy23, dz23, dx31, dy31, dz31;
    float d12,d23,d31;
    double v;

    for (int i=start1; i<end1; i++){
        //Node 1 is never proyected
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        w1 = elements[i].w;
        for (int j=start2; j<end2; j++){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            w2 = elements[j].w;
            dx12 = fabsf(x2-x1) - size_box*fx_2;
            dy12 = fabsf(y2-y1) - size_box*fy_2;
            dz12 = fabsf(z2-z1) - size_box*fz_2;
            d12 = dx12*dx12 + dy12*dy12 + dz12*dz12;
            if (d12 < dd_max && d12>0){
                v = w1*w2;
                d12 = sqrtf(d12);
                for (int k=start3; k<end3; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dx23 = fabsf(x3-x2) + (size_box*fx_2 - size_box*fx_3);
                    dy23 = fabsf(y3-y2) + (size_box*fy_2 - size_box*fy_3);
                    dz23 = fabsf(z3-z2) + (size_box*fz_2 - size_box*fz_3);
                    d23 = dx23*dx23 + dy23*dy23 + dz23*dz23;
                    if (d23 < dd_max && d23>0){
                        dx31 = fabsf(x3-x1) - size_box*fx_3;
                        dy31 = fabsf(y3-y1) - size_box*fy_3;
                        dz31 = fabsf(z3-z1) - size_box*fz_3;
                        d31 = dx31*dx31 + dy31*dy31 + dz31*dz31;
                        if (d31 < dd_max && d31>0){
                            d23 = sqrtf(d23);
                            d31 = sqrtf(d31);
                            v *= elements[k].w;
                            bin = (int)(d12*ds)*bn*bn + (int)(d23*ds)*bn + (int)(d31*ds);
                            atomicAdd(&XXX[bin],v);
                        }
                    }
                }
            }
        }
    }
}

__global__ void make_histoXXX(double *XXX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, float size_box){
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
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<nonzero_nodes && idx2<nonzero_nodes && idx3<nonzero_nodes){
        int end1 = nodeD[idx1].end, end2 = nodeD[idx2].end, end3 = nodeD[idx3].end;
        int start1 = nodeD[idx1].start, start2 = nodeD[idx2].start, start3 = nodeD[idx3].start;
        float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;

        //Vas aquí y te falta considerar las proyecciones de los nodos para ver si se llama count123 o no
        float dxn12 = fabsf(nodeD[idx2].nodepos.x-nodeD[idx1].nodepos.x), dyn12 = fabsf(nodeD[idx2].nodepos.y-nodeD[idx1].nodepos.y), dzn12 = fabsf(nodeD[idx2].nodepos.z-nodeD[idx1].nodepos.z);
        float dxn23 = fabsf(nodeD[idx3].nodepos.x-nodeD[idx2].nodepos.x), dyn23 = fabsf(nodeD[idx3].nodepos.y-nodeD[idx2].nodepos.y), dzn23 = fabsf(nodeD[idx3].nodepos.z-nodeD[idx2].nodepos.z);
        float dxn31 = fabsf(nodeD[idx1].nodepos.x-nodeD[idx3].nodepos.x), dyn31 = fabsf(nodeD[idx1].nodepos.y-nodeD[idx3].nodepos.y), dzn31 = fabsf(nodeD[idx1].nodepos.z-nodeD[idx3].nodepos.z);

        float dd_nod23, dd_nod31, dd_nod12 = dxn12*dxn12+dyn12*dyn12+dzn12*dzn12;
        
        if (dd_nod12 <= d_max_node){
            dd_nod23 = dxn23*dxn23+dyn23*dyn23+dzn23*dzn23;
            if (dd_nod23 <= d_max_node){
                dd_nod31 = dxn31*dxn31+dyn31*dyn31+dzn31*dzn31;
                if (dd_nod31 <= d_max_node){
                    count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, false, false, false, false, false);
                }
            }
        }

    }
}