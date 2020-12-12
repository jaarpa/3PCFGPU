
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================
__device__ void count_frontXX(double *XX, PointW3D *elements, DNode *nodeD, int idx1, int idx2, float dd_nod, float dn_x, float dn_y, float dn_z, bool front_x, bool front_y, bool front_z, float ddmax, float ds, float d_max_node, float size_box){
    
    float dd_nod_f = dd_nod + (front_x + front_y + front_z)*size_box*size_box - 2*size_box*(front_x*dn_x+front_y*dn_y+front_z*dn_z);
    if (dd_nod_f <= d_max_node){
        int bin;
        double v;
        float x1,y1,z1,x2,y2,z2,d;
        int start1 = nodeD[idx1].start, end1 = nodeD[idx1].end;
        int start2 = nodeD[idx2].start, end2 = nodeD[idx2].end;
        for(int i=start1; i<end2; i++){
            x1 = elements[i].x;
            y1 = elements[i].y;
            z1 = elements[i].z;
            for(int j=start2; i<end2; j++){
                x2 = elements[i].x;
                y2 = elements[i].y;
                z2 = elements[i].z;
                d = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
                if (d<dd_max && d>0){
                    bin = (int)(sqrtf(d)*ds);
                    v = elements[i].w*elements[j].w;
                    atomicAdd(&XX[bin],v);
                }
            }
        }
    }
}

__global__ void make_histoXX(double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, float size_box){
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
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<nonzero_nodes && idx2<nonzero_nodes){
        
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float dx12=nx2-nx1, dy12=ny2-ny1, dz12=nz2-nz1;
        float dd_nod12 = dx12*dx12 + dy12*dy12 + dz12*dz12;
        float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;

        float f_d_max_node = dmax + size_node;
        float _f_d_max_node = size_box - f_d_max_node;
        bool front_x=((nx1<=f_d_max_node && nx2=>_f_d_max_node)||(nx2<=f_d_max_node && nx1=>_f_d_max_node));
        bool front_y=((ny1<=f_d_max_node && ny2=>_f_d_max_node)||(ny2<=f_d_max_node && ny1=>_f_d_max_node));
        bool front_z=((nz1<=f_d_max_node && nz2=>_f_d_max_node)||(nz2<=f_d_max_node && nz1=>_f_d_max_node));
        /*
        if (dd_nod12 <= d_max_node){

            float x1,y1,z1,x2,y2,z2,d;
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
                    if (d<=dd_max && d>0){
                        bin = (int)(sqrtf(d)*ds);
                        v = elements[i].w*elements[j].w;
                        atomicAdd(&XX[bin],v);
                    }
                }
            }
        }
        */

        if (front_x){
            //Count x proyection
            count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, front_x, false, false, dd_max, ds, d_max_node, size_box)

            if (front_y){
                //Counts the y proyection
                count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, false, front_y, false, dd_max, ds, d_max_node, size_box)

                //Counts the xy proyection
                count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, front_x, front_y, false, dd_max, ds, d_max_node, size_box)

                if (front_z){
                    //Counts the z proyection
                    count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, false, false, front_z, dd_max, ds, d_max_node, size_box)

                    //Counts the xz proyection
                    count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, front_x, false, front_z, dd_max, ds, d_max_node, size_box)

                    //Counts the yz proyection
                    count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, false, front_y, front_z, dd_max, ds, d_max_node, size_box)

                    //Counts the xyz proyection
                    count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, front_x, front_y, front_z, dd_max, ds, d_max_node, size_box)

                }
            } else if (front_z) {
                //Counts the z proyection
                count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, false, false, front_z, dd_max, ds, d_max_node, size_box)
                //Counts the xz proyection
                count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, front_x, false, front_z, dd_max, ds, d_max_node, size_box)
            }
        } else if (front_y) {
            //Counts the y proyection
            count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, false, front_y, false, dd_max, ds, d_max_node, size_box)

            if (front_z){
                //Counts the z proyection
                count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, false, false, front_z, dd_max, ds, d_max_node, size_box)

                //Counts the yz proyection
                count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, false, front_y, front_z, dd_max, ds, d_max_node, size_box)

            }
        } else if (front_z) {
            //Counts the z proyection
            count_frontXX(XX, elements, nodeD, idx1, idx2, dd_nod12, dx12, dy12, dz12, false, false, front_z, dd_max, ds, d_max_node, size_box)

        }

    }
}

__global__ void make_histoXY(double *XY, PointW3D *elementsD, DNode *nodeD, int nonzero_Dnodes, PointW3D *elementsR,  DNode *nodeR, int nonzero_Rnodes, int bn, float dmax, float d_max_node, float size_box){
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
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<nonzero_Dnodes && idx2<nonzero_Rnodes){
        
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeR[idx2].nodepos.x, ny2=nodeR[idx2].nodepos.y, nz2=nodeR[idx2].nodepos.z;
        float dx12=nx2-nx1, dy12=ny2-ny1, dz12=nz2-nz1;
        float dd_nod12 = dx12*dx12 + dy12*dy12 + dz12*dz12;

        if (dd_nod12 <= d_max_node){

            float x1,y1,z1,x2,y2,z2;
            float d, ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
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
                    if (d<=dd_max && d>0){
                        bin = (int)(sqrtf(d)*ds);
                        v = elementsD[i].w*elementsR[j].w;
                        atomicAdd(&XY[bin],v);
                    }
                }
            }
        }
    }
}
