
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__device__ void count_distances11(double *XX, PointW3D *elements, int start, int end, float ds, float dd_max, int sum){
    /*
    This device function counts the distances betweeen points within the same node. This function is used 
    to compute the XX histogram

    Args:
    XX: (double*) The histogram where the distances are counted in
    elements: (PointW3D*)  Array of PointW3D points orderered coherently by the nodes
    start: (int) index at which the nodeA starts to be defined by elements1. Inclusive.
    end: (int) index at which the nodeA stops being defined by elements1. Non inclusive.
    ds: (float) number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: (float) The maximum distance of interest.
    sum: (int) State if each distance is counted twice or once
    */
    
    int bin;
    double v;
    float d;
    float x1, y1, z1, w1;
    float x2,y2,z2,w2;

    for (int i=start; i<end-1; ++i){
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        w1 = elements[i].w;
        for (int j=i+1; j<end; ++j){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            w2 = elements[j].w;
            d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (d<=dd_max){
                bin = (int)(sqrt(d)*ds);
                v = sum*w1*w2;
                atomicAdd(&XX[bin],v);
            }
        }
    }
}

__device__ void count_distances12(double *XX, PointW3D *elements, int start1, int end1, int start2, int end2, float ds, float dd_max, int sum){
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
    ds: (float) number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: (float) The maximum distance of interest.
    sum: (int) State if each distance is counted twice or once
    */

    int bin;
    double v;
    float d;
    float x1,y1,z1,w1,x2,y2,z2,w2;

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
            d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (d<=dd_max){
                bin = (int)(sqrt(d)*ds);
                v = sum*w1*w2;
                atomicAdd(&XX[bin],v);
            }
        }
    }
}

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
                bin = (int)(sqrt(d)*ds);
                v = sum*w1*w2;
                atomicAdd(&XY[bin],v);
            }
        }
    }
}

//=================================================================== 
__device__ void boundaries_XX(double *XX, PointW3D *elements, int start1, int end1, int start2, int end2, float disn, float dn_x, float dn_y, float dn_z, bool con_in_x, bool con_in_y, bool con_in_z, float size_box, float ds, float dd_max, float ddmax_nod){
    /*
    Function to consider periodic boundary conditions contributions.
    */

    if (con_in_x || con_in_y || con_in_z){
        int bin;
        float dis_f, dis, d_x, d_y, d_z, x, y, z, w1;
        float ll = size_box*size_box;
        double v;


        dis_f = disn + (con_in_x + con_in_y + con_in_z)*ll - 2*(dn_x*con_in_x+dn_y*con_in_y+dn_z*con_in_z)*size_box;
        if (dis_f <= ddmax_nod){
            for (int i=start1; i<end1; ++i){
                x = elements[i].x;
                y = elements[i].y;
                z = elements[i].z;
                w1 = elements[i].w;
                for (int j=start2; j<end2; ++j){
                    d_x = fabs(x-elements[j].x)-(size_box*con_in_x);
                    d_y = fabs(y-elements[j].y)-(size_box*con_in_y);
                    d_z = fabs(z-elements[j].z)-(size_box*con_in_z);
                    dis = d_x*d_x + d_y*d_y + d_z*d_z;
                    if (dis < dd_max){
                        bin = (int)(sqrt(dis)*ds);
                        v = 2*w1*elements[j].w;
                        atomicAdd(&XX[bin],v);
                    }
                }
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
            bool con_x, con_y, con_z;
            float size_box = partitions*size_node;
            float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
            float nx1=nodeD[idx].nodepos.x, ny1=nodeD[idx].nodepos.y, nz1=nodeD[idx].nodepos.z;
            float d_front = size_box - dmax - size_node;
            float d_max_node = dmax + size_node*sqrt(3.0);
            d_max_node*=d_max_node;
            
            if(idx==0){
                printf("Length node 0 : %i", nodeD[idx].len);
                for (int i = 0; i<nodeD[idx].len; i++){
                    printf("%f, %f, %f, %f \n", elements[i].x, elements[i].y, elements[i].z, elements[i].w);
                }
            }

            // Counts distances within the same node
            count_distances11(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, ds, dd_max, 2);
            
            /*
            int idx2, u=row,v=col,w=mom; // Position index of the second node
            float dx_nod12, dy_nod12, dz_nod12, dd_nod12; //Internodal distance
            float nx2, ny2, nz2;

            //Second node mobil in Z direction
            for(w = mom+1; w<partitions; w++){
                idx2 = row + col*partitions + w*partitions*partitions;
                nz2 = nodeD[idx2].nodepos.z;
                dz_nod12 = nz2 - nz1;
                dd_nod12 = dz_nod12*dz_nod12;
                if (dd_nod12 <= d_max_node){
                    count_distances12(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, ds, dd_max, 2);
                }
                
                // Boundary node conditions:
                con_z = ((nz1<=dmax)&&(nz2>=d_front))||((nz2<=dmax)&&(nz1>=d_front));
                if(con_z){
                    boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, 0.0, 0.0, dz_nod12, false, false, con_z, size_box, ds, dd_max, d_max_node);
                }
            }

            //Second node mobil in YZ
            for(v=col+1; v<partitions; v++){
                idx2 = row + col*partitions;
                ny2 = nodeD[idx2].nodepos.y;
                dy_nod12 = ny2 - ny1;
                for(w=0; w<partitions; w++){
                    idx2 = row + v*partitions + w*partitions*partitions;
                    nz2 = nodeD[idx2].nodepos.z;
                    dz_nod12 = nz2 - nz1;
                    dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12;
                    if (dd_nod12<=d_max_node){
                        count_distances12(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, ds, dd_max, 2);
                    }
                    // Boundary node conditions:
                    con_y = ((ny1<=dmax)&&(ny2>=d_front))||((ny2<=dmax)&&(ny1>=d_front));
                    con_z = ((nz1<=dmax)&&(nz2>=d_front))||((nz2<=dmax)&&(nz1>=d_front));
                    if(con_y){ 
                        boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, 0.0, dy_nod12, dz_nod12, false, con_y, false, size_box, ds, dd_max, d_max_node);
                        if (con_z){
                            boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, 0.0, dy_nod12, dz_nod12, false, false, con_z, size_box, ds, dd_max, d_max_node);
                            boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, 0.0, dy_nod12, dz_nod12, false, con_y, con_z, size_box, ds, dd_max, d_max_node);
                        }
                    } else if (con_z){
                        boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, 0.0, dy_nod12, dz_nod12, false, false, con_z, size_box, ds, dd_max, d_max_node);
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
                            count_distances12(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, ds, dd_max, 2);
                        }
                        // Boundary node conditions:
                        con_x = ((nx1<=dmax)&&(nx2>=d_front))||((nx2<=dmax)&&(nx1>=d_front));
                        con_y = ((ny1<=dmax)&&(ny2>=d_front))||((ny2<=dmax)&&(ny1>=d_front));
                        con_z = ((nz1<=dmax)&&(nz2>=d_front))||((nz2<=dmax)&&(nz1>=d_front));
                        if (con_x){
                            boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, con_x, false, false, size_box, ds, dd_max, d_max_node);
                            if(con_y){
                                boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, false, con_y, false, size_box, ds, dd_max, d_max_node);
                                boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, con_x, con_y, false, size_box, ds, dd_max, d_max_node);
                                if (con_z){
                                    boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, false, false, con_z, size_box, ds, dd_max, d_max_node);
                                    boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, con_x, false, con_z, size_box, ds, dd_max, d_max_node);
                                    boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, false, con_y, con_z, size_box, ds, dd_max, d_max_node);
                                    boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, con_x, con_y, con_z, size_box, ds, dd_max, d_max_node);
                                }

                            } else if(con_z){
                                boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, false, false, con_z, size_box, ds, dd_max, d_max_node);
                                boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, con_x, false, con_z, size_box, ds, dd_max, d_max_node);
                            }
                        } else if (con_y) {
                            boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, false, con_y, false, size_box, ds, dd_max, d_max_node);
                            if (con_z){
                                boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, false, false, con_z, size_box, ds, dd_max, d_max_node);
                                boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, false, con_y, con_z, size_box, ds, dd_max, d_max_node);
                            }
                        } else if (con_z){
                            boundaries_XX(XX, elements, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, nodeD[idx2].prev_i, nodeD[idx2].prev_i + nodeD[idx2].len, dd_nod12, dx_nod12, dy_nod12, dz_nod12, false, false, con_z, size_box, ds, dd_max, d_max_node);
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
            float d_max_node = dmax + size_node*sqrt(3.0);
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
                            count_distancesXY(XY, elementsD, nodeD[idx].prev_i, nodeD[idx].prev_i+nodeD[idx].len, elementsR, nodeR[idx2].prev_i, nodeR[idx2].prev_i + nodeR[idx2].len, ds, dd_max, 1);
                        }
                    }
                }
            }
            
        }
    }
}
