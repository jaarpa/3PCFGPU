
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__global__ void XX2ani_BPC(double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, float size_box, float size_node, int node_offset=0, int bn_offset=0){
    /*
    Kernel function to calculate the pure histograms for the 2 point anisotropic correlation function WITH 
    boundary periodic conditions. It stores the counts in the XX histogram.

    args:
    XX: (double*) The histogram where the distances are counted.
    elements: (PointW3D*) Array of the points ordered coherently with the nodes.
    nodeD: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node.
    nonzero_nodes: (int) Number of nonzero nodes where the points have been classificated.
    bn: (int) NUmber of bins in the XY histogram.
    dmax: (float) The maximum distance of interest between points.
    d_max_node: (float) The maximum internodal distance.
    size_box: (float) The size of the box where the points were contained. It is used for the boundary periodic conditions
    size_node: (float) Size of the nodes.
    */

    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<(nonzero_nodes+node_offset) && idx2<(nonzero_nodes+node_offset)){
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
        float dxn12=fabsf(nx2-nx1), dyn12=fabsf(ny2-ny1), dzn12=fabsf(nz2-nz1);
        float dd_nod12_ort = dxn12*dxn12 + dyn12*dyn12;
        float dd_nod12_z = dzn12*dzn12;
        
        float x1,y1,z1,w1,x2,y2,z2;
        float dd_z, dd_ort;
        float dx,dy,dz;
        int bnz, bnort, bin, end1=nodeD[idx1].end, end2=nodeD[idx2].end;
        double v;
        
        //Front vars
        float f_dmax = dmax+size_node;
        float _f_dmax = size_box - f_dmax;
        bool boundx = ((nx1<=f_dmax)&&(nx2>=_f_dmax))||((nx2<=f_dmax)&&(nx1>=_f_dmax));
        bool boundy = ((ny1<=f_dmax)&&(ny2>=_f_dmax))||((ny2<=f_dmax)&&(ny1>=_f_dmax));
        bool boundz = ((nz1<=f_dmax)&&(nz2>=_f_dmax))||((nz2<=f_dmax)&&(nz1>=_f_dmax));
        float f_dxn12, f_dyn12, f_dzn12;

        //Regular histogram calculation
        if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

            for (int i=nodeD[idx1].start; i<end1; ++i){
                x1 = elements[i].x;
                y1 = elements[i].y;
                z1 = elements[i].z;
                w1 = elements[i].w;
                for (int j=nodeD[idx2].start; j<end2; ++j){
                    x2 = elements[j].x;
                    y2 = elements[j].y;
                    z2 = elements[j].z;
                    dd_ort=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);
                    dd_z=(z2-z1)*(z2-z1);
                    if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                        
                        bnz = (int)(sqrt(dd_z)*ds)*bn;
                        if (bnz>(bn*(bn-1))) continue;
                        bnort = (int)(sqrtf(dd_ort)*ds);
                        if (bnort>(bn-1)) continue;
                        bin = bnz + bnort;
                        bin += bn*bn*bn_offset;

                        v = w1*elements[j].w;
                        atomicAdd(&XX[bin],v);
                    }
                }
            }
        }
        
        //Z front proyection
        if (boundz){
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = dxn12*dxn12+dyn12*dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }

        //Y front proyection
        if (boundy){
            f_dyn12 = size_box-dyn12;
            dd_nod12_ort = dxn12*dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dy = size_box-fabsf(y2-y1);
                        dd_ort=(x2-x1)*(x2-x1)+dy*dy;
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }
        
        //X front proyection
        if (boundx){
            f_dxn12 = size_box-dxn12;
            dd_nod12_ort = f_dxn12*f_dxn12+dyn12*dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dd_ort=dx*dx+(y2-y1)*(y2-y1);
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }
        
        //XY front proyection
        if (boundx && boundy){
            f_dxn12 = size_box-dxn12;
            f_dyn12 = size_box-dyn12;
            dd_nod12_ort = f_dxn12*f_dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        dd_ort=dx*dx+dy*dy;
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }

                
        //XZ front proyection
        if (boundx && boundz){
            f_dxn12 = size_box-dxn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = f_dxn12*f_dxn12+dyn12*dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=dx*dx+(y2-y1)*(y2-y1);
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }
        
        //YZ front proyection
        if (boundy && boundz){
            f_dyn12 = size_box-dyn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = dxn12*dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=(x2-x1)*(x2-x1)+dy*dy;
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }
        
        //XYZ front proyection
        if (boundx && boundy && boundz){
            f_dxn12 = size_box-dxn12;
            f_dyn12 = size_box-dyn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = f_dxn12*f_dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elements[i].x;
                    y1 = elements[i].y;
                    z1 = elements[i].z;
                    w1 = elements[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elements[j].x;
                        y2 = elements[j].y;
                        z2 = elements[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=dx*dx+dy*dy;
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_z>0 && dd_ort<=dd_max && dd_ort>0){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elements[j].w;
                            atomicAdd(&XX[bin],v);
                        }
                    }
                }
            }
        }

    }
}

__global__ void XY2ani_BPC(double *XY, PointW3D *elementsD, DNode *nodeD, int nonzero_Dnodes, PointW3D *elementsR,  DNode *nodeR, int nonzero_Rnodes, int bn, float dmax, float d_max_node, float size_box, float size_node, int node_offset, int bn_offset){
    /*
    Kernel function to calculate the mixed histograms for the 2 point anisotropic correlation function with 
    boundary periodic conditions. It stores the counts in the XY histogram.

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
    size_box: (float) The size of the box where the points were contained. It is used for the boundary periodic conditions
    size_node: (float) Size of the nodes.
    */
    
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    if (idx1<nonzero_Dnodes && idx2<(nonzero_Rnodes+node_offset)){
        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeR[idx2].nodepos.x, ny2=nodeR[idx2].nodepos.y, nz2=nodeR[idx2].nodepos.z;
        float dxn12=fabsf(nx2-nx1), dyn12=fabsf(ny2-ny1), dzn12=fabsf(nz2-nz1);
        float dd_nod12_ort = dxn12*dxn12+dyn12*dyn12;
        float dd_nod12_z = dzn12*dzn12;
        float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
        
        float x1,y1,z1,w1,x2,y2,z2;
        float dx,dy,dz;
        float dd_z, dd_ort;
        int bnz, bnort, bin, end1=nodeD[idx1].end, end2=nodeR[idx2].end;
        double v;
        
        //Front vars
        float f_dmax = dmax+size_node;
        float _f_dmax = size_box - f_dmax;
        bool boundx = ((nx1<=f_dmax)&&(nx2>=_f_dmax))||((nx2<=f_dmax)&&(nx1>=_f_dmax));
        bool boundy = ((ny1<=f_dmax)&&(ny2>=_f_dmax))||((ny2<=f_dmax)&&(ny1>=_f_dmax));
        bool boundz = ((nz1<=f_dmax)&&(nz2>=_f_dmax))||((nz2<=f_dmax)&&(nz1>=_f_dmax));
        float f_dxn12, f_dyn12, f_dzn12;

        //Regular no BPC counting
        if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

            for (int i=nodeD[idx1].start; i<end1; ++i){
                x1 = elementsD[i].x;
                y1 = elementsD[i].y;
                z1 = elementsD[i].z;
                for (int j=nodeR[idx2].start; j<end2; ++j){
                    x2 = elementsR[j].x;
                    y2 = elementsR[j].y;
                    z2 = elementsR[j].z;
                    dd_ort=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);
                    dd_z=(z2-z1)*(z2-z1);
                    if (dd_z<=dd_max && dd_ort<=dd_max){
                        
                        bnz = (int)(sqrt(dd_z)*ds)*bn;
                        if (bnz>(bn*(bn-1))) continue;
                        bnort = (int)(sqrtf(dd_ort)*ds);
                        if (bnort>(bn-1)) continue;
                        bin = bnz + bnort;
                        bin += bn*bn*bn_offset;

                        v = w1*elementsR[j].w;
                        atomicAdd(&XY[bin],v);
                    }
                }
            }
        }

                
        //Z front proyection
        if (boundz){
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = dxn12*dxn12+dyn12*dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1);
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }

        //Y front proyection
        if (boundy){
            f_dyn12 = size_box-dyn12;
            dd_nod12_ort = dxn12*dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dy = size_box-fabsf(y2-y1);
                        dd_ort=(x2-x1)*(x2-x1)+dy*dy;
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }
        
        //X front proyection
        if (boundx){
            f_dxn12 = size_box-dxn12;
            dd_nod12_ort = f_dxn12*f_dxn12+dyn12*dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dd_ort=dx*dx+(y2-y1)*(y2-y1);
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }
        
        //XY front proyection
        if (boundx && boundy){
            f_dxn12 = size_box-dxn12;
            f_dyn12 = size_box-dyn12;
            dd_nod12_ort = f_dxn12*f_dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = dzn12*dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        dd_ort=dx*dx+dy*dy;
                        dd_z=(z2-z1)*(z2-z1);
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }

                
        //XZ front proyection
        if (boundx && boundz){
            f_dxn12 = size_box-dxn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = f_dxn12*f_dxn12+dyn12*dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=dx*dx+(y2-y1)*(y2-y1);
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }
        
        //YZ front proyection
        if (boundy && boundz){
            f_dyn12 = size_box-dyn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = dxn12*dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=(x2-x1)*(x2-x1)+dy*dy;
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }
        
        //XYZ front proyection
        if (boundx && boundy && boundz){
            f_dxn12 = size_box-dxn12;
            f_dyn12 = size_box-dyn12;
            f_dzn12 = size_box-dzn12;
            dd_nod12_ort = f_dxn12*f_dxn12+f_dyn12*f_dyn12;
            dd_nod12_z = f_dzn12*f_dzn12;
            if (dd_nod12_ort <= d_max_node && dd_nod12_z <= d_max_node){

                for (int i=nodeD[idx1].start; i<end1; ++i){
                    x1 = elementsD[i].x;
                    y1 = elementsD[i].y;
                    z1 = elementsD[i].z;
                    w1 = elementsD[i].w;
                    for (int j=nodeD[idx2].start; j<end2; ++j){
                        x2 = elementsR[j].x;
                        y2 = elementsR[j].y;
                        z2 = elementsR[j].z;
                        dx = size_box-fabsf(x2-x1);
                        dy = size_box-fabsf(y2-y1);
                        dz = size_box-fabsf(z2-z1);
                        dd_ort=dx*dx+dy*dy;
                        dd_z=dz*dz;
                        if (dd_z<=dd_max && dd_ort<=dd_max){
                            
                            bnz = (int)(sqrt(dd_z)*ds)*bn;
                            if (bnz>(bn*(bn-1))) continue;
                            bnort = (int)(sqrtf(dd_ort)*ds);
                            if (bnort>(bn-1)) continue;
                            bin = bnz + bnort;
                            bin += bn*bn*bn_offset;

                            v = w1*elementsR[j].w;
                            atomicAdd(&XY[bin],v);
                        }
                    }
                }
            }
        }

    }
}
