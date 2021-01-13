
#include <math.h>

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__device__ void count123_ani(double *XXX, PointW3D *elements, int start1, int end1, int start2, int end2, int start3, int end3, int bn, double ds, float ds_th, float dd_max, float size_box, bool fx_2, bool fy_2, bool fz_2, bool fx_3, bool fy_3, bool fz_3, int bn_offset=0){
    /*
    Device function only callable from GPU.

    Given that three nodes are closer to each other than the maximum distance this function counts all the possible triangles 
    with one point in each node. If the triangles within one nodes are required simply pass the same node three times. This 
    function can compute the trieangles from nodes where one or two of the nodes are proyected to a boundary. It considers 
    only one kind of nodes, either data nodes or random nodes but not both.

    args:
    XXX: (double*) The histogram where the distances are counted.
    elements: (PointW3D*) Array of the points ordered coherently with the nodes.
    start1: (int) Index where the points of the node 1 start. Locatedin the elements array.
    end1: (int) Index where the points of the node 1 end. Locatedin the elements array.
    start2: (int) Index where the points of the node 2 start. Locatedin the elements array.
    end2: (int) Index where the points of the node 2 end. Locatedin the elements array.
    start3: (int) Index where the points of the node 3 start. Locatedin the elements array.
    end3: (int) Index where the points of the node 3 end. Locatedin the elements array.
    bn: (int) NUmber of bins in the XY histogram.
    ds: (double) Constant to calculate the bin index where the triangle count will be stored.
    ddmax: (float) The square of the maximum distance of interest between points.
    size_box: (float) The size of the box where the points were contained. It is used to calculate the proyected nodes.
    fx_2: (bool) True if the node number 2 is proyected in the x direction.
    fy_2: (bool) True if the node number 2 is proyected in the y direction.
    fz_2: (bool) True if the node number 2 is proyected in the z direction.
    fx_3: (bool) True if the node number 3 is proyected in the x direction.
    fy_3: (bool) True if the node number 3 is proyected in the y direction.
    fz_3: (bool) True if the node number 3 is proyected in the z direction.
    bn_offset: (int) Number of randomfile if there are many random files
    */
    int a, b, c, t, p, bin;
    float x1,y1,z1,w1,x2,y2,z2,w2,x3,y3,z3;
    float dx12, dy12, dz12, dx23, dy23, dz23, dx31, dy31, dz31;
    double d12,d23,d31, cth12, cth31;
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
                d12 = sqrt(d12);
                for (int k=start3; k<end3; k++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dx23 = fabsf(x3-x2) - fabsf(size_box*fx_2 - size_box*fx_3);
                    dy23 = fabsf(y3-y2) - fabsf(size_box*fy_2 - size_box*fy_3);
                    dz23 = fabsf(z3-z2) - fabsf(size_box*fz_2 - size_box*fz_3);
                    d23 = dx23*dx23 + dy23*dy23 + dz23*dz23;
                    if (d23 < dd_max && d23>0){
                        dx31 = fabsf(x3-x1) - size_box*fx_3;
                        dy31 = fabsf(y3-y1) - size_box*fy_3;
                        dz31 = fabsf(z3-z1) - size_box*fz_3;
                        d31 = dx31*dx31 + dy31*dy31 + dz31*dz31;
                        if (d31 < dd_max && d31>0){

                            d23 = sqrt(d23);
                            d31 = sqrt(d31);
                            cth12 = 1 + dz12/d12;
                            cth31 = 1 + dz31/d31;
                                                                        
                            // Indices 
                            a = (int) (d12*ds);
                            if (a>(bn-1)) continue;
                            b = (int) (d31*ds);
                            if (b>(bn-1)) continue;
                            c = (int) (d23*ds);
                            if (c>(bn-1)) continue;
                            t = (int) (cth12*ds_th);
                            if (t>(bn-1)) continue;
                            p = (int) (cth31*ds_th);
                            if (p>(bn-1)) continue;

                            //Atomic add
                            bin = a*bn*bn*bn*bn + b*bn*bn*bn + c*bn*bn + t*bn + p;
                            bin += bn_offset*bn*bn*bn*bn*bn;

                            v *= elements[k].w;
                            atomicAdd(&XXX[bin],v);

                        }
                    }
                }
            }
        }
    }
}
__device__ void count123_animixed(double *XXY, PointW3D *elementsX, PointW3D *elementsY, int start1, int end1, int start2, int end2, int start3, int end3, int bn, double ds, float ds_th, float dd_max, float size_box, bool fx_2, bool fy_2, bool fz_2, bool fx_3, bool fy_3, bool fz_3, int bn_offset=0){
    /*
    Device function only callable from GPU.

    Given that three nodes are closer to each other than the maximum distance this function counts all the possible triangles 
    with one point in each node. If the triangles within one nodes are required simply pass the same node three times. This 
    function can compute the trieangles from nodes where one or two of the nodes are proyected to a boundary. It considers 
    only one kind of nodes, either data nodes or random nodes but not both.

    args:
    XXY: (double*) The histogram where the distances are counted.
    elementsX: (PointW3D*) Array of the points ordered coherently with the X nodes.
    elementsY: (PointW3D*) Array of the points ordered coherently with the Y nodes.
    start1: (int) Index where the points of the node 1 start. Locatedin the elements array.
    end1: (int) Index where the points of the node 1 end. Locatedin the elements array.
    start2: (int) Index where the points of the node 2 start. Locatedin the elements array.
    end2: (int) Index where the points of the node 2 end. Locatedin the elements array.
    start3: (int) Index where the points of the node 3 start. Locatedin the elements array.
    end3: (int) Index where the points of the node 3 end. Locatedin the elements array.
    bn: (int) NUmber of bins in the XY histogram.
    ds: (double) Constant to calculate the bin index where the triangle count will be stored.
    ddmax: (float) The square of the maximum distance of interest between points.
    size_box: (float) The size of the box where the points were contained. It is used to calculate the proyected nodes.
    fx_2: (bool) True if the node number 2 is proyected in the x direction.
    fy_2: (bool) True if the node number 2 is proyected in the y direction.
    fz_2: (bool) True if the node number 2 is proyected in the z direction.
    fx_3: (bool) True if the node number 3 is proyected in the x direction.
    fy_3: (bool) True if the node number 3 is proyected in the y direction.
    fz_3: (bool) True if the node number 3 is proyected in the z direction.
    bn_offset: (int) Number of randomfile if there are many random files
    */

    int a, b, c, t, p, bin;
    float x1,y1,z1,w1,x2,y2,z2,w2,x3,y3,z3;
    float dx12, dy12, dz12, dx23, dy23, dz23, dx31, dy31, dz31;
    double d12,d23,d31, cth12, cth31;
    double v;

    for (int i=start1; i<end1; i++){
        //Node 1 is never proyected
        x1 = elementsX[i].x;
        y1 = elementsX[i].y;
        z1 = elementsX[i].z;
        w1 = elementsX[i].w;
        for (int j=start2; j<end2; j++){
            x2 = elementsX[j].x;
            y2 = elementsX[j].y;
            z2 = elementsX[j].z;
            w2 = elementsX[j].w;
            dx12 = fabsf(x2-x1) - size_box*fx_2;
            dy12 = fabsf(y2-y1) - size_box*fy_2;
            dz12 = fabsf(z2-z1) - size_box*fz_2;
            d12 = dx12*dx12 + dy12*dy12 + dz12*dz12;
            if (d12 < dd_max && d12>0){
                v = w1*w2;
                d12 = sqrt(d12);
                for (int k=start3; k<end3; k++){
                    x3 = elementsY[k].x;
                    y3 = elementsY[k].y;
                    z3 = elementsY[k].z;
                    dx23 = fabsf(x3-x2) - fabsf(size_box*fx_2 - size_box*fx_3);
                    dy23 = fabsf(y3-y2) - fabsf(size_box*fy_2 - size_box*fy_3);
                    dz23 = fabsf(z3-z2) - fabsf(size_box*fz_2 - size_box*fz_3);
                    d23 = dx23*dx23 + dy23*dy23 + dz23*dz23;
                    if (d23 < dd_max && d23>0){
                        dx31 = fabsf(x3-x1) - size_box*fx_3;
                        dy31 = fabsf(y3-y1) - size_box*fy_3;
                        dz31 = fabsf(z3-z1) - size_box*fz_3;
                        d31 = dx31*dx31 + dy31*dy31 + dz31*dz31;
                        if (d31 < dd_max && d31>0){
                            d23 = sqrt(d23);
                            d31 = sqrt(d31);
                            cth12 = 1 + dz12/d12;
                            cth31 = 1 + dz31/d31;
                                                                        
                            // Indices 
                            a = (int) (d12*ds);
                            if (a>(bn-1)) continue;
                            b = (int) (d31*ds);
                            if (b>(bn-1)) continue;
                            c = (int) (d23*ds);
                            if (c>(bn-1)) continue;
                            t = (int) (cth12*ds_th);
                            if (t>(bn-1)) continue;
                            p = (int) (cth31*ds_th);
                            if (p>(bn-1)) continue;

                            //Atomic add
                            bin = a*bn*bn*bn*bn + b*bn*bn*bn + c*bn*bn + t*bn + p;
                            bin += bn_offset*bn*bn*bn*bn*bn;

                            v *= elementsY[k].w;
                            atomicAdd(&XXY[bin],v);
                        }
                    }
                }
            }
        }
    }
}

//DDD pure histogram
__global__ void XXX3ani_BPC(double *XXX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, float size_box, float size_node, int node_offset=0, int bn_offset=0){
    /*
    Kernel function to calculate the pure histograms for the 3 point isotropic correlation function. 
    This version does considers boudary periodic conditions. It stores the counts in the XXX histogram.

    args:
    XXX: (double*) The histogram where the distances are counted.
    elementsX: (PointW3D*) Array of the points ordered coherently with the nodes. For the X points.
    nodeX: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node. For the X points
    nonzero_Xnodes: (int) Number of nonzero nodes where the points have been classificated. For the X points
    bn: (int) NUmber of bins in the XY histogram.
    dmax: (float) The maximum distance of interest between points.
    d_max_node: (float) The maximum internodal distance.
    size_box: (float) The size of the box where the points were contained. It is used for the boundary periodic conditions
    size_node: (float) Size of the nodes.
    */

    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = node_offset + blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<(nonzero_nodes+node_offset) && idx2<(nonzero_nodes+node_offset) && idx3<(nonzero_nodes+node_offset)){
        int end1 = nodeD[idx1].end, end2 = nodeD[idx2].end, end3 = nodeD[idx3].end;
        int start1 = nodeD[idx1].start, start2 = nodeD[idx2].start, start3 = nodeD[idx3].start;
        float dd_max=dmax*dmax, ds_th = (float)(bn)/2;
        double ds = floor(((double)(bn)/dmax)*1000000)/1000000;

        float nx1=nodeD[idx1].nodepos.x, ny1=nodeD[idx1].nodepos.y, nz1=nodeD[idx1].nodepos.z;
        float nx2=nodeD[idx2].nodepos.x, ny2=nodeD[idx2].nodepos.y, nz2=nodeD[idx2].nodepos.z;
        float nx3=nodeD[idx3].nodepos.x, ny3=nodeD[idx3].nodepos.y, nz3=nodeD[idx3].nodepos.z;
        float dxn12 = fabsf(nx2-nx1), dyn12 = fabsf(ny2-ny1), dzn12 = fabsf(nz2-nz1);
        float dxn23 = fabsf(nx3-nx2), dyn23 = fabsf(ny3-ny2), dzn23 = fabsf(nz3-nz2);
        float dxn31 = fabsf(nx1-nx3), dyn31 = fabsf(ny1-ny3), dzn31 = fabsf(nz1-nz3);
        float dd_nod12 = dxn12*dxn12+dyn12*dyn12+dzn12*dzn12;
        float dd_nod23 = dxn23*dxn23+dyn23*dyn23+dzn23*dzn23;
        float dd_nod31 = dxn31*dxn31+dyn31*dyn31+dzn31*dzn31;

        //Front vars
        float f_dmax = dmax+size_node;
        float size_box2=size_box*size_box, _f_dmax = size_box - f_dmax;
        bool fx_2 = ((nx1<=f_dmax)&&(nx2>=_f_dmax))||((nx2<=f_dmax)&&(nx1>=_f_dmax));
        bool fy_2 = ((ny1<=f_dmax)&&(ny2>=_f_dmax))||((ny2<=f_dmax)&&(ny1>=_f_dmax));
        bool fz_2 = ((nz1<=f_dmax)&&(nz2>=_f_dmax))||((nz2<=f_dmax)&&(nz1>=_f_dmax));
        bool fx_3 = ((nx1<=f_dmax)&&(nx3>=_f_dmax))||((nx3<=f_dmax)&&(nx1>=_f_dmax));;
        bool fy_3 = ((ny1<=f_dmax)&&(ny3>=_f_dmax))||((ny3<=f_dmax)&&(ny1>=_f_dmax));;
        bool fz_3 = ((nz1<=f_dmax)&&(nz3>=_f_dmax))||((nz3<=f_dmax)&&(nz1>=_f_dmax));;
        float f_dd_nod12, f_dd_nod23, f_dd_nod31;

        //No proyection
        if (dd_nod12 <= d_max_node && dd_nod23 <= d_max_node && dd_nod31 <= d_max_node){
            //Regular counting. No BPC
            count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, false, false, false, bn_offset);
        }

        //============ Only node 3 proyections ================
        if (dd_nod12 <= d_max_node && (fx_3 || fy_3 || fz_3)){
            //x proyection
            if (fx_3){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dxn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, true, false, false, bn_offset);
                    }
                }
            }
            //y proyection
            if (fy_3){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dyn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, false, true, false, bn_offset);
                    }
                }
            }
            //z proyection
            if (fz_3){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dzn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, false, false, true, bn_offset);
                    }
                }
            }
            //xy proyection
            if (fx_3 && fy_3){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, true, true, false, bn_offset);
                    }
                }
            }
            //xz proyection
            if (fx_3 && fz_3){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, true, false, true, bn_offset);
                    }
                }
            }
            //yz proyection
            if (fy_3 && fz_3){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, false, true, true, bn_offset);
                    }
                }
            }
            //xyz proyection
            if (fx_3 && fy_3 && fz_3){
                f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, true, true, true, bn_offset);
                    }
                }
            }
        }
        
        //============ Only node 2 proyections ================
        
        if (dd_nod31 <= d_max_node && (fx_2 || fy_2 || fz_2)){
            //x proyection
            if (fx_2){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dxn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + size_box2 - 2*dxn12*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, false, false, false, bn_offset);
                    }
                }
            }
            //y proyection
            if (fy_2){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dyn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + size_box2 - 2*dyn12*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, false, false, false, bn_offset);
                    }
                }
            }
            //z proyection
            if (fz_2){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dzn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + size_box2 - 2*dzn12*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, false, false, false, bn_offset);
                    }
                }
            }
            //xy proyection
            if (fx_2 && fy_2){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dyn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, false, false, false, bn_offset);
                    }
                }
            }
            //xz proyection
            if (fx_2 && fz_2){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dzn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, false, false, false, bn_offset);
                    }
                }
            }
            //yz proyection
            if (fy_2 && fz_2){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dyn12 + dzn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, false, false, false, bn_offset);
                    }
                }
            }
            //xyz proyection
            if (fx_2 && fy_2 && fz_2){
                f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 3*size_box2 - 2*(dxn12 + dyn12 + dzn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, false, false, false, bn_offset);
                    }
                }
            }
        }
        
        //============ Both nodes are proyected ===============
        if ((fx_2 || fy_2 || fz_2) && (fx_3 || fy_3 || fz_3)){
            
            //======== Both nodes are proyected the same ========
            if (dd_nod23 <= d_max_node){
                //x proyection
                if (fx_2 && fx_3){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + size_box2 - 2*dxn12*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, true, false, false, bn_offset);
                        }
                    }
                }
                //y proyection
                if (fy_2 && fy_3){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + size_box2 - 2*dyn12*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, false, true, false, bn_offset);
                        }
                    }
                }
                //z proyection
                if (fz_2 && fz_3){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + size_box2 - 2*dzn12*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, false, false, true, bn_offset);
                        }
                    }
                }
                //xy proyection
                if (fx_2 && fx_3 && fy_2 && fy_3){
                    f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dyn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, true, true, false, bn_offset);
                        }
                    }
                }
                //xz proyection
                if (fx_2 && fx_3 && fz_2 && fz_3){
                    f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dzn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, true, false, true, bn_offset);
                        }
                    }
                }
                //yz proyection
                if (fy_2 && fy_3 && fz_2 && fz_3){
                    f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dyn12 + dzn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, false, true, true, bn_offset);
                        }
                    }
                }
                //xyz proyection
                if (fx_2 && fx_3 && fy_2 && fy_3 && fz_2 && fz_3){
                    f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 3*size_box2 - 2*(dxn12 + dyn12 + dzn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, true, true, true, bn_offset);
                        }
                    }
                }
            }

            //====== Both nodes are proyected differently =======
            //node 2 x proyection
            if (fx_2){
                f_dd_nod12 = dd_nod12 + size_box2 - 2*dxn12*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, false, true, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //node 2 y proyection
            if (fy_2){
                f_dd_nod12 = dd_nod12 + size_box2 - 2*dyn12*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, false, true, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //node 2 z proyection
            if (fz_2){
                f_dd_nod12 = dd_nod12 + size_box2 - 2*dzn12*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, false, true, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //xy proyection
            if (fx_2 && fy_2){
                f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dyn12)*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dzn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, false, true, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //node 2 xz proyection
            if (fx_2 && fz_2){
                f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dzn12)*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, false, true, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //node 2 yz proyection
            if (fy_2 && fz_2){
                f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dzn12 + dyn12)*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //node 2 xyz proyection
            if (fx_2 && fy_2 && fz_2){
                f_dd_nod12 = dd_nod12 + 3*size_box2 - 2*(dxn12 + dyn12 + dzn12)*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_ani(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, false, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
        }

    }
}

__global__ void XXY3ani_BPC(double *XXY, PointW3D *elementsX, DNode *nodeX, int nonzero_Xnodes, PointW3D *elementsY, DNode *nodeY, int nonzero_Ynodes, int bn, float dmax, float d_max_node, float size_box, float size_node, int node_offset, int bn_offset, bool isDDR){
    /*
    Kernel function to calculate the mixed histograms for the 3 point isotropic correlation function. 
    This version does considers boudary periodic conditions. It stores the counts in the XXY histogram.

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
    size_box: (float) The size of the box where the points were contained. It is used for the boundary periodic conditions
    size_node: (float) Size of the nodes.
    isDDR: (bool) To know if the XX or Y node may have many entries from many random nodes
    */

    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx1 = (!isDDR)*node_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = (!isDDR)*node_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = isDDR*node_offset + blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx1<(nonzero_Xnodes + (!isDDR)*node_offset) && idx2<(nonzero_Xnodes + (!isDDR)*node_offset) && idx3<(nonzero_Ynodes + isDDR*node_offset)){
        int end1 = nodeX[idx1].end, end2 = nodeX[idx2].end, end3 = nodeY[idx3].end;
        int start1 = nodeX[idx1].start, start2 = nodeX[idx2].start, start3 = nodeY[idx3].start;
        float dd_max=dmax*dmax, ds_th = (float)(bn)/2;
        double ds = floor(((double)(bn)/dmax)*1000000)/1000000;

        float nx1=nodeX[idx1].nodepos.x, ny1=nodeX[idx1].nodepos.y, nz1=nodeX[idx1].nodepos.z;
        float nx2=nodeX[idx2].nodepos.x, ny2=nodeX[idx2].nodepos.y, nz2=nodeX[idx2].nodepos.z;
        float nx3=nodeY[idx3].nodepos.x, ny3=nodeY[idx3].nodepos.y, nz3=nodeY[idx3].nodepos.z;
        float dxn12 = fabsf(nx2-nx1), dyn12 = fabsf(ny2-ny1), dzn12 = fabsf(nz2-nz1);
        float dxn23 = fabsf(nx3-nx2), dyn23 = fabsf(ny3-ny2), dzn23 = fabsf(nz3-nz2);
        float dxn31 = fabsf(nx1-nx3), dyn31 = fabsf(ny1-ny3), dzn31 = fabsf(nz1-nz3);
        float dd_nod12 = dxn12*dxn12+dyn12*dyn12+dzn12*dzn12;
        float dd_nod23 = dxn23*dxn23+dyn23*dyn23+dzn23*dzn23;
        float dd_nod31 = dxn31*dxn31+dyn31*dyn31+dzn31*dzn31;

        //Front vars
        float f_dmax = dmax+size_node;
        float size_box2=size_box*size_box, _f_dmax = size_box - f_dmax;
        bool fx_2 = ((nx1<=f_dmax)&&(nx2>=_f_dmax))||((nx2<=f_dmax)&&(nx1>=_f_dmax));
        bool fy_2 = ((ny1<=f_dmax)&&(ny2>=_f_dmax))||((ny2<=f_dmax)&&(ny1>=_f_dmax));
        bool fz_2 = ((nz1<=f_dmax)&&(nz2>=_f_dmax))||((nz2<=f_dmax)&&(nz1>=_f_dmax));
        bool fx_3 = ((nx1<=f_dmax)&&(nx3>=_f_dmax))||((nx3<=f_dmax)&&(nx1>=_f_dmax));;
        bool fy_3 = ((ny1<=f_dmax)&&(ny3>=_f_dmax))||((ny3<=f_dmax)&&(ny1>=_f_dmax));;
        bool fz_3 = ((nz1<=f_dmax)&&(nz3>=_f_dmax))||((nz3<=f_dmax)&&(nz1>=_f_dmax));;
        float f_dd_nod12, f_dd_nod23, f_dd_nod31;

        //No proyection
        if (dd_nod12 <= d_max_node && dd_nod23 <= d_max_node && dd_nod31 <= d_max_node){
            //Regular counting. No BPC
            count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, false, false, false, bn_offset);
        }

        //============ Only node 3 proyections ================
        if (dd_nod12 <= d_max_node && (fx_3 || fy_3 || fz_3)){
            //x proyection
            if (fx_3){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dxn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, true, false, false, bn_offset);
                    }
                }
            }
            //y proyection
            if (fy_3){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dyn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, false, true, false, bn_offset);
                    }
                }
            }
            //z proyection
            if (fz_3){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dzn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, false, false, true, bn_offset);
                    }
                }
            }
            //xy proyection
            if (fx_3 && fy_3){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, true, true, false, bn_offset);
                    }
                }
            }
            //xz proyection
            if (fx_3 && fz_3){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, true, false, true, bn_offset);
                    }
                }
            }
            //yz proyection
            if (fy_3 && fz_3){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, false, true, true, bn_offset);
                    }
                }
            }
            //xyz proyection
            if (fx_3 && fy_3 && fz_3){
                f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, false, true, true, true, bn_offset);
                    }
                }
            }
        }
        
        //============ Only node 2 proyections ================
        
        if (dd_nod31 <= d_max_node && (fx_2 || fy_2 || fz_2)){
            //x proyection
            if (fx_2){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dxn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + size_box2 - 2*dxn12*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, false, false, false, bn_offset);
                    }
                }
            }
            //y proyection
            if (fy_2){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dyn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + size_box2 - 2*dyn12*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, false, false, false, bn_offset);
                    }
                }
            }
            //z proyection
            if (fz_2){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dzn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + size_box2 - 2*dzn12*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, false, false, false, bn_offset);
                    }
                }
            }
            //xy proyection
            if (fx_2 && fy_2){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dyn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, false, false, false, bn_offset);
                    }
                }
            }
            //xz proyection
            if (fx_2 && fz_2){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dzn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, false, false, false, bn_offset);
                    }
                }
            }
            //yz proyection
            if (fy_2 && fz_2){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dyn12 + dzn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, false, false, false, bn_offset);
                    }
                }
            }
            //xyz proyection
            if (fx_2 && fy_2 && fz_2){
                f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 3*size_box2 - 2*(dxn12 + dyn12 + dzn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, false, false, false, bn_offset);
                    }
                }
            }
        }
        
        //============ Both nodes are proyected ===============
        if ((fx_2 || fy_2 || fz_2) && (fx_3 || fy_3 || fz_3)){
            
            //======== Both nodes are proyected the same ========
            if (dd_nod23 <= d_max_node){
                //x proyection
                if (fx_2 && fx_3){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + size_box2 - 2*dxn12*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, true, false, false, bn_offset);
                        }
                    }
                }
                //y proyection
                if (fy_2 && fy_3){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + size_box2 - 2*dyn12*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, false, true, false, bn_offset);
                        }
                    }
                }
                //z proyection
                if (fz_2 && fz_3){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + size_box2 - 2*dzn12*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, false, false, true, bn_offset);
                        }
                    }
                }
                //xy proyection
                if (fx_2 && fx_3 && fy_2 && fy_3){
                    f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dyn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, true, true, false, bn_offset);
                        }
                    }
                }
                //xz proyection
                if (fx_2 && fx_3 && fz_2 && fz_3){
                    f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dzn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, true, false, true, bn_offset);
                        }
                    }
                }
                //yz proyection
                if (fy_2 && fy_3 && fz_2 && fz_3){
                    f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dyn12 + dzn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, false, true, true, bn_offset);
                        }
                    }
                }
                //xyz proyection
                if (fx_2 && fx_3 && fy_2 && fy_3 && fz_2 && fz_3){
                    f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 3*size_box2 - 2*(dxn12 + dyn12 + dzn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, true, true, true, bn_offset);
                        }
                    }
                }
            }

            //====== Both nodes are proyected differently =======
            //node 2 x proyection
            if (fx_2){
                f_dd_nod12 = dd_nod12 + size_box2 - 2*dxn12*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, false, true, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //node 2 y proyection
            if (fy_2){
                f_dd_nod12 = dd_nod12 + size_box2 - 2*dyn12*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, false, false, true, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, false, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //node 2 z proyection
            if (fz_2){
                f_dd_nod12 = dd_nod12 + size_box2 - 2*dzn12*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, false, true, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, false, true, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //xy proyection
            if (fx_2 && fy_2){
                f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dyn12)*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dzn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, false, true, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, false, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //node 2 xz proyection
            if (fx_2 && fz_2){
                f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dzn12)*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, false, true, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, false, true, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //node 2 yz proyection
            if (fy_2 && fz_2){
                f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dzn12 + dyn12)*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xyz proyection
                    if (fx_3 && fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, false, true, true, true, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
            //node 2 xyz proyection
            if (fx_2 && fy_2 && fz_2){
                f_dd_nod12 = dd_nod12 + 3*size_box2 - 2*(dxn12 + dyn12 + dzn12)*size_box;
                if (f_dd_nod12 <= d_max_node){
                    //node 3 x proyection
                    if (fx_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, true, false, false, bn_offset);
                            }
                        }
                    }
                    //node 3 y proyection
                    if (fy_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, false, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 z proyection
                    if (fz_3){
                        f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, false, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 xy proyection
                    if (fx_3 && fy_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dzn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, true, true, false, bn_offset);
                            }
                        }
                    }
                    //node 3 xz proyection
                    if (fx_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dyn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, true, false, true, bn_offset);
                            }
                        }
                    }
                    //node 3 yz proyection
                    if (fy_3 && fz_3){
                        f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                        if (f_dd_nod31 <= d_max_node){
                            f_dd_nod23 = dd_nod23 + size_box2 - 2*(dxn23)*size_box;
                            if (f_dd_nod23 <= d_max_node){
                                count123_animixed(XXY, elementsX, elementsY, start1, end1, start2, end2, start3, end3, bn, ds, ds_th, dd_max, size_box, true, true, true, false, true, true, bn_offset);
                            }
                        }
                    }
                }
            }
        }

    }
}