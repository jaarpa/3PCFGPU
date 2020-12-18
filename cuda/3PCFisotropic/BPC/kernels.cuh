
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

__global__ void make_histoXXX(double *XXX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, float dmax, float d_max_node, float size_box, float size_node){
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
            count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, false, false, false, false, false);
        }

        //============ Only node 3 proyections ================
        if (dd_nod12 <= d_max_node && (fx_3 || fy_3 || fz_3)){
            //x proyection
            if (fx_3){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dxn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dxn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, false, false, true, false, false);
                    }
                }
            }
            //y proyection
            if (fy_3){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dyn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, false, false, false, true, false);
                    }
                }
            }
            //z proyection
            if (fz_3){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dzn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, false, false, false, false, true);
                    }
                }
            }
            //xy proyection
            if (fx_3 && fy_3){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dyn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, false, false, true, true, false);
                    }
                }
            }
            //xz proyection
            if (fx_3 && fz_3){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dxn31 + dzn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, false, false, true, false, true);
                    }
                }
            }
            //yz proyection
            if (fy_3 && fz_3){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 2*size_box2 - 2*(dyn31 + dzn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, false, false, false, true, true);
                    }
                }
            }
            //xyz proyection
            if (fx_3 && fy_3 && fz_3){
                f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod31 = dd_nod31 + 3*size_box2 - 2*(dxn31 + dyn31 + dzn31)*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, false, false, true, true, true);
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
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, true, false, false, false, false, false);
                    }
                }
            }
            //y proyection
            if (fy_2){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dyn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + size_box2 - 2*dyn12*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, true, false, false, false, false);
                    }
                }
            }
            //z proyection
            if (fz_2){
                f_dd_nod23 = dd_nod23 + size_box2 - 2*dzn23*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + size_box2 - 2*dzn12*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, false, true, false, false, false);
                    }
                }
            }
            //xy proyection
            if (fx_2 && fy_2){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dyn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, true, true, false, false, false, false);
                    }
                }
            }
            //xz proyection
            if (fx_2 && fz_2){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dzn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, true, false, true, false, false, false);
                    }
                }
            }
            //yz proyection
            if (fy_2 && fz_2){
                f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dyn12 + dzn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, true, true, false, false, false);
                    }
                }
            }
            //xyz proyection
            if (fx_2 && fy_2 && fz_2){
                f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                if (f_dd_nod23 <= d_max_node){
                    f_dd_nod12 = dd_nod12 + 3*size_box2 - 2*(dxn12 + dyn12 + dzn12)*size_box;
                    if (f_dd_nod12 <= d_max_node){
                        count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, true, true, true, false, false, false);
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
                            count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, true, false, false, true, false, false);
                        }
                    }
                }
                //y proyection
                if (fy_2 && fy_3){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dyn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + size_box2 - 2*dyn12*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, true, false, false, true, false);
                        }
                    }
                }
                //z proyection
                if (fz_2 && fz_3){
                    f_dd_nod31 = dd_nod31 + size_box2 - 2*dzn31*size_box;
                    if (f_dd_nod31 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + size_box2 - 2*dzn12*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, false, true, false, false, true);
                        }
                    }
                }
                //xy proyection
                if (fx_2 && fx_3 && fy_2 && fy_3){
                    f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dyn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dyn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, true, true, false, true, true, false);
                        }
                    }
                }
                //xz proyection
                if (fx_2 && fx_3 && fz_2 && fz_3){
                    f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dxn23 + dzn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dxn12 + dzn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, true, false, true, true, false, true);
                        }
                    }
                }
                //yz proyection
                if (fy_2 && fy_3 && fz_2 && fz_3){
                    f_dd_nod23 = dd_nod23 + 2*size_box2 - 2*(dyn23 + dzn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 2*size_box2 - 2*(dyn12 + dzn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, false, true, true, false, true, true);
                        }
                    }
                }
                //xyz proyection
                if (fx_2 && fx_3 && fy_2 && fy_3 && fz_2 && fz_3){
                    f_dd_nod23 = dd_nod23 + 3*size_box2 - 2*(dxn23 + dyn23 + dzn23)*size_box;
                    if (f_dd_nod23 <= d_max_node){
                        f_dd_nod12 = dd_nod12 + 3*size_box2 - 2*(dxn12 + dyn12 + dzn12)*size_box;
                        if (f_dd_nod12 <= d_max_node){
                            count123(XXX, elements, start1, end1, start2, end2, start3, end3, bn, ds, dd_max, size_box, true, true, true, true, true, true);
                        }
                    }
                }
            }

        }

    }
}