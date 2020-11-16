// nvcc iso2PCF.cu -o par.out && ./par.out data_5K.dat rand0_5K.dat 5000 30 180
#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>

using namespace std;

//Point with weight value. Structure
struct PointW3D{
    float x;
    float y; 
    float z;
    float w;
};

struct Node{
    int len;		// Number of points in the node
    PointW3D *elements;	// Points in the node
};


void open_files(string name_file, int pts, PointW3D *datos){
    /* Opens the daya files. Receives the file location, number of points to read and the array of points where the data is stored */
    ifstream file;

    string mypathto_files = "../../../fake_DATA/DATOS/";
    //This creates the full path to where I have my data files
    name_file.insert(0,mypathto_files);

    file.open(name_file.c_str(), ios::in | ios::binary); //Tells the program this is a binary file using ios::binary
    if (file.fail()){
        cout << "Failed to load the file in " << name_file << endl;
        exit(1);
    }

    for ( int c = 0; c < pts; c++) //Reads line by line and stores each c line in the c PointW3D element of the array
    {
        file >> datos[c].x >> datos[c].y >> datos[c].z >> datos[c].w; 
    }
    file.close();
}

//====================================================================

void save_histogram(string name, int bns, double *histo){
    /* This function saves a one dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */

    ofstream file2;
    file2.open(name.c_str(), ios::out | ios::binary);

    if (file2.fail()){
        cout << "Failed to save the the histogram in " << name << endl;
        exit(1);
    }
    for (int i = 0; i < bns; i++){
        file2 << histo[i] << endl;
    }
    file2.close();
}

//====================================================================

void save_histogram(string name, int bns, float *histo){
    /* This function saves a one dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */

    ofstream file2;
    file2.open(name.c_str(), ios::out | ios::binary);

    if (file2.fail()){
        cout << "Failed to save the the histogram in " << name << endl;
        exit(1);
    }
    for (int i = 0; i < bns; i++){
        file2 << histo[i] << endl;
    }
    file2.close();
}

//=================================================================== 
void add(PointW3D *&array, int &lon, float _x, float _y, float _z, float _w){
    /*
    This function manages adding points to an specific Node. It receives the previous array, longitude and point to add
    and updates the previous array and length with the same array with the new point at the end and adds +1 to the length +1

    It manages the memory allocation and free of the previous and new elements.
    */
    lon++;
    PointW3D *array_aux;
    cudaMallocManaged(&array_aux, lon*sizeof(PointW3D)); 
    for (int i=0; i<lon-1; i++){
        array_aux[i].x = array[i].x;
        array_aux[i].y = array[i].y;
        array_aux[i].z = array[i].z;
        array_aux[i].w = array[i].w;
    }

    cudaFree(array);
    array = array_aux;
    array[lon-1].x = _x;
    array[lon-1].y = _y;
    array[lon-1].z = _z;
    array[lon-1].w = _w;
}

void make_nodos(Node ***nod, PointW3D *dat, unsigned int partitions, float size_node, unsigned int np){
    /*
    This function classifies the data in the nodes

    Args
    nod: Node 3D array where the data will be classified
    dat: array of PointW3D data to be classified and stored in the nodes
    partitions: number nodes in each direction
    size_node: dimensions of a single node
    np: number of points in the dat array
    */

    int row, col, mom;

    // First allocate memory as an empty node:
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){
                nod[row][col][mom].len = 0;
                cudaMallocManaged(&nod[row][col][mom].elements, sizeof(PointW3D));
            }
        }
    }

    // Classificate the ith elment of the data into a node and add that point to the node with the add function:
    for (int i=0; i<np; i++){
        row = (int)(dat[i].x/size_node);
        col = (int)(dat[i].y/size_node);
        mom = (int)(dat[i].z/size_node);
        add(nod[row][col][mom].elements, nod[row][col][mom].len, dat[i].x, dat[i].y, dat[i].z, dat[i].w);
    }
}

//====================================================================
//============ Kernels Section ======================================= 
//====================================================================

__device__ void count_distances11(float *XX, PointW3D *elements, int len, float ds, float dd_max, int sum){
    /*
    This device function counts the distances betweeen points within one node.

    Args:
    XX: The histogram where the distances are counted in
    elements:  Array of PointW3D points inside the node
    len: lenght of the elements array
    ds: number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: The maximum distance of interest.
    */

    int bin;
    float d, v;
    float x1,y1,z1,w1,x2,y2,z2,w2;

    for (int i=0; i<len-1; ++i){
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        w1 = elements[i].w;
        for (int j=i+1; j<len; ++j){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            w2 = elements[j].w;
            d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (d<=dd_max+1){
                bin = (int)(sqrt(d)*ds);
                v = 2*w1*w2;
                atomicAdd(&XX[bin],sum);
            }
        }
    }
}

__device__ void count_distances12(float *XX, PointW3D *elements1, int len1, PointW3D *elements2, int len2, float ds, float dd_max, int sum){
    /*
    This device function counts the distances betweeen points between two different nodes.

    Args:
    XX: The histogram where the distances are counted in
    elements1:  Array of PointW3D points inside the first node
    len1: lenght of the first elements array
    elements2:  Array of PointW3D points inside the second node
    len2: lenght of the second elements array
    ds: number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: The maximum distance of interest.
    */

    int bin;
    float d, v;
    float x1,y1,z1,w1,x2,y2,z2,w2;

    for (int i=0; i<len1; ++i){
        x1 = elements1[i].x;
        y1 = elements1[i].y;
        z1 = elements1[i].z;
        w1 = elements1[i].w;
        for (int j=0; j<len2; ++j){
            x2 = elements2[j].x;
            y2 = elements2[j].y;
            z2 = elements2[j].z;
            w2 = elements2[j].w;
            d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (d<=dd_max+1){
                bin = (int)(sqrt(d)*ds);
                v = 2*w1*w2;
                atomicAdd(&XX[bin],sum);
            }
        }
    }
}

__device__ void BPC_loop(float *XX, Node ***nodeD, int row, int col, int mom, int partitions, int did_max, float dd_max, float ds, int sum, float size_box, bool x_border, bool y_border, bool z_border, bool x_upperborder, bool y_upperborder, bool z_upperborder, bool x_lowerborder, bool y_lowerborder, bool z_lowerborder){
    /*
    This device function counts the distances betweeen points between two different nodes from periodic boundary conditiojns.

    Args:
    XX: The histogram where the distances are counted in
    elements1:  Array of PointW3D points inside the first node
    len1: lenght of the first elements array
    elements2:  Array of PointW3D points inside the second node
    len2: lenght of the second elements array
    ds: number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    dd_max: The maximum distance of interest.
    */
    int bin, d_node, u, v, w, did_max2=did_max*did_max;
    float d, s;
    float x1,y1,z1,w1,dx12,dy12,dz12,w2;


    int x_from = ((row-did_max)*(row>did_max))*(!x_border) + (partitions-(did_max-row))*(x_lowerborder&&!x_upperborder);
    int x_to = (partitions-1)*((row+did_max>partitions-1 && !x_upperborder)||x_lowerborder) + (row+did_max)*((row+did_max<partitions)&&!x_border) + (!x_lowerborder&&x_upperborder)*(x_from+(did_max-(partitions-1-row)));
    int y_from = ((col-did_max)*(col>did_max))*(!y_border) + (partitions-(did_max-col))*(y_lowerborder&&!y_upperborder);
    int y_to = (partitions-1)*((col+did_max>partitions-1 && !y_upperborder)||y_lowerborder) + (col+did_max)*((col+did_max<partitions)&&!y_border) + (!y_lowerborder&&y_upperborder)*(y_from+(did_max-(partitions-1-col)));
    int z_from = ((mom-did_max)*(mom>did_max))*(!z_border) + (partitions-(did_max-mom))*(z_lowerborder&&!z_upperborder);
    int z_to = (partitions-1)*((mom+did_max>partitions-1 && !z_upperborder)||z_lowerborder) + (mom+did_max)*((mom+did_max<partitions)&&!z_border) + (!z_lowerborder&&z_upperborder)*(z_from+(did_max-(partitions-1-mom)));
    //If the z direction is not the nearest border the z index it is 0 if mom<did_max or mom-did-max otherwise.
    //If both z borders or ONLY the upper z border are the nearest borders the z index starts from 0
    //If ONLY the lower z border is the nearest the z index starts from partitions-(did_max-mom)
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //If both borders are the nearest the highest limit is partitions-1
    //If the lower border is the nearest the highes limit is partitions-1
    //If the upper border is not the nerarest and mom+did_max>partitions the highest limit is partitions-1
    //If this is not the border side and mom+did_max< paritions then the highest limit is mom+did_max
    //If only the upper border is the nearest border the higher limit is the lower limit + (did_max-(partitions-1-mom))

    for (u=x_from; u<=x_to; u++){
        for (v=y_from; v<=y_to; v++){
            for (w=z_from; w<=z_to; w++){
                d_node=(w-mom)*(w-mom) + (v-col)*(v-col) + (u-row)*(u-row);
                if (d_node<=did_max2){
                    for (int i=0; i<nodeD[row][col][mom].len; ++i){
                        x1 = nodeD[row][col][mom].elements[i].x;
                        y1 = nodeD[row][col][mom].elements[i].y;
                        z1 = nodeD[row][col][mom].elements[i].z;
                        w1 = nodeD[row][col][mom].elements[i].w;
                        for (int j=0; j<nodeD[u][v][w].len; ++j){
                            dx12 = fabsf(x1-nodeD[u][v][w].elements[j].x) - size_box*x_border;
                            dy12 = fabsf(y1-nodeD[u][v][w].elements[j].y) - size_box*y_border;
                            dz12 = fabsf(z1-nodeD[u][v][w].elements[j].z) - size_box*z_border;
                            w2 = nodeD[u][v][w].elements[j].w;
                            d = dx12*dx12 + dy12*dy12 + dz12*dz12;
                            if (d<=dd_max+1){
                                bin = (int)(sqrt(d)*ds);
                                s = 2*w1*w2;
                                atomicAdd(&XX[bin],sum);
                            }
                        }
                    }
                }
            }
        }
    }
}
__global__ void BPC_XX(float *XX_A, float *XX_B, Node ***nodeD, float ds, float d_max, float size_node, float size_box){
    /*
    This device function counts the distances betweeen points between a node and a node reproduction in the border.

    Args:
    XX: The histogram where the distances are counted in
    nodeD: Full array of nodes
    ds: number of bins divided by the maximum distance. Used to calculate the bin it should be counted at
    
    dd_max: The maximum distance of interest.
    did_max: maximum number of node between two nodes be considered
    did_max2: did_max*did_max
    size_box:  Size of the whole box
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int partitions = (int)(ceilf(size_box/size_node));
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon in this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;
        //printf("%i, %i, %i \n", mom, col,row)

        //This may see redundant but with this these often checked values are upgraded to device memory
        float dd_max = d_max*d_max;
        int did_max = (int)(ceilf((d_max+size_node*sqrt(3.0))/size_node));
        
        if (nodeD[row][col][mom].len > 0 && (row<did_max-1 || partitions-row<did_max || col<did_max-1 || partitions-col<did_max || mom<did_max-1 || partitions-mom<did_max)){
            //Only if the current node has elements and it is near to any border does the thread will be active
            bool x_border=false, y_border=false, z_border=false, x_upperborder=false, y_upperborder=false, z_upperborder=false, x_lowerborder=false, y_lowerborder=false, z_lowerborder=false;
            
            x_border=(row<did_max-1 || partitions-row<did_max);
            if (x_border){
                x_upperborder=partitions-row<did_max;
                x_lowerborder=row<did_max-1;
                BPC_loop(XX_A, nodeD, row, col, mom, partitions, did_max, dd_max, ds, 2, size_box, x_border, false, false, x_upperborder, false, false, x_lowerborder, false, false);
            }
            
            y_border=(col<did_max-1 || partitions-col<did_max);
            if (y_border){
                y_upperborder=partitions-col<did_max;
                y_lowerborder=col<did_max-1;

                //Only Y boundaries
                BPC_loop(XX_B, nodeD, row, col, mom, partitions, did_max, dd_max, ds, 2, size_box, false, y_border, false, false, y_upperborder, false, false, y_lowerborder, false); 
                if (x_border){
                    //Boundaries in the XY walls
                    BPC_loop(XX_A, nodeD, row, col, mom, partitions, did_max, dd_max, ds, 2, size_box, x_border, y_border, false, x_upperborder, y_upperborder, false, x_lowerborder, y_lowerborder, false);
                }
            }
            
            z_border=(mom<did_max-1 || partitions-mom<did_max);
            if (z_border){
                z_upperborder=partitions-mom<did_max;
                z_lowerborder=mom<did_max-1;
                
                //Only Z boundaries
                BPC_loop(XX_B, nodeD, row, col, mom, partitions, did_max, dd_max, ds, 2, size_box, false, false, z_border, false, false, z_upperborder, false, false, z_lowerborder); 
                if (x_border){
                    //For the ZY corner
                    BPC_loop(XX_A, nodeD, row, col, mom, partitions, did_max, dd_max, ds, 2, size_box, x_border, false, z_border, x_upperborder, false, z_upperborder, x_lowerborder, false, z_lowerborder); 
                    if (y_border){
                        //For the XYZ corner
                        BPC_loop(XX_B, nodeD, row, col, mom, partitions, did_max, dd_max, ds, 2, size_box, x_border, y_border, z_border, x_upperborder, y_upperborder, z_upperborder, x_lowerborder, y_lowerborder, z_lowerborder); 
                    }

                }

                if (y_border){
                    //For the YZ
                    BPC_loop(XX_A, nodeD, row, col, mom, partitions, did_max, dd_max, ds, 2, size_box, false, y_border, z_border, false, y_upperborder, z_upperborder, false, y_lowerborder, z_lowerborder); 
                }
            }


        }
    }
}


__global__ void make_histoXX(float *XX_A, float *XX_B, Node ***nodeD, float ds, float d_max, float size_node, float size_box){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int partitions = (int)(ceilf(size_box/size_node));
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon in this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;
        //printf("%i, %i, %i \n", mom, col,row)
        
        if (nodeD[row][col][mom].len > 0){
            
            //This may see redundant but with this these often checked values are upgraded to device memory
            float dd_max = d_max*d_max;
            int did_max = (int)(ceilf((d_max+size_node*sqrt(3.0))/size_node));
            int did_max2 = did_max*did_max;

            // Counts distances betweeen the same node
            if (idx%2==0){ //If the main index is even stores the countings in the XX_A subhistogram
                count_distances11(XX_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, ds, dd_max, 2);
            } else { //If the main index is odd stores the countings in the XX_B subhistogram
                count_distances11(XX_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, ds, dd_max, 2);
            }
            
            
            int u,v,w; // Position index of the second node
            int dx_nod12, dy_nod12, dz_nod12, dd_nod12; //Internodal distance

            //Second node movil in Z direction
            for(w = mom+1; w<partitions && w-row<=did_max; w++){
                if (idx%2==0){ //If the main index is even stores the countings in the XX_A subhistogram
                    count_distances12(XX_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[row][col][w].elements, nodeD[row][col][w].len, ds, dd_max, 2);
                } else { //If the main index is odd stores the countings in the XX_B subhistogram
                    count_distances12(XX_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[row][col][w].elements, nodeD[row][col][w].len, ds, dd_max, 2);
                }
            }

            //Second node movil in YZ
            for(v=col+1; v<partitions && v-col<=did_max; v++){
                dy_nod12 = v-col;
                for(w=(mom-did_max)*(mom>did_max); w<partitions && w-mom<=did_max; w++){
                    dz_nod12 = w-mom;
                    dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12;
                    if (dd_nod12<=did_max2){
                        if (idx%2==0){
                            count_distances12(XX_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[row][v][w].elements, nodeD[row][v][w].len, ds, dd_max, 2);
                        } else {
                            count_distances12(XX_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[row][v][w].elements, nodeD[row][v][w].len, ds, dd_max, 2);
                        }
                    }
                    //}
                }
            }

            //Second node movil in XYZ
            for(u = row+1; u < partitions && u-row< did_max; u++){
                dx_nod12 = u-row;
                for(v = (col-did_max)*(col>did_max); v < partitions && v-col< did_max; v++){
                    dy_nod12 = v-col;
                    for(w = (mom-did_max)*(mom>did_max); w < partitions && w-mom< did_max; w++){
                        dz_nod12 = w-mom;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=did_max2){
                            if (idx%2==0){
                                count_distances12(XX_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[u][v][w].elements, nodeD[u][v][w].len, ds, dd_max, 2);
                            } else {
                                count_distances12(XX_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[u][v][w].elements, nodeD[u][v][w].len, ds, dd_max, 2);
                            }
                        }
                    }
                }
            }

        }
    }
}
__global__ void make_histoXY(float *XY_A, float *XY_B, Node ***nodeD, Node ***nodeR, float ds, float d_max, float size_node, float size_box){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int partitions = (int)(ceilf(size_box/size_node));
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon in this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;
        
        if (nodeD[row][col][mom].len > 0){
            
            //This may see redundant but with this these often checked values are upgraded to device memory
            float dd_max = d_max*d_max;
            int did_max = (int)(ceilf((d_max+size_node*sqrt(3.0))/size_node));
            int did_max2 = did_max*did_max;
            
            int u,v,w; //Position of the second node
            unsigned int dx_nod12, dy_nod12, dz_nod12, dd_nod12;


            //Second node movil in XYZ
            //for(u = (row-did_max)*(row>did_max); u < partitions && u-row< did_max; u++){
            for(u = 0; u < partitions && u-row< did_max; u++){
                dx_nod12 = u-row;
                //for(v = (col-did_max)*(col>did_max); v < partitions && v-col< did_max; v++){
                for(v = 0; v < partitions && v-col< did_max; v++){
                    dy_nod12 = v-col;
                    //for(w = (mom-did_max)*(mom>did_max); w < partitions && w-mom< did_max; w++){
                    for(w = 0; w < partitions && w-mom< did_max; w++){
                        dz_nod12 = w-mom;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=did_max2){
                            if (idx%2==0){
                                count_distances12(XY_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[u][v][w].elements, nodeR[u][v][w].len, ds, dd_max, 1);
                            } else {
                                count_distances12(XY_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[u][v][w].elements, nodeR[u][v][w].len, ds, dd_max, 1);
                            }
                        }
                    }
                }
            }
            
        }
    }
}

__global__ void make_analyticRR(float *RR, float d_max, int bn, float size_box, int n_pts){
    /*
    Analytic calculation of the RR histogram

    */
    int a = threadIdx.x;
    if (a < bn){
        float dr = (d_max/bn);
        float V = size_box*size_box*size_box;
        float beta1 = n_pts*n_pts/V;
        float alph = 4*(2*acosf(0.0))*(beta1)*dr*dr*dr/3;
        float r1, r2;
        r2 = (float) a;
        r1 = r2+1;
        float sum = alph*((r1*r1*r1)-(r2*r2*r2));
        atomicAdd(&RR[a],sum);
    }
}

int main(int argc, char **argv){
	
    unsigned int np = stoi(argv[3]), bn = stoi(argv[4]);
    float dmax = stof(argv[5]);
    float ds = ((float)(bn))/dmax, size_box = 250.0, alpha = 2.176;
    float size_node = alpha*(size_box/pow((float)(np),1/3.));
    unsigned int partitions = (int)(ceil(size_box/size_node));

    float *DD_A, *DR_A, *DD_B, *DR_B, *RR;
    double *DD, *DR;
    PointW3D *dataD;
    PointW3D *dataR;
    cudaMallocManaged(&dataD, np*sizeof(PointW3D));
    cudaMallocManaged(&dataR, np*sizeof(PointW3D));

    // Name of the files where the results are saved
    string nameDD = "DDiso.dat", nameRR = "RRiso.dat", nameDR = "DRiso.dat";

    // Allocate memory for the histogram as double
    // And the subhistograms as simple presision floats
    DD = new double[bn];
    RR = new float[bn];
    DR = new double[bn];
    cudaMallocManaged(&DD_A, bn*sizeof(float));
    cudaMallocManaged(&DR_A, bn*sizeof(float));
    cudaMallocManaged(&DD_B, bn*sizeof(float));
    cudaMallocManaged(&DR_B, bn*sizeof(float));
    
    //Initialize the histograms in 0
    for (int i = 0; i < bn; i++){
        *(DD+i) = 0;
        *(RR+i) = 0;
        *(DR+i) = 0;
        *(DD_A+i) = 0;
        *(DR_A+i) = 0;
        *(DD_B+i) = 0;
        *(DR_B+i) = 0;
    }
	
	// Open and read the files to store the data in the arrays
	open_files(argv[1], np, dataD);
    open_files(argv[2], np, dataR);

    //Init the nodes arrays
    Node ***nodeD;
    Node ***nodeR;
    cudaMallocManaged(&nodeR, partitions*sizeof(Node**));
    cudaMallocManaged(&nodeD, partitions*sizeof(Node**));
    for (int i=0; i<partitions; i++){
        cudaMallocManaged(&*(nodeR+i), partitions*sizeof(Node*));
        cudaMallocManaged(&*(nodeD+i), partitions*sizeof(Node*));
        for (int j=0; j<partitions; j++){
            cudaMallocManaged(&*(*(nodeR+i)+j), partitions*sizeof(Node));
            cudaMallocManaged(&*(*(nodeD+i)+j), partitions*sizeof(Node));
        }
    }
    
    //Classificate the data into the nodes
    make_nodos(nodeD, dataD, partitions, size_node, np);
    make_nodos(nodeR, dataR, partitions, size_node, np);

    //Get the dimensions of the GPU grid
    int blocks = (int)(ceil((float)((partitions*partitions*partitions)/(float)(1024))));
    dim3 grid(blocks,1,1);
    dim3 block(1024,1,1);

    clock_t begin = clock();
    //Launch the kernels
    make_histoXX<<<grid,block>>>(DD_A, DD_B, nodeD, ds, dmax, size_node, size_box);
    BPC_XX<<<grid,block>>>(DD_A, DD_B, nodeD, ds, dmax, size_node, size_box);

    //Waits for the GPU to finish
    cudaDeviceSynchronize();  

    //Check here for errors
    cudaError_t error = cudaGetLastError(); 
    cout << "The error code is " << error << endl;
    if(error != 0)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    //make_histoXY<<<grid,block>>>(DR_A, DR_B, nodeD, nodeR, ds, dmax, size_node, size_box);
    
    blocks = (int)(ceil((float)(bn)/1024.0));
    dim3 grid_a(blocks,1,1);
    dim3 block_a(1024,1,1);

    make_analyticRR<<<grid_a,block_a>>>(RR, dmax, bn, size_box, np);

    //Waits for the GPU to finish
    cudaDeviceSynchronize();  

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\nSpent time = %.4f seg.\n", time_spent );

    //Collect the subhistograms data into the double precision main histograms
    //THis has to be done in CPU since GPU only allows single precision
    for (int i = 0; i < bn; i++){
        DD[i] = (double)(DD_A[i]+ DD_B[i]);
        DR[i] = (double)(DR_A[i]+ DR_B[i]);
    }

    cout << "Termine de hacer todos los histogramas" << endl;
    /*
    // Shows the histograms
    cout << "\nHistograma DD:" << endl;
    int sum = 0;
    for (int k = 0; k<bn; k++){
        cout << DD[k] << "\t";
        sum += DD[k];
    }
    cout << "Total: " << sum << endl;

    cout << "\nHistograma RR:" << endl;
    for (int k = 0; k<bn; k++){
        cout << RR[k] << "\t";
    }

    cout << "\nHistograma DR:" << endl;
    for (int k = 0; k<bn; k++){
        cout << DR[k] << "\t";
    }
    */
	
	// Guardamos los histogramas
	save_histogram(nameDD, bn, DD);
	cout << "Guarde histograma DD..." << endl;
	save_histogram(nameRR, bn, RR);
	cout << "Guarde histograma RR..." << endl;
	save_histogram(nameDR, bn, DR);
	cout << "Guarde histograma DR..." << endl;

    //Free the memory
    cudaFree(&dataD);
    cudaFree(&dataR);

    delete[] DD;
    delete[] DR;
    delete[] RR;
    cudaFree(&DD_A);
    cudaFree(&DR_A);
    cudaFree(&DD_B);
    cudaFree(&DR_B);


    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            cudaFree(&*(*(nodeR+i)+j));
            cudaFree(&*(*(nodeD+i)+j));
        }
        cudaFree(&*(nodeR+i));
        cudaFree(&*(nodeD+i));
    }
    cudaFree(&nodeR);
    cudaFree(&nodeD);

    cout << "Programa Terminado..." << endl;
    return 0;
}

