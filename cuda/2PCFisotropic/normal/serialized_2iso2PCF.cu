// nvcc serialized_iso2PCF.cu -o par.out && ./par.out data_5K.dat rand0_5K.dat 5000 30 180
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
	PointW3D nodepos;
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
    float p_med = size_node*0.5;

    // First allocate memory as an empty node:
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){
                nod[row][col][mom].nodepos.w = 0;
                nod[row][col][mom].nodepos.z = ((float)(mom)*(size_node))+p_med;
                nod[row][col][mom].nodepos.y = ((float)(col)*(size_node))+p_med;
                nod[row][col][mom].nodepos.x = ((float)(row)*(size_node))+p_med;
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

__device__ void count_distances11(float *XX, PointW3D *elements, int len, float ds, float dd_max){
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
                //printf("%f,%f,%f and %f,%f,%f bin: %i \n",x1,y1,z1, x2,y2,z2, bin);
                atomicAdd(&XX[bin],2);
            }
        }
    }
}

__device__ void count_distances12(float *XX, PointW3D *elements1, int len1, PointW3D *elements2, int len2, float ds, float dd_max){
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
                atomicAdd(&XX[bin],2);
            }
        }
    }
}

__global__ void make_histoXX(float *XX_A, float *XX_B, Node ***nodeD, int partitions, float ds, float dd_max, int did_max, int did_max2){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon in this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;

        int bin, u, v, w;
        float dis, dis_nod;
        float x1D, y1D, z1D, x2D, y2D, z2D;
        x1D = nodeX[row][0][0].nodepos.x;
        y1D = nodeX[0][col][0].nodepos.y;
        z1D = nodeX[0][0][mom].nodepos.z;
        float x, y, z, w1;
        float dx, dy, dz, dx_nod, dy_nod, dz_nod;
        //printf("%i, %i, %i \n", mom, col,row)
        
        if (nodeD[row][col][mom].len > 0){
            
            //==================================================
            // Distancias entre puntos del mismo nodo:
            //==================================================
            for ( i= 0; i <nodeD[row][col][mom].len - 1; ++i){
                x = nodeD[row][col][mom].elements[i].x;
                y = nodeD[row][col][mom].elements[i].y;
                z = nodeD[row][col][mom].elements[i].z;
                w1 = nodeD[row][col][mom].elements[i].w;
                    for ( j = i+1; j < nodeD[row][col][mom].len; ++j){
                    dx = x-nodeD[row][col][mom].elements[j].x;
                    dy = y-nodeD[row][col][mom].elements[j].y;
                    dz = z-nodeD[row][col][mom].elements[j].z;
                    dis = dx*dx+dy*dy+dz*dz;
                    if (dis <= dd_max){
                    bin = (int)(sqrt(dis)*ds)
                    atomicAdd(&XX_A[bin],2);
                    }
                }
            }

            //==================================================
            // Distancias entre puntos del diferente nodo:
            //==================================================
            u = row;
            v = col;

            //=========================
            // N2 movil en Z
            //=========================
            for (w=mom+1;  w<partitions ; ++w){	
            z2D = nodeD[u][v][w].nodepos.z;
            dz_nod = z1D-z2D;
            dis_nod = dz_nod*dz_nod;
            if (dis_nod <= ddmax_nod){
                for ( i = 0; i < nodeD[row][col][mom].len; ++i){
                x = nodeD[row][col][mom].elements[i].x;
                y = nodeD[row][col][mom].elements[i].y;
                z = nodeD[row][col][mom].elements[i].z;
                w1 = nodeD[row][col][mom].elements[i].w;
                    for ( j = 0; j < nodeD[u][v][w].len; ++j){
                    dx = x-nodeD[u][v][w].elements[j].x;
                    dy = y-nodeD[u][v][w].elements[j].y;
                    dz = z-nodeD[u][v][w].elements[j].z;
                    dis = dx*dx+dy*dy+dz*dz;
                    if (dis <= dd_max){
                    *(XX + (int)(sqrt(dis)*ds)) += 2*w1*nodeD[u][v][w].elements[j].w;
                    }
                    }
                    }
                }
            }

            //=========================
            // N2 movil en ZY
            //=========================
            for (v = col + 1; v < partitions ; ++v){
            y2D = nodeD[u][v][0].nodepos.y;
            dy_nod = y1D-y2D;
            dy_nod *= dy_nod;
                for (w = 0; w < partitions ; ++w){		
                z2D = nodeD[u][v][w].nodepos.z;
                dz_nod = z1D-z2D;
                dz_nod *= dz_nod;
                dis_nod = dy_nod + dz_nod;
                if (dis_nod <= ddmax_nod){
                    for ( i = 0; i < nodeD[row][col][mom].len; ++i){
                    x = nodeD[row][col][mom].elements[i].x;
                    y = nodeD[row][col][mom].elements[i].y;
                    z = nodeD[row][col][mom].elements[i].z;
                    w1 = nodeD[row][col][mom].elements[i].w;
                        for ( j = 0; j < nodeD[u][v][w].len; ++j){	
                        dx =  x-nodeD[u][v][w].elements[j].x;
                        dy =  y-nodeD[u][v][w].elements[j].y;
                        dz =  z-nodeD[u][v][w].elements[j].z;
                        dis = dx*dx+dy*dy+dz*dz;
                        if (dis <= dd_max){
                        *(XX + (int)(sqrt(dis)*ds)) += 2*w1*nodeD[u][v][w].elements[j].w;
                        }
                        }
                    }
                }
                }
            }

            //=========================
            // N2 movil en ZYX
            //=========================
            for ( u = row + 1; u < partitions; ++u){
            x2D = nodeD[u][0][0].nodepos.x;
            dx_nod = x1D-x2D;
            dx_nod *= dx_nod;	
                for ( v = 0; v < partitions; ++v){
                y2D = nodeD[u][v][0].nodepos.y;
                dy_nod = y1D-y2D;
                dy_nod *= dy_nod;	
                    for ( w = 0; w < partitions; ++w){
                    z2D = nodeD[u][v][w].nodepos.z;
                    dz_nod = z1D-z2D;
                    dz_nod *= dz_nod;
                    dis_nod = dx_nod + dy_nod + dz_nod;
                    if (dis_nod <= ddmax_nod){
                        for ( i = 0; i < nodeD[row][col][mom].len; ++i){
                        x = nodeD[row][col][mom].elements[i].x;
                        y = nodeD[row][col][mom].elements[i].y;
                        z = nodeD[row][col][mom].elements[i].z;
                        w1 = nodeD[row][col][mom].elements[i].w;
                            for ( j = 0; j < nodeD[u][v][w].len; ++j){	
                            dx = x-nodeD[u][v][w].elements[j].x;
                            dy = y-nodeD[u][v][w].elements[j].y;
                            dz = z-nodeD[u][v][w].elements[j].z;
                            dis = dx*dx + dy*dy + dz*dz;
                            if (dis <= dd_max){
                                *(XX + (int)(sqrt(dis)*ds)) += 2*w1*nodeD[u][v][w].elements[j].w;
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
__global__ void make_histoXY(float *XY_A, float *XY_B, Node ***nodeD, Node ***nodeR, int partitions, float ds, float dd_max, int did_max, int did_max2){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon in this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;
        
        if (nodeD[row][col][mom].len > 0){
            
            int u,v,w; //Position of the second node
            unsigned int dx_nod12, dy_nod12, dz_nod12, dd_nod12;

             //Second node movil in Z
            for(w = (mom-did_max)*(mom>did_max); w<partitions && w-row<=did_max; w++){
                if (idx%2==0){
                    count_distances12(XY_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[row][col][w].elements, nodeR[row][col][w].len, ds, dd_max);
                } else {
                    count_distances12(XY_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[row][col][w].elements, nodeR[row][col][w].len, ds, dd_max);
                }
            }

            //Second node movil in YZ
            for(v = (col-did_max)*(col>did_max); v<partitions && v-col<=did_max; v++){
                dy_nod12 = v-col;
                for(w = (mom-did_max)*(mom>did_max); w<partitions && w-mom<=did_max; w++){
                    dz_nod12 = w-mom;
                    dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12;
                    if (dd_nod12<=did_max2){
                        if (idx%2==0){
                            count_distances12(XY_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[row][v][w].elements, nodeR[row][v][w].len, ds, dd_max);
                        } else {
                            count_distances12(XY_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[row][v][w].elements, nodeR[row][v][w].len, ds, dd_max);
                        }
                    }
                }
            }

            //Second node movil in XYZ
            for(u = (row-did_max)*(row>did_max); u < partitions && u-row< did_max; u++){
                dx_nod12 = u-row;
                for(v = (col-did_max)*(col>did_max); v < partitions && v-col< did_max; v++){
                    dy_nod12 = v-col;
                    for(w = (mom-did_max)*(mom>did_max); w < partitions && w-mom< did_max; w++){
                        dz_nod12 = w-mom;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=did_max2){
                            if (idx%2==0){
                                count_distances12(XY_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[u][v][w].elements, nodeR[u][v][w].len, ds, dd_max);
                            } else {
                                count_distances12(XY_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[u][v][w].elements, nodeR[u][v][w].len, ds, dd_max);
                            }
                        }
                    }
                }
            }
            
        }
    }
}

int main(int argc, char **argv){
	
    unsigned int np = stoi(argv[3]), bn = stoi(argv[4]);
    float dmax = stof(argv[5]);
    float ds = ((float)(bn))/dmax, dd_max=dmax*dmax, size_box = 250.0, alpha = 2.176;
    float size_node = alpha*(size_box/pow((float)(np),1/3.));
    int did_max = (int)(ceil(dmax/size_node));
    int did_max2 = (int)(ceil(dd_max/(size_node*size_node)));
    cout << "did_max" << did_max << "did_max2" << did_max2 << endl;
    unsigned int partitions = (int)(ceil(size_box/size_node));

    float *DD_A, *RR_A, *DR_A, *DD_B, *RR_B, *DR_B;
    double *DD, *RR, *DR;
    PointW3D *dataD;
    PointW3D *dataR;
    cudaMallocManaged(&dataD, np*sizeof(PointW3D));
    cudaMallocManaged(&dataR, np*sizeof(PointW3D));

    // Name of the files where the results are saved
    string nameDD = "DDiso.dat", nameRR = "RRiso.dat", nameDR = "DRiso.dat";

    // Allocate memory for the histogram as double
    // And the subhistograms as simple presision floats
    DD = new double[bn];
    RR = new double[bn];
    DR = new double[bn];
    cudaMallocManaged(&DD_A, bn*sizeof(float));
    cudaMallocManaged(&RR_A, bn*sizeof(float));
    cudaMallocManaged(&DR_A, bn*sizeof(float));
    cudaMallocManaged(&DD_B, bn*sizeof(float));
    cudaMallocManaged(&RR_B, bn*sizeof(float));
    cudaMallocManaged(&DR_B, bn*sizeof(float));
    
    //Initialize the histograms in 0
    for (int i = 0; i < bn; i++){
        *(DD+i) = 0;
        *(RR+i) = 0;
        *(DR+i) = 0;
        *(DD_A+i) = 0;
        *(RR_A+i) = 0;
        *(DR_A+i) = 0;
        *(DD_B+i) = 0;
        *(RR_B+i) = 0;
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
    make_histoXX<<<grid,block>>>(DD_A, DD_B, nodeD, partitions, ds, dd_max, did_max, did_max2);
    make_histoXX<<<grid,block>>>(RR_A, RR_B, nodeR, partitions, ds, dd_max, did_max, did_max2);
    make_histoXY<<<grid,block>>>(DR_A, DR_B, nodeD, nodeR, partitions, ds, dd_max, did_max, did_max2);

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

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\nSpent time = %.4f seg.\n", time_spent );

    //Collect the subhistograms data into the double precision main histograms
    //THis has to be done in CPU since GPU only allows single precision
    for (int i = 0; i < bn; i++){
        DD[i] = (double)(DD_A[i]+ DD_B[i]);
        RR[i] = (double)(RR_A[i]+ RR_B[i]);
        DR[i] = (double)(DR_A[i]+ DR_B[i]);
    }

    cout << "Termine de hacer todos los histogramas" << endl;
	
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
    cudaFree(&RR_A);
    cudaFree(&DR_A);
    cudaFree(&DD_B);
    cudaFree(&RR_B);
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

