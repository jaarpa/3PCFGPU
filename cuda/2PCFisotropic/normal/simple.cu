// nvcc -arch=sm_75 simple.cu -o par_s.out && ./par_s.out data_1GPc.dat data_1GPc.dat 405224 20 160
// nvcc -arch=sm_75 simple.cu -o par_s.out && ./par_s.out data_5K.dat rand0_5K.dat 5000 30 180

// For dynamic parallelism
// nvcc -arch=sm_35 -rdc=true dynamic.cu -lcudadevrt -o par_d.out && ./par_d.out data_5K.dat rand0_5K.dat 5000 30 50
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>

using namespace std;

/** CUDA check macro */
#define cucheck(call) \
	{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
	const char* err_str = cudaGetErrorString(res);\
	fprintf(stderr, "%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);	\
	exit(-1);\
	}\
	}

#define cucheck_dev(call) \
	{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
	const char* err_str = cudaGetErrorString(res);\
	printf("%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);	\
	assert(0);																												\
	}\
	}

//Point with weight value. Structure

struct Point3D{
	float x;
	float y; 
	float z;
};

struct PointW3D{
    float x;
    float y; 
    float z;
    float w;
};

struct Node{
    Point3D nodepos; //Position of the node
    int len;		// Number of points in the node
    PointW3D *elements;	// Points in the node
};

struct DNode{ //Define the node in the device without using elements to avoid deep copy
    Point3D nodepos; //Position of the node
    int len;		// Number of points in the node
    int prev_i; //prev element idx
};

void open_files(string name_file, int pts, PointW3D *datos, float &size_box){
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

    double candidate_size_box=0;
    double max_component;
    for ( int c = 0; c < pts; c++) //Reads line by line and stores each c line in the c PointW3D element of the array
    {
        file >> datos[c].x >> datos[c].y >> datos[c].z >> datos[c].w;

        if (datos[c].x>datos[c].y){
            if (datos[c].x>datos[c].z){
                max_component = datos[c].x;
            } else {
                max_component = datos[c].z;
            }

        } else {
            if (datos[c].y>datos[c].z){
                max_component = datos[c].y;
            } else {
                max_component = datos[c].z;
            }
        }

        if (max_component>candidate_size_box){
            candidate_size_box = max_component;
        }
    }

    size_box=ceil(candidate_size_box+1);

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
    PointW3D *array_aux = new PointW3D[lon];
    for (int i=0; i<lon-1; i++){
        array_aux[i].x = array[i].x;
        array_aux[i].y = array[i].y;
        array_aux[i].z = array[i].z;
        array_aux[i].w = array[i].w;
    }
    delete[] array;
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
                nod[row][col][mom].nodepos.z = ((float)(mom)*(size_node));
                nod[row][col][mom].nodepos.y = ((float)(col)*(size_node));
                nod[row][col][mom].nodepos.x = ((float)(row)*(size_node));
                nod[row][col][mom].len = 0;
                nod[row][col][mom].elements = new PointW3D[0];
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

__device__ void count_distances11(double *XX, PointW3D *elements, int start, int end, float ds, float dd_max, int sum){
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

__global__ void make_histoXX(double *XX, PointW3D *elements, DNode *nodeD, int partitions, int bn, float dmax, float size_node){
    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon of this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;

        //idx = row + col*partitions + mom*partitions*partitions;

        if (nodeD[idx].len > 0){

            float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
            float nx1=nodeD[idx].nodepos.x, ny1=nodeD[idx].nodepos.y, nz1=nodeD[idx].nodepos.z;
            float d_max_node = dmax + size_node*sqrt(3.0);
            d_max_node*=d_max_node;
            
            // Counts distances within the same node
            count_distances11(XX, elements, nodeD[idx].prev_i, nodeD[idx+1].prev_i, ds, dd_max, 2);
            
            int idx2, u=row,v=col,w=mom; // Position index of the second node
            float dx_nod12, dy_nod12, dz_nod12, dd_nod12; //Internodal distance

            //Second node mobil in Z direction
            for(w = mom+1; w<partitions; w++){
                idx2 = row + col*partitions + w*partitions*partitions;
                dz_nod12 = nodeD[idx2].nodepos.z - nz1;
                dd_nod12 = dz_nod12*dz_nod12;
                if (dd_nod12 <= d_max_node){
                    count_distances12(XX, elements, nodeD[idx].prev_i, nodeD[idx+1].prev_i, nodeD[idx2].prev_i, nodeD[idx2+1].prev_i, ds, dd_max, 2);
                }
            }

            //Second node mobil in YZ
            for(v=col+1; v<partitions; v++){
                idx2 = row + col*partitions;
                dy_nod12 = nodeD[idx2].nodepos.y - ny1;
                for(w=0; w<partitions; w++){
                    idx2 = row + v*partitions + w*partitions*partitions
                    dz_nod12 = nodeD[idx2].nodepos.z - nz1;
                    dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12;
                    if (dd_nod12<=d_max_node){
                        count_distances12(XX, elements, nodeD[idx].prev_i, nodeD[idx+1].prev_i, nodeD[idx2].prev_i, nodeD[idx2+1].prev_i, ds, dd_max, 2);
                    }
                }
            }

            //Second node mobil in XYZ
            for(u = row+1; u < partitions; u++){
                dx_nod12 = nodeD[u].nodepos.x - nx1;
                for(v = 0; v < partitions; v++){
                    idx2 = u + v*partitions;
                    dy_nod12 = nodeD[idx2].nodepos.y - ny1;
                    for(w = 0; w < partitions; w++){
                        idx2 = row + col*partitions + mom*partitions*partitions;
                        dz_nod12 = nodeD[idx2].nodepos.z - nz1;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=d_max_node){
                            count_distances12(XX, elements, nodeD[idx].prev_i, nodeD[idx+1].prev_i, nodeD[idx2].prev_i, nodeD[idx2+1].prev_i, ds, dd_max, 2);
                        }
                    }
                }
            }

        }
    }
}
/*
__global__ void make_histoXY(double *XY, Node ***nodeD, Node ***nodeR, int partitions, int bn, float dmax, float size_node){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon in this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;
        
        if (nodeD[row][col][mom].len > 0){

            float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
            float nx1=nodeD[row][col][mom].nodepos.x, ny1=nodeD[row][col][mom].nodepos.y, nz1=nodeD[row][col][mom].nodepos.z;
            float d_max_node = dmax + size_node*sqrt(3.0);
            d_max_node*=d_max_node;
            
            int u,v,w; //Position of the second node
            unsigned int dx_nod12, dy_nod12, dz_nod12, dd_nod12;


            //Second node mobil in XYZ
            for(u = 0; u < partitions; u++){
                dx_nod12 = nodeR[u][0][0].nodepos.x - nx1;
                for(v = 0; v < partitions; v++){
                    dy_nod12 = nodeR[u][v][0].nodepos.y - ny1;
                    for(w = 0; w < partitions; w++){
                        dz_nod12 = nodeR[u][v][w].nodepos.z - nz1;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=d_max_node){
                            count_distances12(XY, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[u][v][w].elements, nodeR[u][v][w].len, ds, dd_max, 1);
                        }
                    }
                }
            }
            
        }
    }
}*/

int main(int argc, char **argv){

    /* =======================================================================*/
    /* =====================   Var declaration ===============================*/
    /* =======================================================================*/

    unsigned int np = stoi(argv[3]), bn = stoi(argv[4]), partitions;

    float time_spent, size_node, dmax = stof(argv[5]), size_box = 0, r_size_box=0;
    //float **subDD, **subRR, **subDR;

    double *DD, *RR, *DR, *d_DD, *d_RR, *d_DR,;

    //n_kernel_calls should depend of the number of points, its density, and the number of bins
    int threads_perblock, blocks;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    cucheck(cudaEventCreate(&start_timmer));
    cucheck(cudaEventCreate(&stop_timmer));

    clock_t stop_timmer_host, start_timmer_host;

    PointW3D *dataD;
    //PointW3D *dataR;

    Node ***hnodeD; //, hnodeR;
    DNode *hnodeD_s;//, *hnodeR_s,
    PointW3D *h_ordered_pointsD_s;//, h_ordered_pointsR_s;
    cudaStream_t streamDD, streamRR, streamDR;
    cudaStreamCreate(&streamDD);
    cudaStreamCreate(&streamDR);
    cudaStreamCreate(&streamRR);
    DNode *dnodeD_s1;//, *dnodeD_s3, *dnodeR_s2, *dnodeR_s3;
    int row, col, mom, k_element, last_pointD;//, last_pointR;
    PointW3D *d_ordered_pointsD_s1; //, *d_ordered_pointsD_s3, *d_ordered_pointsR_s2, *d_ordered_pointsR_s3;

    // Name of the files where the results are saved
    string nameDD = "DDiso.dat", nameRR = "RRiso.dat", nameDR = "DRiso.dat";

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/
    start_timmer_host = clock();
    dataD = new PointW3D[np];
    //dataR = new PointW3D[np];

    // Open and read the files to store the data in the arrays
    open_files(argv[1], np, dataD, size_box); //This function also gets the real size of the box
    //open_files(argv[2], np, dataR, r_size_box);

    // Allocate memory for the histogram as double
    DD = new double[bn];
    RR = new double[bn];
    DR = new double[bn];

    cucheck(cudaMalloc(&d_DD, bn*sizeof(double)));
    cucheck(cudaMalloc(&d_RR, bn*sizeof(double)));
    cucheck(cudaMalloc(&d_DR, bn*sizeof(double)));


    //Restarts the main histograms in host to zero
    for (int i = 0; i<bn; i++){
        *(DD+i) = 0.0;
        *(RR+i) = 0.0;
        *(DR+i) = 0.0;
    }

    cucheck(cudaMemcpyAsync(d_DD, DD, bn*sizeof(double), cudaMemcpyHostToDevice, streamDD));
    cucheck(cudaMemcpyAsync(d_RR, RR, bn*sizeof(double), cudaMemcpyHostToDevice, streamRR));
    cucheck(cudaMemcpyAsync(d_DR, DR, bn*sizeof(double), cudaMemcpyHostToDevice, streamDR));

    //Sets the number of partitions of the box and the size of each node
    size_node = 2.176*(size_box/pow((float)(np),1/3.));
    partitions = (int)(ceil(size_box/size_node));

    //Allocate memory for the nodes depending of how many partitions there are.
    cucheck(cudaMalloc(&dnodeD_s1, partitions*partitions*partitions*sizeof(DNode)));
    cucheck(cudaMalloc(&d_ordered_pointsD_s1, np*sizeof(PointW3D)));
    //cucheck(cudaMalloc(&dnodeR_s2, partitions*partitions*partitions*sizeof(DNode)));
    //cucheck(cudaMalloc(&d_ordered_pointsR_s2, np*sizeof(PointW3D)));
    //cucheck(cudaMalloc(&dnodeD_s3, partitions*partitions*partitions*sizeof(DNode)));
    //cucheck(cudaMalloc(&d_ordered_pointsD_s3, np*sizeof(PointW3D)));
    //cucheck(cudaMalloc(&dnodeR_s3, partitions*partitions*partitions*sizeof(DNode)));
    //cucheck(cudaMalloc(&d_ordered_pointsR_s3, np*sizeof(PointW3D)));

    hnodeD_s = new DNode[partitions*partitions*partitions];
    h_ordered_pointsD_s = new PointW3D[np];

    hnodeD = new Node**[partitions];
    //hnodeR = new Node**[partitions];
    for (int i=0; i<partitions; i++){
        *(hnodeD+i) = new Node*[partitions];
        //*(hnodeR+i) = new Node*[partitions];
        for (int j=0; j<partitions; j++){
            *(*(hnodeD+i)+j) = new Node[partitions];
            //*(*(hnodeR+i)+j) = new Node[partitions];
        }
    }

    //Classificate the data into the nodes in the host side
    //The node classification is made in the host
    make_nodos(hnodeD, dataD, partitions, size_node, np);
    //make_nodos(hnodeR, dataR, partitions, size_node, np);
    
    //Deep copy to device memory

    last_pointR = 0;
    last_pointD = 0;
    for (int idx=0; idx<partitions*partitions*partitions; idx++){
        mom = (int) (idx/(partitions*partitions));
        col = (int) ((idx%(partitions*partitions))/partitions);
        row = idx%partitions;
        
        hnodeD_s[idx].nodepos = hnodeD[row][col][mom].nodepos;
        hnodeD_s[idx].prev_i = last_pointR;
        last_pointR = last_pointR + hnodeD[row][col][mom].len;
        hnodeD_s[idx].len = hnodeD[row][col][mom].len;
        for (int j=hnodeD_s[idx].prev_i; j<last_pointR; j++){
            k_element = j-hnodeD_s[idx].prev_i;
            h_ordered_pointsD_s[j] = hnodeD[row][col][mom].elements[k_element];
        }

        // hnodeR_s[idx].nodepos = hnodeR[row][col][mom].nodepos;
        // hnodeR_s[idx].prev_i = last_pointR;
        // last_pointR = last_pointR + hnodeR[row][col][mom].len;
        // hnodeR_s[idx].len = hnodeR[row][col][mom].len;
        // for (int j=hnodeR_s[idx].prev_i; j<last_pointR; j++){
        //     k_element = j-hnodeR_s[idx].prev_i;
        //     h_ordered_pointsR_s[j] = hnodeR[row][col][mom].elements[k_element];
        // }
    }

    cucheck(cudaMemcpyAsync(d_ordered_pointsD_s1, h_ordered_pointsD_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDD));
    cucheck(cudaMemcpyAsync(dnodeD_s1, hnodeD_s, partitions*partitions*partitions*sizeof(DNode), cudaMemcpyHostToDevice, streamDD));
    //cucheck(cudaMemcpyAsync(d_ordered_pointsD_s3, h_ordered_pointsD_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDR));
    //cucheck(cudaMemcpyAsync(dnodeD_s3, hnodeD_s, partitions*partitions*partitions*sizeof(DNode), cudaMemcpyHostToDevice, streamDR));

    //cucheck(cudaMemcpyAsync(d_ordered_pointsR_s3, h_ordered_pointsR_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDR));
    //cucheck(cudaMemcpyAsync(dnodeR_s3, hnodeR_s, partitions*partitions*partitions*sizeof(DNode), cudaMemcpyHostToDevice, streamDR));
    //cucheck(cudaMemcpyAsync(d_ordered_pointsR_s2, h_ordered_pointsR_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamRR));
    //cucheck(cudaMemcpyAsync(dnodeR_s2, hnodeR_s, partitions*partitions*partitions*sizeof(DNode), cudaMemcpyHostToDevice, streamRR));

    stop_timmer_host = clock();
    time_spent = ((float)(stop_timmer_host-start_timmer_host))/CLOCKS_PER_SEC;
    cout << "Succesfully readed the data. All set to compute the histograms in " << time_spent*1000 << " miliseconds" << endl;


    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/


    //Compute the dimensions of the GPU grid
    //One thread for each node
    threads_perblock = 512;
    blocks = (int)(ceil((float)((float)(partitions*partitions*partitions)/(float)(threads_perblock))));

    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    make_histoXX<<<blocks,threads_perblock,0,streamDD>>>(d_DD, d_ordered_pointsD_s1, dnodeD_s1, partitions, bn, dmax, size_node);
    //make_histoXX<<<blocks,threads_perblock>>>(RR, dnodeR, partitions, bn, dmax, size_node);
    //make_histoXY<<<blocks,threads_perblock>>>(DR, dnodeD, dnodeR, partitions, bn, dmax, size_node);

    cucheck(cudaMemcpyAsync(DD, d_DD, bn*sizeof(double), cudaMemcpyDeviceToHost, streamDD));
    cucheck(cudaMemcpyAsync(RR, d_RR, bn*sizeof(double), cudaMemcpyDeviceToHost, streamRR));
    cucheck(cudaMemcpyAsync(DR, d_DR, bn*sizeof(double), cudaMemcpyDeviceToHost, streamDR));

    //Waits for all the kernels to complete
    cucheck(cudaDeviceSynchronize());

    cudaEventRecord(stop_timmer);
    cudaEventSynchronize(stop_timmer);
    cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer);

    cout << "Spent "<< time_spent << " miliseconds to compute all the histograms." << endl;
    
    /* =======================================================================*/
    /* =======================  Save the results =============================*/
    /* =======================================================================*/

	// Guardamos los histogramas
	save_histogram(nameDD, bn, DD);
	save_histogram(nameRR, bn, RR);
	save_histogram(nameDR, bn, DR);
    cout << "Saved the histograms" << endl;
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory
    cudaStreamDestroy(streamDD);
    cudaStreamDestroy(streamDR);
    cudaStreamDestroy(streamRR);

    cucheck(cudaEventDestroy(start_timmer));
    cucheck(cudaEventDestroy(stop_timmer));

    delete[] dataD;
    //delete[] dataR;

    delete[] DD;
    delete[] RR;    
    delete[] DR;    
    
    cucheck(cudaFree(d_DD));
    cucheck(cudaFree(d_RR));
    cucheck(cudaFree(d_DR));

    
    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            delete[] hnodeD[i][j];
            //delete[] hnodeR[i][j];
        }
        delete[] hnodeD[i];
        //delete[] hnodeR[i];
    }    
    delete[] hnodeD;
    //delete[] hnodeR;

    cucheck(cudaFree(d_ordered_pointsD_s1));
    cucheck(cudaFree(dnodeD_s1));
    //cucheck(cudaFree(d_ordered_pointsR_s2));
    //cucheck(cudaFree(dnodeR_s2));
    //cucheck(cudaFree(d_ordered_pointsD_s3));
    //cucheck(cudaFree(dnodeD_s3));
    //cucheck(cudaFree(d_ordered_pointsR_s3));
    //cucheck(cudaFree(dnodeR_s3));
    
    delete[] hnodeD_s;
    delete[] h_ordered_pointsD_s;
    //delete[] hnodeR_s;
    //delete[] h_ordered_pointsR_s;

    cout << "Programa Terminado..." << endl;
    return 0;
}

