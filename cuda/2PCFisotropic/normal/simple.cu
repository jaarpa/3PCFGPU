// nvcc simple.cu -o par_s.out && ./par_s.out data_1GPc.dat data_1GPc.dat 405224 20 160
// nvcc simple.cu -o par_s.out && ./par_s.out data_5K.dat rand0_5K.dat 5000 30 180

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
    PointW3D *array_aux;
    array_aux = new PointW3D[lon];
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

void make_nodos(Node *nod, PointW3D *dat, unsigned int partitions, float size_node, unsigned int np){
    /*
    This function classifies the data in the nodes

    Args
    nod: Node 3D array where the data will be classified
    dat: array of PointW3D data to be classified and stored in the nodes
    partitions: number nodes in each direction
    size_node: dimensions of a single node
    np: number of points in the dat array
    */

    int row, col, mom, idx;

    // First allocate memory as an empty node:
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){
                idx = mom*partitions*partitions + col*partitions + row;
                nod[idx].nodepos.z = ((float)(mom)*(size_node));
                nod[idx].nodepos.y = ((float)(col)*(size_node));
                nod[idx].nodepos.x = ((float)(row)*(size_node));
                nod[idx].len = 0;
                nod[idx].elements = new PointW3D[0];
            }
        }
    }

    // Classificate the ith elment of the data into a node and add that point to the node with the add function:
    for (int i=0; i<np; i++){
        row = (int)(dat[i].x/size_node);
        col = (int)(dat[i].y/size_node);
        mom = (int)(dat[i].z/size_node);
        idx = mom*partitions*partitions + col*partitions + row;
        add(nod[idx].elements, nod[idx].len, dat[i].x, dat[i].y, dat[i].z, dat[i].w);
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
    
    //printf("The id is: %i . The len: %i The blockid: %i \n. ", idx, len, blockIdx.x);
    int bin;
    float d, v;
    float x1, y1, z1, w1;
    float x2,y2,z2,w2;

    for (int idx=0; idx<len-1; ++idx){
        x1 = elements[idx].x;
        y1 = elements[idx].y;
        z1 = elements[idx].z;
        w1 = elements[idx].w;
        for (int j=idx+1; j<len; ++j){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            w2 = elements[j].w;
            d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (d<=dd_max+1){
                bin = (int)(sqrt(d)*ds);
                v = sum*w1*w2;
                atomicAdd(&XX[bin],v);
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
                v = sum*w1*w2;
                atomicAdd(&XX[bin],v);
            }
        }
    }
}

__global__ void make_histoXX(float *XX, Node *nodeD, int partitions, int bn, float dmax, float size_node, int start_at, int n_kernel_calls){
    //Distributes all the indexes equitatively into the n_kernelc_calls.
    int idx = start_at + n_kernel_calls*(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon of this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;
        
        idx = mom*partitions*partitions + col*partitions + row;

        if (nodeD[idx].len > 0){

            float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
            float nx1=nodeD[idx].nodepos.x, ny1=nodeD[idx].nodepos.y, nz1=nodeD[idx].nodepos.z;
            float d_max_node = dmax + size_node*sqrt(3.0);
            d_max_node*=d_max_node;
            
            // Counts distances within the same node
            count_distances11(XX, nodeD[idx].elements, nodeD[idx].len, ds, dd_max, 2);
            
            
            int idx2, u=row,v=col,w=mom; // Position index of the second node
            float dx_nod12, dy_nod12, dz_nod12, dd_nod12; //Internodal distance

            //Second node mobil in Z direction
            for(w = mom+1; w<partitions; w++){
                idx2 = w*partitions*partitions + v*partitions + u;
                dz_nod12 = nodeD[idx2].nodepos.z - nz1;
                dd_nod12 = dz_nod12*dz_nod12;
                if (dd_nod12 <= d_max_node){
                    count_distances12(XX, nodeD[idx].elements, nodeD[idx].len, nodeD[idx2].elements, nodeD[idx2].len, ds, dd_max, 2);
                }
            }

            //Second node mobil in YZ
            for(v=col+1; v<partitions; v++){
                dy_nod12 = nodeD[v*partitions + u].nodepos.y - ny1;
                for(w=0; w<partitions; w++){
                    idx2=w*partitions*partitions + v*partitions + u;
                    dz_nod12 = nodeD[idx2].nodepos.z - nz1;
                    dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12;
                    if (dd_nod12<=d_max_node){
                        count_distances12(XX, nodeD[idx].elements, nodeD[idx].len, nodeD[idx2].elements, nodeD[idx2].len, ds, dd_max, 2);
                    }
                }
            }

            //Second node mobil in XYZ
            for(u = row+1; u < partitions; u++){
                dx_nod12 = nodeD[u].nodepos.x - nx1;
                for(v = 0; v < partitions; v++){
                    dy_nod12 = nodeD[v*partitions + u].nodepos.y - ny1;
                    for(w = 0; w < partitions; w++){
                        idx2=w*partitions*partitions + v*partitions + u;
                        dz_nod12 = nodeD[idx2].nodepos.z - nz1;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=d_max_node){
                            count_distances12(XX, nodeD[idx].elements, nodeD[idx].len, nodeD[idx2].elements, nodeD[idx2].len, ds, dd_max, 2);
                        }
                    }
                }
            }
            
        }
    }
}
__global__ void make_histoXY(float *XY, Node *nodeD, Node *nodeR, int partitions, int bn, float dmax, float size_node, int start_at, int n_kernel_calls){
    int idx = start_at + n_kernel_calls*(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon in this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;

        idx = mom*partitions*partitions + col*partitions + row;
        
        if (nodeD[idx].len > 0){

            float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
            float nx1=nodeD[idx].nodepos.x, ny1=nodeD[idx].nodepos.y, nz1=nodeD[idx].nodepos.z;
            float d_max_node = dmax + size_node*sqrt(3.0);
            d_max_node*=d_max_node;
            
            int idx2, u,v,w; //Position of the second node
            unsigned int dx_nod12, dy_nod12, dz_nod12, dd_nod12;


            //Second node mobil in XYZ
            for(u = 0; u < partitions; u++){
                dx_nod12 = nodeD[u].nodepos.x - nx1;
                for(v = 0; v < partitions; v++){
                    dy_nod12 = nodeD[v*partitions + u].nodepos.y - ny1;
                    for(w = 0; w < partitions; w++){
                        idx2 = w*partitions*partitions + v*partitions + u;
                        dz_nod12 = nodeD[idx2].nodepos.z - nz1;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=d_max_node){
                            count_distances12(XY, nodeD[idx].elements, nodeD[idx].len, nodeR[idx2].elements, nodeR[idx2].len, ds, dd_max, 1);
                        }
                    }
                }
            }
            
        }
    }
}

int main(int argc, char **argv){

    /* =======================================================================*/
    /* =====================   Var declaration ===============================*/
    /* =======================================================================*/

    unsigned int np = stoi(argv[3]), bn = stoi(argv[4]), partitions;

    float time_spent, size_node, dmax = stof(argv[5]), size_box = 0, r_size_box=0;
    float **subDD, **subRR, **subDR;

    double *DD, *RR, *DR;

    //n_kernel_calls should depend of the number of points, its density, and the number of bins
    int threads_perblock, blocks, n_kernel_calls = 2 + (np==405224)*3 + (np>405224)*42;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    cucheck(cudaEventCreate(&start_timmer));
    cucheck(cudaEventCreate(&stop_timmer));

    bool enough_kernels = false;

    PointW3D *dataD;
    PointW3D *dataR;

    // Name of the files where the results are saved
    string nameDD = "DDiso.dat", nameRR = "RRiso.dat", nameDR = "DRiso.dat";

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    //Allocate memory to read the data
    dataD = new PointW3D[np];
    dataR = new PointW3D[np];
    // Open and read the files to store the data in the arrays
    open_files(argv[1], np, dataD, size_box); //This function also gets the real size of the box
    open_files(argv[2], np, dataR, r_size_box);

    // Allocate memory for the histogram as double
    DD = new double[bn];
    RR = new double[bn];
    DR = new double[bn];
    
    //Sets the number of partitions of the box and the size of each node
    size_node = 2.176*(size_box/pow((float)(np),1/3.));
    partitions = (int)(ceil(size_box/size_node));

    //Allocate memory for the nodes depending of how many partitions there are.
    //Flatened 3D Node arrays
    Node *hnodeD, *dnodeD;
    //Node *hnodeR, *dnodeR;
    hnodeD = new Node[partitions*partitions*partitions];
    //hnodeR = new Node[partitions*partitions*partitions];
    cucheck(cudaMallocManaged(&dnodeD, partitions*partitions*partitions*sizeof(Node**)));
    //cucheck(cudaMallocManaged(&dnodeR, partitions*partitions*partitions*sizeof(Node**)));
    
    //Classificate the data into the nodes in the host side
    //The node classification is made in the host
    make_nodos(hnodeD, dataD, partitions, size_node, np);
    //make_nodos(hnodeR, dataR, partitions, size_node, np);

    //Copy nodes to global memory
    for (int i=0; i<partitions*partitions*partitions; i++){
        dnodeD[i] = hnodeD[i];
        //dnodeR[i] = hnodeR[i];
        if (hnodeD[i].len>0){
            //Deep copy for the Node elements. If the node has no elements no memory is allocated!!!
            cucheck(cudaMallocManaged(&dnodeD[i].elements, hnodeD[i].len*sizeof(PointW3D)));
            for (int j=0; j<hnodeD[i].len; j++){
                dnodeD[i].elements[j] = hnodeD[i].elements[j];
            }
        }

        /*
        if (hnodeR[i].len>0){
            //Deep copy for the Node elements. If the node has no elements no memory is allocated!!!
            cucheck(cudaMallocManaged(&dnodeR[i].elements, hnodeR[i].len*sizeof(PointW3D)));
            for (int j=0; j<hnodeR[i].len; j++){
                dnodeR[i].elements[j] = hnodeR[i].elements[j];
            }
        }
        */
    }
    
    cout << "Succesfully readed the data" << endl;
    cout << "All set to compute the histograms" << endl;


    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/

    //Starts loop to ensure the float histograms are not being overfilled.
    while (!enough_kernels){

        //Allocate an array of histograms to a different histogram to each kernel launch
        cucheck(cudaMallocManaged(&subDD, n_kernel_calls*sizeof(float*)));
        cucheck(cudaMallocManaged(&subRR, n_kernel_calls*sizeof(float*)));
        cucheck(cudaMallocManaged(&subDR, n_kernel_calls*sizeof(float*)));
        for (int i=0; i<n_kernel_calls; ++i){
            cucheck(cudaMallocManaged(&*(subDD+i), bn*sizeof(float)));
            cucheck(cudaMallocManaged(&*(subRR+i), bn*sizeof(float)));
            cucheck(cudaMallocManaged(&*(subDR+i), bn*sizeof(float)));

            //Restarts the subhistograms in 0
            for (int j = 0; j < bn; j++){
                subDD[i][j] = 0.0;
                subRR[i][j] = 0.0;
                subDR[i][j] = 0.0;
            }

            //Restarts the main histograms in host to zero
            *(DD+i) = 0;
            *(RR+i) = 0;
            *(DR+i) = 0;
        }

        for (int i=0; i<n_kernel_calls; i++){
            for (int j=0; j<bn; j++){
                cout << subDD[i][j] << endl;
            }
        }
        //Compute the dimensions of the GPU grid
        //One thread for each node
        threads_perblock = 512;
        blocks = (int)(ceil((float)((float)(partitions*partitions*partitions)/(float)(n_kernel_calls*threads_perblock))));

        //Launch the kernels
        time_spent=0; //Restarts timmer
        cudaEventRecord(start_timmer);
        for (int j=0; j<n_kernel_calls; j++){
            make_histoXX<<<blocks,threads_perblock>>>(subDD[j], dnodeD, partitions, bn, dmax, size_node, j, n_kernel_calls);
            //make_histoXX<<<blocks,threads_perblock>>>(subRR[j], dnodeR, partitions, bn, dmax, size_node, j, n_kernel_calls);
            //make_histoXY<<<blocks,threads_perblock>>>(subDR[j], dnodeD, dnodeR, partitions, bn, dmax, size_node, j, n_kernel_calls);
        }

        //Waits for all the kernels to complete
        cucheck(cudaDeviceSynchronize());

        //Sums all the subhistograms in CPU and tests if any of the subhistograms reached the maximum posible value of a float type data
        for (int j=0; j<n_kernel_calls; j++){
            for (int i = 0; i < bn; i++){

                //Test precision and max float value
                if ((subDD[j][i]+1)<=subDD[j][i] || (subRR[j][i]+1)<=subRR[j][i] || (subDR[j][i]+1)<=subDR[j][i]){
                    enough_kernels = false;
                    cout << "Not enough kernels launched the bin " << i << " exceeds the maximum value " << endl;
                    cout << "Restarting the hitogram calculations. Now trying with " << n_kernel_calls+2 << "kernel launches" << endl;
                    n_kernel_calls=n_kernel_calls+2;
                    break;
                } else {
                    enough_kernels = true;
                }

                DD[i] += (double)(subDD[j][i]);
                RR[i] += (double)(subRR[j][i]);
                DR[i] += (double)(subDR[j][i]);

            }

            if (!enough_kernels){
                break;
            }

        }

        cudaEventRecord(stop_timmer);
        cudaEventSynchronize(stop_timmer);
        cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer);
        cout << "Took "<< time_spent << " miliseconds to compute all the distances using " << n_kernel_calls << " kernel launches" << endl;
        
        //Free the subhistograms
        //If there were not enough kernel launches the subhistograms will be allocated again.
        for (int i=0; i<n_kernel_calls; ++i){
            cucheck(cudaFree(subDD[i]));
            cucheck(cudaFree(subRR[i]));
            cucheck(cudaFree(subDR[i]));
        }
        cucheck(cudaFree(subDD));
        cucheck(cudaFree(subRR));
        cucheck(cudaFree(subDR));

    }

    cout << "Spent "<< time_spent << " miliseconds to compute all the distances using " << n_kernel_calls << " kernel launches" << endl;
    
    /* =======================================================================*/
    /* =======================  Save the results =============================*/
    /* =======================================================================*/

    cout << "Saving results" << endl;
	// Guardamos los histogramas
	save_histogram(nameDD, bn, DD);
	cout << "DD histogram saved" << endl;
	save_histogram(nameRR, bn, RR);
	cout << "RR histogram saved" << endl;
	save_histogram(nameDR, bn, DR);
    cout << "DR histogram saved" << endl;
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory

    cucheck(cudaEventDestroy(start_timmer));
    cucheck(cudaEventDestroy(stop_timmer));

    delete[] dataD;
    delete[] dataR;

    delete[] DD;
    delete[] DR;
    delete[] RR;

    /*
    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            delete[] hnodeD[i][j];
            //delete[] hnodeR[i][j];

            cucheck(cudaFree(*(*(dnodeD+i)+j)));
            //cucheck(cudaFree(*(*(dnodeR+i)+j)));
        }
        delete[] hnodeD[i];
        //delete[] hnodeR[i];

        cucheck(cudaFree(*(dnodeD+i)));
        //cucheck(cudaFree(*(dnodeR+i)));
    }
    */
    
    delete[] hnodeD;
    //delete[] hnodeR;

    cucheck(cudaFree(dnodeD));
    //cucheck(cudaFree(dnodeR));

    cout << "Programa Terminado..." << endl;
    return 0;
}

