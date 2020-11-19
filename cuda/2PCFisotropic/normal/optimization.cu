// nvcc simple.cu -o par_s.out && ./par_s.out data_5K.dat rand0_5K.dat 5000 30 180

// For dynamic parallelism
// nvcc -arch=sm_35 -rdc=true dynamic.cu -lcudadevrt -o par_d.out && ./par_d.out data_5K.dat rand0_5K.dat 5000 30 50
#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>
#include <limits>

using namespace std;

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

void make_nodos(Node ***nod, PointW3D *dat, int partitions, float size_node, unsigned int np){
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

__global__ void make_histoXX(float *XX, Node ***nodeD, int partitions, int bn, float dmax, float size_node, int start_at){
    //If start at is 0 it does every even index, it does every odd index otherwise
    int idx = 2*(blockIdx.x * blockDim.x + threadIdx.x) + start_at;
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon of this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;
        
        if (nodeD[row][col][mom].len > 0){

            float ds = ((float)(bn))/dmax, dd_max=dmax*dmax;
            float nx1=nodeD[row][col][mom].nodepos.x, ny1=nodeD[row][col][mom].nodepos.y, nz1=nodeD[row][col][mom].nodepos.z;
            float d_max_node = dmax + size_node*sqrt(3.0);
            d_max_node*=d_max_node;
            
            // Counts distances within the same node
            count_distances11(XX, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, ds, dd_max, 2);
            
            
            int u=row,v=col,w=mom; // Position index of the second node
            float dx_nod12, dy_nod12, dz_nod12, dd_nod12; //Internodal distance

            //Second node mobil in Z direction
            for(w = mom+1; w<partitions; w++){
                dz_nod12 = nodeD[u][v][w].nodepos.z - nz1;
                dd_nod12 = dz_nod12*dz_nod12;
                if (dd_nod12 <= d_max_node){
                    count_distances12(XX, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[row][col][w].elements, nodeD[row][col][w].len, ds, dd_max, 2);
                }
            }

            //Second node mobil in YZ
            for(v=col+1; v<partitions; v++){
                dy_nod12 = nodeD[u][v][0].nodepos.y - ny1;
                for(w=0; w<partitions; w++){
                    dz_nod12 = nodeD[u][v][w].nodepos.z - nz1;
                    dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12;
                    if (dd_nod12<=d_max_node){
                        count_distances12(XX, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[row][v][w].elements, nodeD[row][v][w].len, ds, dd_max, 2);
                    }
                }
            }

            //Second node mobil in XYZ
            for(u = row+1; u < partitions; u++){
                dx_nod12 = nodeD[u][0][0].nodepos.x - nx1;
                for(v = 0; v < partitions; v++){
                    dy_nod12 = nodeD[u][v][0].nodepos.y - ny1;
                    for(w = 0; w < partitions; w++){
                        dz_nod12 = nodeD[u][v][w].nodepos.z - nz1;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=d_max_node){
                            count_distances12(XX, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[u][v][w].elements, nodeD[u][v][w].len, ds, dd_max, 2);
                        }
                    }
                }
            }
            
        }
    }
}
__global__ void make_histoXY(float *XY, Node ***nodeD, Node ***nodeR, int partitions, int bn, float dmax, float size_node, int start_at){
    int idx = 2*(blockIdx.x * blockDim.x + threadIdx.x) + start_at;
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
                dx_nod12 = nodeD[u][0][0].nodepos.x - nx1;
                for(v = 0; v < partitions; v++){
                    dy_nod12 = nodeD[u][v][0].nodepos.y - ny1;
                    for(w = 0; w < partitions; w++){
                        dz_nod12 = nodeD[u][v][w].nodepos.z - nz1;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=d_max_node){
                            count_distances12(XY, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[u][v][w].elements, nodeR[u][v][w].len, ds, dd_max, 1);
                        }
                    }
                }
            }
            
        }
    }
}

int main(int argc, char **argv){
	
    int np = stoi(argv[3]), bn = stoi(argv[4]);
    float dmax; // = stof(argv[5]);

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
	
    // Open and read the files to store the data in the arrays
    float size_box = 0; //, dummy_size_box=0; //Will be obtained from the data
    open_files(argv[1], np, dataD, size_box);
    cout << "Successfully readed the file "<< argv[1] << endl;
    cout << "Size of the box of " << size_box << endl; 
    //open_files(argv[2], np, dataR, dummy_size_box);
    
    float size_node;//, prev_time = numeric_limits<double>::infinity();// = 2.176;
    //float size_node = alpha*(size_box/pow((float)(np),1/3.));
    int partitions;// = (int)(ceil(size_box/size_node));
    ofstream outfile; //To write optimization results

    for (dmax=20; dmax<140; dmax=dmax+10){
        cout << "Computing for a dmax of: "<< dmax <<endl;

        for (partitions=10; partitions<100; partitions+=5){
            size_node = size_box/((float)(partitions));
            cout << "Trying with " << partitions << " partitions and a size node of " << size_node << endl;
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

            //Init the nodes arrays
            Node ***nodeD;
            //Node ***nodeR;
            //cudaMallocManaged(&nodeR, partitions*sizeof(Node**));
            cudaMallocManaged(&nodeD, partitions*sizeof(Node**));
            for (int i=0; i<partitions; i++){
                //cudaMallocManaged(&*(nodeR+i), partitions*sizeof(Node*));
                cudaMallocManaged(&*(nodeD+i), partitions*sizeof(Node*));
                for (int j=0; j<partitions; j++){
                    //cudaMallocManaged(&*(*(nodeR+i)+j), partitions*sizeof(Node));
                    cudaMallocManaged(&*(*(nodeD+i)+j), partitions*sizeof(Node));
                }
            }

            
            //Classificate the data into the nodes
            make_nodos(nodeD, dataD, partitions, size_node, np);
            //make_nodos(nodeR, dataR, partitions, size_node, np);

            //Get the dimensions of the GPU grid
            int threads = 512;
            int blocks = (int)(ceil((float)((partitions*partitions*partitions)/(float)(2*threads))));
            dim3 grid(blocks,1,1);
            dim3 block(threads,1,1);
            //One thread for each node

            clock_t begin = clock();
            //Launch the kernels
            make_histoXX<<<grid,block>>>(DD_A, nodeD, partitions, bn, dmax, size_node, 0);
            make_histoXX<<<grid,block>>>(DD_B, nodeD, partitions, bn, dmax, size_node, 1);
            //make_histoXX<<<grid,block>>>(RR_A, nodeR, partitions, bn, dmax, size_node, 0);
            //make_histoXX<<<grid,block>>>(RR_B, nodeR, partitions, bn, dmax, size_node, 1);
            //make_histoXY<<<grid,block>>>(DR_A, nodeD, nodeR, partitions, bn, dmax, size_node, 0);
            //make_histoXY<<<grid,block>>>(DR_B, nodeD, nodeR, partitions, bn, dmax, size_node, 1);

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

            //"time_results.dat"
            //n_points, size_box, dmax, partitions, node_size, time [s]
            outfile.open("time_results.dat", ios_base::app); // append instead of overwrite
            outfile << np << ", " << size_box << ", " << dmax << ", " << partitions << ", " << size_node << ", " <<  time_spent << endl;
            outfile.close();

            //Collect the subhistograms data into the double precision main histograms
            //THis has to be done in CPU since GPU only allows single precision
            for (int i = 0; i < bn; i++){
                DD[i] = (double)(DD_A[i]+ DD_B[i]);
                RR[i] = (double)(RR_A[i]+ RR_B[i]);
                DR[i] = (double)(DR_A[i]+ DR_B[i]);
            }


            for (int i=0; i<partitions; i++){
                for (int j=0; j<partitions; j++){
                    //cudaFree(&*(*(nodeR+i)+j));
                    cudaFree(*(*(nodeD+i)+j));
                }
                //cudaFree(&*(nodeR+i));
                cudaFree(*(nodeD+i));
            }
            //cudaFree(&nodeR);
            cudaFree(nodeD);

            //Check here for errors
            error = cudaGetLastError(); 
            cout << "The error code is " << error << endl;
        }

    }

    /*
    cout << "Termine de hacer todos los histogramas" << endl;
	
	// Guardamos los histogramas
	save_histogram(nameDD, bn, DD);
	cout << "Guarde histograma DD..." << endl;
	save_histogram(nameRR, bn, RR);
	cout << "Guarde histograma RR..." << endl;
	save_histogram(nameDR, bn, DR);
    cout << "Guarde histograma DR..." << endl;
    */

    //Free the memory
    cudaFree(dataD);
    cudaFree(dataR);

    delete[] DD;
    delete[] DR;
    delete[] RR;
    cudaFree(DD_A);
    cudaFree(RR_A);
    cudaFree(DR_A);
    cudaFree(DD_B);
    cudaFree(RR_B);
    cudaFree(DR_B);

    cout << "Programa Terminado..." << endl;
    return 0;
}

