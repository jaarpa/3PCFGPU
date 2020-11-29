//nvcc test.cu -o t.out && ./t.out data_5K.dat 5000

//nvcc test.cu -o t.out && ./t.out data_1GPc.dat 405224 
//Spent time = 0.0334 seg (ONly host. Nodes allocation and calculation. dataD in globalmemory )

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

    string mypathto_files = "fake_DATA/DATOS/";
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

//=================================================================== 
void add(PointW3D *&array, int &lon, float _x, float _y, float _z, float _w){
    /*
    This function manages adding points to an specific Node. It receives the previous array, longitude and point to add
    and updates the previous array and length with the same array with the new point at the end and adds +1 to the length +1

    It manages the memory allocation and free of the previous and new elements.
    */

    lon++;
    PointW3D *array_aux;
    //cucheck(cudaMallocManaged(&array_aux, lon*sizeof(PointW3D))); 
    array_aux = new PointW3D[lon];
    for (int i=0; i<lon-1; i++){
        array_aux[i].x = array[i].x;
        array_aux[i].y = array[i].y;
        array_aux[i].z = array[i].z;
        array_aux[i].w = array[i].w;
    }
    //cucheck(cudaFree(array));
    delete[] array;
    array = array_aux;
    array[lon-1].x = _x;
    array[lon-1].y = _y; 
    array[lon-1].z = _z;
    array[lon-1].w = _w; 
}

__global__ void pnodestest(Node *dnodeD, int partitions){
    int i = blockIdx.x + threadIdx.x;
    if (i==0){
        int px=1,py=2,pz=3;
        int idx = pz*partitions*partitions + py*partitions + px;
        printf("In GPU... \n Node 1,2,3 len: %i Position %f, %f, %f \n Elements:\n", dnodeD[idx].len, dnodeD[idx].nodepos.x, dnodeD[idx].nodepos.y, dnodeD[idx].nodepos.z);
        for (int i=0; i<dnodeD[idx].len; i++){
            printf("%f,%f,%f \n", dnodeD[idx].elements[i].x, dnodeD[idx].elements[i].y, dnodeD[idx].elements[i].z);
        }

        px=3,py=3,pz=3;
        idx = pz*partitions*partitions + py*partitions + px;
        printf("In GPU... \n Node 3,3,3 len: %i Position %f, %f, %f \n Elements:\n", dnodeD[idx].len, dnodeD[idx].nodepos.x, dnodeD[idx].nodepos.y, dnodeD[idx].nodepos.z);
        for (int i=0; i<dnodeD[idx].len; i++){
            printf("%f,%f,%f \n", dnodeD[idx].elements[i].x, dnodeD[idx].elements[i].y, dnodeD[idx].elements[i].z);
        }

        px=3,py=2,pz=1;
        idx = pz*partitions*partitions + py*partitions + px;
        printf("In GPU... \n Node 3,2,1 len: %i Position %f, %f, %f \n Elements:\n", dnodeD[idx].len, dnodeD[idx].nodepos.x, dnodeD[idx].nodepos.y, dnodeD[idx].nodepos.z);
        for (int i=0; i<dnodeD[idx].len; i++){
            printf("%f,%f,%f \n", dnodeD[idx].elements[i].x, dnodeD[idx].elements[i].y, dnodeD[idx].elements[i].z);
        }
    }
}

void make_nodos(Node *nod, PointW3D *dat, float size_node, float partitions, unsigned int n_pts){
    /*
    Función para crear los nodos con los datos y puntos random

    Argumentos
    nod: arreglo donde se crean los nodos.
    dat: datos a dividir en nodos.

    */
    int idx;
    int i, row, col, mom;

    // Inicializamos los nodos vacíos:
    for (mom=0; mom<partitions; mom++){
    for (col=0; col<partitions; col++){
    for (row=0; row<partitions; row++){
        idx = mom*partitions*partitions + col*partitions + row;
        nod[idx].nodepos.z = ((float)(mom)*(size_node));
        nod[idx].nodepos.y = ((float)(col)*(size_node));
        nod[idx].nodepos.x = ((float)(row)*(size_node));
        nod[idx].len = 0;
        
        //cucheck(cudaMallocManaged(&nod[idx].elements, sizeof(PointW3D)));
        nod[idx].elements = new PointW3D[0];
    }
    }
    }
    // Llenamos los nodos con los puntos de dat:
    for (i=0; i<n_pts; ++i){
        row = (int)(dat[i].x/size_node);
        col = (int)(dat[i].y/size_node);
        mom = (int)(dat[i].z/size_node);
        idx = mom*partitions*partitions + col*partitions + row;
        if (idx>partitions*partitions*partitions){
            cout << "For point " << i << endl;
            cout << "Got idx out of range " << idx << " with row,col, mom: "<< row <<", " << col << ", "<< mom << endl;
        }
        add( nod[idx].elements, nod[idx].len, dat[i].x, dat[i].y, dat[i].z, dat[i].w);
    }
}

int main(int argc, char **argv){
    unsigned int np = stoi(argv[2]), partitions;
    float size_node, size_box = 0;//, r_size_box;
    clock_t start_timmer, stop_timmer;
    double time_spent;

    PointW3D *dataD;
    cucheck(cudaMallocManaged(&dataD, np*sizeof(PointW3D)));
    //dataD = new PointW3D[np];
    
    open_files(argv[1], np, dataD, size_box);
    size_node = 2.176*(size_box/pow((float)(np),1/3.));
    partitions = (int)(ceil(size_box/size_node));
    int partitions3 = partitions*partitions*partitions;

    start_timmer = clock();
    //Allocate memory for the nodes depending of how many partitions there are.
    Node *hnodeD;
    //Node ***dnodeD;
    hnodeD = new Node[partitions3];
    //cucheck(cudaMallocManaged(&dnodeD, partitions*sizeof(Node**)));
    //for (int i=0; i<partitions; i++){
        //*(hnodeD+i) = new Node*[partitions];
        //cucheck(cudaMallocManaged(&*(dnodeD+i), partitions*sizeof(Node*)));
        //for (int j=0; j<partitions; j++){
            //*(*(hnodeD+i)+j) = new Node[partitions];
            //cucheck(cudaMallocManaged(&*(*(dnodeD+i)+j), partitions*sizeof(Node)));
        //}
    //}

    make_nodos(hnodeD, dataD, size_node, partitions, np);

    //Copy to device memory
    Node *dnodeD;
    cucheck(cudaMallocManaged(&dnodeD, partitions*partitions*partitions*sizeof(Node)));
    for (int i=0; i<partitions3; i++){
        dnodeD[i] = hnodeD[i];
        if (hnodeD[i].len>0){
            cucheck(cudaMallocManaged(&dnodeD[i].elements, hnodeD[i].len*sizeof(PointW3D)));
            for (int j=0; j<hnodeD[i].len; j++){
                dnodeD[i].elements[j] = hnodeD[i].elements[j];
            }
        }
    }

    stop_timmer = clock();
    time_spent = (double)(stop_timmer - start_timmer) / CLOCKS_PER_SEC;
    printf("\nSpent time = %.4f seg.\n", time_spent );

    int idx;
    int px=1,py=2,pz=3;
    idx = pz*partitions*partitions + py*partitions + px;
    cout << "Node 1,2,3 " << "len: " << hnodeD[idx].len << "Position: " << hnodeD[idx].nodepos.x << ", " << hnodeD[idx].nodepos.y << ", " << hnodeD[idx].nodepos.z << endl;
    cout << "Elements: " << endl;
    for (int i=0; i<hnodeD[idx].len; i++){
        cout << hnodeD[idx].elements[i].x << ", " << hnodeD[idx].elements[i].y << ", " << hnodeD[idx].elements[i].z << endl;
    }

    px=3,py=3,pz=3;
    idx = pz*partitions*partitions + py*partitions + px;
    cout << "Node 3,3,3 " << "len: " << hnodeD[idx].len << "Position: " << hnodeD[idx].nodepos.x << ", " << hnodeD[idx].nodepos.y << ", " << hnodeD[idx].nodepos.z << endl;
    cout << "Elements: " << endl;
    for (int i=0; i<hnodeD[idx].len; i++){
        cout << hnodeD[idx].elements[i].x << ", " << hnodeD[idx].elements[i].y << ", " << hnodeD[idx].elements[i].z << endl;
    }
    
    px=3,py=2,pz=1;
    idx = pz*partitions*partitions + py*partitions + px;
    cout << "Node 3,2,1 " << "len: " << hnodeD[idx].len << "Position: " << hnodeD[idx].nodepos.x << ", " << hnodeD[idx].nodepos.y << ", " << hnodeD[idx].nodepos.z << endl;
    cout << "Elements: " << endl;
    for (int i=0; i<hnodeD[idx].len; i++){
        cout << hnodeD[idx].elements[i].x << ", " << hnodeD[idx].elements[i].y << ", " << hnodeD[idx].elements[i].z << endl;
    }

    pnodestest<<<1,32>>>(dnodeD, partitions);
    cucheck(cudaDeviceSynchronize());
    //for (int i=0; i<partitions; i++){
        //for (int j=0; j<partitions; j++){
            //delete[] hnodeD[i][j];
            //cucheck(cudaFree(*(*(dnodeD+i)+j)));
        //}
        //delete[] hnodeD[i];
        //cucheck(cudaFree(*(dnodeD+i)));
    //}
    delete[] hnodeD;
    cucheck(cudaFree(dnodeD));
    
    cucheck(cudaFree(dataD));
    //delete[] dataD;
    
    cout << "Finished" << endl;
    return 0;

}
