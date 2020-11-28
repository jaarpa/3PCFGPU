//nvcc test.cu -o t.out && ./t.out data_5K.dat 5000
//nvcc test.cu -o t.out && ./t.out data_1GPc.dat 405224
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

void make_nodos(Node ***nod, PointW3D *dat, float size_node, float size_box, unsigned int n_pts){
    /*
    Función para crear los nodos con los datos y puntos random

    Argumentos
    nod: arreglo donde se crean los nodos.
    dat: datos a dividir en nodos.

    */
    int i, row, col, mom, partitions = (int)((size_box/size_node)+1);
    float p_med = size_node/2;

    // Inicializamos los nodos vacíos:
    for (row=0; row<partitions; row++){
    for (col=0; col<partitions; col++){
    for (mom=0; mom<partitions; mom++){
        nod[row][col][mom].nodepos.z = ((float)(mom)*(size_node))+p_med;
        nod[row][col][mom].nodepos.y = ((float)(col)*(size_node))+p_med;
        nod[row][col][mom].nodepos.x = ((float)(row)*(size_node))+p_med;
        nod[row][col][mom].len = 0;
        nod[row][col][mom].elements = new PointW3D[0];
    }
    }
    }
    // Llenamos los nodos con los puntos de dat:
    for (i=0; i<n_pts; ++i){
        row = (int)(dat[i].x/size_node);
            col = (int)(dat[i].y/size_node);
            mom = (int)(dat[i].z/size_node);
        add( nod[row][col][mom].elements, nod[row][col][mom].len, dat[i].x, dat[i].y, dat[i].z, dat[i].w);
    }
}

int main(int argc, char **argv){
    unsigned int np = stoi(argv[2]), partitions;
    float size_node, size_box = 0;//, r_size_box;
    PointW3D *dataD;
    dataD = new PointW3D[np];
    
    open_files(argv[1], np, dataD, size_box);
    size_node = 2.176*(size_box/pow((float)(np),1/3.));
    partitions = (int)(ceil(size_box/size_node));


    //Allocate memory for the nodes depending of how many partitions there are.
    Node ***hnodeD, ***dnodeD;
    hnodeD = new Node**[partitions];
    //cucheck(cudaMalloc((void**)&dnodeD, partitions*sizeof(Node**))); 
    //cucheck(cudaMallocManaged(&dnodeD, partitions*sizeof(Node**)));
    for (int i=0; i<partitions; i++){
        *(hnodeD+i) = new Node*[partitions];
        //cucheck(cudaMalloc((void**)&*(dnodeD+i), partitions*sizeof(Node*)));
        //cucheck(cudaMallocManaged(&*(dnodeD+i), partitions*sizeof(Node*)));
        for (int j=0; j<partitions; j++){
            *(*(hnodeD+i)+j) = new Node[partitions];
            //cucheck(cudaMalloc((void**)&*(*(dnodeD+i)+j), partitions*sizeof(Node)));
            //cucheck(cudaMallocManaged(&*(*(dnodeD+i)+j), partitions*sizeof(Node)));
        }
    }

    make_nodos(hnodeD, dataD, size_node, size_box, np);

    //Copy to device memory
    cucheck(cudaMalloc((void**)&dnodeD, partitions*partitions*partitions*sizeof(Node**))); //1D array
    int idx;
    PointW3D *d_node_elements;	// Points in the node
    for(int row=0; row<partitions; row++) { for(int col=0; col<partitions; col++) { for(int mom=0; mom<partitions; mom++) {
        cucheck(cudaMalloc((void**)&d_node_elements, hnodeD[row][col][mom].len*sizeof(PointW3D))); //1D array
        idx = mom*partitions*partitions+ col*partitions +row;
        cucheck(cudaMemcpy(dnodeD[idx], hnodeD[row][col][mom], sizeof(Node), cudaMemcpyHostToDevice));
        cucheck(cudaMemcpy(d_node_elements, hnodeD[row][col][mom]->elements,  hnodeD[row][col][mom].len*sizeof(PointW3D), cudaMemcpyHostToDevice));
        cucheck(cudaMemcpy(&(dnodeD[idx]->elements), &d_node_elements, hnodeD[row][col][mom].len*sizeof(PointW3D), cudaMemcpyDeviceToDevice));
        cucheck(cudaFree(d_node_elements))
    }}}


    int px=1,py=2,pz=3;
    cout << "Node 1,2,3 " << "len: " << hnodeD[px][py][pz].len << "Position: " << hnodeD[px][py][pz].nodepos.x << ", " << hnodeD[px][py][pz].nodepos.y << ", " << hnodeD[px][py][pz].nodepos.z << endl;
    cout << "Elements: " << endl;
    for (int i=0; i<hnodeD[px][py][pz].len; i++){
        cout << hnodeD[px][py][pz].elements[i].x << ", " << hnodeD[px][py][pz].elements[i].y << ", " << hnodeD[px][py][pz].elements[i].z << endl;
    }

    px=3,py=3,pz=3;
    cout << "Node 3,3,3 " << "len: " << hnodeD[px][py][pz].len << "Position: " << hnodeD[px][py][pz].nodepos.x << ", " << hnodeD[px][py][pz].nodepos.y << ", " << hnodeD[px][py][pz].nodepos.z << endl;
    cout << "Elements: " << endl;
    for (int i=0; i<hnodeD[px][py][pz].len; i++){
        cout << hnodeD[px][py][pz].elements[i].x << ", " << hnodeD[px][py][pz].elements[i].y << ", " << hnodeD[px][py][pz].elements[i].z << endl;
    }
    
    px=3,py=2,pz=1;
    cout << "Node 3,2,1 " << "len: " << hnodeD[px][py][pz].len << "Position: " << hnodeD[px][py][pz].nodepos.x << ", " << hnodeD[px][py][pz].nodepos.y << ", " << hnodeD[px][py][pz].nodepos.z << endl;
    cout << "Elements: " << endl;
    for (int i=0; i<hnodeD[px][py][pz].len; i++){
        cout << hnodeD[px][py][pz].elements[i].x << ", " << hnodeD[px][py][pz].elements[i].y << ", " << hnodeD[px][py][pz].elements[i].z << endl;
    }

    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            delete[] hnodeD[i][j];
            //cucheck(cudaFree(*(*(dnodeD+i)+j)));
        }
        delete[] hnodeD[i];
        //cucheck(cudaFree(*(dnodeD+i)));
    }
    delete[] hnodeD;
    
    cucheck(cudaFree(dnodeD));

    delete[] dataD;
    
    cout << "Finished" << endl;
    return 0;

}
