// nvcc distances.cu -o o.out && ./o.out data.dat rand0.dat 32768 30 180
#include<iostream>
#include<fstream>
#include<string.h>
#include <stdio.h>
#include <math.h>

using namespace std;

//Structura que define un punto 3D
//Accesa a cada componente con var.x, var.y, var.z
struct Punto{
    double x,y,z;
};

struct Node{
    //Punto nodepos;	// Coordenadas del nodo (posición del nodo) // Se obtiene con las coordenadas del nodo.
    //int in_vicinage;    //Cantidad de nodos vecinos.
    //int *nodes_vicinage;     // Array con los master id de localizacion de los nodos vecinos.
    int len;		// Cantidad de elementos en el nodo.
    Punto *elements;
};

void read_file(string file_loc, Punto *data){
    //cout << file_loc << endl;
    string line; //No uso esta variable realmente, pero con eof() no se detenía el loop
    
    ifstream archivo(file_loc);
    
    if (archivo.fail() | !archivo ){
        cout << "Error al cargar el archivo " << endl;
        exit(1);
    }
    
    
    int n_line = 1;
    if (archivo.is_open() && archivo.good()){
        archivo >> data[0].x >> data[0].y >> data[0].z;
        while(getline(archivo, line)){
            archivo >> data[n_line].x >> data[n_line].y >> data[n_line].z;
            n_line++;
        }
    }
    //cout << "Succesfully readed " << file_loc << endl;
}

void save_histogram(string name, int bns, unsigned int ***histo){
    int i, j, k;
    unsigned int **reshape = new unsigned int*[bns];
    for (i=0; i<bns; i++){
        *(reshape+i) = new unsigned int[bns*bns];
        }
    for (i=0; i<bns; i++){
    for (j=0; j<bns; j++){
    for (k=0; k<bns; k++){
        reshape[i][bns*j+k] = histo[i][j][k];
    }
    }
    }
    ofstream file;
    file.open(name.c_str(),ios::out | ios::binary);
    if (file.fail()){
        cout << "Error al guardar el archivo " << endl;
        exit(1);
    }
    for (i=0; i<bns; i++){
        for (j=0; j<bns*bns; j++){
            file << reshape[i][j] << " "; 
        }
        file << endl;
    }
    file.close();
}

__global__
void count_3_N111(Punto *elements, int len, float dmax2, float ds, unsigned int ***XXX){
    /*
    Funcion para contar los triangulos en un mismo Nodo.

    row, col, mom => posición del Nodo. Esto define al Nodo.

    */
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<len){
        float x1,y1,z1,x2,y2,z2,x3,y3,z3;
        float d12,d13,d23;

        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;

        for (int j=i; j<len; j++){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            d12 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);

            if (d12<=dmax2){
                d12 = sqrt(d12);
                for (int k=j; k<len; j++){
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    d23 = (x2-x3)*(x2-x3) + (y2-y3)*(y2-y3) + (z2-z3)*(z2-z3);
                    if (d23<=dmax2){
                        d23 = sqrt(d23);
                        d13 = (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1) + (z3-z1)*(z3-z1);
                        if (d13<=dmax2){
                            d13 = sqrt(d13);
                            unsigned int a = (unsigned int)(d12*ds);
                            unsigned int b = (unsigned int)(d13*ds);
                            unsigned int c = (unsigned int)(d23*ds);
                            atomicAdd(&XXX[a][b][c],1);
                        }
                    }
                }
            }
        }
    }
}

// Kernel function to populate the grid of nodes
__global__
void histo_XXX(Node ***tensor_node, unsigned int ***XXX, unsigned int partitions, float dmax2, float ds)
{
    if (blockIdx.x<partitions && threadIdx.x<partitions && threadIdx.y<partitions ){
        unsigned int row, col, mom;
        row = threadIdx.x;
        col = threadIdx.y;
        mom = blockIdx.x;
        if (tensor_node[row][col][mom].len<=1024){
            dim3 grid_N111(1,1,1);
            dim3 block_N111(1024,1,1);
        } else {
            unsigned int N111_blocks;
            N111_blocks = (int)(tensor_node[row][col][mom].len/1024 + 1);
            dim3 grid_N111(N111_blocks,1,1);
            dim3 block_N111(1024,1,1);
            
        }
        count_3_N111<<<1,1024>>>(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len, dmax2, ds, XXX);
        
        __syncthreads();

        printf("Exit the kernel \n");
    }
}

void add_neighbor(int *&array, int &lon, int id){
    lon++;
    /*
    int *array_aux;
    cudaMallocManaged(&array_aux, lon*sizeof(int)); 
    for (int i=0; i<lon-1; i++){
        array_aux[i] = array[i];
    }
    cudaFree(&array);
    array = array_aux;
    */
    array[lon-1] = id;
}

//=================================================================== 
void add(Punto *&array, int &lon, float _x, float _y, float _z){
    lon++;
    Punto *array_aux; // = new Punto[lon];
    cudaMallocManaged(&array_aux, lon*sizeof(Punto)); 
    for (int i=0; i<lon-1; i++){
        array_aux[i].x = array[i].x;
        array_aux[i].y = array[i].y;
        array_aux[i].z = array[i].z;
    }
    cudaFree(&array);
    array = array_aux;
    array[lon-1].x = _x;
    array[lon-1].y = _y; 
    array[lon-1].z = _z; 
}

void make_nodos(Node ***nod, Punto *dat, unsigned int partitions, float size_node, unsigned int n_pts, float dmax){
    /*
    Función para crear los nodos con los datos y puntos random

    Argumentos
    nod: arreglo donde se crean los nodos.
    dat: datos a dividir en nodos.

    */
    int row, col, mom, id_max = pow((int) dmax/size_node + 1,2);
    //int n_row, n_col, n_mom, internodal_distance; // Row, Col and Mom of the possible node in the neighborhood

    // Inicializamos los nodos vacíos:
    cout << "Initialize empty nodes" << endl;
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){

                nod[row][col][mom].len = 0;
                cudaMallocManaged(&nod[row][col][mom].elements, sizeof(Punto));
            }
        }
    }
    cout << "The nodes have 0 elements each and 0 neighbors" << endl;

    // Llenamos los nodos con los puntos de dat:
    cout << "Started the classification" << endl;
    for (int i=0; i<n_pts; i++){
        row = (int)(dat[i].x/size_node);
        col = (int)(dat[i].y/size_node);
        mom = (int)(dat[i].z/size_node);
        add(nod[row][col][mom].elements, nod[row][col][mom].len, dat[i].x, dat[i].y, dat[i].z);
    }
    cout << "Finished the classification" << endl;
}

int main(int argc, char **argv){
        
    string data_loc = argv[1];
    string rand_loc = argv[2];
    string mypathto_files = "../../fake_DATA/DATOS/";
    //This creates the full path to where I have my data files
    data_loc.insert(0,mypathto_files);
    rand_loc.insert(0,mypathto_files);
    
    unsigned int n_pts = stoi(argv[3]), bn=stoi(argv[4]);
    unsigned int n_even = n_pts+(n_pts%2!=0);
    float dmax=stof(argv[5]), size_box = 250.0, size_node = 2.17*size_box/bn;
    float ds = ((float)(bn))/dmax, dmax2=dmax*dmax;
    unsigned int partitions = (int)(ceil(size_box/size_node));
    double dbin = dmax/(double)bn;
    
    // Crea los histogramas
    //cout << "Histograms initialization" << endl;
    unsigned int ***DDD;
    // inicializamos los histogramas
    cudaMallocManaged(&DDD, bn*sizeof(unsigned int**));
    for (int i=0; i<bn; i++){
        cudaMallocManaged(&*(DDD+i), bn*sizeof(unsigned int*));
        for (int j = 0; j < bn; j++){
            cudaMallocManaged(&*(*(DDD+i)+j), bn*sizeof(unsigned int));
        }
    }
    //Inicializa en 0
    for (int i=0; i<bn; i++){
        for (int j=0; j<bn; j++){
            for (int k = 0; k < bn; k++){
                DDD[i][j][k]= 0;
            }
        }
    }
    //cout << "Finished histograms initialization" << endl;

    //cout << "Starting to read the data files" << endl;
    Punto *data, *rand; //Crea un array de n_pts puntos
    cudaMallocManaged(&data, n_pts*sizeof(Punto));
    cudaMallocManaged(&rand, n_pts*sizeof(Punto));
    //Llama a una funcion que lee los puntos y los guarda en la memoria asignada a data y rand
    read_file(data_loc,data);
    read_file(rand_loc,rand);
    cout << "Successfully readed the data" << endl;

    //Create Nodes
    //cout << "Started nodes initialization" << endl;
    Node ***nodeD;
    cudaMallocManaged(&nodeD, partitions*sizeof(Node**));
    for (int i=0; i<partitions; i++){
        cudaMallocManaged(&*(nodeD+i), partitions*sizeof(Node*));
        for (int j=0; j<partitions; j++){
            cudaMallocManaged(&*(*(nodeD+i)+j), partitions*sizeof(Node));
        }
    }
    //cout << "Finished nodes initialization" << endl;
    //cout << "Started the data classification into the nodes." << endl;
    make_nodos(nodeD, data, partitions, size_node, n_pts, dmax);
    cout << "Finished the data classification in node" << endl;

    //cout << "Calculating the nuber of blocks and threads for the kernel for XXX" << endl;
    //Sets GPU arrange of threads
    int threads=1, blocks=n_even, threads_test, blocks_test;
    float score=pow(blocks,2)+pow((blocks*threads)-n_even,2), score_test;
    for (int i=1; i<6; i++){
        threads_test = pow(2,i);
        blocks_test = (int)(n_even/threads_test)+1;
        score_test = pow(blocks_test,2)+pow((blocks_test*threads_test)-n_even,2);
        
        if (score_test<score){
            threads=threads_test;
            blocks=blocks_test;
            score=score_test;
        }
    }
    
    cout << "Entering to the kernel" << endl;
    dim3 grid(16,1,1);
    dim3 block(16,16);
    histo_XXX<<<grid,block>>>(nodeD, DDD, partitions, dmax2, ds);

    //Waits for the GPU to finish
    cudaDeviceSynchronize();

    cout << DDD[0][0][0] << endl;

    // Free memory
    // Free the histogram arrays
    cout << "Free the histograms allocated memory" << endl;
    for (int i=0; i<bn; i++){
        for (int j = 0; j < bn; j++){
            cudaFree(&*(*(DDD+i)+j));
        }
        cudaFree(&*(DDD+i));
    }
    cudaFree(&DDD);
    //Free the nodes and their inner arrays.
    cout << "Free the nodes allocated memory" << endl;
    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            cudaFree(&*(*(nodeD+i)+j));
        }
        cudaFree(&*(nodeD+i));
    }
    cudaFree(&nodeD);
    //Free data and random arrays
    cout << "Free the data allocated memory" << endl;
    cudaFree(&data);
    cudaFree(&rand);

    cout << "Finished the program" << endl;

    return 0;
}
