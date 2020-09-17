
#include<iostream>
#include<fstream>
#include<vector>
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
    //Punto nodepos;	// Coordenadas del nodo (posición del nodo)
    int len=0;		// Cantidad de elementos en el nodo.
    //vector<Punto> elements;	// Elementos del nodo.
    Punto *elements = new Punto[110];
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

void guardar_Histograma(string nombre,int dim, long int *histograma){
    ofstream archivo;
    archivo.open(nombre.c_str(),ios::out | ios::binary);
    if (archivo.fail()){
        cout << "Error al guardar el archivo " << endl;
        exit(1);
    }
    for (int i = 0; i < dim; i++)
    {
        archivo << histograma[i] << endl;
    }
    archivo.close();
}

float distance(Punto p1, Punto p2){
    float x = p1.x-p2.x, y=p1.y-p2.y, z=p1.z-p2.z;
    return sqrt(x*x + y*y + z*z);
}

__global__
void XY(float *dest, float *a, float *b, int *N){
    int p_id = threadIdx.x + blockDim.x*blockIdx.x;
    int id = threadIdx.y + blockDim.y*blockIdx.y;

    if (id < *N && p_id <*N){
        int x = id*3;
        int y = x+1;
        int z = y+1;

        int p_x = p_id*3;
        int p_y = p_x+1;
        int p_z = p_y+1;
        float d;
        //float histo[30];
        int bin;
        d = sqrt(pow(a[p_x] - b[x],2)+pow(a[p_y]-b[y],2) + pow(a[p_z]-b[z],2));
        if (d<=180){
            bin = (int) (d/6.0);
            atomicAdd(&dest[bin],1);
        }
    }
}

__global__
void XX(float *dest, float *a, int *N){
    int p_id = threadIdx.x + blockDim.x*blockIdx.x;
    int id = threadIdx.y + blockDim.y*blockIdx.y;

    if (p_id<*N && id<*N && p_id<id){

        int p_x = p_id*3;
        int p_y = p_x+1;
        int p_z = p_y+1;

        float d;
        int bin;

        int x = id*3;
        int y = x+1;
        int z = y+1;

        d = sqrt(pow(a[p_x] - a[x],2)+pow(a[p_y]-a[y],2) + pow(a[p_z]-a[z],2));
        if (d<=180){
            bin = (int) (d/6.0);
            atomicAdd(&dest[bin],2);
        }
    }
}

// Kernel function to populate the grid of nodes
__global__
void create_grid(Node *node_grid, Punto *datos, unsigned int n_pts, float size_node, unsigned int partitions)
{
    unsigned int nodeid;
    for(int i=0; i<n_pts;i++){
        nodeid = (int)(datos[i].x/size_node) + (int)((datos[i].y/size_node))*partitions + (int)((datos[i].z/size_node))*partitions*partitions;
        //node_grid[nodeid].elements[node_grid[nodeid].len]=datos[i];
        node_grid[nodeid].len++;
        printf("El valor es %d.\n", *(node_grid[nodeid].elements+1));
    }
}

int main(int argc, char **argv){
        
    string data_loc = argv[1];
    string rand_loc = argv[2];
    string mypathto_files = "../fake_DATA/DATOS/";
    //This creates the full path to where I have my data files
    data_loc.insert(0,mypathto_files);
    rand_loc.insert(0,mypathto_files);
    
    unsigned int n_pts = stoi(argv[3]), bn=stoi(argv[4]);
    unsigned int N_even = n_pts+(n_pts%2!=0);
    float d_max=stof(argv[5]), size_box = 250.0, size_node = 2.17*size_box/bn;
    unsigned int partitions = (int)(ceil(size_box/size_node));
    double dbin = d_max/(double)bn;

    //Punto *data = new Punto[n_pts]; //Crea un array de n_pts puntos
    //Punto *rand = new Punto[n_pts]; //Crea un array de N puntos

    Punto *data, *rand;
    cudaMallocManaged(&data, n_pts*sizeof(Punto));
    cudaMallocManaged(&rand, n_pts*sizeof(Punto));

    //Llama a una funcion que lee los puntos y los guarda en la memoria asignada a data y rand
    read_file(data_loc,data);
    read_file(rand_loc,rand);
    //read_file(data_loc,d_data);
    //read_file(rand_loc,d_rand);

    // Crea los histogramas
    long int ***DDD, ***DDR, ***DRR, ***RRR;
    // inicializamos los histogramas
    DDD = new long int**[bn];
    RRR = new long int**[bn];
    DDR = new long int**[bn];
    DRR = new long int**[bn];

    for (int i=0; i<bn; i++){
        *(DDD+i) = new long int*[bn];
        *(RRR+i) = new long int*[bn];
        *(DDR+i) = new long int*[bn];
        *(DRR+i) = new long int*[bn];
        for (int j = 0; j < bn; j++){
            *(*(DDD+i)+j) = new long int[bn];
            *(*(RRR+i)+j) = new long int[bn];
            *(*(DDR+i)+j) = new long int[bn];
            *(*(DRR+i)+j) = new long int[bn];
        }
    }
    
    //Inicializa en 0
    for (int i=0; i<bn; i++){
        for (int j=0; j<bn; j++){
            for (int k = 0; k < bn; k++){
                DDD[i][j][k]= 0;
                DDR[i][j][k]= 0;   
                DRR[i][j][k]= 0;
                RRR[i][j][k]= 0;
            }
        }
    }

    int threads=1, blocks=N_even, threads_test, blocks_test;
    float score=pow(blocks,2)+pow((blocks*threads)-N_even,2), score_test;

    for (int i=1; i<6; i++){
        threads_test = pow(2,i);
        blocks_test = (int)(N_even/threads_test)+1;
        score_test = pow(blocks_test,2)+pow((blocks_test*threads_test)-N_even,2);
        
        if (score_test<score){
            threads=threads_test;
            blocks=blocks_test;
            score=score_test;
        }
    }

    //Inicializar nodos
    Node *h_node_grid, *d_node_grid;
    h_node_grid = new Node[partitions*partitions*partitions];

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMalloc((void**)&d_node_grid,partitions*partitions*partitions*sizeof(Node));
    cudaMemcpy(d_node_grid,h_node_grid,partitions*partitions*partitions*sizeof(Node),cudaMemcpyHostToDevice);
    //cudaMallocManaged(&node_grid, partitions*partitions*partitions*sizeof(Node));

    //node_grid = new Node[partitions*partitions*partitions];
    create_grid<<<1,1>>>(d_node_grid, data, n_pts, size_node, partitions);
    //Waits for the GPU to finish
    cudaDeviceSynchronize();

    // Sustituir por kernel
    
    /*
    cout << "Im in the loop" << endl;
    int nodeid;
    for(int i = 0; i<n_pts; i++){
        nodeid = (int)(data[i].x/size_node) + (int)((data[i].y/size_node))*partitions + (int)((data[i].z/size_node))*partitions*partitions;
        cout << node_grid[nodeid].len << endl;
        //node_grid[nodeid].elements.push_back(data[i]);
        node_grid[nodeid].elements[node_grid[nodeid].len]=data[i];
        node_grid[nodeid].len++;
    }
    */
    
    cudaMemcpy(h_node_grid,d_node_grid,partitions*partitions*partitions*sizeof(Node),cudaMemcpyDeviceToHost);
    for(int j=0; j<10; j++){
        cout<<h_node_grid[j].len<<endl;
        cout<<h_node_grid[j].elements[0].x << endl;
    }
    for(int i=0; i<h_node_grid[1].len; i++){ 
       cout << h_node_grid[1].elements[i].x << " " << h_node_grid[1].elements[i].y << " " << h_node_grid[1].elements[i].z << endl; 
    }

    // Free memory
    cudaFree(&d_node_grid);
    cudaFree(&data);
    cudaFree(&rand);

    return 0;
}
