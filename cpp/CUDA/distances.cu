
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
    //Punto nodepos;	// Coordenadas del nodo (posición del nodo) // Se obtiene con las coordenadas del nodo.
    int in_vicinage;    //Cantidad de nodos vecinos.
    int *nodes_vicinage;     // Array con los master id de localizacion de los nodos vecinos.
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
void create_grid(Node ***XXX, Punto *data_node, long int ***DDD, unsigned int n_pts)
{
    if (blockIdx.x==0 && blockIdx.y==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0 ){
       //printf("%i \n", threadIdx.x);
       //XXX[0][0][0].elements[1].x = data_node[1].x + data_node[1].y +data_node[1].z;
       //XXX[0][0][0].elements[1].y = data_node[1].x + data_node[1].y +data_node[1].z;
       //XXX[0][0][0].elements[1].z = data_node[1].x + data_node[1].y +data_node[1].z;
       DDD[0][0][0]=100;
       XXX[0][0][0].len = (int) (XXX[0][0][0].elements[1].x + XXX[0][0][0].elements[1].y +XXX[0][0][0].elements[1].z);
       printf("Exit the kernel");
    }
    
    /*
    for(int i=0; i<n_pts;i++){
        nodeid = (int)(datos[i].x/size_node) + (int)((datos[i].y/size_node))*partitions + (int)((datos[i].z/size_node))*partitions*partitions;
        //node_grid[nodeid].elements[node_grid[nodeid].len]=datos[i];
        node_grid[nodeid].len++;
        printf("El valor es %d.\n", *(node_grid[nodeid].elements+1));
    }
    */
}

void add_neighbor(int *&array, int &lon, int id){
    lon++;
    int *array_aux; // = new Punto[lon];
    cudaMallocManaged(&array_aux, lon*sizeof(int)); 
    for (int i=0; i<lon-1; i++){
        array_aux[i] = array[i];
    }
    cudaFree(&array);
    array = array_aux;
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
void make_nodos(Node ***nod, Punto *dat, unsigned int partitions, float size_node, unsigned int n_pts, float d_max){
    /*
    Función para crear los nodos con los datos y puntos random

    Argumentos
    nod: arreglo donde se crean los nodos.
    dat: datos a dividir en nodos.

    */
    int row, col, mom, id_max = pow((int) d_max/size_node + 1,2);

    // Inicializamos los nodos vacíos:
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){

                nod[row][col][mom].len = 0;
                cudaMallocManaged(&nod[row][col][mom].elements, sizeof(Punto));
                nod[row][col][mom].in_vicinage = 0;
                cudaMallocManaged(&nod[row][col][mom].nodes_vicinage, sizeof(int));
                //nod[row][col][mom].elements = new Punto[0];
            }
        }
    }

    // Agregar los nodos que seran vecinos. Aquí podría agregar las condiciones periodicas de frontera
    /*
    int t_row,t_col,t_mom;
    for (int i=0; i<partitions*partitions*partitions; i++){
        row = i%partitions;
        col = (int) (i%(partitions*partitions))/partitions;
        mom = (int) i/(partitions*partitions);
        for (int j=i; j<partitions*partitions*partitions; j++){
            t_row = j%partitions;
            t_col = (int) (j%(partitions*partitions))/partitions;
            t_mom = (int) j/(partitions*partitions);
            if ( (t_row-row)*(t_row-row) + (t_col-col)*(t_col-col) + (t_mom-mom)*(t_mom-mom) < id_max ){
                add_neighbor(nod[row][col][mom].nodes_vicinage, nod[row][col][mom].in_vicinage, j);
            }
        }
    }
*/

    // Llenamos los nodos con los puntos de dat:
    for (int i=0; i<n_pts; i++){
        row = (int)(dat[i].x/size_node);
        col = (int)(dat[i].y/size_node);
        mom = (int)(dat[i].z/size_node);
        add(nod[row][col][mom].elements, nod[row][col][mom].len, dat[i].x, dat[i].y, dat[i].z);
    }
}

int main(int argc, char **argv){
        
    string data_loc = argv[1];
    string rand_loc = argv[2];
    string mypathto_files = "../../fake_DATA/DATOS/";
    //This creates the full path to where I have my data files
    data_loc.insert(0,mypathto_files);
    rand_loc.insert(0,mypathto_files);
    
    unsigned int n_pts = stoi(argv[3]), bn=stoi(argv[4]);
    unsigned int N_even = n_pts+(n_pts%2!=0);
    float d_max=stof(argv[5]), size_box = 250.0, size_node = 2.17*size_box/bn;
    unsigned int partitions = (int)(ceil(size_box/size_node));
    double dbin = d_max/(double)bn;
    
    // Crea los histogramas
    long int ***DDD;
    // inicializamos los histogramas
    cudaMallocManaged(&DDD, bn*sizeof(long int**));
    for (int i=0; i<bn; i++){
        cudaMallocManaged(&*(DDD+i), bn*sizeof(long int*));
        for (int j = 0; j < bn; j++){
            cudaMallocManaged(&*(*(DDD+i)+j), bn*sizeof(long int));
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

    //Sets GPU arrange of threads
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

    Punto *data, *rand; //Crea un array de n_pts puntos
    cudaMallocManaged(&data, n_pts*sizeof(Punto));
    cudaMallocManaged(&rand, n_pts*sizeof(Punto));
    //Llama a una funcion que lee los puntos y los guarda en la memoria asignada a data y rand
    read_file(data_loc,data);
    read_file(rand_loc,rand);

    //Create Nodes
    Node ***nodeD;
    cudaMallocManaged(&nodeD, partitions*sizeof(Node**));
    for (int i=0; i<partitions; i++){
        cudaMallocManaged(&*(nodeD+i), partitions*sizeof(Node*));
        for (int j=0; j<partitions; j++){
            cudaMallocManaged(&*(*(nodeD+i)+j), partitions*sizeof(Node));
        }
    }
    make_nodos(nodeD, data, partitions, size_node, n_pts, d_max);
    
    cout<< DDD[0][0][0] << endl;
    cout<< nodeD[0][0][0].elements[1].x << ' ' << nodeD[0][0][0].elements[1].y << ' ' << nodeD[0][0][0].elements[1].z << endl;
    cout<< (int) (data[1].x + data[1].y +data[1].z) << endl;
    cout<< nodeD[0][0][0].len << endl;
    cout << "Entering to the kernel" << endl;

    create_grid<<<1,256>>>(nodeD, data, DDD, n_pts);

    //Waits for the GPU to finish
    cudaDeviceSynchronize();

    cout<< nodeD[0][0][0].elements[1].x << ' ' << nodeD[0][0][0].elements[1].y << ' ' << nodeD[0][0][0].elements[1].z << endl; //-(double)24.909824 << endl;
    cout<< nodeD[0][0][0].len << endl;
    cout<< DDD[0][0][0] << endl;

    // Free memory
    // Free the histogram arrays
    for (int i=0; i<bn; i++){
        for (int j = 0; j < bn; j++){
            cudaFree(&*(*(DDD+i)+j));
        }
        cudaFree(&*(DDD+i));
    }
    cudaFree(&DDD);
    //Free the nodes and their inner arrays.
    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            cudaFree(&*(*(nodeD+i)+j));
        }
        cudaFree(&*(nodeD+i));
    }
    cudaFree(&nodeD);
    //Free data and random arrays
    cudaFree(&data);
    cudaFree(&rand);

    return 0;
}
