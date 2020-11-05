#include <iostream>
#include <fstream> //manejo de archivos
#include <string.h>
#include <time.h>
#include <math.h>

using namespace std;

struct PointW3D{
    float x;
    float y; 
    float z;
    float w;
};

struct Node{
    int len;		// Cantidad de elementos en el nodo.
    PointW3D *elements;	// Elementos del nodo.
};


void open_files(string name_file, int pts, PointW3D *datos){
    /* Función para abrir nuestros archivos de datos */
    ifstream file;

    string mypathto_files = "../../../fake_DATA/DATOS/";
    //This creates the full path to where I have my data files
    name_file.insert(0,mypathto_files);

    file.open(name_file.c_str(), ios::in | ios::binary); //le indico al programa que se trata de un archivo binario con ios::binary
    if (file.fail()){
        cout << "Error al cargar el archivo " << endl;
        exit(1);
    }

    for ( int c = 0; c < pts; c++)
    {
        file >> datos[c].x >> datos[c].y >> datos[c].z >> datos[c].w; 
    }
    file.close();
}

//====================================================================

void save_histogram(string name, int bns, unsigned long int *histo){
    /* Función para guardar nuestros archivos de histogramas */
    ofstream file2;
    file2.open(name.c_str(), ios::out | ios::binary);

    if (file2.fail()){
        cout << "Error al guardar el archivo " << endl;
        exit(1);
    }
    for (int i = 0; i < bns; i++){
        file2 << histo[i] << endl;
    }
    file2.close();
}

//=================================================================== 
void add(PointW3D *&array, int &lon, float _x, float _y, float _z, float _w){
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
    array[lon-1].z = _w;
}

void make_nodos(Node ***nod, PointW3D *dat, unsigned int partitions, float size_node, unsigned int np){
    /*
    Función para crear los nodos con los datos y puntos random

    Argumentos
    nod: arreglo donde se crean los nodos.
    dat: datos a dividir en nodos.
    */

    int row, col, mom;

    // Inicializamos los nodos vacíos:
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){
                nod[row][col][mom].len = 0;
                cudaMallocManaged(&nod[row][col][mom].elements, sizeof(PointW3D));
            }
        }
    }

    // Llenamos los nodos con los puntos de dat:
    for (int i=0; i<np; i++){
        row = (int)(dat[i].x/size_node);
        col = (int)(dat[i].y/size_node);
        mom = (int)(dat[i].z/size_node);
        add(nod[row][col][mom].elements, nod[row][col][mom].len, dat[i].x, dat[i].y, dat[i].z, dat[i].w);
    }
}

//====================================================================
//============ Sección de Kernels ================================== 
//====================================================================
__global__ void make_histoXX(unsigned int *XX_A, unsigned int *XX_B, PointW3D *data, int np, float ds, float dd_max){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<np-1){
        //printf("%f \n",  blockDim.x);
        int pos; // Posición de apuntador.
        float dis, dx, dy, dz;
        for(int j = idx+1; j < np; j++){
            dx = data[idx].x-data[j].x;
            dy = data[idx].y-data[j].y;
            dz = data[idx].z-data[j].z;
            dis = dx*dx + dy*dy + dz*dz;
            if(dis <= dd_max){
                pos = (int)(sqrt(dis)*ds);
                if (idx%2==0){ //Si es par lo guarda en histograma A, si no en el B
                    pos = (int)(sqrt(dis)*ds);
                    atomicAdd(&XX_A[pos],2);
                } else {
                    pos = (int)(sqrt(dis)*ds);
                    atomicAdd(&XX_B[pos],2);
                }
            }
        }
    }
}
__global__ void make_histoXY(unsigned int *XY_A, unsigned int *XY_B, PointW3D *dataD, PointW3D *dataR, int np, float ds, float dd_max){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<np-1){
        int pos;
        float dis, dx, dy, dz;
        for(int j = 0; j < np; j++){
            dx = dataD[idx].x-dataR[j].x;
            dy = dataD[idx].y-dataR[j].y;
            dz = dataD[idx].z-dataR[j].z;
            dis = dx*dx + dy*dy + dz*dz;
            if(dis <= dd_max){
                if (idx%2==0){
                    pos = (int)(sqrt(dis)*ds);
                    atomicAdd(&XY_A[pos],1);
                } else {
                    pos = (int)(sqrt(dis)*ds);
                    atomicAdd(&XY_B[pos],1);
                }
            }
        }
    }
}

int main(int argc, char **argv){
	
    int np = stoi(argv[3]), bn = stoi(argv[4]);
    float dmax = stof(argv[5]);
    float ds = (float)(bn)/dmax, dd_max=dmax*dmax, size_box = 250.0, alpha = 2.176;
    float size_node = alpha*(size_box/pow((float)(np),1/3.));
    int partitions = (int)(ceil(size_box/size_node));
    //int np = 32768, bn = 10;
    //float dmax = 180.0;

    unsigned int *DD_A, *RR_A, *DR_A, *DD_B, *RR_B, *DR_B;
    unsigned long int *DD, *RR, *DR;
    PointW3D *dataD;
    PointW3D *dataR;
    cudaMallocManaged(&dataD, np*sizeof(PointW3D));// Asignamos meoria a esta variable
    cudaMallocManaged(&dataR, np*sizeof(PointW3D));

    // Nombre de los archivos 
    string nameDD = "DDiso.dat", nameRR = "RRiso.dat", nameDR = "DRiso.dat";

    // Asignamos memoria para los histogramas
    DD = new unsigned long int[bn];
    RR = new unsigned long int[bn];
    DR = new unsigned long int[bn];
    cudaMallocManaged(&DD_A, bn*sizeof(unsigned int));
    cudaMallocManaged(&RR_A, bn*sizeof(unsigned int));
    cudaMallocManaged(&DR_A, bn*sizeof(unsigned int));
    cudaMallocManaged(&DD_B, bn*sizeof(unsigned int));
    cudaMallocManaged(&RR_B, bn*sizeof(unsigned int));
    cudaMallocManaged(&DR_B, bn*sizeof(unsigned int));
    
    //Inicializar en 0 los histogramas
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
	
	// Abrimos y guardamos los datos en los en los arrays correspondientes
	open_files(argv[1], np, dataD);
    open_files(argv[2], np, dataR);

    //Iniciar los nodos.
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
    
    //Clasificar los puntos en los nodos
    make_nodos(nodeD, dataD, partitions, size_node, np);
    make_nodos(nodeR, dataR, partitions, size_node, np);

    int blocks = (int)(ceil((float)(np/(float)(1024))));
    dim3 grid(blocks,1,1);
    dim3 block(1024,1,1);

    clock_t begin = clock();
    //make_histoXX<<<grid,block>>>(DD_A, DD_B, dataD, np, ds, dd_max);
    //make_histoXX<<<grid,block>>>(RR_A, RR_B, dataR, np, ds, dd_max);
    //make_histoXY<<<grid,block>>>(DR_A, DR_B, dataD, dataR, np, ds, dd_max);

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
    printf("\nTiempo en CPU usado = %.4f seg.\n", time_spent );

    for (int i = 0; i < bn; i++){
        DD[i] = DD_A[i]+ DD_B[i];
        RR[i] = RR_A[i]+ RR_B[i];
        DR[i] = DR_A[i]+ DR_B[i];
    }

    cout << "Termine de hacer todos los histogramas" << endl;
    // Mostramos los histogramas 
    cout << "\nHistograma DD:" << endl;
    int sum = 0;
    for (int k = 0; k<bn; k++){
        cout << DD[k] << "\t"
        sum += DD[k];
    }
    cout << "Total: " << endl;
    cout << sum << endl;

    cout << "\nHistograma RR:" << endl;
    for (int k = 0; k<bn; k++){
        cout << RR[k] << "\t"
    }
    cout << "\nHistograma DR:" << endl;
    for (int k = 0; k<bn; k++){
        cout << DR[k] << "\t"
    }
	
	
	// Guardamos los histogramas
	save_histogram(nameDD, bn, DD);
	cout << "Guarde histograma DD..." << endl;
	save_histogram(nameRR, bn, RR);
	cout << "Guarde histograma RR..." << endl;
	save_histogram(nameDR, bn, DR);
	cout << "Guarde histograma DR..." << endl;

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

