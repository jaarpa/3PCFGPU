#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>

using namespace std;

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
    Point3D nodepos;	// Coordenadas del nodo (posición del nodo).
    int len;		// Cantidad de elementos en el nodo.
    PointW3D *elements;	// Elementos del nodo.
};

//====================================================================
//============ Sección de Funciones ================================== 
//====================================================================
void open_files(string name_file, int pts, PointW3D *datos){
    /* Función para abrir nuestros archivos de datos */
    string mypathto_files = "../../fake_DATA/DATOS/";
    name_file.insert(0,mypathto_files);

    ifstream file;
    file.open(name_file.c_str(), ios::in | ios::binary); //le indico al programa que se trata de un archivo binario con ios::binary
    if (file.fail()){
        cout << "Error al cargar el archivo " << endl;
        exit(1);
    }
    for (int c = 0; c < pts; ++c) file >> datos[c].x >> datos[c].y >> datos[c].z >> datos[c].w; 
    file.close();
}

//====================================================================
void save_histogram(string name, int bns, unsigned int *histo){
    /* Función para guardar nuestros archivos de histogramas */
    ofstream file2;
    file2.open(name.c_str(), ios::out | ios::binary);

    if (file2.fail()){
        cout << "Error al guardar el archivo " << endl;
        exit(1);
    }
    for (int i=0; i<bns; ++i) file2 << histo[i] << endl;
    file2.close();
}
//====================================================================
void save_histogram_analitic(string name, int bns, float *histo){
    /* Función para guardar nuestros archivos de histogramas */
    ofstream file2;
    file2.open(name.c_str(), ios::out | ios::binary);

    if (file2.fail()){
        cout << "Error al guardar el archivo " << endl;
        exit(1);
    }
    for (int i=0; i<bns; ++i) file2 << histo[i] << endl;
    file2.close();
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

    cudaFree(array);
    array = array_aux;
    array[lon-1].x = _x;
    array[lon-1].y = _y; 
    array[lon-1].z = _z; 
}

void make_nodos(Node ***nod, Punto *dat, unsigned int partitions, float size_node, unsigned int n_pts){
    /*
    Función para crear los nodos con los datos y puntos random

    Argumentos
    nod: arreglo donde se crean los nodos.
    dat: datos a dividir en nodos.

    */
    int row, col, mom;
    //int node_id, n_row, n_col, n_mom, internode_max, id_max = pow((int) dmax/size_node + 1,2); // Row, Col and Mom of the possible node in the neighborhood

    // Inicializamos los nodos vacíos:
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){

                nod[row][col][mom].len = 0;
                cudaMallocManaged(&nod[row][col][mom].elements, sizeof(Punto));
            }
        }
    }

    // Llenamos los nodos con los puntos de dat:
    for (int i=0; i<n_pts; i++){
        row = (int)(dat[i].x/size_node);
        col = (int)(dat[i].y/size_node);
        mom = (int)(dat[i].z/size_node);
        add(nod[row][col][mom].elements, nod[row][col][mom].len, dat[i].x, dat[i].y, dat[i].z);
    }
}

int main(int argc, char **argv){

    PointW3D *dataD;
    unsigned int  *DD; 
    float *RR;

    int n_pts = stoi(argv[3]), bn = stoi(argv[4]);
    float d_max = stof(argv[5]);
    //int n_pts = 32*32*32, bn = 1000;
    float size_box = 250.0, alpha = 2.176;
    float size_node = alpha*(size_box/pow((float)(n_pts),1/3.));

    cudaMallocManaged(&dataD, n_pts*sizeof(PointW3D));

    //Mensaje a usuario
    cout << "::::::::::::::::::::::::::::::::::::::::::::::::::::::" << endl;
    cout << "::::::::::::::::::::::::::::::::::::::::::::::::::::::" << endl;
    cout << "Construcción de Histogramas DD, RR para calcular" << endl;
    cout << "la función de correlación de 2 puntos isotrópica" << endl;
    cout << "implementando el método de mallas con condiciones" << endl;
    cout << "periódicas de frontera" << endl;
    cout << "::::::::::::::::::::::::::::::::::::::::::::::::::::::" << endl;
    cout << "Parametros usados: \n" << endl;
    cout << "	Cantidad de puntos: " << n_pts << endl;
    cout << "	Bins de histogramas: " << bn << endl;
    cout << "	Distancia máxima: " << d_max << endl;
    cout << "	Tamaño de nodos: " << size_node << endl;
    cout << "\n::::::::::::::::::::::::::::::::::::::::::::::::::::::" << endl;
    cout << "::::::::::::::::::::::::::::::::::::::::::::::::::::::" << endl;
    // Nombre de los archivos 
    string nameDD = "DDiso_mesh_3D.dat", nameRR = "RRiso_mesh_3D.dat";

    // inicializamos los histogramas
    cudaMallocManaged(&DD, bn*sizeof(unsigned int));
    cudaMallocManaged(&RR, bn*sizeof(float));
    int i, j;
    for (i = 0; i < bn; ++i){
        *(DD+i) = 0; // vector[i]
        *(RR+i) = 0.0;
    }

    // Abrimos y trabajamos los datos en los histogramas
    open_files(argv[1],n_pts,dataD); // guardo los datos en los Struct

    // inicializamos las mallas
    int partitions = (int)(ceil(size_box/size_node));
    //Create Nodes
    cout << "Started nodes initialization" << endl;
    Node ***nodeD;
    cudaMallocManaged(&nodeD, partitions*sizeof(Node**));
    for (int i=0; i<partitions; i++){
        cudaMallocManaged(&*(nodeD+i), partitions*sizeof(Node*));
        for (int j=0; j<partitions; j++){
            cudaMallocManaged(&*(*(nodeD+i)+j), partitions*sizeof(Node));
        }
    }

    make_nodos(nodeD, dataD, partitions, size_node, n_pts);

    clock_t begin = clock();

    //my_hist.make_histoXX(DD, RR, my_hist.meshData()); //hace histogramas XX
    //make_histoRR_analytic

    //cudaDeviceSynchronize(); 

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
    cout << "Termine de hacer todos los histogramas" << endl;

    cout << "::::::::::::::::::::::::::::::::::::::::::::::::::::::" << endl;
    cout << "::::::::::::::::::::::::::::::::::::::::::::::::::::::" << endl;
    save_histogram(nameDD, bn, DD);
    cout << "Guarde histograma DD..." << endl;
    save_histogram_analitic(nameRR, bn, RR);
    cout << "Guarde histograma RR..." << endl;

    // Eliminamos los hitogramas y nodos
    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            cudaFree(&*(*(nodeD+i)+j));
        }
        cudaFree(&*(nodeD+i));
    }
    cudaFree(&nodeD);

    for (int i=0; i<bn; i++)
        cudaFree(&*(DD+i));
        cudaFree(&*(RR+i));
    }
    cudaFree(&DD);
    cudaFree(&RR);

    cout << "Programa finalizado..." << endl;
    return 0;
}
