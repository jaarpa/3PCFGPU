// nvcc iso2PCF.cu -o par.out && ./par.out data.dat rand0.dat 32768 30 180
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

void save_histogram(string name, int bns, double *histo){
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
//===================================================================

__device__ void count_distances11(float *XX, PointW3D *elements, int len, float ds, float dd_max){
    /*
    Funcion para contar las distancias entre puntos en un mismo Nodo.
    */

    int bin;
    float d, v;
    float x1,y1,z1,w1,x2,y2,z2,w2;

    for (int i=0; i<len-1; ++i){
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        w1 = elements[i].w;
        for (int j=i+1; j<len; ++j){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            w2 = elements[j].w;
            d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (d<=dd_max){
                d = sqrt(d);
                bin = (int)(d*ds);
                v = 2*w1*w2;
                atomicAdd(&XX[bin],1);
            }
        }
    }
}

__device__ void count_distances12(float *XX, PointW3D *elements1, int len1, PointW3D *elements2, int len2, float ds, float dd_max){
    /*
    Funcion para contar las distancias entre puntos en un mismo Nodo.
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
            if (d<=dd_max){
                d = sqrt(d);
                bin = (int)(d*ds);
                v = 2*w1*w2;
                atomicAdd(&XX[bin],1);
            }
        }
    }
}

__global__ void make_histoXX(float *XX_A, float *XX_B, Node ***nodeD, int partitions, float ds, float dd_max, int did_max, int did_max2){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 7) { printf("%f \n",nodeD[1][1][2].elements[2].x); }
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon in this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;
        
        if (nodeD[row][col][mom].len > 0){
            
            if (idx%2==0){
                count_distances11(XX_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, ds, dd_max);
            } else {
                count_distances11(XX_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, ds, dd_max);
            }
            
            int u,v,w; //Posicion del nodo 2
            unsigned int dx_nod12, dy_nod12, dz_nod12, dd_nod12;

            //Nodo2 solo movil en z
            for(w = mom+1; w<partitions && w-row<=did_max; w++){
                if (idx%2==0){ //Si es par lo guarda en histograma A, si no en el B
                    count_distances12(XX_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[row][col][w].elements, nodeD[row][col][w].len, ds, dd_max);
                } else {
                    count_distances12(XX_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[row][col][w].elements, nodeD[row][col][w].len, ds, dd_max);
                }
            }

            //Nodo2 movil en ZY
            for(v=col+1; v<partitions && v-col<=did_max; v++){
                dy_nod12 = v-col;
                for(w=(mom-did_max)*(mom>did_max); w<partitions && w-mom<=did_max; w++){
                    dz_nod12 = w-mom;
                    dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12;
                    if (dd_nod12<=did_max2){
                        if (idx%2==0){ //Si es par lo guarda en histograma A, si no en el B
                            count_distances12(XX_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[row][v][w].elements, nodeD[row][v][w].len, ds, dd_max);
                        } else {
                            count_distances12(XX_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[row][v][w].elements, nodeD[row][v][w].len, ds, dd_max);
                        }
                    }
                }
            }

            //Nodo movil en XYZ
            for(u = row+1; u < partitions && u-row< did_max; u++){
                dx_nod12 = u-row;
                for(v = (col-did_max)*(col>did_max); v < partitions && v-col< did_max; v++){
                    dy_nod12 = v-col;
                    for(w = (mom-did_max)*(mom>did_max); w < partitions && w-mom< did_max; w++){
                        dz_nod12 = w-mom;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=did_max2){
                            if (idx%2==0){ //Si es par lo guarda en histograma A, si no en el B
                                count_distances12(XX_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[u][v][w].elements, nodeD[u][v][w].len, ds, dd_max);
                            } else {
                                count_distances12(XX_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeD[u][v][w].elements, nodeD[u][v][w].len, ds, dd_max);
                            }
                        }
                    }
                }
            }

        }
        if (idx == 7) { printf("Exit the kernel"); }
    }
}
__global__ void make_histoXY(float *XY_A, float *XY_B, Node ***nodeD, Node ***nodeR, int partitions, float ds, float dd_max, int did_max, int did_max2){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<(partitions*partitions*partitions)){
        //Get the node positon in this thread
        int mom = (int) (idx/(partitions*partitions));
        int col = (int) ((idx%(partitions*partitions))/partitions);
        int row = idx%partitions;
        
        if (nodeD[row][col][mom].len > 0){
            
            int u,v,w; //Posicion del nodo 2
            unsigned int dx_nod12, dy_nod12, dz_nod12, dd_nod12;

            //Nodo2 solo movil en z
            w = 0;//(mom-did_max)*(mom>did_max);
            for(w = (mom-did_max)*(mom>did_max); w<partitions && w-row<=did_max; w++){
                if (idx%2==0){ //Si es par lo guarda en histograma A, si no en el B
                    count_distances12(XY_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[row][col][w].elements, nodeR[row][col][w].len, ds, dd_max);
                } else {
                    count_distances12(XY_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[row][col][w].elements, nodeR[row][col][w].len, ds, dd_max);
                }
            }

            //Nodo2 movil en ZY
            for(v = (col-did_max)*(col>did_max); v<partitions && v-col<=did_max; v++){
                dy_nod12 = v-col;
                for(w = (mom-did_max)*(mom>did_max); w<partitions && w-mom<=did_max; w++){
                    dz_nod12 = w-mom;
                    dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12;
                    if (dd_nod12<=did_max2){
                        if (idx%2==0){ //Si es par lo guarda en histograma A, si no en el B
                            count_distances12(XY_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[row][v][w].elements, nodeR[row][v][w].len, ds, dd_max);
                        } else {
                            count_distances12(XY_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[row][v][w].elements, nodeR[row][v][w].len, ds, dd_max);
                        }
                    }
                }
            }

            //Nodo movil en XYZ
            for(u = (row-did_max)*(row>did_max); u < partitions && u-row< did_max; u++){
                dx_nod12 = u-row;
                for(v = (col-did_max)*(col>did_max); v < partitions && v-col< did_max; v++){
                    dy_nod12 = v-col;
                    for(w = (mom-did_max)*(mom>did_max); w < partitions && w-mom< did_max; w++){
                        dz_nod12 = w-mom;
                        dd_nod12 = dz_nod12*dz_nod12 + dy_nod12*dy_nod12 + dx_nod12*dx_nod12;
                        if (dd_nod12<=did_max2){
                            if (idx%2==0){ //Si es par lo guarda en histograma A, si no en el B
                                count_distances12(XY_A, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[u][v][w].elements, nodeR[u][v][w].len, ds, dd_max);
                            } else {
                                count_distances12(XY_B, nodeD[row][col][mom].elements, nodeD[row][col][mom].len, nodeR[u][v][w].elements, nodeR[u][v][w].len, ds, dd_max);
                            }
                        }
                    }
                }
            }
            
        }
    }
}

int main(int argc, char **argv){
	
    unsigned int np = stoi(argv[3]), bn = stoi(argv[4]);
    float dmax = stof(argv[5]);
    float ds = (float)(bn)/dmax, dd_max=dmax*dmax, size_box = 250.0, alpha = 2.176;
    float size_node = alpha*(size_box/pow((float)(np),1/3.));
    int did_max = (int)(ceil(dmax/size_node));
    int did_max2 = (int)(ceil(dd_max/(size_node*size_node)));
    unsigned int partitions = (int)(ceil(size_box/size_node));
    //int np = 32768, bn = 10;
    //float dmax = 180.0;

    float *DD_A, *RR_A, *DR_A, *DD_B, *RR_B, *DR_B;
    double *DD, *RR, *DR;
    PointW3D *dataD;
    PointW3D *dataR;
    cudaMallocManaged(&dataD, np*sizeof(PointW3D));// Asignamos meoria a esta variable
    cudaMallocManaged(&dataR, np*sizeof(PointW3D));

    // Nombre de los archivos 
    string nameDD = "DDiso.dat", nameRR = "RRiso.dat", nameDR = "DRiso.dat";

    // Asignamos memoria para los histogramas
    DD = new double[bn];
    RR = new double[bn];
    DR = new double[bn];
    cudaMallocManaged(&DD_A, bn*sizeof(float));
    cudaMallocManaged(&RR_A, bn*sizeof(float));
    cudaMallocManaged(&DR_A, bn*sizeof(float));
    cudaMallocManaged(&DD_B, bn*sizeof(float));
    cudaMallocManaged(&RR_B, bn*sizeof(float));
    cudaMallocManaged(&DR_B, bn*sizeof(float));
    
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

    int blocks = (int)(ceil((float)((partitions*partitions*partitions)/(float)(1024))));
    dim3 grid(blocks,1,1);
    dim3 block(1024,1,1);

    clock_t begin = clock();
    make_histoXX<<<grid,block>>>(DD_A, DD_B, nodeD, partitions, ds, dd_max, did_max, did_max2);
    make_histoXX<<<grid,block>>>(RR_A, RR_B, nodeR, partitions, ds, dd_max, did_max, did_max2);
    make_histoXY<<<grid,block>>>(DR_A, DR_B, nodeD, nodeR, partitions, ds, dd_max, did_max, did_max2);

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
        DD[i] = (double)(DD_A[i]+ DD_B[i]);
        RR[i] = (double)(RR_A[i]+ RR_B[i]);
        DR[i] = (double)(DR_A[i]+ DR_B[i]);
    }

    cout << "Termine de hacer todos los histogramas" << endl;
    // Mostramos los histogramas 
    cout << "\nHistograma DD:" << endl;
    int sum = 0;
    for (int k = 0; k<bn; k++){
        cout << DD[k] << "\t";
        sum += DD[k];
    }
    cout << "Total: " << sum << endl;

    cout << "\nHistograma DD_A:" << endl;
    for (int k = 0; k<bn; k++){
        cout << DD_A[k] << "\t";
        sum += DD_A[k];
    }
    cout << "Total: " << sum << endl;

    cout << "\nHistograma RR:" << endl;
    for (int k = 0; k<bn; k++){
        cout << RR[k] << "\t";
    }

    cout << "\nHistograma DR:" << endl;
    for (int k = 0; k<bn; k++){
        cout << DR[k] << "\t";
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

