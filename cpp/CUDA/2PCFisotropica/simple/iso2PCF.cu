#include <iostream>
#include <fstream> //manejo de archivos
#include <string.h>
#include <time.h>
#include <math.h>

using namespace std;

struct Point3D{
    float x;
    float y;
    float z;
};

//====================================================================
//============ Sección de Funciones ================================== 
//====================================================================

void open_files(string name_file, int pts, Point3D *datos){
    /* Función para abrir nuestros archivos de datos */
    ifstream file;

    string mypathto_files = "../../../../fake_DATA/DATOS/";
    //This creates the full path to where I have my data files
    name_file.insert(0,mypathto_files);

    file.open(name_file.c_str(), ios::in | ios::binary); //le indico al programa que se trata de un archivo binario con ios::binary
    if (file.fail()){
        cout << "Error al cargar el archivo " << endl;
        exit(1);
    }

    //int c=0,remove;
    int remove;
    //while (!file.eof())
    for ( int c = 0; c < pts; c++)
    {
        file >> datos[c].x >> datos[c].y >> datos[c].z >> remove; 
        //c++;
    }
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
    for (int i = 0; i < bns; i++){
        file2 << histo[i] << endl;
    }
    file2.close();
}

// Métodos para hacer histogramas.
__global__ void make_histoXX(unsigned int *XX, Point3D *data, int n_pts, float ds, float dd_max){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<n_pts-1){
        printf("%f \n",  blockDim.x);
        int pos; // Posición de apuntador.
        float dis, dx, dy, dz;
        for(int j = idx+1; j < n_pts; j++){
            dx = data[idx].x-data[j].x;
            dy = data[idx].y-data[j].y;
            dz = data[idx].z-data[j].z;
            dis = dx*dx + dy*dy + dz*dz;
            //printf("%f \n", dis);
            //printf("%f \n", sqrt(dis));
            //printf("%f \n", sqrt(dis)*ds);

            if(dis <= dd_max){
                pos = (int)(sqrt(dis)*ds);
                atomicAdd(&XX[pos],2);
            }
        }
    }
}
__global__ void make_histoXY(unsigned int *XY, Point3D *dataD, Point3D *dataR, int n_pts, float ds, float dd_max){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<n_pts-1){
        int pos;
        float dis, dx, dy, dz;
        for(int j = 0; j < n_pts; j++){
            dx = dataD[idx].x-dataR[j].x;
            dy = dataD[idx].y-dataR[j].y;
            dz = dataD[idx].z-dataR[j].z;
            dis = dx*dx + dy*dy + dz*dz;
            if(dis <= dd_max){
                pos = (int)(sqrt(dis)*ds);
                atomicAdd(&XY[pos],1);
            }
        }
    }
}

int main(int argc, char **argv){
	
    int np = stoi(argv[3]), bn = stoi(argv[4]);
    float dmax = stof(argv[5]);
    float ds = (float)(bn)/dmax, dd_max=dmax*dmax;
    //int np = 32768, bn = 10;
    //float dmax = 180.0;

    unsigned int *DD, *RR, *DR;
    Point3D *dataD;
    Point3D *dataR;
    cudaMallocManaged(&dataD, np*sizeof(Point3D));// Asignamos meoria a esta variable
    cudaMallocManaged(&dataR, np*sizeof(Point3D));

    // Nombre de los archivos 
    string nameDD = "DDiso.dat", nameRR = "RRiso.dat", nameDR = "DRiso.dat";
    /*
    nameDD.append(argv[3]);
    nameRR.append(argv[3]);
    nameDR.append(argv[3]);
    nameDD += ".dat";
    nameRR += ".dat";
    nameDR += ".dat";
    */

    // Creamos los histogramas
    cudaMallocManaged(&DD, bn*sizeof(unsigned int));
    cudaMallocManaged(&RR, bn*sizeof(unsigned int));
    cudaMallocManaged(&DR, bn*sizeof(unsigned int));
    
    for (int i = 0; i < bn; i++){
        *(DD+i) = 0.0; // vector[i]
        *(RR+i) = 0.0;
        *(DR+i) = 0.0;
    }
	
	// Abrimos y trabajamos los datos en los histogramas
	open_files(argv[1], np, dataD);
    open_files(argv[2], np, dataR); // guardo los datos en los Struct
    
    dim3 grid((int)(ceil((float)(np/(float)(1024))),1,1));
    dim3 block(1024,1,1);

    clock_t begin = clock();
    cout << ds << endl;
    make_histoXX<<<grid,block>>>(DD, dataD, np, ds, dd_max);
    make_histoXX<<<grid,block>>>(RR, dataR, np, ds, dd_max);
    make_histoXY<<<grid,block>>>(DR, dataD, dataR, np, ds, dd_max);

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
    
	cout << "Termine de hacer todos los histogramas" << endl;
	// Mostramos los histogramas 
	cout << "\nHistograma DD:" << endl;
	for (int k = 0; k<bn; k++){
		printf("%d \t",DD[k]);
	}
	cout << "\nHistograma RR:" << endl;
	for (int k = 0; k<bn; k++){
		printf("%d \t",RR[k]);
	}
	cout << "\nHistograma DR:" << endl;
	for (int k = 0; k<bn; k++){
		printf("%d \t",DR[k]);
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
    cudaFree(&DD);
    cudaFree(&RR);
    cudaFree(&DR);

    cout << "Programa Terminado..." << endl;
    return 0;
}

