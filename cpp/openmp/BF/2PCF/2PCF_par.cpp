#include<iostream>
#include<fstream>
#include<string.h>
#include <cmath>
#include<omp.h>

using namespace std;

//Structura que define un punto 3D
//Accesa a cada componente con var.x, var.y, var.z
struct Punto{
    double x,y,z;
};

void read_file(string file_loc, Punto *data){
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
}

void guardar_Histograma(string nombre,int dim, long int*histograma){
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

/*
Parameters:
1 Data file name
2 Random file name
3 Number of points
4 Numero de bins
5 Distancia maxima
*/
int main(int argc, char **argv){
    
    double tot_start = omp_get_wtime();
    
    string mypathto_files = "../../../../fake_DATA/DATOS/";
    
    unsigned int N = stoi(argv[3]), bins=stoi(argv[4]);
    float d_max=stof(argv[5]);
    Punto *data = new Punto[N]; //Crea un array de N puntos
    Punto *rand = new Punto[N]; //Crea un array de N puntos

    //Llama a una funcion que lee los puntos y los guarda en la memoria asignada a data y rand
    
    #pragma omp parallel num_threads(2)
    {
        int ID = omp_get_thread_num();
        string file = argv[ID+1];
        file.insert(0,mypathto_files);
        
        cout << "Number of threads to read file: " << omp_get_num_threads() << endl;
        if (ID==0) read_file(file,data);
        if (ID==1) read_file(file,rand);
    }
    
    //Ahora puedo trabajar con data y rand como una lista de puntos con coordenadas xy

    /* Para revisar que sí haya leido los archivos
    int c = 0;
    cout << "First 3 in data \n" << endl;
    for (int i=0; i<3; i++){
        cout << c << endl;
        cout << "x: " << data[i].x <<"  y: " << data[i].y <<"  z: " << data[i].z <<endl;
        c++;
    }
    
    c = 0;
    cout << "First 3 in random \n" << endl;
    for (int i=0; i<3; i++){
        cout << c << endl;
        cout << "x: " << rand[i].x <<"  y: " << rand[i].y <<"  z: " << rand[i].z <<endl;
        c++;
    }
    */

    // Crea los histogramas
    long int *DD, *DR, *RR;
    DD = new long int[bins];
    omp_lock_t DD_lock[bins];
    DR = new long int[bins];
    omp_lock_t DR_lock[bins];
    RR = new long int[bins];
    omp_lock_t RR_lock[bins];
    
    //omp_init_lock()
    //omp_set_lock()
    //omp_unset_lock()
    //omp_destroy_lock()
    //omp_test_lock()
    
    //Inicializa en 0 y bloquea cada bin en el histograma
    
    #pragma openmp parallel for
    for (int i=0; i<bins; i++){
        omp_init_lock(&DD_lock[i]),omp_init_lock(&RR_lock[i]),omp_init_lock(&DR_lock[i]);
        DD[i] = 0, RR[i] = 0, DR[i] = 0;
    }
    
    
    double dbin = (double)bins/d_max;
    
    //Hace el conteo para el histograma DD
    #pragma openmp parallel for
    for (int i=0; i<N-1; i++){
        float d;
        int idh;
        for (int j=i+1; j<N; j++){
            d = distance(data[i],data[j]);
            if (d<=d_max){
                idh = (int)(d*dbin);
                omp_set_lock(&DD_lock[idh]);
                    DD[(int)(d*dbin)]+=2;
                omp_unset_lock(&DD_lock[idh]);
            }
	        d = distance(rand[i],rand[j]);
            if (d<=d_max){
                idh = (int)(d*dbin);
                omp_set_lock(&RR_lock[idh]);
                    RR[(int)(d*dbin)]+=2;
                omp_unset_lock(&RR_lock[idh]);
            }
        }
    }
    
    //Hace el conteo para el histograma DR
    #pragma openmp parallel for
    for (int i=0; i<N; i++){
        float d;
        int idh;
        for (int j=0; j<N; j++){
            d = distance(rand[i],data[j]);
            if (d<=d_max){
                idh = (int)(d*dbin);
                omp_set_lock(&DR_lock[idh]);
                    DR[(int)(d*dbin)]+=1;
                omp_unset_lock(&DR_lock[idh]);
            }
        }
    }
    
    //Destruye los bloqueos
    for (int i=0; i<bins; i++){
    	omp_destroy_lock(&DD_lock[i]),omp_destroy_lock(&RR_lock[i]),omp_destroy_lock(&DR_lock[i]);
    }
    
    // Guarda los histogramas
    guardar_Histograma("DD_par.dat", bins, DD);
    guardar_Histograma("RR_par.dat", bins, RR);
    guardar_Histograma("DR_par.dat", bins, DR);
    
    double tot_end = omp_get_wtime();
    cout << tot_start-tot_end << endl;
    
    return 0;
}
