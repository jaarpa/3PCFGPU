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
    //cout << "Succesfully readed " << file_loc << endl;
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
    
    string data_loc = argv[1];
    string rand_loc = argv[2];
    string mypathto_files = "../../../../fake_DATA/DATOS/";
    //This creates the full path to where I have my data files
    data_loc.insert(0,mypathto_files);
    rand_loc.insert(0,mypathto_files);
    
    unsigned int N = stoi(argv[3]), bins=stoi(argv[4]);
    float d_max=stof(argv[5]);
    Punto *data = new Punto[N]; //Crea un array de N puntos
    Punto *rand = new Punto[N]; //Crea un array de N puntos

    //Llama a una funcion que lee los puntos y los guarda en la memoria asignada a data y rand
    read_file(data_loc,data);
    read_file(rand_loc,rand);
    
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
    DR = new long int[bins];
    RR = new long int[bins];
    //Inicializa en 0
    for (int i=0; i<bins; i++){
        DD[i] = 0.0, RR[i] = 0.0, DR[i] = 0.0;     
    }
    double dbin = d_max/(double)bins;
    
    float d;
    //Hace el conteo para el histograma DD
    //cout << "Todo listo para hacer los histogramas" << endl;
    for (int i=0; i<N-1; i++){
        for (int j=i+1; j<N; j++){
            d = distance(data[i],data[j]);
            if (d<=d_max){
                //cout << (int)(d/dbin) <<endl;
                DD[(int)(d/dbin)]+=2;
            }
	        d = distance(rand[i],rand[j]);
            if (d<=d_max){
                //cout << (int)(d/dbin) <<endl;
                RR[(int)(d/dbin)]+=2;
            }
        }
    }
    
    //Hace el conteo para el histograma DR
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            d = distance(rand[i],data[j]);
            if (d<=d_max){
                //cout << (int)(d/dbin) <<endl;
                DR[(int)(d/dbin)]+=1;
            }
        }
    }
    
    // Guarda los histogramas
    guardar_Histograma("DD.dat", bins, DD);
    guardar_Histograma("RR.dat", bins, RR);
    guardar_Histograma("DR.dat", bins, DR);

    double tot_end = omp_get_wtime();
    cout << tot_start-tot_end << endl;
    return 0;
}
