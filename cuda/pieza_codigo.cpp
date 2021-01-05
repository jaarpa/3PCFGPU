#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

int main(int argc, char *argv[]){
    /*
        Esta pieza de código usa el estándar c++17:
        compilar con g++ -std=c++17 pieza_codigo.cpp -o ejecutable
        requiere las librerias: sstream, fstream, string y filesystem

        Recibe en carpeta: el nombre de la carpeta donde se localizan los archivos
        Muestra: todos los archivos con sus respectivos datos localizados en la carpeta especificada
    */

    //Varible que recibe el nombre de la carpeta donde se localizan los ficheros
    string carpeta = "/home/alejandrogoper/Documentos/RepoDePrueba/JuanCODE/DESI/cuda/datatest";
    //Variables para guardar los datos y mostrarlos en pantalla 
    float x,y;
    //Variable que cuenta el numero de puntos en cada archivo y numero de archivos en la carpeta
    int num_puntos=0, num_archivos=0;
    for (auto & dir_archivo : filesystem::directory_iterator(carpeta))
    {
        num_archivos++;
        //Convertimos a string el iterador dir_archivo para poder leer el fichero
        ostringstream str;
        str << dir_archivo;
        string nombre = str.str(), nombre_archivo = "";
        //eliminamos el primer y ultimo caracter de la cadena nombre (pues contienen " " al inicio y al final y esto genera problemas al leer el fichero)
        for (int i = 1; i < nombre.length()-1; i++)
        {
            nombre_archivo += nombre[i];
        }
        cout << "======================================================"<< endl;
        //Abrimos cada archivo        
        ifstream archivo;
        archivo.open(nombre_archivo.c_str(), ios::in | ios::binary); //le indico al programa que se trata de un archivo binario con ios::binary
        if (archivo.fail()){
            cout << "Error al cargar el archivo: "<< nombre_archivo<< endl;
            exit(1);
        }
        int c = 0;
        //accedemos a los datos de cada archivo
        while (!archivo.eof())
        {
            archivo >> x >> y;
            cout << x << "\t" << y << endl; 
            c++;
        }
        cout << "Archivo: "<<nombre_archivo << "\n" << "Longitud: " << c << " datos"<<endl;
        cout << "======================================================"<< endl;
        archivo.close();
    }
    cout << "Numero de archivos = " << num_archivos << endl;   
    return 0;
}