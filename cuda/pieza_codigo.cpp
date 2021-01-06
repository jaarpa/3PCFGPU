#include <iostream>
#include <dirent.h>
#include <string>
#include <fstream>

using namespace std;

int main(int argc, char *argv[]){
    /*
        Esta pieza de código utiliza la libreria dirent.h que es compatible con el estandar c++11.

        Intrucciones: Solo cambiar la variable ruta_carpeta por la dirección de la carpeta deseada.

        Nota: funciona con archivos .dat de dos columnas de datos x e y (por eso las variables float x, y )

        Regresa: la cantidad de puntos de cada archivo, imprime los puntos de cada archivo e imprime la cantidad de archivos abiertos.
    */

   string ruta_carpeta = "/home/alejandrogoper/Documentos/RepoDePrueba/JuanCODE/DESI/cuda/datatest";
   
   /*
    ======================================================================================================================
                                                Variables útiles
    ======================================================================================================================
   */
   string nombre_archivo, ruta_archivo;
   float x,y;
   int num_puntos=0, num_archivos=0;    
   /*
    ======================================================================================================================
                                                Lógica del progama
    ======================================================================================================================
   */
    if(DIR *carpeta = opendir(ruta_carpeta.c_str()))
    {
        num_archivos = 0;
        while(dirent *archivos = readdir(carpeta))
        {
            nombre_archivo = archivos->d_name; //regresa el nombre de un archivo perteneciente al directorio "carpeta"
            //Este if es para evitar leer archivos ocultos de linux que en principio comienzan con .src o .bin etc.
            if( nombre_archivo != "." && nombre_archivo != ".." )
            {
                num_archivos++;
                //De esta manera nos aseguramos que abrimos el archivo correcto dentro de la carpeta.
                ruta_archivo = ruta_carpeta + "/" + nombre_archivo;
                ifstream archivo;
                archivo.open(ruta_archivo.c_str(), ios::in | ios::binary);
                if (archivo.fail())
                {
                    cout << "Error al abrir el archivo: " << nombre_archivo << endl;
                    exit(1);
                }
                num_puntos=0;
                while (!archivo.eof())
                {
                    archivo >> x >> y;
                    cout << x << "\t" << y << endl;
                    num_puntos++;
                }
                cout << "Archivo: "<< nombre_archivo << "\n" << "Longitud: " << num_puntos << " datos." << endl;
                cout << "======================================================"<< endl;
                archivo.close();
            }
        }
        cout << "\n\n\tCantidad de archivos: " << num_archivos << endl;
        closedir(carpeta);
    }
    else
    {
        cout << "No se pudo abrir el directorio" << endl;
    }
    
}