/*
Esta funcion lee los datos y crea el grid de nodos y luego llama a la función correspondiente 
para crear y guardar los histogramas correspondientes.
*/

/** CUDA check macro */
#define cucheck(call){\
    cudaError_t res = (call);\
    if(res != cudaSuccess) {\
        const char* err_str = cudaGetErrorString(res);\
        fprintf(stderr, "%s (%d): %s in %s \n", __FILE__, __LINE__, err_str, #call);\
        exit(-1);\
    }\
}\

#include <stdio.h>
#include <string.h>
#include "create_grid.cuh"
#include "PCF_help.cuh"
using namespace std;

int main(int argc, char **argv){
    /*
    Main function to calculate the correlation function of 2 and 3 points either isotropic or anisotropic. This is the master
    script which calls the correct function. The file must contain 4 columns, the first 3 
    are the x,y,z coordinates and the 4 the weigh of the measurment.
    */

    if(argc == 2 && (strcmp(argv[1], "--help")==0 || strcmp(argv[1], "-h")==0)){
        show_help();
        return 0;
    } else if (argc >= 12 && (strcmp(argv[1],"3iso")==0 || strcmp(argv[1],"3ani")==0 || strcmp(argv[1],"2iso")==0 || strcmp(argv[1],"2ani")==0)) {

        //Read the parameters from command line
        int np=0, bn=0, partitions=35;
        bool bpc = false, analytic=false, rand_dir = false;
        float size_box = 0, dmax=0;
        string data_name, rand_name="";
        for (int idpar=2; idpar<argc; idpar++){     
            
            if (strcmp(argv[idpar],"-n")==0){
                np = stoi(argv[idpar+1]);
                idpar++;
            } else if (strcmp(argv[idpar],"-f")==0){
                data_name = argv[idpar+1];
                idpar++;
            } else if (strcmp(argv[idpar],"-r")==0 || strcmp(argv[idpar],"-rd")==0){
                rand_dir = (strcmp(argv[idpar],"-rd")==0);
                rand_name = argv[idpar+1];
                idpar++;
            } else if (strcmp(argv[idpar],"-b")==0){
                bn = stoi(argv[idpar+1]);
                idpar++;
            } else if (strcmp(argv[idpar],"-d")==0){
                dmax = stof(argv[[idpar+1]]);
                idpar++;
            } else if (strcmp(argv[idpar],"-bpc")==0){
                bpc = true;
            } else if (strcmp(argv[idpar],"-a")==0){
                analytic = true;
            } else if (strcmp(argv[idpar],"-p")==0){
                partitions = stof(argv[[idpar+1]]);
                idpar++;
            } else if (strcmp(argv[idpar],"-s")==0){
                size_box = stof(argv[[idpar+1]]);
                idpar++;
            }

        }

        //Figure out if something very necessary is missing
        if (np==0){
            cout << "Missing number of points argument." << endl;
            exit(1);
        }
        if (bn==0){
            cout << "Missing number of bins argument." << endl;
            exit(1);
        }
        if (dmax==0){
            cout << "Missing maximum distance argument." << endl;
            exit(1);
        }
        if (!(bpc && analytic) && rand_name==""){
            cout << "Missing random file(s) location." << endl;
            exit(1);
        }

        cout << "Random file(s) location."<< rand_name << endl;

    } else {
        cout << "Invalid <cal_type> option or not enough parameters. \nSee --help for more information." << endl;
        exit(1);
    }
    return 0;
}