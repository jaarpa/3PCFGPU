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
        int col=argc;
        cout << col << endl;
    } else {
        cout << "Invalid <cal_type> option. See --help for more information." << endl;
        exit(1);
    }
    return 0;
}