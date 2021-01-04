
/*
Esta funcion lee los datos y crea el grid de nodos y luego llama a la función correspondiente 
para crear y guardar los histogramas correspondientes.

nvcc main.cu -o PCF.out && ./PCF.out 2iso -f data.dat -r rand0.dat -n 32768 -b 20 -d 150
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
        bool bpc = false, analytic=false, rand_dir = false, size_box_provided=false, rand_required=false;
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
                dmax = stof(argv[idpar+1]);
                idpar++;
            } else if (strcmp(argv[idpar],"-bpc")==0){
                bpc = true;
            } else if (strcmp(argv[idpar],"-a")==0){
                analytic = true;
            } else if (strcmp(argv[idpar],"-p")==0){
                partitions = stof(argv[idpar+1]);
                idpar++;
            } else if (strcmp(argv[idpar],"-s")==0){
                size_box = stof(argv[idpar+1]);
                size_box_provided = true;
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
        if (!(bpc && analytic && (strcmp(argv[1],"3iso")==0 || strcmp(argv[1],"2iso")==0))){
            rand_required=true;
            if (rand_name==""){
                cout << "Missing random file(s) location." << endl;
                exit(1);
            }
        }


        //Everything should be set to read the data and call the funtions to make and save the histograms.
        
        /* =======================================================================*/
        /* ========================= Declare variables ===========================*/
        /* =======================================================================*/

        clock_t stop_timmer_host, start_timmer_host;
        start_timmer_host = clock();

        DNode *hnodeD_s, *dnodeD;
        PointW3D *dataD, *h_ordered_pointsD, *d_ordered_pointsD;
        Node ***hnodeD;

        int nonzero_Dnodes, k_element=0, idxD=0, last_pointD = 0;
        float size_node, htime;

        //Declare variables for random.
        DNode **hnodeR_s;
        PointW3D **dataR, **h_ordered_pointsR_s;
        Node ****hnodeR;
        int *nonzero_Rnodes, *idxR, *last_pointR, r_size_box=0, n_randfiles=1;


        /* =======================================================================*/
        /* ================== Define and prepare variables =======================*/
        /* =======================================================================*/

        //Read data
        dataD = new PointW3D[np];
        open_files(data_name, np, dataD, size_box);
        
        //Read rand only if rand was required.
        if (rand_required){
            
//Check if a directory of random files was provided to change n_randfiles
//Instead of rand name should be an array with the name of each rand array or something like that.
if (rand_dir){
    cout << "You are dealing with multiple random files" << endl;
    cout << rand_name << endl;
}
            
            dataR = new PointW3D*[n_randfiles];
            nonzero_Rnodes = new int[n_randfiles];
            idxR = new int[n_randfiles];
            last_pointR = new int[n_randfiles];
            for (int i=0; i<n_randfiles; i++){
                nonzero_Rnodes[i] = 0;
                idxR[i] = 0;
                last_pointR[i] = 0;
                dataR[i] = new PointW3D[np];
                open_files(rand_name, np, dataR[i], r_size_box);
                
                //Set box size
                if (!size_box_provided){
                    if (r_size_box>size_box) size_box=r_size_box;
                }
            }

        }

        //Set nodes size
        size_node = size_box/(float)(partitions);

        //Make nodes
        if (rand_required){
            hnodeR = new Node***[n_randfiles];
            for (int i=0; i<n_randfiles; i++){
                hnodeR[i] = new Node**[partitions];
                for (int j=0; j<partitions; j++){
                    hnodeR[i][j] = new Node*[partitions];
                    for (int k=0; k<partitions; k++){
                        hnodeR[i][j][k] = new Node[partitions];
                    }
                }
            }
        }
        
        hnodeD = new Node**[partitions];
        for (int i=0; i<partitions; i++){
            *(hnodeD+i) = new Node*[partitions];
            for (int j=0; j<partitions; j++){
                *(*(hnodeD+i)+j) = new Node[partitions];
            }
        }
        
        cout << "Properly assigned and read the data. Justbefore making the nodes" << endl;

        make_nodos(hnodeD, dataD, partitions, size_node, np);
        if (rand_required){
            for (int i=0; i<n_randfiles; i++){
                make_nodos(hnodeR[i], dataR[i], partitions, size_node, np);
            }
        };

        //Count nonzero data nodes
        for(int row=0; row<partitions; row++){
            for(int col=0; col<partitions; col++){
                for(int mom=0; mom<partitions; mom++){
                    if(hnodeD[row][col][mom].len>0){
                        nonzero_Dnodes+=1;
                    }

                    if (rand_required){
                        for (int i=0; i<n_randfiles; i++){
                            if(hnodeR[i][row][col][mom].len>0) nonzero_Rnodes[i]+=1;
                        }
                    }

                }
            }
        }

        //Deep copy into linear nodes and an ordered elements array
        hnodeD_s = new DNode[nonzero_Dnodes];
        h_ordered_pointsD = new PointW3D[np];
        if (rand_required){
            hnodeR_s = new DNode*[n_randfiles];
            h_ordered_pointsR_s = new PointW3D*[n_randfiles];
            for (int i=0; i<n_randfiles; i++){
                hnodeR_s[i] = new DNode[nonzero_Rnodes[i]];
                h_ordered_pointsR_s[i] = new PointW3D[np];
            }
        }

        for(int row=0; row<partitions; row++){
            for(int col=0; col<partitions; col++){
                for(int mom=0; mom<partitions; mom++){
            
                    if (hnodeD[row][col][mom].len>0){
                        hnodeD_s[idxD].nodepos = hnodeD[row][col][mom].nodepos;
                        hnodeD_s[idxD].start = last_pointD;
                        hnodeD_s[idxD].len = hnodeD[row][col][mom].len;
                        last_pointD = last_pointD + hnodeD[row][col][mom].len;
                        hnodeD_s[idxD].end = last_pointD;
                        for (int j=hnodeD_s[idxD].start; j<last_pointD; j++){
                            k_element = j-hnodeD_s[idxD].start;
                            h_ordered_pointsD[j] = hnodeD[row][col][mom].elements[k_element];
                        }
                        idxD++;
                    }

                    if (rand_required){
                        for (int i=0; i<n_randfiles; i++){
                            if (hnodeR[i][row][col][mom].len>0){
                                hnodeR_s[i][idxR[i]].nodepos = hnodeR[i][row][col][mom].nodepos;
                                hnodeR_s[i][idxR[i]].start = last_pointR[i];
                                hnodeR_s[i][idxR[i]].len = hnodeR[i][row][col][mom].len;
                                last_pointR[i] = last_pointR[i] + hnodeR[i][row][col][mom].len;
                                hnodeR_s[i][idxR[i]].end = last_pointR[i];
                                for (int j=hnodeR_s[i][idxR[i]].start; j<last_pointR[i]; j++){
                                    k_element = j-hnodeR_s[i][idxR[i]].start;
                                    h_ordered_pointsR_s[i][j] = hnodeR[i][row][col][mom].elements[k_element];
                                }
                                idxR[i]++;
                            }
                        }
                    }

                }
            }
        }

        //Allocate and copy the nodes into device memory
        cucheck(cudaMalloc(&dnodeD, nonzero_Dnodes*sizeof(DNode)));
        cucheck(cudaMalloc(&d_ordered_pointsD, np*sizeof(PointW3D)));
        cucheck(cudaMemcpy(dnodeD, hnodeD_s, nonzero_Dnodes*sizeof(DNode), cudaMemcpyHostToDevice));
        cucheck(cudaMemcpy(d_ordered_pointsD, h_ordered_pointsD, np*sizeof(PointW3D), cudaMemcpyHostToDevice));
        //Memory allocation and copy of random files in the loop of the functions
        stop_timmer_host = clock();
        htime = ((float)(stop_timmer_host-start_timmer_host))/CLOCKS_PER_SEC;
        cout << "Succesfully readed the data. All set to compute the histograms in " << htime*1000 << " miliseconds" << endl;
        cout << "(Does not include host to device copies of nodes of random data)" << endl;

        //Launch the correct function
        
        
        //Free the host memory
        cucheck(cudaFree(dnodeD));
        cucheck(cudaFree(d_ordered_pointsD));
        
        for (int i=0; i<partitions; i++){
            for (int j=0; j<partitions; j++){
                delete[] hnodeD[i][j];
            }
            delete[] hnodeD[i];
        }    
        delete[] hnodeD;
        delete[] dataD;
        delete[] hnodeD_s;
        delete[] h_ordered_pointsD;
        
        if (rand_required){
            for (int i=0; i<n_randfiles; i++){
                delete[] hnodeR_s[i];
                delete[] dataR[i];
                delete[] h_ordered_pointsR_s[i];
                for (int j=0; j<partitions; j++){
                    for (int k=0; k<partitions; k++){
                        delete[] hnodeR[i][j][k];
                    }
                    delete[] hnodeR[i][j];
                }
                delete[] hnodeR[i];
            }
            delete[] hnodeR;
            delete[] hnodeR_s;
            delete[] dataR;
            delete[] h_ordered_pointsR_s;
            delete[] nonzero_Rnodes;
            delete[] idxR;
            delete[] last_pointR;
        }

    } else {
        cout << "Invalid <cal_type> option or not enough parameters. \nSee --help for more information." << endl;
        exit(1);
    }
    return 0;
}