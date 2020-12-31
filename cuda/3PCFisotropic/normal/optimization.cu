//Simple compilation
// 01:07
//nvcc -arch=sm_75 main.cu -o par_d.out && ./par_d.out data.dat rand0.dat 10000 30 50
//nvcc -arch=sm_75 main.cu -o par_d.out && ./par_d.out data_1GPc.dat rand_1GPc.dat 405224 30 150 1024

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <time.h>
#include <math.h>
#include "create_grid.cuh"
#include "kernels.cuh"

using namespace std;

/** CUDA check macro */
#define cucheck(call) \
	{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
	const char* err_str = cudaGetErrorString(res);\
	fprintf(stderr, "%s (%d): %s in %s \n", __FILE__, __LINE__, err_str, #call);	\
	exit(-1);\
	}\
	}


int main(int argc, char **argv){
    /*
    Main function to calculate the isotropic 3 point correlation function. Saves three different histograms in the same location of this script
    with the names DD.dat DR.dat RR.dat. This program do not consider periodic boundary conditions. The file must contain 4 columns, the first 3 
    are the x,y,z coordinates and the 4 the weigh of the measurment.

    Args:
    arg[1]: name or path to the data file relative to ../../../fake_DATA/DATOS/. 
    arg[2]: name or path to the random file relative to ../../../fake_DATA/DATOS/
    arg[3]: integer of the number of points in the files.
    arg[4]: integer. Number of bins where the distances are classified
    arg[5]: float. Maximum distance of interest. It has to have the same units as the points in the files.
    */

    /* =======================================================================*/
    /* =====================   Var declaration ===============================*/
    /* =======================================================================*/

    unsigned int np = stoi(argv[3]), bn = stoi(argv[4]), partitions;

    float time_spent, d_max_node, size_node, dmax = stof(argv[5]), size_box = 0, r_size_box=0;

    double *DDD;
    double *d_DDD;

    //n_kernel_calls should depend of the number of points, its density, and the number of bins
    int nonzero_Dnodes = 0, threads_perblock_dim = 8, idxD=0;
    int blocks_D;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    cucheck(cudaEventCreate(&start_timmer));
    cucheck(cudaEventCreate(&stop_timmer));

    clock_t stop_timmer_host, start_timmer_host;

    PointW3D *dataD;

    int k_element, last_pointD;
    Node ***hnodeD;
    DNode *hnodeD_s;
    PointW3D *h_ordered_pointsD_s;

    cudaStream_t streamDDD;
    cucheck(cudaStreamCreate(&streamDDD));
    DNode *dnodeD_DDD;
    PointW3D *d_ordered_pointsD_DDD;

    // Name of the files where the results are saved
    string nameDDD = "DDDiso.dat";

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/
    start_timmer_host = clock();
    dataD = new PointW3D[np];

    // Open and read the files to store the data in the arrays
    open_files(argv[1], np, dataD, size_box); //This function also gets the real size of the box
    if (r_size_box>size_box){
        size_box=r_size_box;
    }

    if (argc>6){
        r_size_box = stof(argv[6]);
        if (r_size_box>0){
            size_box=r_size_box;
        }
    }

    //Sets the number of partitions of the box and the size of each node
    if (argc>7){
        //Partitions entered by the user
        partitions = stof(argv[7]);
    } else {
        //Calculate optimum partitions
        partitions = 35;
    }
    size_node = size_box/(float)(partitions);
    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    // Allocate memory for the histogram as double
    DDD = new double[bn*bn*bn];

    cucheck(cudaMalloc(&d_DDD, bn*bn*bn*sizeof(double)));

    //Restarts the main histograms in host to zero
    cucheck(cudaMemsetAsync(d_DDD, 0, bn*bn*bn*sizeof(double), streamDDD));

    hnodeD = new Node**[partitions];
    for (int i=0; i<partitions; i++){
        *(hnodeD+i) = new Node*[partitions];
        for (int j=0; j<partitions; j++){
            *(*(hnodeD+i)+j) = new Node[partitions];
        }
    }

    //Classificate the data into the nodes in the host side
    //The node classification is made in the host
    make_nodos(hnodeD, dataD, partitions, size_node, np);

    for(int row=0; row<partitions; row++){
        for(int col=0; col<partitions; col++){
            for(int mom=0; mom<partitions; mom++){
                if(hnodeD[row][col][mom].len>0){
                    nonzero_Dnodes+=1;
                }
            }
        }
    }

    //Allocate memory for the nodes depending of how many partitions there are.
    cucheck(cudaMalloc(&dnodeD_DDD, nonzero_Dnodes*sizeof(DNode)));
    cucheck(cudaMalloc(&d_ordered_pointsD_DDD, np*sizeof(PointW3D)));

    hnodeD_s = new DNode[nonzero_Dnodes];
    h_ordered_pointsD_s = new PointW3D[np];
    
    //Deep copy to device memory
    last_pointD = 0;
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
                        h_ordered_pointsD_s[j] = hnodeD[row][col][mom].elements[k_element];
                    }
                    idxD++;
                }
            }
        }
    }


    cucheck(cudaMemcpyAsync(dnodeD_DDD, hnodeD_s, nonzero_Dnodes*sizeof(DNode), cudaMemcpyHostToDevice, streamDDD));
    cucheck(cudaMemcpyAsync(d_ordered_pointsD_DDD, h_ordered_pointsD_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDDD));

    stop_timmer_host = clock();
    time_spent = ((float)(stop_timmer_host-start_timmer_host))/CLOCKS_PER_SEC;
    //cout << "Succesfully readed the data. All set to compute the histograms in " << time_spent*1000 << " miliseconds" << endl;


    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/


    //Compute the dimensions of the GPU grid
    //One thread for each node
    
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));

    dim3 threads_perblock(threads_perblock_dim,threads_perblock_dim,threads_perblock_dim);
    
    dim3 gridDDD(blocks_D,blocks_D,blocks_D);

    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    make_histoXXX<<<gridDDD,threads_perblock,0,streamDDD>>>(d_DDD, d_ordered_pointsD_DDD, dnodeD_DDD, nonzero_Dnodes, bn, dmax, d_max_node);

    cucheck(cudaMemcpyAsync(DDD, d_DDD, bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost, streamDDD));

    //Waits for all the kernels to complete
    cucheck(cudaStreamSynchronize(streamDDD));

    cucheck(cudaEventRecord(stop_timmer));
    cucheck(cudaEventSynchronize(stop_timmer));
    cucheck(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    //cout << "Spent "<< time_spent << " miliseconds to compute and save all the histograms." << endl;

    string name = "time_results.dat";
    ofstream time_results_file;
    time_results_file.open(name.c_str(), ios_base::app);
    time_results_file << np << "\t" << size_box << "\t" << dmax << "\t" << partitions << "\t" << time_spent/1000 << endl;
    time_results_file.close();
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory

    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            delete[] hnodeD[i][j];
        }
        delete[] hnodeD[i];
    }    
    delete[] hnodeD;

    delete[] dataD;
    
    delete[] hnodeD_s;
    delete[] h_ordered_pointsD_s;
    
    cucheck(cudaStreamDestroy(streamDDD));

    cucheck(cudaEventDestroy(start_timmer));
    cucheck(cudaEventDestroy(stop_timmer));

    delete[] DDD;  
    
    cucheck(cudaFree(d_DDD));

    cucheck(cudaFree(dnodeD_DDD));
    cucheck(cudaFree(d_ordered_pointsD_DDD));
    
    return 0;
}

