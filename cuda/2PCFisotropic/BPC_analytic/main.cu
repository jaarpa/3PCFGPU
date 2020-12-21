// nvcc -arch=sm_75 main.cu -o par_s.out && ./par_s.out data_2GPc.dat data_2GPc.dat 3241792 20 160
// nvcc -arch=sm_75 main.cu -o par_s.out && ./par_s.out data_1GPc.dat data_1GPc.dat 405224 20 160
// nvcc -arch=sm_75 main.cu -o par_s.out && ./par_s.out data.dat rand0.dat 32768 30 150
// nvcc -arch=sm_75 main.cu -o par_s.out && ./par_s.out data_5K.dat rand0_5K.dat 5000 30 180

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
    Main function to calculate the isotropic 2 point correlation function. Saves three different histograms in the same location of this script
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

    double *DD, *RR, *DR, *d_DD, *d_RR, *d_DR;
    double alpha1, beta1, dr; //For analytic RR

    //n_kernel_calls should depend of the number of points, its density, and the number of bins
    int  blocks_D, blocks_analytic, nonzero_Dnodes, threads_perblock_dim = 32, threads_perblock_analytic = 512, idxD=0;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    cucheck(cudaEventCreate(&start_timmer));
    cucheck(cudaEventCreate(&stop_timmer));

    clock_t stop_timmer_host, start_timmer_host;

    PointW3D *dataD;

    Node ***hnodeD;
    DNode *hnodeD_s;
    PointW3D *h_ordered_pointsD_s;
    cudaStream_t streamDD, streamRR, streamDR;
    cucheck(cudaStreamCreate(&streamDD));
    cucheck(cudaStreamCreate(&streamDR));
    cucheck(cudaStreamCreate(&streamRR));
    DNode *dnodeD_DD;
    int k_element, last_pointD;
    PointW3D *d_ordered_pointsD_DD;

    // Name of the files where the results are saved
    string nameDD = "DDiso_", nameRR = "RRiso_", nameDR = "DRiso_";
    string data_name = argv[1], rand_name = argv[2];
    nameDD.append(data_name);
    nameRR.append(rand_name);
    nameDR.append(data_name);

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/
    start_timmer_host = clock();
    dataD = new PointW3D[np];

    // Open and read the files to store the data in the arrays
    open_files(data_name, np, dataD, size_box); //This function also gets the real size of the box
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

    dr = (dmax/bn);
    beta1 = (np*np)/(size_box*size_box*size_box);
    alpha = 8*dr*dr*dr*(acos(0.0))*(beta1)/3;
    
    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    // Allocate memory for the histogram as double
    DD = new double[bn];
    RR = new double[bn];
    DR = new double[bn];

    cucheck(cudaMalloc(&d_DD, bn*sizeof(double)));
    cucheck(cudaMalloc(&d_RR, bn*sizeof(double)));
    cucheck(cudaMalloc(&d_DR, bn*sizeof(double)));

    //Restarts the main histograms in device to zero
    cucheck(cudaMemsetAsync(d_DD, 0, bn*sizeof(double), streamDD));
    cucheck(cudaMemsetAsync(d_RR, 0, bn*sizeof(double), streamRR));
    cucheck(cudaMemsetAsync(d_DR, 0, bn*sizeof(double), streamDR));

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

    nonzero_Dnodes=0;
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
    cucheck(cudaMalloc(&dnodeD_DD, nonzero_Dnodes*sizeof(DNode)));
    cucheck(cudaMalloc(&d_ordered_pointsD_DD, np*sizeof(PointW3D)));

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

    cucheck(cudaMemcpyAsync(d_ordered_pointsD_DD, h_ordered_pointsD_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDD));
    cucheck(cudaMemcpyAsync(dnodeD_DD, hnodeD_s, nonzero_Dnodes*sizeof(DNode), cudaMemcpyHostToDevice, streamDD));

    stop_timmer_host = clock();
    time_spent = ((float)(stop_timmer_host-start_timmer_host))/CLOCKS_PER_SEC;
    cout << "Succesfully readed the data. All set to compute the histograms in " << time_spent*1000 << " miliseconds" << endl;

    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/

    //Compute the dimensions of the GPU grid
    //One thread for each node
    
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));
    dim3 threads_perblock_D(threads_perblock_dim,threads_perblock_dim,1);
    dim3 gridD(blocks_D,blocks_D,1);
    
    blocks_analytic = (int)(ceil((float)((float)(bn)/(float)(threads_perblock_analytic))));

    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    make_histoXX<<<gridD,threads_perblock_D,0,streamDD>>>(d_DD, d_ordered_pointsD_DD, dnodeD_DD, nonzero_Dnodes, bn, dmax, d_max_node, size_box, size_node);
    cucheck(cudaMemcpyAsync(DD, d_DD, bn*sizeof(double), cudaMemcpyDeviceToHost, streamDD));
    make_histoRR<<<blocks_analytic,threads_perblock_analytic,0,streamRR>>>(d_RR, alpha, bn);
    cucheck(cudaMemcpyAsync(RR, d_RR, bn*sizeof(double), cudaMemcpyDeviceToHost, streamRR));
    
    cucheck(cudaStreamSynchronize(streamDD));
    //make_histoDR<<<blocks_analytic,threads_perblock_analytic,0,streamDR>>>(d_DR, d_DD, dnodeD_DR, nonzero_Dnodes, d_ordered_pointsR_DR, dnodeR_DR, nonzero_Rnodes, bn, dmax, d_max_node, size_box, size_node);
    //cucheck(cudaMemcpyAsync(DR, d_DR, bn*sizeof(double), cudaMemcpyDeviceToHost, streamDR));
    
    //Waits for all the kernels to complete
    save_histogram(nameDD, bn, DD);
    cucheck(cudaStreamSynchronize(streamRR));
    save_histogram(nameRR, bn, RR);
    cucheck(cudaStreamSynchronize(streamDR));
    save_histogram(nameDR, bn, DR);

    cucheck(cudaEventRecord(stop_timmer));
    cucheck(cudaEventSynchronize(stop_timmer));
    cucheck(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    cout << "Spent "<< time_spent << " miliseconds to compute and save all the histograms." << endl;
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory
    cucheck(cudaStreamDestroy(streamDD));
    cucheck(cudaStreamDestroy(streamDR));
    cucheck(cudaStreamDestroy(streamRR));

    cucheck(cudaEventDestroy(start_timmer));
    cucheck(cudaEventDestroy(stop_timmer));

    delete[] dataD;

    delete[] DD;
    delete[] RR;    
    delete[] DR;    
    
    cucheck(cudaFree(d_DD));
    cucheck(cudaFree(d_RR));
    cucheck(cudaFree(d_DR));

    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            delete[] hnodeD[i][j];
        }
        delete[] hnodeD[i];
    }    
    delete[] hnodeD;

    cucheck(cudaFree(d_ordered_pointsD_DD));
    cucheck(cudaFree(dnodeD_DD));

    delete[] hnodeD_s;
    delete[] h_ordered_pointsD_s;

    cout << "Program terminated..." << endl;
    return 0;
}

