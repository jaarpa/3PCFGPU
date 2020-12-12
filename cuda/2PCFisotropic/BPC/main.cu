// nvcc -arch=sm_75 main.cu -o par_s.out && ./par_s.out data_2GPc.dat data_2GPc.dat 3241792 20 160
// nvcc -arch=sm_75 main.cu -o par_s.out && ./par_s.out data_1GPc.dat data_1GPc.dat 405224 20 160
// nvcc -arch=sm_75 main.cu -o par_s.out && ./par_s.out data.dat rand0.dat 32768 20 150
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

    float time_spent, size_node, dmax = stof(argv[5]), size_box = 0, r_size_box=0;

    double *DD, *RR, *DR, *d_DD, *d_RR, *d_DR;

    //n_kernel_calls should depend of the number of points, its density, and the number of bins
    int threads_perblock, blocks;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    cucheck(cudaEventCreate(&start_timmer));
    cucheck(cudaEventCreate(&stop_timmer));

    clock_t stop_timmer_host, start_timmer_host;

    PointW3D *dataD;
    PointW3D *dataR;

    Node ***hnodeD, ***hnodeR;
    DNode *hnodeD_s, *hnodeR_s;
    PointW3D *h_ordered_pointsD_s, *h_ordered_pointsR_s;
    cudaStream_t streamDD, streamRR, streamDR;
    cucheck(cudaStreamCreate(&streamDD));
    cucheck(cudaStreamCreate(&streamDR));
    cucheck(cudaStreamCreate(&streamRR));
    DNode *dnodeD_s1, *dnodeD_s3, *dnodeR_s2, *dnodeR_s3;
    int row, col, mom, k_element, last_pointD, last_pointR;
    PointW3D *d_ordered_pointsD_s1, *d_ordered_pointsD_s3, *d_ordered_pointsR_s2, *d_ordered_pointsR_s3;

    // Name of the files where the results are saved
    string nameDD = "DDiso.dat", nameRR = "RRiso.dat", nameDR = "DRiso.dat";

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/
    start_timmer_host = clock();
    dataD = new PointW3D[np];
    dataR = new PointW3D[np];

    // Open and read the files to store the data in the arrays
    open_files(argv[1], np, dataD, size_box); //This function also gets the real size of the box
    open_files(argv[2], np, dataR, r_size_box);

    //Sets the number of partitions of the box and the size of each node
    partitions = 35;
    size_node = size_box/(float)(partitions);

    // Allocate memory for the histogram as double
    DD = new double[bn];
    RR = new double[bn];
    DR = new double[bn];

    cucheck(cudaMalloc(&d_DD, bn*sizeof(double)));
    cucheck(cudaMalloc(&d_RR, bn*sizeof(double)));
    cucheck(cudaMalloc(&d_DR, bn*sizeof(double)));

    //Restarts the main histograms in host to zero
    cucheck(cudaMemsetAsync(d_DD, 0, bn*sizeof(double), streamDD));
    cucheck(cudaMemsetAsync(d_RR, 0, bn*sizeof(double), streamRR));
    cucheck(cudaMemsetAsync(d_DR, 0, bn*sizeof(double), streamDR));

    //Allocate memory for the nodes depending of how many partitions there are.
    cucheck(cudaMalloc(&dnodeD_s1, partitions*partitions*partitions*sizeof(DNode)));
    cucheck(cudaMalloc(&d_ordered_pointsD_s1, np*sizeof(PointW3D)));
    cucheck(cudaMalloc(&dnodeR_s2, partitions*partitions*partitions*sizeof(DNode)));
    cucheck(cudaMalloc(&d_ordered_pointsR_s2, np*sizeof(PointW3D)));
    cucheck(cudaMalloc(&dnodeD_s3, partitions*partitions*partitions*sizeof(DNode)));
    cucheck(cudaMalloc(&d_ordered_pointsD_s3, np*sizeof(PointW3D)));
    cucheck(cudaMalloc(&dnodeR_s3, partitions*partitions*partitions*sizeof(DNode)));
    cucheck(cudaMalloc(&d_ordered_pointsR_s3, np*sizeof(PointW3D)));

    hnodeD_s = new DNode[partitions*partitions*partitions];
    h_ordered_pointsD_s = new PointW3D[np];
    hnodeR_s = new DNode[partitions*partitions*partitions];
    h_ordered_pointsR_s = new PointW3D[np];

    hnodeD = new Node**[partitions];
    hnodeR = new Node**[partitions];
    for (int i=0; i<partitions; i++){
        *(hnodeD+i) = new Node*[partitions];
        *(hnodeR+i) = new Node*[partitions];
        for (int j=0; j<partitions; j++){
            *(*(hnodeD+i)+j) = new Node[partitions];
            *(*(hnodeR+i)+j) = new Node[partitions];
        }
    }

    //Classificate the data into the nodes in the host side
    //The node classification is made in the host
    make_nodos(hnodeD, dataD, partitions, size_node, np);
    make_nodos(hnodeR, dataR, partitions, size_node, np);
    
    //Deep copy to device memory

    last_pointR = 0;
    last_pointD = 0;
    for (int idx=0; idx<partitions*partitions*partitions; idx++){
        mom = (int) (idx/(partitions*partitions));
        col = (int) ((idx%(partitions*partitions))/partitions);
        row = idx%partitions;
        
        hnodeD_s[idx].nodepos = hnodeD[row][col][mom].nodepos;
        hnodeD_s[idx].prev_i = last_pointD;
        last_pointD = last_pointD + hnodeD[row][col][mom].len;
        hnodeD_s[idx].len = hnodeD[row][col][mom].len;
        for (int j=hnodeD_s[idx].prev_i; j<last_pointD; j++){
            k_element = j-hnodeD_s[idx].prev_i;
            h_ordered_pointsD_s[j] = hnodeD[row][col][mom].elements[k_element];
        }

        hnodeR_s[idx].nodepos = hnodeR[row][col][mom].nodepos;
        hnodeR_s[idx].prev_i = last_pointR;
        last_pointR = last_pointR + hnodeR[row][col][mom].len;
        hnodeR_s[idx].len = hnodeR[row][col][mom].len;
        for (int j=hnodeR_s[idx].prev_i; j<last_pointR; j++){
            k_element = j-hnodeR_s[idx].prev_i;
            h_ordered_pointsR_s[j] = hnodeR[row][col][mom].elements[k_element];
        }
    }

    cucheck(cudaMemcpyAsync(d_ordered_pointsD_s1, h_ordered_pointsD_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDD));
    cucheck(cudaMemcpyAsync(dnodeD_s1, hnodeD_s, partitions*partitions*partitions*sizeof(DNode), cudaMemcpyHostToDevice, streamDD));

    cucheck(cudaMemcpyAsync(d_ordered_pointsR_s2, h_ordered_pointsR_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamRR));
    cucheck(cudaMemcpyAsync(dnodeR_s2, hnodeR_s, partitions*partitions*partitions*sizeof(DNode), cudaMemcpyHostToDevice, streamRR));

    cucheck(cudaMemcpyAsync(d_ordered_pointsR_s3, h_ordered_pointsR_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDR));
    cucheck(cudaMemcpyAsync(dnodeR_s3, hnodeR_s, partitions*partitions*partitions*sizeof(DNode), cudaMemcpyHostToDevice, streamDR));
    cucheck(cudaMemcpyAsync(d_ordered_pointsD_s3, h_ordered_pointsD_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDR));
    cucheck(cudaMemcpyAsync(dnodeD_s3, hnodeD_s, partitions*partitions*partitions*sizeof(DNode), cudaMemcpyHostToDevice, streamDR));

    stop_timmer_host = clock();
    time_spent = ((float)(stop_timmer_host-start_timmer_host))/CLOCKS_PER_SEC;
    cout << "Succesfully readed the data. All set to compute the histograms in " << time_spent*1000 << " miliseconds" << endl;


    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/


    //Compute the dimensions of the GPU grid
    //One thread for each node
    threads_perblock = 512;
    blocks = (int)(ceil((float)((float)(partitions*partitions*partitions)/(float)(threads_perblock))));

    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    make_histoXX<<<blocks,threads_perblock,0,streamDD>>>(d_DD, d_ordered_pointsD_s1, dnodeD_s1, partitions, bn, dmax, size_node);
    make_histoXX<<<blocks,threads_perblock,0,streamRR>>>(d_RR, d_ordered_pointsR_s2, dnodeR_s2, partitions, bn, dmax, size_node);
    make_histoXY<<<blocks,threads_perblock,0,streamDR>>>(d_DR, d_ordered_pointsD_s3, dnodeD_s3, d_ordered_pointsR_s3, dnodeR_s3, partitions, bn, dmax, size_node);

    cucheck(cudaMemcpyAsync(RR, d_RR, bn*sizeof(double), cudaMemcpyDeviceToHost, streamRR));
    cucheck(cudaMemcpyAsync(DR, d_DR, bn*sizeof(double), cudaMemcpyDeviceToHost, streamDR));
    cucheck(cudaMemcpyAsync(DD, d_DD, bn*sizeof(double), cudaMemcpyDeviceToHost, streamDD));

    //Waits for all the kernels to complete
    cucheck(cudaDeviceSynchronize());


    cucheck(cudaEventRecord(stop_timmer));
    cucheck(cudaEventSynchronize(stop_timmer));
    cucheck(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    cout << "Spent "<< time_spent << " miliseconds to compute all the histograms." << endl;
    
    /* =======================================================================*/
    /* =======================  Save the results =============================*/
    /* =======================================================================*/

	save_histogram(nameDD, bn, DD);
	save_histogram(nameRR, bn, RR);
	save_histogram(nameDR, bn, DR);
    cout << "Saved the histograms" << endl;
    
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
    delete[] dataR;

    delete[] DD;
    delete[] RR;    
    delete[] DR;    
    
    cucheck(cudaFree(d_DD));
    cucheck(cudaFree(d_RR));
    cucheck(cudaFree(d_DR));

    
    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            delete[] hnodeD[i][j];
            delete[] hnodeR[i][j];
        }
        delete[] hnodeD[i];
        delete[] hnodeR[i];
    }    
    delete[] hnodeD;
    delete[] hnodeR;

    cucheck(cudaFree(d_ordered_pointsD_s1));
    cucheck(cudaFree(dnodeD_s1));
    cucheck(cudaFree(d_ordered_pointsR_s2));
    cucheck(cudaFree(dnodeR_s2));
    cucheck(cudaFree(d_ordered_pointsD_s3));
    cucheck(cudaFree(dnodeD_s3));
    cucheck(cudaFree(d_ordered_pointsR_s3));
    cucheck(cudaFree(dnodeR_s3));
    
    delete[] hnodeD_s;
    delete[] h_ordered_pointsD_s;
    delete[] hnodeR_s;
    delete[] h_ordered_pointsR_s;

    cout << "Program terminated..." << endl;
    return 0;
}

