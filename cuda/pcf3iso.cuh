//Simple compilation
// 01:07
//nvcc -arch=sm_75 main.cu -o par_d.out && ./par_d.out data.dat rand0.dat 10000 30 50
//nvcc -arch=sm_75 main.cu -o par_d.out && ./par_d.out data_1GPc.dat rand_1GPc.dat 405224 30 150 1024

#include <string.h>
#include <math.h>
#include "kernels/3iso_k.cuh"

void pcf_3iso(string *histo_names, DNode *dnodeD, PointW3D *d_ordered_pointsD, int nonzero_Dnodes, DNode *dnodeR, PointW3D *d_ordered_pointsR, int *nonzero_Rnodes, int *acum_nonzero_Rnodes, int n_randfiles, int bn, float size_node, float dmax){
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

    float time_spent, d_max_node;

    double *DDD, *RRR, *DRR, *DDR;
    double *d_DDD, *d_RRR, *d_DRR, *d_DDR;

    //n_kernel_calls should depend of the number of points, its density, and the number of bins
    int threads_perblock_dim = 8;
    int blocks_D, blocks_R;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    cucheck(cudaEventCreate(&start_timmer));
    cucheck(cudaEventCreate(&stop_timmer));

    cudaStream_t streamDDD, *streamRRR, *streamDDR, *streamDRR;
    streamRRR = new cudaStream_t[n_randfiles];
    streamDDR = new cudaStream_t[n_randfiles];
    streamDRR = new cudaStream_t[n_randfiles];
    cucheck(cudaStreamCreate(&streamDDD));
    for (int i = 0; i < n_randfiles; i++){
        cucheck(cudaStreamCreate(&streamRRR[i]));
        cucheck(cudaStreamCreate(&streamDDR[i]));
        cucheck(cudaStreamCreate(&streamDRR[i]));
    }

    // Name of the files where the results are saved
    string nameDDD = "DDDiso_", nameRRR = "RRRiso_", nameDDR = "DDRiso_", nameDRR = "DRRiso_";

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    // Allocate memory for the histogram as double
    DDD = new double[bn*bn*bn];
    RRR = new double[n_randfiles*bn*bn*bn];
    DDR = new double[n_randfiles*bn*bn*bn];
    DRR = new double[n_randfiles*bn*bn*bn];

    cucheck(cudaMalloc(&d_DDD, bn*bn*bn*sizeof(double)));
    cucheck(cudaMalloc(&d_RRR, n_randfiles*bn*bn*bn*sizeof(double)));
    cucheck(cudaMalloc(&d_DRR, n_randfiles*bn*bn*bn*sizeof(double)));
    cucheck(cudaMalloc(&d_DDR, n_randfiles*bn*bn*bn*sizeof(double)));

    //Restarts the main histograms in host to zero
    cucheck(cudaMemsetAsync(d_DDD, 0, bn*bn*bn*sizeof(double), streamDDD));
    cucheck(cudaMemsetAsync(d_RRR, 0, n_randfiles*bn*bn*bn*sizeof(double), streamRRR[0]));
    cucheck(cudaMemsetAsync(d_DRR, 0, n_randfiles*bn*bn*bn*sizeof(double), streamDRR[0]));
    cucheck(cudaMemsetAsync(d_DDR, 0, n_randfiles*bn*bn*bn*sizeof(double), streamDDR[0]));
    cucheck(cudaStreamSynchronize(streamRRR[0]));
    cucheck(cudaStreamSynchronize(streamDRR[0]));
    cucheck(cudaStreamSynchronize(streamDDR[0]));

    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/


    //Compute the dimensions of the GPU grid
    //One thread for each node
    
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));
    dim3 gridDDD(blocks_D,blocks_D,blocks_D);
    dim3 threads_perblock(threads_perblock_dim,threads_perblock_dim,threads_perblock_dim);
    
    //Dummy declaration
    dim3 gridRRR(2,2,1);
    dim3 gridDDR(blocks_D,blocks_D,1);
    dim3 gridDRR(blocks_D,2,1);

    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    XXX3iso<<<gridDDD,threads_perblock,0,streamDDD>>>(d_DDD, d_ordered_pointsD, dnodeD, nonzero_Dnodes, bn, dmax, d_max_node);
    for (int i=0; i<n_randfiles; i++){
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridRRR.x = blocks_R;
        gridRRR.y = blocks_R;
        gridRRR.z = blocks_R;
        XXX3iso<<<gridRRR,threads_perblock,0,streamRRR[i]>>>(d_RRR, d_ordered_pointsR, dnodeR, nonzero_Rnodes[i], bn, dmax, d_max_node, acum_nonzero_Rnodes[i], i);
        gridDDR.z = blocks_R;
        cout << "i: " << i << "grid ddr: "<< gridDDR.x << ", " << gridDDR.y << ", " << gridDDR.z << endl;
        XXY3iso<<<gridDDR,threads_perblock,0,streamDDR[i]>>>(d_DDR, d_ordered_pointsD, dnodeD, nonzero_Dnodes, d_ordered_pointsR, dnodeR, nonzero_Rnodes[i], bn, dmax, d_max_node, acum_nonzero_Rnodes[i], i, true);
        gridDRR.y = blocks_R;
        gridDRR.z = blocks_R;
        cout << "i: " << i << "grid drr: "<< gridDRR.x << ", " << gridDRR.y << ", " << gridDRR.z << endl;
        XXY3iso<<<gridDRR,threads_perblock,0,streamDRR[i]>>>(d_DRR, d_ordered_pointsR, dnodeR, nonzero_Rnodes[i], d_ordered_pointsD, dnodeD, nonzero_Dnodes, bn, dmax, d_max_node, acum_nonzero_Rnodes[i], i, false);
    }

    //Waits for all the kernels to complete
    cucheck(cudaDeviceSynchronize());

    //Save the results
    cucheck(cudaMemcpy(DDD, d_DDD, bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost));
    nameDDD.append(histo_names[0]);
    save_histogram3D(nameDDD, bn, DDD);

    cucheck(cudaMemcpy(RRR, d_RRR, n_randfiles*bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameRRR.append(histo_names[i+1]);
        save_histogram3D(nameRRR, bn, RRR, i);
    }
    cucheck(cudaMemcpy(DDR, d_DDR, n_randfiles*bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameDDR.append(histo_names[i+1]);
        save_histogram3D(nameDDR, bn, DDR, i);
    }
    cucheck(cudaMemcpy(DRR, d_DRR, n_randfiles*bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameDRR.append(histo_names[i+1]);
        save_histogram3D(nameDRR, bn, DRR, i);
    }

    cucheck(cudaEventRecord(stop_timmer));
    cucheck(cudaEventSynchronize(stop_timmer));
    cucheck(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    cout << "Spent "<< time_spent << " miliseconds to compute and save all the histograms." << endl;
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory
    cucheck(cudaStreamDestroy(streamDDD));
    for (int i = 0; i < n_randfiles; i++){
        cucheck(cudaStreamDestroy(streamDDR[i]));
        cucheck(cudaStreamDestroy(streamDRR[i]));
        cucheck(cudaStreamDestroy(streamRRR[i]));
    }
    delete[] streamDDR;
    delete[] streamDRR;
    delete[] streamRRR;
    
    cucheck(cudaEventDestroy(start_timmer));
    cucheck(cudaEventDestroy(stop_timmer));

    delete[] DDD;
    delete[] RRR;
    delete[] DRR;    
    delete[] DDR;    
    
    cucheck(cudaFree(d_DDD));
    cucheck(cudaFree(d_RRR));
    cucheck(cudaFree(d_DRR));
    cucheck(cudaFree(d_DDR));

}

