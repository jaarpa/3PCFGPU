// nvcc -arch=sm_75 main.cu -o par_s.out && ./par_s.out data_2GPc.dat data_2GPc.dat 3241792 20 160
// nvcc -arch=sm_75 main.cu -o par_s.out && ./par_s.out data_1GPc.dat data_1GPc.dat 405224 20 160
// nvcc -arch=sm_75 main.cu -o par_s.out && ./par_s.out data.dat rand0.dat 32768 20 150
// nvcc -arch=sm_75 main.cu -o par_s.out && ./par_s.out data_5K.dat rand0_5K.dat 5000 30 180

#include <string.h>
#include <math.h>
#include "kernels/2iso_k.cuh"


void pcf_2iso(string *histo_names, DNode *dnodeD, PointW3D *d_ordered_pointsD, int nonzero_Dnodes, DNode *dnodeR, PointW3D *d_ordered_pointsR, int *nonzero_Rnodes, int *acum_nonzero_Rnodes, int n_randfiles, int bn, float size_node, float dmax){
    /*
    Main function to calculate the isotropic 2 point correlation function. Saves three different histograms in the same location of this script
    with the names DD.dat DR.dat RR.dat. This program do not consider periodic boundary conditions. The file must contain 4 columns, the first 3 
    are the x,y,z coordinates and the 4 the weigh of the measurment.

    Args:
    dnodeD: (DNode)
    d_ordered_pointsD: (PointW3D)
    nonzero_Dnodes: (int)
    hnodeR_s: (DNode)
    h_ordered_pointsR_s: (PointW3D)
    nonzero_Rnodes: (int)
    n_randfiles: (int)
    bn: (int)
    dmax: (float)

    */

    /* =======================================================================*/
    /* ======================  Var declaration ===============================*/
    /* =======================================================================*/

    float d_max_node, time_spent;
    double *DD, *RR, *DR, *d_DD, *d_RR, *d_DR;
    int  blocks_D, blocks_R, threads_perblock_dim = 32;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    cucheck(cudaEventCreate(&start_timmer));
    cucheck(cudaEventCreate(&stop_timmer));

    cudaStream_t streamDD, *streamRR, *streamDR;
    streamRR = new cudaStream_t[n_randfiles];
    streamDR = new cudaStream_t[n_randfiles];
    cucheck(cudaStreamCreate(&streamDD));
    for (int i = 0; i < n_randfiles; i++){
        cucheck(cudaStreamCreate(&streamDR[i]));
        cucheck(cudaStreamCreate(&streamRR[i]));
    }

    std::string nameDD, nameRR, nameDR;

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    // Allocate memory for the histogram as double
    DD = new double[bn];
    RR = new double[n_randfiles*bn];
    DR = new double[n_randfiles*bn];

    cucheck(cudaMalloc(&d_DD, bn*sizeof(double)));
    cucheck(cudaMalloc(&d_RR, n_randfiles*bn*sizeof(double)));
    cucheck(cudaMalloc(&d_DR, n_randfiles*bn*sizeof(double)));

    //Restarts the main histograms in device to zero
    cucheck(cudaMemsetAsync(d_DD, 0, bn*sizeof(double), streamDD));
    cucheck(cudaMemsetAsync(d_RR, 0, n_randfiles*bn*sizeof(double), streamRR[0]));
    cucheck(cudaMemsetAsync(d_DR, 0, n_randfiles*bn*sizeof(double), streamDR[0]));
    cucheck(cudaStreamSynchronize(streamRR[0]));
    cucheck(cudaStreamSynchronize(streamDR[0]));

    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/


    //Compute the dimensions of the GPU grid
    //One thread for each node
    
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));
    dim3 threads_perblock(threads_perblock_dim,threads_perblock_dim,1);
    dim3 gridD(blocks_D,blocks_D,1);
    
    //Dummy declaration
    dim3 gridR(2,2,1);
    dim3 gridDR(blocks_D,2,1);

    //Launch the kernels
    time_spent=0; //Restarts timmer
    cucheck(cudaEventRecord(start_timmer));
    XX2iso<<<gridD,threads_perblock,0,streamDD>>>(d_DD, d_ordered_pointsD, dnodeD, nonzero_Dnodes, bn, dmax, d_max_node);
    for (int i=0; i<n_randfiles; i++){
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridR.x = blocks_R;
        gridR.y = blocks_R;
        XX2iso<<<gridR,threads_perblock,0,streamRR[i]>>>(d_RR, d_ordered_pointsR, dnodeR, nonzero_Rnodes[i], bn, dmax, d_max_node, acum_nonzero_Rnodes[i], i);
        gridDR.y = blocks_R;
        XY2iso<<<gridDR,threads_perblock,0,streamDR[i]>>>(d_DR, d_ordered_pointsD, dnodeD, nonzero_Dnodes, d_ordered_pointsR, dnodeR, nonzero_Rnodes[i], bn, dmax, d_max_node, acum_nonzero_Rnodes[i], i);
    }

    //Waits for all the kernels to complete
    cucheck(cudaDeviceSynchronize());
    


    cucheck(cudaMemcpy(DD, d_DD, bn*sizeof(double), cudaMemcpyDeviceToHost));
    nameDD = "DDiso_":
    nameDD.append(histo_names[0]);
    save_histogram1D(nameDD, bn, DD);
    cucheck(cudaMemcpy(RR, d_RR, n_randfiles*bn*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameRR = "RRiso_";
        nameRR.append(histo_names[i+1]);
        save_histogram1D(nameRR, bn, RR, i);
    }
    cucheck(cudaMemcpy(DR, d_DR, n_randfiles*bn*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<n_randfiles; i++){
        nameDR = "DRiso_";
        nameDR.append(histo_names[i+1]);
        save_histogram1D(nameDR, bn, DR, i);
    }


    cucheck(cudaEventRecord(stop_timmer));
    cucheck(cudaEventSynchronize(stop_timmer));
    cucheck(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    cout << "Spent "<< time_spent << " miliseconds to compute and save all the histograms." << endl;
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory
    cucheck(cudaStreamDestroy(streamDD));
    for (int i = 0; i < n_randfiles; i++){
        cucheck(cudaStreamDestroy(streamDR[i]));
        cucheck(cudaStreamDestroy(streamRR[i]));
    }
    delete[] streamDR;
    delete[] streamRR;

    cucheck(cudaEventDestroy(start_timmer));
    cucheck(cudaEventDestroy(stop_timmer));

    delete[] DD;
    delete[] RR;    
    delete[] DR;    
    
    cucheck(cudaFree(d_DD));
    cucheck(cudaFree(d_RR));
    cucheck(cudaFree(d_DR));

}

