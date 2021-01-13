//Simple compilation
// 01:07
//nvcc -arch=sm_75 main.cu -o par_d.out && ./par_d.out data.dat rand0.dat 10000 30 50
//nvcc -arch=sm_75 main.cu -o par_d.out && ./par_d.out data_1GPc.dat rand_1GPc.dat 405224 30 150 1024

#include <string.h>
#include <math.h>
#include "kernels/3isoBPC_k.cuh"

void pcf_3isoBPC(string *histo_names, DNode *dnodeD, PointW3D *d_ordered_pointsD, int nonzero_Dnodes, DNode *dnodeR, PointW3D *d_ordered_pointsR, int *nonzero_Rnodes, int *acum_nonzero_Rnodes, int n_randfiles, int bn, float size_node, float size_box, float dmax){
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
    string nameDDD = "DDDiso_BPC_", nameRRR = "RRRiso_BPC_", nameDDR = "DDRiso_BPC_", nameDRR = "DRRiso_BPC_";

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
    dim3 gridDRR(2,2,blocks_D);

    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    XXX3iso_BPC<<<gridDDD,threads_perblock,0,streamDDD>>>(d_DDD, d_ordered_pointsD, dnodeD, nonzero_Dnodes, bn, dmax, d_max_node, size_box, size_node);
    for (int i=0; i<n_randfiles; i++){
        //Calculates grid dim for each file
        blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes[i])/(float)(threads_perblock_dim))));
        gridRRR.x = blocks_R;
        gridRRR.y = blocks_R;
        gridRRR.z = blocks_R;
        XXX3iso_BPC<<<gridRRR,threads_perblock,0,streamRRR[i]>>>(d_RRR, d_ordered_pointsR, dnodeR, nonzero_Rnodes[i], bn, dmax, d_max_node, size_box, size_node, acum_nonzero_Rnodes[i], i);
        gridDDR.z = blocks_R;
        XXY3iso_BPC<<<gridDDR,threads_perblock,0,streamDDR[i]>>>(d_DDR, d_ordered_pointsD, dnodeD, nonzero_Dnodes, d_ordered_pointsR, dnodeR, nonzero_Rnodes[i], bn, dmax, d_max_node, size_box, size_node, acum_nonzero_Rnodes[i], i, true);
        gridDRR.x = blocks_R;
        gridDRR.y = blocks_R;
        XXY3iso_BPC<<<gridDRR,threads_perblock,0,streamDRR[i]>>>(d_DRR, d_ordered_pointsR, dnodeR, nonzero_Rnodes[i], d_ordered_pointsD, dnodeD, nonzero_Dnodes, bn, dmax, d_max_node, size_box, size_node, acum_nonzero_Rnodes[i], i, false);
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

void pcf_3isoBPC_analytic(string data_name, DNode *dnodeD, PointW3D *d_ordered_pointsD, int nonzero_Dnodes, int bn, int np, float size_node, float size_box, float dmax){
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

    double *DDD, *RRR, *DDR;
    double *d_DDD, *d_RRR, *d_DDR;
    double *d_DD_ff_av, *d_RR_ff_av, *d_DD_ff_av_ref, *d_RR_ff_av_ref;
    double *d_ff_av, *d_ff_av_ref;
    double dr_ff_av, alpha_ff_av, dr_ff_av_ref, alpha_ff_av_ref, beta;
    double dr, dr_ref, V, beta_3D, gama, alpha, alpha_ref;

    //n_kernel_calls should depend of the number of points, its density, and the number of bins
    int ptt = 100, bn_ref = 200;
    int bn_XX_ff_av = ptt*bn, bn_XX_ff_av_ref = ptt*bn_ref*bn;
    int threads_perblock_dim = 8, threads_bn_ff_av=16, threads_ptt_ff_av=64;
    int gridRR_ff_av, gridRR_ff_av_ref, threads_perblock_RR_ff_av, threads_perblock_RR_ff_av_ref;
    int gridff_av_ref_x, gridff_av_ref_y, gridff_av_ref_z, threadsff_av_ref_x = 8, threadsff_av_ref_y = 16, threadsff_av_ref_z = 8;
    int blocks_D, blocks_analytic;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    cucheck(cudaEventCreate(&start_timmer));
    cucheck(cudaEventCreate(&stop_timmer));
    
    cudaStream_t streamDDD, stream_analytic, streamRR_ff_av, streamRR_ff_av_ref;
    cucheck(cudaStreamCreate(&streamDDD));
    cucheck(cudaStreamCreate(&stream_analytic));
    cucheck(cudaStreamCreate(&streamRR_ff_av));
    cucheck(cudaStreamCreate(&streamRR_ff_av_ref));

    // Name of the files where the results are saved
    string nameDDD = "DDDiso_BPCanalytic_", nameRRR = "RRRiso_BPCanalytic_", nameDDR = "DDRiso_BPCanalytic_", nameDRR = "DRRiso_BPCanalytic_";
    nameDDD.append(data_name);
    nameRRR.append(data_name);
    nameDDR.append(data_name);
    nameDRR.append(data_name);

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/

    d_max_node = dmax + size_node*sqrt(3.0);
    d_max_node*=d_max_node;

    beta = (np*np)/(size_box*size_box*size_box);

    dr_ff_av = (dmax/bn_XX_ff_av);
    alpha_ff_av = 8*dr_ff_av*dr_ff_av*dr_ff_av*(acos(0.0))*(beta)/3;

    dr_ff_av_ref = (dmax/bn_XX_ff_av_ref);
    alpha_ff_av_ref = 8*dr_ff_av_ref*dr_ff_av_ref*dr_ff_av_ref*(acos(0.0))*(beta)/3;

    dr = dmax/(double)bn, dr_ref = dr/bn_ref, V = size_box*size_box, beta_3D = np/V;
    gama = 8*(4*acos(0.0)*acos(0.0))*(np*beta_3D*beta_3D);
    gama /= V;
    alpha_ref = gama*dr_ref*dr_ref*dr_ref, alpha = gama*dr*dr*dr;

    // Allocate memory for the histogram as double
    DDD = new double[bn*bn*bn];
    RRR = new double[bn*bn*bn];
    DDR = new double[bn*bn*bn];

    cucheck(cudaMalloc(&d_DDD, bn*bn*bn*sizeof(double)));
    cucheck(cudaMalloc(&d_RRR, bn*bn*bn*sizeof(double)));
    cucheck(cudaMalloc(&d_DDR, bn*bn*bn*sizeof(double)));
    cucheck(cudaMalloc(&d_DD_ff_av, bn_XX_ff_av*sizeof(double)));
    cucheck(cudaMalloc(&d_RR_ff_av, bn_XX_ff_av*sizeof(double)));
    cucheck(cudaMalloc(&d_DD_ff_av_ref, bn_XX_ff_av_ref*sizeof(double)));
    cucheck(cudaMalloc(&d_RR_ff_av_ref, bn_XX_ff_av_ref*sizeof(double)));
    cucheck(cudaMalloc(&d_ff_av, bn*sizeof(double)));
    cucheck(cudaMalloc(&d_ff_av_ref, bn_ref*bn*sizeof(double)));

    //Restarts the main histograms in host to zero
    cucheck(cudaMemsetAsync(d_DDD, 0, bn*bn*bn*sizeof(double), streamDDD));
    cucheck(cudaMemsetAsync(d_DDR, 0, bn*bn*bn*sizeof(double), stream_analytic));
    cucheck(cudaMemsetAsync(d_RRR, 0, bn*bn*bn*sizeof(double), stream_analytic));
    cucheck(cudaMemsetAsync(d_DD_ff_av, 0, bn_XX_ff_av*sizeof(double), stream_analytic));
    cucheck(cudaMemsetAsync(d_DD_ff_av_ref, 0, bn_XX_ff_av_ref*sizeof(double), stream_analytic));
    cucheck(cudaMemsetAsync(d_RR_ff_av, 0, bn_XX_ff_av*sizeof(double), streamRR_ff_av));
    cucheck(cudaMemsetAsync(d_RR_ff_av_ref, 0, bn_XX_ff_av_ref*sizeof(double), streamRR_ff_av_ref));
    cucheck(cudaMemsetAsync(d_ff_av, 0, bn*sizeof(double), streamRR_ff_av));
    cucheck(cudaMemsetAsync(d_ff_av_ref, 0, bn_ref*bn*sizeof(double), streamRR_ff_av_ref));

    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/


    //Compute the dimensions of the GPU grid
    //One thread for each node
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));

    dim3 threads_perblockDDD(threads_perblock_dim,threads_perblock_dim,threads_perblock_dim);
    dim3 gridDDD(blocks_D,blocks_D,blocks_D);
    dim3 threads_perblockDD(threads_perblock_dim,threads_perblock_dim,1);
    dim3 gridDD(blocks_D,blocks_D,1);

    threads_perblock_RR_ff_av = (bn_XX_ff_av<1024)*bn_XX_ff_av + (bn_XX_ff_av>=1024)*512;
    gridRR_ff_av = (int)(ceil((float)((float)(bn_XX_ff_av)/(float)(threads_perblock_RR_ff_av))));
    threads_perblock_RR_ff_av_ref = (bn_XX_ff_av_ref<1024)*bn_XX_ff_av_ref + (bn_XX_ff_av_ref>=1024)*512;
    gridRR_ff_av_ref = (int)(ceil((float)((float)(bn_XX_ff_av_ref)/(float)(threads_perblock_RR_ff_av_ref))));
    
    dim3 threads_perblockff_av(threads_bn_ff_av,threads_ptt_ff_av,1);
    dim3 gridff_av((int)(ceil((float)((float)(bn)/(float)(threads_bn_ff_av)))),(int)(ceil((float)((float)(ptt)/(float)(threads_ptt_ff_av)))),1);

    dim3 threads_perblockff_av_ref(threadsff_av_ref_x,threadsff_av_ref_y,threadsff_av_ref_z);
    gridff_av_ref_x = (int)(ceil((float)((float)(ptt)/(float)(threadsff_av_ref_x))));
    gridff_av_ref_y = (int)(ceil((float)((float)(bn_ref)/(float)(threadsff_av_ref_y))));
    gridff_av_ref_z = (int)(ceil((float)((float)(bn)/(float)(threadsff_av_ref_z))));
    dim3 gridff_av_ref(gridff_av_ref_x,gridff_av_ref_y,gridff_av_ref_z);

    blocks_analytic = (int)(ceil((float)((float)(bn)/(float)(threads_perblock_dim))));
    dim3 gridanalytic(blocks_analytic,blocks_analytic,blocks_analytic);


    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    XXX3iso_BPC<<<gridDDD,threads_perblockDDD,0,streamDDD>>>(d_DDD, d_ordered_pointsD, dnodeD, nonzero_Dnodes, bn, dmax, d_max_node, size_box, size_node);

    DD2iso_forBPC<<<gridDD,threads_perblockDD,0,stream_analytic>>>(d_DD_ff_av_ref, d_DD_ff_av, d_ordered_pointsD, dnodeD, nonzero_Dnodes, bn_XX_ff_av_ref, bn_XX_ff_av, dmax, d_max_node, size_box, size_node);
    
    RR2iso_BPC_analytic<<<gridRR_ff_av,threads_perblock_RR_ff_av,0,streamRR_ff_av>>>(d_RR_ff_av, alpha_ff_av, bn_XX_ff_av);
    RR2iso_BPC_analytic<<<gridRR_ff_av_ref,threads_perblock_RR_ff_av_ref,0,streamRR_ff_av_ref>>>(d_RR_ff_av_ref, alpha_ff_av_ref, bn_XX_ff_av_ref);
    
    //Wait for the 2PCF histograms to be finished. The DDD could not be finished yet
    cucheck(cudaStreamSynchronize(stream_analytic));
    cucheck(cudaStreamSynchronize(streamRR_ff_av));
    cucheck(cudaStreamSynchronize(streamRR_ff_av_ref));
    
    make_ff_av<<<gridff_av,threads_perblockff_av,0,streamRR_ff_av>>>(d_ff_av, d_DD_ff_av, d_RR_ff_av, dmax, bn, bn_XX_ff_av, ptt);
    make_ff_av_ref<<<gridff_av_ref,threads_perblockff_av_ref,0,streamRR_ff_av_ref>>>(d_ff_av_ref, d_DD_ff_av_ref, d_RR_ff_av_ref, dmax, bn, bn_ref, ptt);

    //Waits to finish the ff_av and ff_av_ref histograms
    cucheck(cudaStreamSynchronize(streamRR_ff_av));
    cucheck(cudaStreamSynchronize(streamRR_ff_av_ref));

    make_histo_analitic<<<gridanalytic,threads_perblockDDD,0,stream_analytic>>>(d_DDR, d_RRR, d_ff_av, d_ff_av_ref, alpha, alpha_ref, dmax, bn, bn_ref);

    cucheck(cudaMemcpyAsync(DDD, d_DDD, bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost, streamDDD));
    cucheck(cudaMemcpyAsync(DDR, d_DDR, bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost, stream_analytic));
    cucheck(cudaMemcpyAsync(RRR, d_RRR, bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost, stream_analytic));

    //Save the results
    cucheck(cudaStreamSynchronize(streamDDD));
    save_histogram3D(nameDDD, bn, DDD);
    cucheck(cudaStreamSynchronize(stream_analytic));
    save_histogram3D(nameDDR, bn, DDR);
    save_histogram3D(nameDRR, bn, DDR);
    save_histogram3D(nameRRR, bn, RRR);

    cucheck(cudaEventRecord(stop_timmer));
    cucheck(cudaEventSynchronize(stop_timmer));
    cucheck(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    cout << "Spent "<< time_spent << " miliseconds to compute and save all the histograms." << endl;
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory
    cucheck(cudaStreamDestroy(streamDDD));
    cucheck(cudaStreamDestroy(stream_analytic));
    cucheck(cudaStreamDestroy(streamRR_ff_av));
    cucheck(cudaStreamDestroy(streamRR_ff_av_ref));
    
    cucheck(cudaEventDestroy(start_timmer));
    cucheck(cudaEventDestroy(stop_timmer));

    delete[] DDD;
    delete[] RRR;
    delete[] DDR;    

    cucheck(cudaFree(d_DDD));
    cucheck(cudaFree(d_RRR));
    cucheck(cudaFree(d_DDR));
    cucheck(cudaFree(d_DD_ff_av));
    cucheck(cudaFree(d_RR_ff_av));
    cucheck(cudaFree(d_DD_ff_av_ref));
    cucheck(cudaFree(d_RR_ff_av_ref));
    cucheck(cudaFree(d_ff_av));
    cucheck(cudaFree(d_ff_av_ref));

}