//Simple compilation
// 01:07
//nvcc -arch=sm_75 main.cu -o par_d.out && ./par_d.out data.dat rand0.dat 5000 30 60
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
    int ptt = 100, bn_ref = 200;
    int bn_XX_ff_av = ptt*bn, bn_XX_ff_av_ref = ptt*bn_ref*bn;;

    float time_spent, d_max_node, size_node, dmax = stof(argv[5]), size_box = 0, r_size_box=0;

    double *DDD, *RRR, *DDR;
    double *d_DDD, *d_RRR, *d_DDR;
    double *d_DD_ff_av, *d_RR_ff_av, *d_DD_ff_av_ref, *d_RR_ff_av_ref;
    double *d_ff_av, *d_ff_av_ref;
    double dr_ff_av, alpha_ff_av, dr_ff_av_ref, alpha_ff_av_ref, beta = (np*np)/(size_box*size_box*size_box);

    int nonzero_Dnodes = 0, threads_perblock_dim = 8, idxD=0;
    int threads_bn_ff_av=16, threads_ptt_ff_av=64;
    int gridRR_ff_av, gridRR_ff_av_ref, threads_perblock_RR_ff_av, threads_perblock_RR_ff_av_ref;
    int gridff_av_ref_x, gridff_av_ref_y, gridff_av_ref_z, threadsff_av_ref_x = 8, threadsff_av_ref_y = 16, threadsff_av_ref_z = 8;
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
    cudaStream_t streamDDD, stream_analytic, streamRR_ff_av, streamRR_ff_av_ref;
    cucheck(cudaStreamCreate(&streamDDD));
    cucheck(cudaStreamCreate(&stream_analytic));
    cucheck(cudaStreamCreate(&streamRR_ff_av));
    cucheck(cudaStreamCreate(&streamRR_ff_av_ref));
    DNode *dnodeD;
    PointW3D *d_ordered_pointsD;

    // Name of the files where the results are saved
    string nameDDD = "DDDiso_BPCanalytic_", nameRRR = "RRRiso_BPCanalytic_", nameDDR = "DDRiso_BPCanalytic_";
    string data_name = argv[1];
    nameDDD.append(data_name);
    nameRRR.append(data_name);
    nameDDR.append(data_name);

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
    
    dr_ff_av = (dmax/bn_XX_ff_av);
    alpha_ff_av = 8*dr_ff_av*dr_ff_av*dr_ff_av*(acos(0.0))*(beta)/3;
    dr_ff_av_ref = (dmax/bn_XX_ff_av_ref);
    alpha_ff_av_ref = 8*dr_ff_av_ref*dr_ff_av_ref*dr_ff_av_ref*(acos(0.0))*(beta)/3;

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
    cucheck(cudaMemsetAsync(d_DD_ff_av, 0, bn_XX_ff_av*sizeof(double), stream_analytic));
    cucheck(cudaMemsetAsync(d_DD_ff_av_ref, 0, bn_XX_ff_av_ref*sizeof(double), stream_analytic));
    cucheck(cudaMemsetAsync(d_RR_ff_av, 0, bn_XX_ff_av*sizeof(double), streamRR_ff_av));
    cucheck(cudaMemsetAsync(d_RR_ff_av_ref, 0, bn_XX_ff_av_ref*sizeof(double), streamRR_ff_av_ref));
    cucheck(cudaMemsetAsync(d_ff_av, 0, bn*sizeof(double), streamRR_ff_av));
    cucheck(cudaMemsetAsync(d_ff_av_ref, 0, bn_ref*bn*sizeof(double), streamRR_ff_av_ref));

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

    //Allocate memory for the nodes depending of how many partitions there are.
    cucheck(cudaMalloc(&dnodeD, nonzero_Dnodes*sizeof(DNode)));
    cucheck(cudaMalloc(&d_ordered_pointsD, np*sizeof(PointW3D)));
    cucheck(cudaMemcpyAsync(dnodeD, hnodeD_s, nonzero_Dnodes*sizeof(DNode), cudaMemcpyHostToDevice, streamDDD));
    cucheck(cudaMemcpyAsync(d_ordered_pointsD, h_ordered_pointsD_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDDD));
    
    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            delete[] hnodeD[i][j];
        }
        delete[] hnodeD[i];
    }    
    delete[] hnodeD;
    
    delete[] dataD;
    
    cucheck(cudaStreamSynchronize(streamDDD)); //Waits to copy all the nodes into device

    delete[] hnodeD_s;
    delete[] h_ordered_pointsD_s;


    stop_timmer_host = clock();
    time_spent = ((float)(stop_timmer_host-start_timmer_host))/CLOCKS_PER_SEC;
    cout << "Succesfully readed the data. All set to compute the histograms in " << time_spent*1000 << " miliseconds" << endl;


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
    i<bn && j<bn_ref && k<ptt
    dim3 threads_perblockff_av_ref(threadsff_av_ref_x,threadsff_av_ref_y,threadsff_av_ref_z);
    gridff_av_ref_x = (int)(ceil((float)((float)(bn)/(float)(threadsff_av_ref_x))));
    gridff_av_ref_y = (int)(ceil((float)((float)(bn_ref)/(float)(threadsff_av_ref_y))));
    gridff_av_ref_z = (int)(ceil((float)((float)(ptt)/(float)(threadsff_av_ref_z))));
    dim3 gridff_av_ref(gridff_av_ref_x,gridff_av_ref_y,gridff_av_ref_z);


    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    make_histoXXX<<<gridDDD,threads_perblockDDD,0,streamDDD>>>(d_DDD, d_ordered_pointsD, dnodeD, nonzero_Dnodes, bn, dmax, d_max_node, size_box, size_node);

    make_histoDD<<<gridDD,threads_perblockDD,0,stream_analytic>>>(d_DD_ff_av_ref, d_DD_ff_av, d_ordered_pointsD, dnodeD, nonzero_Dnodes, bn_XX_ff_av_ref, bn_XX_ff_av, dmax, d_max_node, size_box, size_node);
    make_histoRR<<<gridRR_ff_av,threads_perblock_RR_ff_av,0,streamRR_ff_av>>>(d_RR_ff_av, alpha_ff_av, bn_XX_ff_av);
    make_histoRR<<<gridRR_ff_av_ref,threads_perblock_RR_ff_av_ref,0,streamRR_ff_av_ref>>>(d_RR_ff_av_ref, alpha_ff_av_ref, bn_XX_ff_av_ref);
    
    //Wait for the 2PCF histograms to be finished. The DDD could not be finished yet
    cucheck(cudaStreamSynchronize(stream_analytic));
    cucheck(cudaStreamSynchronize(streamRR_ff_av));
    cucheck(cudaStreamSynchronize(streamRR_ff_av_ref));
    
    make_ff_av<<<gridff_av,threads_perblockff_av,bn*sizeof(double),streamRR_ff_av>>>(d_ff_av, d_DD_ff_av, d_RR_ff_av, dmax, bn, bn_XX_ff_av, ptt);
    make_ff_av_ref<<<gridff_av_ref,threads_perblockff_av_ref,0,streamRR_ff_av_ref>>>(d_ff_av_ref, d_DD_ff_av_ref, d_RR_ff_av_ref, dmax, bn, bn_ref, ptt)

    cucheck(cudaMemcpyAsync(DDD, d_DDD, bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost, streamDDD));

    //Waits for all the kernels to complete
    cucheck(cudaStreamSynchronize(streamDDD));
    save_histogram(nameDDD, bn, DDD);

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
    cucheck(cudaFree(d_DD_ff_av_ref));
    cucheck(cudaFree(d_RR_ff_av_ref));
    cucheck(cudaFree(d_DD_ff_av));
    cucheck(cudaFree(d_RR_ff_av));
    cucheck(cudaFree(d_ff_av));
    cucheck(cudaFree(d_ff_av_ref));

    cucheck(cudaFree(dnodeD));
    cucheck(cudaFree(d_ordered_pointsD));

    cout << "Program terminated..." << endl;
    return 0;
}

