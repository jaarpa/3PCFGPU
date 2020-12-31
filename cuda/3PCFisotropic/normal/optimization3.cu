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

    double *DDD;//, *RRR, *DRR, *DDR;
    double *d_DDD;//, *d_RRR, *d_DRR, *d_DDR;

    //n_kernel_calls should depend of the number of points, its density, and the number of bins
    int threads_perblock_dim = 8, nonzero_Dnodes = 0, idxD=0;//, nonzero_Rnodes = 0, idxR=0;
    int blocks_D;//, blocks_R;

    cudaEvent_t start_timmer, stop_timmer; // GPU timmer
    cucheck(cudaEventCreate(&start_timmer));
    cucheck(cudaEventCreate(&stop_timmer));

    clock_t stop_timmer_host, start_timmer_host;

    PointW3D *dataD;
    //PointW3D *dataR;

    int k_element, last_pointD;//, last_pointR;
    Node ***hnodeD;//, ***hnodeR;
    DNode *hnodeD_s;//, *hnodeR_s;
    PointW3D *h_ordered_pointsD_s;//, *h_ordered_pointsR_s;

    cudaStream_t streamDDD;//, streamDRR, streamDDR, streamRRR;
    cucheck(cudaStreamCreate(&streamDDD));
    //cucheck(cudaStreamCreate(&streamDDR));
    //cucheck(cudaStreamCreate(&streamDRR));
    //cucheck(cudaStreamCreate(&streamRRR));
    DNode *dnodeD_DDD;//, *dnodeD_DDR, *dnodeD_DRR;
    //DNode *dnodeR_RRR, *dnodeR_DDR, *dnodeR_DRR;
    PointW3D *d_ordered_pointsD_DDD;//, *d_ordered_pointsD_DDR, *d_ordered_pointsD_DRR;
    //PointW3D *d_ordered_pointsR_RRR, *d_ordered_pointsR_DDR, *d_ordered_pointsR_DRR;

    // Name of the files where the results are saved
    string nameDDD = "DDDiso.dat";//, nameRRR = "RRRiso.dat", nameDDR = "DDRiso.dat", nameDRR = "DRRiso.dat";

    /* =======================================================================*/
    /* =======================  Memory allocation ============================*/
    /* =======================================================================*/
    start_timmer_host = clock();
    dataD = new PointW3D[np];
    //dataR = new PointW3D[np];

    // Open and read the files to store the data in the arrays
    open_files(argv[1], np, dataD, size_box); //This function also gets the real size of the box
    //open_files(argv[2], np, dataR, r_size_box);
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
    //RRR = new double[bn*bn*bn];
    //DDR = new double[bn*bn*bn];
    //DRR = new double[bn*bn*bn];

    cucheck(cudaMalloc(&d_DDD, bn*bn*bn*sizeof(double)));
    //cucheck(cudaMalloc(&d_RRR, bn*bn*bn*sizeof(double)));
    //cucheck(cudaMalloc(&d_DRR, bn*bn*bn*sizeof(double)));
    //cucheck(cudaMalloc(&d_DDR, bn*bn*bn*sizeof(double)));

    //Restarts the main histograms in host to zero
    cucheck(cudaMemsetAsync(d_DDD, 0, bn*bn*bn*sizeof(double), streamDDD));
    //cucheck(cudaMemsetAsync(d_RRR, 0, bn*bn*bn*sizeof(double), streamRRR));
    //cucheck(cudaMemsetAsync(d_DRR, 0, bn*bn*bn*sizeof(double), streamDRR));
    //cucheck(cudaMemsetAsync(d_DDR, 0, bn*bn*bn*sizeof(double), streamDDR));

    hnodeD = new Node**[partitions];
    //hnodeR = new Node**[partitions];
    for (int i=0; i<partitions; i++){
        *(hnodeD+i) = new Node*[partitions];
        //*(hnodeR+i) = new Node*[partitions];
        for (int j=0; j<partitions; j++){
            *(*(hnodeD+i)+j) = new Node[partitions];
            //*(*(hnodeR+i)+j) = new Node[partitions];
        }
    }

    //Classificate the data into the nodes in the host side
    //The node classification is made in the host
    make_nodos(hnodeD, dataD, partitions, size_node, np);
    //make_nodos(hnodeR, dataR, partitions, size_node, np);

    for(int row=0; row<partitions; row++){
        for(int col=0; col<partitions; col++){
            for(int mom=0; mom<partitions; mom++){
                if(hnodeD[row][col][mom].len>0){
                    nonzero_Dnodes+=1;
                }
                //if(hnodeR[row][col][mom].len>0){
                    //nonzero_Rnodes+=1;
                //}
            }
        }
    }

    //Allocate memory for the nodes depending of how many partitions there are.
    cucheck(cudaMalloc(&dnodeD_DDD, nonzero_Dnodes*sizeof(DNode)));
    cucheck(cudaMalloc(&d_ordered_pointsD_DDD, np*sizeof(PointW3D)));
    //cucheck(cudaMalloc(&dnodeD_DDR, nonzero_Dnodes*sizeof(DNode)));
    //cucheck(cudaMalloc(&d_ordered_pointsD_DDR, np*sizeof(PointW3D)));
    //cucheck(cudaMalloc(&dnodeD_DRR, nonzero_Dnodes*sizeof(DNode)));
    //cucheck(cudaMalloc(&d_ordered_pointsD_DRR, np*sizeof(PointW3D)));

    //cucheck(cudaMalloc(&dnodeR_RRR, nonzero_Rnodes*sizeof(DNode)));
    //cucheck(cudaMalloc(&d_ordered_pointsR_RRR, np*sizeof(PointW3D)));
    //cucheck(cudaMalloc(&dnodeR_DDR, nonzero_Rnodes*sizeof(DNode)));
    //cucheck(cudaMalloc(&d_ordered_pointsR_DDR, np*sizeof(PointW3D)));
    //cucheck(cudaMalloc(&dnodeR_DRR, nonzero_Rnodes*sizeof(DNode)));
    //cucheck(cudaMalloc(&d_ordered_pointsR_DRR, np*sizeof(PointW3D)));

    hnodeD_s = new DNode[nonzero_Dnodes];
    h_ordered_pointsD_s = new PointW3D[np];
    //hnodeR_s = new DNode[nonzero_Rnodes];
    //h_ordered_pointsR_s = new PointW3D[np];
    
    //Deep copy to device memory
    last_pointD = 0;
    //last_pointR = 0;
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

                // if (hnodeR[row][col][mom].len>0){
                //     hnodeR_s[idxR].nodepos = hnodeR[row][col][mom].nodepos;
                //     hnodeR_s[idxR].start = last_pointR;
                //     hnodeR_s[idxR].len = hnodeR[row][col][mom].len;
                //     last_pointR = last_pointR + hnodeR[row][col][mom].len;
                //     hnodeR_s[idxR].end = last_pointR;
                //     for (int j=hnodeR_s[idxR].start; j<last_pointR; j++){
                //         k_element = j-hnodeR_s[idxR].start;
                //         h_ordered_pointsR_s[j] = hnodeR[row][col][mom].elements[k_element];
                //     }
                //     idxR++;
                // }

            }
        }
    }


    cucheck(cudaMemcpyAsync(dnodeD_DDD, hnodeD_s, nonzero_Dnodes*sizeof(DNode), cudaMemcpyHostToDevice, streamDDD));
    cucheck(cudaMemcpyAsync(d_ordered_pointsD_DDD, h_ordered_pointsD_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDDD));
    //cucheck(cudaMemcpyAsync(dnodeD_DDR, hnodeD_s, nonzero_Dnodes*sizeof(DNode), cudaMemcpyHostToDevice, streamDDR));
    //cucheck(cudaMemcpyAsync(d_ordered_pointsD_DDR, h_ordered_pointsD_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDDR));
    //cucheck(cudaMemcpyAsync(dnodeD_DRR, hnodeD_s, nonzero_Dnodes*sizeof(DNode), cudaMemcpyHostToDevice, streamDRR));
    //cucheck(cudaMemcpyAsync(d_ordered_pointsD_DRR, h_ordered_pointsD_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDRR));
    
    //cucheck(cudaMemcpyAsync(dnodeR_RRR, hnodeR_s, nonzero_Rnodes*sizeof(DNode), cudaMemcpyHostToDevice, streamRRR));
    //cucheck(cudaMemcpyAsync(d_ordered_pointsR_RRR, h_ordered_pointsR_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamRRR));
    //cucheck(cudaMemcpyAsync(dnodeR_DDR, hnodeR_s, nonzero_Rnodes*sizeof(DNode), cudaMemcpyHostToDevice, streamDDR));
    //cucheck(cudaMemcpyAsync(d_ordered_pointsR_DDR, h_ordered_pointsR_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDDR));
    //cucheck(cudaMemcpyAsync(dnodeR_DRR, hnodeR_s, nonzero_Rnodes*sizeof(DNode), cudaMemcpyHostToDevice, streamDRR));
    //cucheck(cudaMemcpyAsync(d_ordered_pointsR_DRR, h_ordered_pointsR_s, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDRR));

    //stop_timmer_host = clock();
    //time_spent = ((float)(stop_timmer_host-start_timmer_host))/CLOCKS_PER_SEC;
    //cout << "Succesfully readed the data. All set to compute the histograms in " << time_spent*1000 << " miliseconds" << endl;


    /* =======================================================================*/
    /* ====================== Starts kernel Launches  ========================*/
    /* =======================================================================*/


    //Compute the dimensions of the GPU grid
    //One thread for each node
    
    blocks_D = (int)(ceil((float)((float)(nonzero_Dnodes)/(float)(threads_perblock_dim))));
    //blocks_R = (int)(ceil((float)((float)(nonzero_Rnodes)/(float)(threads_perblock_dim))));

    dim3 threads_perblock(threads_perblock_dim,threads_perblock_dim,threads_perblock_dim);
    
    dim3 gridDDD(blocks_D,blocks_D,blocks_D);
    //dim3 gridRRR(blocks_R,blocks_R,blocks_R);
    //dim3 gridDDR(blocks_D,blocks_D,blocks_R);
    //dim3 gridDRR(blocks_R,blocks_R,blocks_D);

    //Launch the kernels
    time_spent=0; //Restarts timmer
    cudaEventRecord(start_timmer);
    make_histoXXX<<<gridDDD,threads_perblock,0,streamDDD>>>(d_DDD, d_ordered_pointsD_DDD, dnodeD_DDD, nonzero_Dnodes, bn, dmax, d_max_node);
    //make_histoXXX<<<gridRRR,threads_perblock,0,streamRRR>>>(d_RRR, d_ordered_pointsR_RRR, dnodeR_RRR, nonzero_Rnodes, bn, dmax, d_max_node);
    //make_histoXXY<<<gridDRR,threads_perblock,0,streamDRR>>>(d_DRR, d_ordered_pointsR_DRR, dnodeR_DRR, nonzero_Rnodes, d_ordered_pointsD_DRR, dnodeD_DRR, nonzero_Dnodes, bn, dmax, d_max_node);
    //make_histoXXY<<<gridDDR,threads_perblock,0,streamDDR>>>(d_DDR, d_ordered_pointsD_DDR, dnodeD_DDR, nonzero_Dnodes, d_ordered_pointsR_DDR, dnodeR_DDR, nonzero_Rnodes, bn, dmax, d_max_node);

    cucheck(cudaMemcpyAsync(DDD, d_DDD, bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost, streamDDD));
    //cucheck(cudaMemcpyAsync(RRR, d_RRR, bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost, streamRRR));
    //cucheck(cudaMemcpyAsync(DRR, d_DRR, bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost, streamDRR));
    //cucheck(cudaMemcpyAsync(DDR, d_DDR, bn*bn*bn*sizeof(double), cudaMemcpyDeviceToHost, streamDDR));

    //Waits for all the kernels to complete
    cucheck(cudaStreamSynchronize(streamDDD));
    //save_histogram(nameDDD, bn, DDD);
    //cucheck(cudaStreamSynchronize(streamRRR));
    //save_histogram(nameRRR, bn, RRR);
    //cucheck(cudaStreamSynchronize(streamDRR));
    //save_histogram(nameDRR, bn, DRR);
    //cucheck(cudaStreamSynchronize(streamDDR));
    //save_histogram(nameDDR, bn, DDR);

    cucheck(cudaEventRecord(stop_timmer));
    cucheck(cudaEventSynchronize(stop_timmer));
    cucheck(cudaEventElapsedTime(&time_spent, start_timmer, stop_timmer));

    //cout << "Spent "<< time_spent << " miliseconds to compute and save all the histograms." << endl;
    
    /* =======================================================================*/
    /* ==========================  Free memory ===============================*/
    /* =======================================================================*/

    //Free the memory

    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            delete[] hnodeD[i][j];
            //delete[] hnodeR[i][j];
        }
        delete[] hnodeD[i];
        //delete[] hnodeR[i];
    }    
    delete[] hnodeD;
    //delete[] hnodeR;

    delete[] dataD;
    //delete[] dataR;
    
    delete[] hnodeD_s;
    delete[] h_ordered_pointsD_s;
    //delete[] hnodeR_s;
    //delete[] h_ordered_pointsR_s;
    
    cucheck(cudaStreamDestroy(streamDDD));
    //cucheck(cudaStreamDestroy(streamDDR));
    //cucheck(cudaStreamDestroy(streamDRR));
    //cucheck(cudaStreamDestroy(streamRRR));

    cucheck(cudaEventDestroy(start_timmer));
    cucheck(cudaEventDestroy(stop_timmer));

    delete[] DDD;
    //delete[] RRR;
    //delete[] DRR;    
    //delete[] DDR;    
    
    cucheck(cudaFree(d_DDD));
    //cucheck(cudaFree(d_RRR));
    //cucheck(cudaFree(d_DRR));
    //cucheck(cudaFree(d_DDR));

    cucheck(cudaFree(dnodeD_DDD));
    cucheck(cudaFree(d_ordered_pointsD_DDD));
    //cucheck(cudaFree(dnodeD_DDR));
    //cucheck(cudaFree(d_ordered_pointsD_DDR));
    //cucheck(cudaFree(dnodeD_DRR));
    //cucheck(cudaFree(d_ordered_pointsD_DRR));

    //cucheck(cudaFree(dnodeR_RRR));
    //cucheck(cudaFree(d_ordered_pointsR_RRR));
    //cucheck(cudaFree(dnodeR_DDR));
    //cucheck(cudaFree(d_ordered_pointsR_DDR));
    //cucheck(cudaFree(dnodeR_DRR));
    //cucheck(cudaFree(d_ordered_pointsR_DRR));

    //cout << "Program terminated..." << endl;
    return 0;
}

