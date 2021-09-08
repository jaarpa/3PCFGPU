#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "cucheck_macros.cuh"
#include "help.cuh"
#include "create_grid.cuh"
//#include "pcf3iso.cuh"
//#include "pcf3ani.cuh"
#include "pcf2ani.cuh"
#include "pcf2iso.cuh"
//#include "pcf3aniBPC.cuh"
//#include "pcf3isoBPC.cuh"
#include "pcf2aniBPC.cuh"
#include "pcf2isoBPC.cuh"

/*
Main function to calculate the correlation function of 2 and 3 points either isotropic or anisotropic. This is the master
script which calls the correct function, and reads and settles the data. The file must contain 4 columns
the first 3 columns in file are the x,y,z coordinates and the 4 the weight of the measurment. Even if pips calculation (-P)
will be performed the data file (-f) and random files (-r or -rd) must have 4 columns separated by spaces.

The pip files must have a list of int32 integers every integer in row must be separated by spaces and it must have at least the
same number of rows as its data counterpart.

A random sample of the data can be obtained specifying the argument (-n) every file must have at least -n rows, if that is not the case
or -n is not specified the sample size is set to the number of rows of the file with less rows.
*/
int main(int argc, char **argv)
{

    /* ================ Check for compute capability >= 6.x ==================*/
    int devId, major;
    CUCHECK(cudaGetDevice(&devId));
    CUCHECK(cudaDeviceGetAttribute(&major,
                                    cudaDevAttrComputeCapabilityMajor,
                                    devId));
    if (major < 6)
    {
        fprintf(stderr, "Compute major capability > 6 is required. Yours %i\n", major);
        exit(1);
    }

    /* ===================  Read command line args ===========================*/
    if (argc <= 6 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)
    {
        show_help();
        return 0;
    }
    else if (strcmp(argv[1],"3iso") != 0 && strcmp(argv[1],"3ani") != 0 && strcmp(argv[1],"2iso") != 0 && strcmp(argv[1],"2ani") != 0)
    {
        fprintf(stderr, 
            "Invalid <calc_type> option or not enough parameters. \n"
            "See --help for more information. \n"
        );
        exit(1);
    }

    int bins = 0, partitions = 35;
    //Used as bools
    int bpc = 0, analytic = 0, rand_dir = 0, rand_required = 0;
    int IIP = 0, pip_calculation = 0;
    float size_box_provided = 0, dmax = 0;
    char *data_name = NULL, *rand_name = NULL;

    //Read the parameters from command line
    for (int idpar = 2; idpar < argc; idpar++)
    {
        if (strcmp(argv[idpar],"-f") == 0)
        {
            if (strlen(argv[idpar+1]) < 100)
                data_name = strdup(argv[idpar+1]);
            else
            {
                fprintf(stderr, 
                    "String exceded the maximum length or "
                    "is not null terminated. \n");
                exit(1);
            }
            idpar++;
        }
        else if (strcmp(argv[idpar],"-r") == 0 || strcmp(argv[idpar],"-rd") == 0)
        {
            rand_dir = (strcmp(argv[idpar],"-rd") == 0);
            if (strlen(argv[idpar+1]) < 100)
                rand_name = strdup(argv[idpar+1]);
            else
            {
                fprintf(stderr, 
                    "String exceded the maximum length or "
                    "is not null terminated. \n"
                );
                exit(1);
            }
            idpar++;
        }
        else if (strcmp(argv[idpar],"-b") == 0)
        {
            bins = atoi(argv[idpar+1]);
            idpar++;
            if (bins <= 0)
            {
                fprintf(stderr, 
                    "Invalid number of bins (-b). "
                    "The number of bins must be larger than 0. \n"
                );
                exit(1);
            }
        }
        else if (strcmp(argv[idpar],"-d") == 0)
        {
            dmax = atof(argv[idpar+1]);
            idpar++;
            if (dmax <= 0)
            {
                fprintf(stderr, 
                    "Invalid maximum distance (-d). \
                    The maximum distance must be larger than 0. \n"
                );
                exit(1);
            }
        }
        else if (strcmp(argv[idpar],"-p") == 0)
        {
            partitions = atof(argv[idpar+1]);
            idpar++;
            if (partitions <= 0)
            {
                fprintf(stderr, 
                    "Invalid number partitions (-p). "
                    "The number of partitions must be larger than 0. \n"
                );
                exit(1);
            }
        }
        else if (strcmp(argv[idpar],"-sb") == 0)
        {
            size_box_provided = atof(argv[idpar+1]);
            idpar++;
            if (size_box_provided < 0)
            {
                fprintf(stderr, 
                    "Invalid size box (-sb). "
                    "Size box must be larger than 0. \n"
                );
                exit(1);
            }
        }
        else if (strcmp(argv[idpar],"-IIP") == 0)
        {
            pip_calculation = 1;
            IIP = 1;
        }
        else if (strcmp(argv[idpar],"-PIP") == 0)
            pip_calculation = 1;
        else if (strcmp(argv[idpar],"-bpc") == 0)
            bpc = 1;
        else if (strcmp(argv[idpar],"-a") == 0)
            analytic = 1;
    }

    //Figure out if something very necessary is missing
    if (data_name == NULL)
    {
        fprintf(stderr, "Missing data file (-f). \n");
        exit(1);
    }
    if (bins == 0)
    {
        fprintf(stderr, "Missing number of --bins (-b) argument. \n");
        exit(1);
    }
    if (dmax == 0)
    {
        fprintf(stderr, "Missing maximum distance (-d) argument. \n");
        exit(1);
    }
    if (!(bpc && analytic && (strcmp(argv[1],"3iso") == 0 || strcmp(argv[1],"2iso") == 0)))
    {
        //If it is not any of the analytic options
        rand_required = 1; //Then a random file/directory is required
        if (rand_name == NULL)
        {
            fprintf(stderr, "Missing random file(s) location (-r or -rd). \n");
            exit(1);
        }
    }

    //Everything should be set to read the data.
    
    /* =======================================================================*/
    /* ========================= Declare variables ===========================*/
    /* =======================================================================*/

    clock_t stop_timmer_host, start_timmer_host;
    cudaEvent_t DDcopy_done, *RRcopy_done;

    
    // In 3 PCF DD is used for DDD, RR for RRR
    // New streams are created for mixed histograms within functions
    cudaStream_t streamDD, *streamRR;
    start_timmer_host = clock(); //To check time setting up data
    
    float size_node = 0, htime = 0, size_box = 0;
    float max_x = 0, max_y = 0, max_z = 0;
    float min_x = 0, min_y = 0, min_z = 0;
    int np = 0;
    char **histo_names = NULL;

    //Declare variables for data.
    DNode *h_nodeD = NULL, *d_nodeD = NULL;
    PointW3D *dataD = NULL, *d_dataD = NULL;
    int32_t *pipsD = NULL, *d_pipsD = NULL;
    int nonzero_Dnodes = 0, pips_width = 0;
    
    //Declare variables for random.
    char **rand_files = NULL;
    PointW3D **dataR = NULL, **d_dataR=NULL;
    DNode **h_nodeR = NULL, **d_nodeR = NULL;
    int *rnp = NULL, *nonzero_Rnodes = NULL;
    int n_randfiles = 1;

    /* =======================================================================*/
    /* ================== Assign and prepare variables =======================*/
    /* =======================================================================*/

    /* ============================ Read data ================================*/
    open_files(&dataD, &np, data_name);
    //Read PIPs if required
    if (pip_calculation)
        open_pip_files(&pipsD, &pips_width, data_name, np);
    
    //Read rand only if rand was required.
    if (rand_required)
    {

        //read_random_files
        read_random_files(
            &rand_files, &histo_names, &rnp, &dataR, &n_randfiles,
            rand_name, rand_dir
        );
        histo_names[0] = strdup(data_name);
    }

    /* ========================= Set the size box ============================*/
    //Sets the size_box to the largest either the one found or the provided
    if (rand_required)
    {
        for (int i = 0; i < n_randfiles; i++)
        {
            for (int j = 0; j < rnp[i]; j++)
            {
                if (dataR[i][j].x > max_x) max_x = dataR[i][j].x;
                if (dataR[i][j].y > max_y) max_y = dataR[i][j].y;
                if (dataR[i][j].z > max_z) max_z = dataR[i][j].z;

                if (dataR[i][j].x < min_x) min_x = dataR[i][j].x;
                if (dataR[i][j].y < min_y) min_y = dataR[i][j].y;
                if (dataR[i][j].z < min_z) min_z = dataR[i][j].z;
            }
        }
    }
    for (int i = 0; i < np; i++)
    {
        if (dataD[i].x > max_x) max_x = dataD[i].x;
        if (dataD[i].y > max_y) max_y = dataD[i].y;
        if (dataD[i].z > max_z) max_z = dataD[i].z;

        if (dataD[i].x < min_x) min_x = dataD[i].x;
        if (dataD[i].y < min_y) min_y = dataD[i].y;
        if (dataD[i].z < min_z) min_z = dataD[i].z;

    }
    
    if (min_x < 0 || min_y < 0 || min_y < 0)
    {
        
        // Get the largest coordinate and shifts it to the I quadrant
        max_x = max_x - min_x;
        max_y = max_y - min_y;
        max_z = max_z - min_z;
        
        // Shift every point to the I quadrant to have only positive values
        if (rand_required)
        {
            for (int i = 0; i < n_randfiles; i++)
            {
                for (int j = 0; j < rnp[i]; j++)
                {
                    dataR[i][j].x = dataR[i][j].x - min_x;
                    dataR[i][j].y = dataR[i][j].y - min_y;
                    dataR[i][j].z = dataR[i][j].z - min_z;
                }
            }
        }
        for (int i = 0; i < np; i++)
        {
            dataD[i].x = dataD[i].x - min_x;
            dataD[i].y = dataD[i].y - min_y;
            dataD[i].z = dataD[i].z - min_z;
        }

    }

    if (max_x > max_y && max_x > max_z)
        size_box = max_x;
    else if (max_y > max_z)
        size_box = max_y;
    else
        size_box = max_z;

    if (size_box_provided < size_box)
    {
        size_box = ceilf(size_box) + 1;
        printf(
            "Size box set to %f according to the largest register "
            "in provided files. \n", size_box
        );
    }
    else
    {
        size_box = size_box_provided;
    }

    /*
    Would be nice to query the device here to compute the best number of 
    partitions
    */

    //Set nodes size
    size_node = size_box/(float)(partitions);

    /* ================ Make the nodes and copy to device ===================*/

    // Create the streams for async copy
    streamRR = (cudaStream_t*)malloc(n_randfiles*sizeof(cudaStream_t));
    CHECKALLOC(streamRR);
    CUCHECK(cudaStreamCreate(&streamDD));
    for (int i = 0; i < n_randfiles; i++)
        CUCHECK(cudaStreamCreate(&streamRR[i]));
    
    //To watch if the copies are done before attempting to read
    CUCHECK(cudaEventCreate(&DDcopy_done));
    RRcopy_done = (cudaEvent_t*)malloc(n_randfiles*sizeof(cudaEvent_t));
    for (int i = 0; i < n_randfiles; i++)
        CUCHECK(cudaEventCreate(&RRcopy_done[i]));
    
    //Data nodes
    nonzero_Dnodes = create_nodes(&h_nodeD, &dataD, &pipsD, pips_width, partitions, size_node, np);
    if (IIP)
    {
        pip_calculation = 0;
        CUCHECK(cudaFreeHost(pipsD));
        pipsD = NULL;
    }

    //Copy the data nodes to the device asynchronously
    CUCHECK(cudaMalloc(&d_nodeD, nonzero_Dnodes*sizeof(DNode)));
    CUCHECK(cudaMalloc(&d_dataD,  np*sizeof(PointW3D)));
    if (pipsD != NULL)
    {
        CUCHECK(cudaMalloc(&d_pipsD,  np*pips_width*sizeof(int32_t)));
        //This potentially could become async
        CUCHECK(cudaMemcpyAsync(d_pipsD, pipsD, np*pips_width*sizeof(int32_t), cudaMemcpyHostToDevice, streamDD));
    }
    CUCHECK(cudaMemcpyAsync(d_nodeD, h_nodeD, nonzero_Dnodes*sizeof(DNode), cudaMemcpyHostToDevice, streamDD));
    CUCHECK(cudaMemcpyAsync(d_dataD, dataD, np*sizeof(PointW3D), cudaMemcpyHostToDevice, streamDD));
    CUCHECK(cudaEventRecord(DDcopy_done, streamDD));
    
    //Random nodes grid
    if (rand_required)
    {
        nonzero_Rnodes = (int*)calloc(n_randfiles,sizeof(int));
        CHECKALLOC(nonzero_Rnodes);
        h_nodeR =(DNode**)malloc(n_randfiles*sizeof(DNode*));
        CHECKALLOC(h_nodeR);
        d_nodeR =(DNode**)malloc(n_randfiles*sizeof(DNode*));
        CHECKALLOC(d_nodeR);
        d_dataR = (PointW3D**)malloc(n_randfiles*sizeof(PointW3D*));
        CHECKALLOC(d_dataR);

        for (int i = 0; i < n_randfiles; i++)
        {
            nonzero_Rnodes[i] = create_nodes(&h_nodeR[i], &dataR[i], NULL, pips_width, partitions, size_node, rnp[i]);
            CUCHECK(cudaMalloc(&d_nodeR[i], nonzero_Rnodes[i]*sizeof(DNode)));
            CUCHECK(cudaMalloc(&d_dataR[i],  rnp[i]*sizeof(PointW3D)));

            CUCHECK(cudaMemcpyAsync(d_nodeR[i], h_nodeR[i], nonzero_Rnodes[i]*sizeof(DNode), cudaMemcpyHostToDevice, streamRR[i]));
            CUCHECK(cudaMemcpyAsync(d_dataR[i], dataR[i], rnp[i]*sizeof(PointW3D), cudaMemcpyHostToDevice, streamRR[i]));

            CUCHECK(cudaEventRecord(RRcopy_done[i], streamRR[i]));

        }
    }

    stop_timmer_host = clock();
    htime = ((float)(stop_timmer_host-start_timmer_host))/CLOCKS_PER_SEC;
    printf("All set up for computations in %f ms in host. \n", htime*1000);
    
    /* =======================================================================*/
    /* ======================= Launch the cuda code ==========================*/
    /* =======================================================================*/

    /*
    if (strcmp(argv[1],"3iso")==0){
        if (bpc){
            if (analytic){
                pcf_3isoBPC_analytic(data_name, d_nodeD, d_ordered_pointsD, nonzero_Dnodes, bins, np, size_node, size_box, dmax);
            } else {
                pcf_3isoBPC(histo_names, d_nodeD, d_ordered_pointsD, nonzero_Dnodes, d_nodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
            }
        } else {
            pcf_3iso(histo_names, d_nodeD, d_ordered_pointsD, nonzero_Dnodes, d_nodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, dmax);
        }
    } else if (strcmp(argv[1],"3ani")==0){
        if (bpc){
            pcf_3aniBPC(histo_names, d_nodeD, d_ordered_pointsD, nonzero_Dnodes, d_nodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
        } else {
            pcf_3ani(histo_names, d_nodeD, d_ordered_pointsD, nonzero_Dnodes, d_nodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, dmax);
        }
    */
    if (strcmp(argv[1],"2ani")==0)
    {
        if (bpc)
        {
            pcf_2aniBPC(
                d_nodeD, d_dataD,
                nonzero_Dnodes, streamDD, DDcopy_done, 
                d_nodeR, d_dataR,
                nonzero_Rnodes, streamRR, RRcopy_done,
                histo_names, n_randfiles, bins, size_node, dmax,
                size_box
            );
        }
        else
        {
            pcf_2ani(
                d_nodeD, d_dataD, d_pipsD, nonzero_Dnodes, streamDD, DDcopy_done,
                d_nodeR, d_dataR, nonzero_Rnodes, streamRR, RRcopy_done,
                histo_names, n_randfiles, bins, size_node, dmax,
                pips_width
            );
        }
    }
    else if (strcmp(argv[1],"2iso")==0)
    {
        if (bpc)
        {
            if (analytic){
                pcf_2iso_BPCanalytic(
                    d_nodeD, d_dataD,
                    nonzero_Dnodes, streamDD,
                    bins, np, size_node, size_box, dmax,
                    data_name
                );
            }
            else
            {
                pcf_2iso_BPC(
                    d_nodeD, d_dataD,
                    nonzero_Dnodes, streamDD, DDcopy_done, 
                    d_nodeR, d_dataR,
                    nonzero_Rnodes, streamRR, RRcopy_done,
                    histo_names, n_randfiles, bins, size_node, dmax,
                    size_box
                );
            }
        }
        else
        {
            pcf_2iso(
                d_nodeD, d_dataD, d_pipsD, nonzero_Dnodes, streamDD, DDcopy_done,
                d_nodeR, d_dataR, nonzero_Rnodes, streamRR, RRcopy_done,
                histo_names, n_randfiles, bins, size_node, dmax,
                pips_width
            );
        }
    }
    else if (strcmp(argv[1],"3iso")==0)
    {
        if (bpc)
        {
            // if (analytic)
            // {
            //     pcf_3isoBPC_analytic(data_name, d_nodeD, d_ordered_pointsD, nonzero_Dnodes, bins, np, size_node, size_box, dmax);
            // }
            // else
            // {
            //     pcf_3isoBPC(histo_names, d_nodeD, d_ordered_pointsD, nonzero_Dnodes, d_nodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
            // }
        }
        /*
        else
        {
            pcf_3iso(
                d_nodeD, d_dataD, d_pipsD, nonzero_Dnodes, streamDD, DDcopy_done,
                d_nodeR, d_dataR, d_pipsR, nonzero_Rnodes, streamRR, RRcopy_done,
                histo_names, n_randfiles, bins, size_node, dmax,
                pips_width
            );
        }
        */
    }

    /* =======================================================================*/
    /* ========================== Free memory ================================*/
    /* =======================================================================*/

    CUCHECK(cudaEventDestroy(DDcopy_done));
    for (int i = 0; i < n_randfiles; i++)
        CUCHECK(cudaEventDestroy(RRcopy_done[i]));
    free(RRcopy_done);

    free(data_name);
    CUCHECK(cudaFreeHost(dataD));
    CUCHECK(cudaFreeHost(h_nodeD));

    CUCHECK(cudaFree(d_nodeD));
    CUCHECK(cudaFree(d_dataD));

    if (pip_calculation)
    {
        CUCHECK(cudaFreeHost(pipsD));
        CUCHECK(cudaFree(d_pipsD));
    }

    free(rand_name);
    if (rand_required)
    {
        for (int i = 0; i < n_randfiles; i++)
        {
            free(rand_files[i]);
            CUCHECK(cudaFreeHost(dataR[i]));
            CUCHECK(cudaFreeHost(h_nodeR[i]));
            CUCHECK(cudaFree(d_nodeR[i]));
            CUCHECK(cudaFree(d_dataR[i]));
        }
        free(rand_files);
        free(dataR);
        free(rnp);
        free(h_nodeR);

        free(d_nodeR);
        free(d_dataR);
        free(nonzero_Rnodes);
        for (int i = 0; i < n_randfiles + 1; i++)
            free(histo_names[i]);
        free(histo_names);
    }

    printf("Program terminated...\n");

    return 0;
}
