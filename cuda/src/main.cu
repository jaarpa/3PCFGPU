#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dirent.h>
#include <string.h>
#include <time.h>

#include "cucheck_macros.cuh"
#include "help.cuh"
#include "create_grid.cuh"
#include "pcf2ani.cuh"
#include "pcf2aniBPC.cuh"
#include "pcf2iso.cuh"
//#include "pcf2isoBPC.cuh"
#include "pcf3iso.cuh"
//#include "pcf3isoBPC.cuh"
//#include "pcf3ani.cuh"
//#include "pcf3aniBPC.cuh"

int main(int argc, char **argv)
{
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

    /* ================ Check for compute capability >= 6.x ==================*/
    int devId, major;
    CUCHECK(cudaGetDevice(&devId));
    for(int i = 0; i < 25; ++i) {
        CUCHECK(cudaDeviceGetAttribute(&major,
                                        cudaDevAttrComputeCapabilityMajor,
                                        devId));
    }
    printf("%i\n", major);
 
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
    int bpc = 0, analytic = 0, rand_dir = 0, rand_required = 0, pip_calculation = 0;
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
        else if (strcmp(argv[idpar],"-bpc") == 0)
            bpc = 1;
        else if (strcmp(argv[idpar],"-a") == 0)
            analytic = 1;
        else if (strcmp(argv[idpar],"-P") == 0)
            pip_calculation = 1;
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
    start_timmer_host = clock(); //To check time setting up data
    
    float size_node = 0, htime = 0, size_box = 0;
    int np = 0;
    char **histo_names = NULL;

    //Declare variables for data.
    DNode *h_nodeD = NULL, *dnodeD = NULL;
    PointW3D *dataD = NULL, *d_dataD = NULL;
    int32_t *pipsD = NULL, *d_pipsD = NULL;
    int nonzero_Dnodes = 0, pips_width = 0;
    
    //Declare variables for random.
    char **rand_files = NULL;
    PointW3D **dataR = NULL;
    PointW3D *flattened_dataR = NULL, *d_dataR = NULL;
    int32_t **pipsR = NULL;
    int32_t *flattened_pipsR = NULL, *d_pipsR = NULL;
    DNode **nodeR2_2D = NULL;
    DNode *h_nodeR = NULL, *dnodeR = NULL;
    int *nonzero_Rnodes = NULL, *acum_nonzero_Rnodes = NULL, *rnp = NULL;
    int tot_nonzero_Rnodes = 0, pips_widthR = 0, n_randfiles = 1;

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

        if (pip_calculation)
        {
            pipsR = (int32_t**)malloc(n_randfiles*sizeof(int32_t *));
            CHECKALLOC(pipsR);
            for (int i = 0; i < n_randfiles; i++)
            {
                //rnp to check that pip file has at least rnp points
                open_pip_files(&pipsR[i], &pips_widthR, rand_files[i], rnp[i]); 
                if (pips_width != pips_widthR) 
                {
                    fprintf(stderr, 
                        "PIP files have different number of columns. "
                        "%s has %i while %s has %i\n",
                        rand_files[i], pips_widthR, rand_files[i-1],
                        pips_width
                    );
                    exit(1);
                }
            }
        }
    }

    /* ========================= Set the size box ============================*/
    //Sets the size_box to the largest either the one found or the provided
    if (rand_required)
    {
        for (int i = 0; i < n_randfiles; i++)
        {
            for (int j = 0; j < np; j++)
            {
                if (dataR[i][j].x > size_box) size_box = (int)(dataR[i][j].x)+1;
                if (dataR[i][j].y > size_box) size_box = (int)(dataR[i][j].y)+1;
                if (dataR[i][j].z > size_box) size_box = (int)(dataR[i][j].z)+1;
            }
        }
    }
    for (int i = 0; i < np; i++)
    {
        if (dataD[i].x > size_box) size_box = (int)(dataD[i].x)+1;
        if (dataD[i].y > size_box) size_box = (int)(dataD[i].y)+1;
        if (dataD[i].z > size_box) size_box = (int)(dataD[i].z)+1;
    }
    if (size_box_provided < size_box)
        printf(
            "Size box set to %f according to the largest register "
            "in provided files. \n", size_box
        );
    else size_box = size_box_provided;

    //Set nodes size
    size_node = size_box/(float)(partitions);

    /* ========================== Make the nodes =============================*/
    
    //Data nodes
    nonzero_Dnodes = create_nodes(&h_nodeD, &dataD, &pipsD, pips_width, partitions, size_node, np);
    //Copy the data nodes to the device asynchronously
    
    //Random nodes gird
    if (rand_required)
    {
        nonzero_Rnodes = (int*)calloc(n_randfiles,sizeof(int));
        CHECKALLOC(nonzero_Rnodes);
        acum_nonzero_Rnodes = (int*)calloc(n_randfiles,sizeof(int));
        CHECKALLOC(acum_nonzero_Rnodes);
        nodeR2_2D =(DNode**)malloc(n_randfiles*sizeof(DNode*));
        CHECKALLOC(nodeR2_2D);

        for (int i = 0; i < n_randfiles; i++)
            nonzero_Rnodes[i] = create_nodes(&nodeR2_2D[i], &dataR[i], &pipsR[i], pips_width, partitions, size_node, np);

       //Flatten the R Nodes
        flattened_dataR = (PointW3D*)malloc(n_randfiles*np*sizeof(PointW3D));
        CHECKALLOC(flattened_dataR);
        if (pip_calculation)
        {
            flattened_pipsR = (int32_t*)malloc(n_randfiles*np*pips_width*sizeof(int32_t));
            CHECKALLOC(flattened_pipsR);
        }
        for (int i = 0; i<n_randfiles; i++)
        {
            acum_nonzero_Rnodes[i] = tot_nonzero_Rnodes;
            tot_nonzero_Rnodes += nonzero_Rnodes[i];
        }
        h_nodeR = (DNode*)malloc(tot_nonzero_Rnodes*sizeof(DNode));
        CHECKALLOC(h_nodeR);
        for (int i = 0; i < n_randfiles; i++)
        {
            memcpy(&flattened_dataR[i*np], dataR[i], np*sizeof(PointW3D));
            memcpy(&h_nodeR[acum_nonzero_Rnodes[i]], nodeR2_2D[i], nonzero_Rnodes[i]*sizeof(DNode));
            if (pip_calculation) memcpy(&flattened_pipsR[i*np*pips_width], pipsR[i], np*pips_width*sizeof(int32_t));
        }
    }
    
    //Copy to Device
    CUCHECK(cudaMalloc(&dnodeD, nonzero_Dnodes*sizeof(DNode)));
    CUCHECK(cudaMemcpy(dnodeD, h_nodeD, nonzero_Dnodes*sizeof(DNode), cudaMemcpyHostToDevice));
    CUCHECK(cudaMalloc(&d_dataD, np*sizeof(PointW3D)));
    CUCHECK(cudaMemcpy(d_dataD, dataD, np*sizeof(PointW3D), cudaMemcpyHostToDevice));
    if (pip_calculation)
    {
        CUCHECK(cudaMalloc(&d_pipsD, np*pips_width*sizeof(int32_t)));
        CUCHECK(cudaMemcpy(d_pipsD, pipsD, np*pips_width*sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    
    if (rand_required)
    {
        CUCHECK(cudaMalloc(&dnodeR, tot_nonzero_Rnodes*sizeof(DNode)));
        CUCHECK(cudaMemcpy(dnodeR, h_nodeR, tot_nonzero_Rnodes*sizeof(DNode), cudaMemcpyHostToDevice));
        CUCHECK(cudaMalloc(&d_dataR, n_randfiles*np*sizeof(PointW3D)));
        CUCHECK(cudaMemcpy(d_dataR, flattened_dataR,  n_randfiles*np*sizeof(PointW3D), cudaMemcpyHostToDevice));
        if (pip_calculation)
        {
            CUCHECK(cudaMalloc(&d_pipsR, n_randfiles*np*pips_width*sizeof(int32_t)));
            CUCHECK(cudaMemcpy(d_pipsR, flattened_pipsR, n_randfiles*np*pips_width*sizeof(int32_t), cudaMemcpyHostToDevice));
        }
    }

    stop_timmer_host = clock();
    htime = ((float)(stop_timmer_host-start_timmer_host))/CLOCKS_PER_SEC;
    printf("All set up for computations in %f ms in host. \n", htime*1000);

    /* =======================================================================*/
    /* ================== Free unused host memory ============================*/
    /* =======================================================================*/

    cudaFreeHost(dataD);
    cudaFreeHost(h_nodeD);
    if (pip_calculation) cudaFreeHost(pipsD);
    if (rand_required)
    {
        for (int i = 0; i < n_randfiles; i++)
        {
            free(rand_files[i]);
            cudaFreeHost(dataR[i]);
            cudaFreeHost(nodeR2_2D[i]);
            if (pip_calculation) cudaFreeHost(pipsR[i]);
        }
        free(rand_files);
        free(rnp);
        free(dataR);
        free(nodeR2_2D);
        if (pip_calculation) free(pipsR);
        free(flattened_dataR);
        free(h_nodeR);
        if (pip_calculation) free(flattened_pipsR);
    }
    
    free(rand_name);
    
    /* =======================================================================*/
    /* ======================= Launch the cuda code ==========================*/
    /* =======================================================================*/
    if (pip_calculation)
    {   if (strcmp(argv[1],"2ani") == 0)
            pcf_2ani_wpips(histo_names, dnodeD, d_dataD, d_pipsD, nonzero_Dnodes, dnodeR, d_dataR, d_pipsR, nonzero_Rnodes, acum_nonzero_Rnodes, pips_width, n_randfiles, bins, size_node, dmax);
        else if (strcmp(argv[1],"2iso") == 0)
            pcf_2iso_wpips(
                dnodeD, d_dataD, d_pipsD, nonzero_Dnodes,
                dnodeR, d_dataR, d_pipsR, nonzero_Rnodes,
                acum_nonzero_Rnodes, pips_width,
                histo_names, n_randfiles, bins, size_node, dmax
            );
        else if (strcmp(argv[1],"3iso")==0)
            pcf_3iso_wpips(
                dnodeD, d_dataD, d_pipsD, nonzero_Dnodes,
                dnodeR, d_dataR, d_pipsR, nonzero_Rnodes, acum_nonzero_Rnodes,
                histo_names, n_randfiles, bins, size_node, dmax, pips_width
            );
        /*
        if (strcmp(argv[1],"3iso")==0){
            if (bpc){
                if (analytic){
                    pcf_3isoBPC_analytic_wpips(data_name, dnodeD, d_ordered_pointsD, nonzero_Dnodes, bins, np, size_node, size_box, dmax);
                } else {
                    pcf_3isoBPC_wpips(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
                }
            } else {
                pcf_3iso_wpips(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, dmax);
            }
        } else if (strcmp(argv[1],"3ani")==0){
            if (bpc){
                pcf_3aniBPC_wpips(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
            } else {
                pcf_3ani_wpips(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, dmax);
            }
        } else if (strcmp(argv[1],"2iso")==0){
            if (bpc){
                if (analytic){
                    pcf_2iso_BPCanalytic_wpips(data_name, dnodeD, d_ordered_pointsD, nonzero_Dnodes, bins, np, size_node, size_box, dmax);
                } else {
                    pcf_2iso_BPC_wpips(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
                }
            } else {
                pcf_2iso_wpips(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, dmax);
            }
        } else if (strcmp(argv[1],"2ani")==0){
            if (bpc){
                pcf_2aniBPC_wpips(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
            } else {
                pcf_2ani_wpips(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, dmax);
            }
        }
        */
    }
    else
    {
        /*
        if (strcmp(argv[1],"3iso")==0){
            if (bpc){
                if (analytic){
                    pcf_3isoBPC_analytic(data_name, dnodeD, d_ordered_pointsD, nonzero_Dnodes, bins, np, size_node, size_box, dmax);
                } else {
                    pcf_3isoBPC(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
                }
            } else {
                pcf_3iso(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, dmax);
            }
        } else if (strcmp(argv[1],"3ani")==0){
            if (bpc){
                pcf_3aniBPC(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
            } else {
                pcf_3ani(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, dmax);
            }
        } else if (strcmp(argv[1],"2iso")==0){
            if (bpc){
                if (analytic){
                    pcf_2iso_BPCanalytic(data_name, dnodeD, d_ordered_pointsD, nonzero_Dnodes, bins, np, size_node, size_box, dmax);
                } else {
                    pcf_2iso_BPC(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
                }
            } else {
                pcf_2iso(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, dmax);
            }
        } else 
        */
        if (strcmp(argv[1],"2ani")==0)
        {
            if (bpc)
                pcf_2aniBPC(
                    histo_names, dnodeD, d_dataD, nonzero_Dnodes,
                    dnodeR, d_dataR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, 
                    bins, size_node, size_box, dmax
                );
            else
                pcf_2ani(
                    histo_names, dnodeD, d_dataD, nonzero_Dnodes, 
                    dnodeR, d_dataR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, 
                    bins, size_node, dmax
                );
        }
        else if (strcmp(argv[1],"2iso")==0)
        {
            if (bpc){
                /*
                if (analytic){
                    pcf_2iso_BPCanalytic(data_name, dnodeD, d_ordered_pointsD, nonzero_Dnodes, bins, np, size_node, size_box, dmax);
                } else {
                    pcf_2iso_BPC(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
                }
                */
            } else {
                pcf_2iso(
                    dnodeD, d_dataD, nonzero_Dnodes,
                    dnodeR, d_dataR, nonzero_Rnodes, acum_nonzero_Rnodes,
                    histo_names, n_randfiles, bins, size_node, dmax
                );
            }
        }
        else if (strcmp(argv[1],"3iso")==0){
            if (bpc){
                /*
                if (analytic){
                    pcf_3isoBPC_analytic_wpips(data_name, dnodeD, d_ordered_pointsD, nonzero_Dnodes, bins, np, size_node, size_box, dmax);
                } else {
                    pcf_3isoBPC_wpips(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
                }
                */
            } else {
                pcf_3iso(
                    dnodeD, d_dataD, nonzero_Dnodes,
                    dnodeR, d_dataR, nonzero_Rnodes, acum_nonzero_Rnodes,
                    histo_names, n_randfiles, bins, size_node, dmax
                );
            }
        }
    }
    

    /* =======================================================================*/
    /* ========================== Free memory ================================*/
    /* =======================================================================*/

    CUCHECK(cudaFree(dnodeD));
    CUCHECK(cudaFree(d_dataD));
    if (pip_calculation)
        CUCHECK(cudaFree(d_pipsD));
    free(data_name);

    if (rand_required)
    {
        CUCHECK(cudaFree(dnodeR));
        CUCHECK(cudaFree(d_dataR));
        CUCHECK(cudaFree(d_pipsR));
        free(nonzero_Rnodes);
        for (int i = 0; i < n_randfiles + 1; i++) free(histo_names[i]);
        free(histo_names);
        free(acum_nonzero_Rnodes);
    }

    printf("Program terminated...\n");

    return 0;
}
