/** CUDA check macro */
#define CUCHECK(call){\
    cudaError_t res = (call);\
    if(res != cudaSuccess) {\
        const char* err_str = cudaGetErrorString(res);\
        fprintf(stderr, "%s (%d): %s in %s \n", __FILE__, __LINE__, err_str, #call);\
        exit(-1);\
    }\
}\

/* Complains if it cannot allocate the array */
#define CHECKALLOC(p)  if(p == NULL) {\
    fprintf(stderr, "%s (line %d): Error - unable to allocate required memory \n", __FILE__, __LINE__);\
    exit(1);\
}\

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dirent.h>
#include <string.h>
#include <time.h>

#include "help.cuh"
#include "create_grid.cuh"
//#include "pcf2ani.cuh"
//#include "pcf2aniBPC.cuh"
//#include "pcf2iso.cuh"
//#include "pcf2isoBPC.cuh"
//#include "pcf3iso.cuh"
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

    /* =======================================================================*/
    /* ===================  Read command line args ===========================*/
    /* =======================================================================*/

    if (argc <= 6)
    {
        show_help();
        return 0;
    } 
    else if (strcmp(argv[1], "--help")==0 || strcmp(argv[1], "-h")==0)
    {
        show_help();
        return 0;
    }
    else if (strcmp(argv[1],"3iso")==0 || strcmp(argv[1],"3ani")==0 || strcmp(argv[1],"2iso")==0 || strcmp(argv[1],"2ani")==0)
    {
        //Read the parameters from command line
        int sample_size=0, bins=0, partitions=35;
        int bpc = 0, analytic=0, rand_dir = 0, rand_required=0, pip_calculation=0; //Used as bools
        float size_box_provided = 0, dmax=0;
        char *data_name=NULL, *rand_name=NULL;
        for (int idpar=2; idpar<argc; idpar++)
        {
            
            if (strcmp(argv[idpar],"-n")==0)
            {
                sample_size = atoi(argv[idpar+1]);
                idpar++;
                if (sample_size < 0)
                {
                    fprintf(stderr, "Invalid sample size (-n). Sample size must be larger than 0. \n");
                    exit(1);
                }
            }
            else if (strcmp(argv[idpar],"-f")==0)
            {
                if (strlen(argv[idpar+1])<100)
                    data_name = strdup(argv[idpar+1]);
                else
                {
                    fprintf(stderr, "String exceded the maximum length or is not null terminated. \n");
                    exit(1);
                }
                idpar++;
            }
            else if (strcmp(argv[idpar],"-r")==0 || strcmp(argv[idpar],"-rd")==0)
            {
                rand_dir = (strcmp(argv[idpar],"-rd")==0);
                if (strlen(argv[idpar+1])<100)
                    rand_name = strdup(argv[idpar+1]);
                else
                {
                    fprintf(stderr, "String exceded the maximum length or is not null terminated. \n");
                    exit(1);
                }
                idpar++;
            }
            else if (strcmp(argv[idpar],"-b")==0)
            {
                bins = atoi(argv[idpar+1]);
                idpar++;
                if (bins <= 0)
                {
                    fprintf(stderr, "Invalid number of bins (-b). The number of bins must be larger than 0. \n");
                    exit(1);
                }
            }
            else if (strcmp(argv[idpar],"-d")==0)
            {
                dmax = atof(argv[idpar+1]);
                idpar++;
                if (dmax <= 0)
                {
                    fprintf(stderr, "Invalid maximum distance (-d). The maximum distance must be larger than 0. \n");
                    exit(1);
                }
            }
            else if (strcmp(argv[idpar],"-bpc")==0) bpc = 1;
            else if (strcmp(argv[idpar],"-a")==0) analytic = 1;
            else if (strcmp(argv[idpar],"-p")==0)
            {
                partitions = atof(argv[idpar+1]);
                idpar++;
                if (partitions <= 0)
                {
                    fprintf(stderr, "Invalid number partitions (-p). The number of partitions must be larger than 0. \n");
                    exit(1);
                }
            }
            else if (strcmp(argv[idpar],"-sb")==0)
            {
                size_box_provided = atof(argv[idpar+1]);
                idpar++;
                if (size_box_provided < 0)
                {
                    fprintf(stderr, "Invalid size box (-sb). Size box must be larger than 0. \n");
                    exit(1);
                }
            }
            else if (strcmp(argv[idpar],"-P")==0) pip_calculation = 1;
        }

        //Figure out if something very necessary is missing
        if (data_name==NULL)
        {
            fprintf(stderr, "Missing data file (-f). \n");
            exit(1);
        }
        if (bins==0)
        {
            fprintf(stderr, "Missing number of --bins (-b) argument. \n");
            exit(1);
        }
        if (dmax==0)
        {
            fprintf(stderr, "Missing maximum distance (-d) argument. \n");
            exit(1);
        }
        if (!(bpc && analytic && (strcmp(argv[1],"3iso")==0 || strcmp(argv[1],"2iso")==0)))
        {
            //If it is not any of the analytic options
            rand_required=1; //Then a random file/directory is required
            if (rand_name==NULL)
            {
                fprintf(stderr, "Missing random file(s) location (-r or -rd). \n");
                exit(1);
            }
        }


        //Everything should be set to read the data and call the funtions to make and save the histograms.
        
        /* =======================================================================*/
        /* ========================= Declare variables ===========================*/
        /* =======================================================================*/

        clock_t stop_timmer_host, start_timmer_host;
        start_timmer_host = clock(); //To check time setting up data
        
        float size_node, htime, size_box=0;
        int np = 0, minimum_number_lines;
        char **histo_names = NULL;

        //Declare variables for data.
        PointW3D *dataD = NULL, *d_dataD = NULL;
        int32_t *pipsD = NULL, *dpipsD = NULL;
        DNode *hnodeD_s = NULL, *dnodeD = NULL;
        int nonzero_Dnodes, n_pips=0;
        
        //Declare variables for random.
        char **rand_files = NULL;
        PointW3D **dataR = NULL, *flattened_dataR = NULL, *d_dataR = NULL;
        int32_t **pipsR = NULL, *flattened_pipsR = NULL, *dpipsR = NULL;
        DNode **hnodeR_s = NULL, *flattened_hnodeR_s = NULL, *dnodeR = NULL;
        int *nonzero_Rnodes = NULL, *acum_nonzero_Rnodes = NULL, tot_nonzero_Rnodes=0, rnp=0, n_randfiles=1, n_pipsR=0;

        /* =======================================================================*/
        /* ================== Assign and prepare variables =======================*/
        /* =======================================================================*/

        //Read data
        open_files(data_name, &dataD, &np, &size_box);
        
        //Read rand only if rand was required.
        if (rand_required)
        {

            //Check if a directory of random files was provided to change n_randfiles
            //Instead of rand name should be an array with the name of each rand array or something like that.
            if (rand_dir)
            {
                char *directory_path = (char*)malloc((9+strlen(rand_name))*sizeof(char));
                CHECKALLOC(directory_path);
                char data_path[] = "../data/";
                strcpy(directory_path, data_path);
                strcat(directory_path, rand_name); //Set up the full path
                DIR *folder = opendir(directory_path);

                if(folder != NULL)
                {
                    n_randfiles = 0;
                    struct dirent *archivo;
                    while( (archivo=readdir(folder)) )
                        if (strcmp(archivo->d_name,".") != 0 && strcmp(archivo->d_name,"..") != 0)
                            if (strcmp(&(archivo->d_name)[strlen((archivo->d_name))-4],".pip") != 0) n_randfiles++;

                    if (!n_randfiles)
                    {
                        fprintf(stderr, "There are no suitable files in %s \n", directory_path);
                        exit(1);
                    }

                    //Reset the folder stream to actually read the files
                    closedir(folder);
                    folder = opendir(directory_path);

                    rand_files = (char **)malloc(n_randfiles * sizeof(char *));
                    CHECKALLOC(rand_files);
                    histo_names = (char **)malloc((1 + n_randfiles) * sizeof(char *));
                    CHECKALLOC(histo_names);
                    histo_names[0] = strdup(data_name);

                    int j = 0;
                    char *nombre_archivo;
                    while( (archivo=readdir(folder)) )
                    {
                        nombre_archivo = archivo->d_name;
                        if (strcmp(nombre_archivo,".") == 0 || strcmp(nombre_archivo,"..") == 0) continue;
                        if (strcmp(&(nombre_archivo)[strlen((nombre_archivo))-4],".pip") == 0) continue;
                        histo_names[j+1] = strdup(nombre_archivo);

                        rand_files[j] = (char*)malloc((strlen(rand_name)+strlen(nombre_archivo)+1)*sizeof(char));
                        CHECKALLOC(rand_files[j]);
                        strcpy(rand_files[j], rand_name);
                        strcat(rand_files[j], nombre_archivo); //Set up the full path

                        j++;
                    }
                    closedir(folder);
                    
                }
                else
                {
                    fprintf(stderr, "Unable to open directory %s \n", directory_path);
                    exit(1);
                }
                free(directory_path);
            }
            else
            {
                rand_files = (char**)malloc(n_randfiles * sizeof(char *));
                CHECKALLOC(rand_files);
                rand_files[0] = strdup(rand_name);
                histo_names = (char**)malloc((1 + n_randfiles) * sizeof(char *));
                CHECKALLOC(histo_names);
                histo_names[0] = strdup(data_name);
                histo_names[1] = strdup(rand_name);
            }

            dataR = (PointW3D**)calloc(n_randfiles, sizeof(PointW3D *));
            CHECKALLOC(dataR);
            if (pip_calculation)
            {
                pipsR = (int32_t**)calloc(n_randfiles, sizeof(int32_t *));
                CHECKALLOC(pipsR);
            }

            rnp = get_smallest_file(rand_files, n_randfiles); // Get the number of lines in the file with less entries
            minimum_number_lines = rnp<np ? rnp : np;
            if (sample_size == 0 || sample_size>minimum_number_lines)
            {
                sample_size = minimum_number_lines;
                printf("Sample size set to %i according to the file with the least amount of entries \n", sample_size);
            }

            for (int i=0; i<n_randfiles; i++)
            {
                open_files(rand_files[i], &dataR[i],&rnp, &size_box);

                //Read pips files of random data if required
                if (pip_calculation)
                {
                    open_pip_files(&pipsR[i], rand_files[i], rnp, &n_pipsR); //rnp is used to check that the pip file has at least the same number of points as the data file
                    
                    if (i==0) n_pips = n_pipsR; //It has nothing to compare against in the first reading
                    if (n_pips != n_pipsR) 
                    {
                        fprintf(stderr, "PIP files have different number of columns. %s has %i while %s has %i\n", rand_files[i], n_pipsR, rand_files[i-1], n_pips);
                        exit(1);
                    }

                    //Takes a sample if sample_size != rnp is less than np
                    if (rnp > sample_size) random_sample_wpips(&dataR[i], &pipsR[i], rnp, n_pipsR, sample_size);
                } 
                else if (rnp > sample_size) random_sample(&dataR[i], rnp, sample_size);
            }
        }

        //Sets the size_box to the larges either the one found or the provided
        if (size_box_provided < size_box) printf("Size box set to %f according to the largest register in provided files. \n", size_box);
        else size_box = size_box_provided;
        
        //Read PIPs if required
        if (pip_calculation)
        {
            open_pip_files(&pipsD, data_name, np, &n_pips);
            if (n_pips != n_pipsR && rand_required)
            {
                fprintf(stderr, "Length of data PIPs and random PIPs are not the same. \n Data pips has %i columns but the random pip has %i columns. \n", n_pips, n_pipsR);
                exit(1);
            }
        }

        //Take a random sample from data
        if (sample_size > np) printf("Sample size set to %i according to the file with the least amount of entries \n", np);
        if (sample_size != 0 && sample_size < np)
        {
            if (pip_calculation) random_sample_wpips(&dataD, &pipsD, rnp, n_pips, sample_size);
            else random_sample(&dataD, rnp, sample_size);
            np = sample_size;
        }

        //Set nodes size
        size_node = size_box/(float)(partitions);

        //Make the nodes
        if (pip_calculation)
            nonzero_Dnodes = create_nodes_wpips(&hnodeD_s, &dataD, &pipsD, n_pips, partitions, size_node, np);
        else
            nonzero_Dnodes = create_nodes(&hnodeD_s, &dataD, partitions, size_node, np);

        if (rand_required)
        {
            nonzero_Rnodes = (int*)calloc(n_randfiles,sizeof(int));
            CHECKALLOC(nonzero_Rnodes);
            acum_nonzero_Rnodes = (int*)calloc(n_randfiles,sizeof(int));
            CHECKALLOC(acum_nonzero_Rnodes);
            hnodeR_s =(DNode**)malloc(n_randfiles*sizeof(DNode*));
            CHECKALLOC(hnodeR_s);

            if (pip_calculation)
                for (int i = 0; i < n_randfiles; i++)
                    nonzero_Rnodes[i] = create_nodes_wpips(&hnodeR_s[i], &dataR[i], &pipsR[i], n_pips, partitions, size_node, np);
            else
                for (int i = 0; i < n_randfiles; i++)
                    nonzero_Rnodes[i] = create_nodes(&hnodeR_s[i], &dataR[i], partitions, size_node, np);
        }

        //Flatten the R Nodes
        if (rand_required)
        {
            flattened_dataR = (PointW3D*)malloc(n_randfiles*np*sizeof(PointW3D));
            CHECKALLOC(flattened_dataR);
            if (pip_calculation)
            {
                flattened_pipsR = (int32_t*)malloc(n_randfiles*np*n_pips*sizeof(int32_t));
                CHECKALLOC(flattened_pipsR);
            }
            for (int i = 0; i<n_randfiles; i++)
            {
                acum_nonzero_Rnodes[i] = tot_nonzero_Rnodes;
                tot_nonzero_Rnodes += nonzero_Rnodes[i];
            }
            flattened_hnodeR_s = (DNode*)malloc(tot_nonzero_Rnodes*sizeof(DNode));
            CHECKALLOC(flattened_hnodeR_s);
            for (int i = 0; i < n_randfiles; i++)
            {
                memcpy(&flattened_dataR[i*np], dataR[i], np*sizeof(PointW3D));
                memcpy(&flattened_hnodeR_s[acum_nonzero_Rnodes[i]], hnodeR_s[i], nonzero_Rnodes[i]*sizeof(DNode));
                if (pip_calculation) memcpy(&flattened_pipsR[i*np*n_pips], pipsR[i], np*n_pips*sizeof(int32_t));
            }
        }
        
        //Copy to Device
        CUCHECK(cudaMalloc(&dnodeD, nonzero_Dnodes*sizeof(DNode)));
        CUCHECK(cudaMemcpy(dnodeD, hnodeD_s, nonzero_Dnodes*sizeof(DNode), cudaMemcpyHostToDevice));
        CUCHECK(cudaMalloc(&d_dataD, np*sizeof(PointW3D)));
        CUCHECK(cudaMemcpy(d_dataD, hnodeD_s, nonzero_Dnodes*sizeof(PointW3D), cudaMemcpyHostToDevice));
        if (pip_calculation)
        {
            CUCHECK(cudaMalloc(&dpipsD, np*n_pips*sizeof(int32_t)));
            CUCHECK(cudaMemcpy(dpipsD, pipsD, np*n_pips*sizeof(int32_t), cudaMemcpyHostToDevice));
        }
        
        if (rand_required)
        {
            CUCHECK(cudaMalloc(&dnodeR, tot_nonzero_Rnodes*sizeof(DNode)));
            CUCHECK(cudaMemcpy(dnodeR, flattened_hnodeR_s, tot_nonzero_Rnodes*sizeof(DNode), cudaMemcpyHostToDevice));
            CUCHECK(cudaMalloc(&d_dataR, n_randfiles*np*sizeof(PointW3D)));
            CUCHECK(cudaMemcpy(d_dataR, flattened_dataR,  n_randfiles*np*sizeof(PointW3D), cudaMemcpyHostToDevice));
            if (pip_calculation)
            {
                CUCHECK(cudaMalloc(&dpipsR, n_randfiles*np*n_pips*sizeof(int32_t)));
                CUCHECK(cudaMemcpy(dpipsR, flattened_pipsR, n_randfiles*np*n_pips*sizeof(int32_t), cudaMemcpyHostToDevice));
            }
        }
 
        stop_timmer_host = clock();
        htime = ((float)(stop_timmer_host-start_timmer_host))/CLOCKS_PER_SEC;
        printf("All set up for computations in %f ms in host. \n", htime*1000);       

        /* =======================================================================*/
        /* ================== Free unused host memory ============================*/
        /* =======================================================================*/

        free(dataD);
        free(hnodeD_s);
        if (pip_calculation) free(pipsD);
        if (rand_required)
        {
            for (int i = 0; i < n_randfiles; i++)
            {
                free(rand_files[i]);
                free(dataR[i]);
                free(hnodeR_s[i]);
                if (pip_calculation) free(pipsR[i]);
            }
            free(rand_files);
            free(dataR);
            free(hnodeR_s);
            if (pip_calculation) free(pipsR);
            free(flattened_dataR);
            free(flattened_hnodeR_s);
            if (pip_calculation) free(flattened_pipsR);
        }
        
        free(rand_name);
        
        /* =======================================================================*/
        /* ======================= Launch the cuda code ==========================*/
        /* =======================================================================*/
        
        /*
        if (strcmp(argv[1],"3iso")==0){
            if (bpc){
                if (analytic){
                    pcf_3isoBPC_analytic(data_name, dnodeD, d_ordered_pointsD, nonzero_Dnodes, bn, np, size_node, size_box, dmax);
                } else {
                    pcf_3isoBPC(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bn, size_node, size_box, dmax);
                }
            } else {
                pcf_3iso(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bn, size_node, dmax);
            }
        } else if (strcmp(argv[1],"3ani")==0){
            if (bpc){
                pcf_3aniBPC(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bn, size_node, size_box, dmax);
            } else {
                pcf_3ani(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bn, size_node, dmax);
            }
        } else if (strcmp(argv[1],"2iso")==0){
            if (bpc){
                if (analytic){
                    pcf_2iso_BPCanalytic(data_name, dnodeD, d_ordered_pointsD, nonzero_Dnodes, bn, np, size_node, size_box, dmax);
                } else {
                    pcf_2iso_BPC(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bn, size_node, size_box, dmax);
                }
            } else {
                pcf_2iso(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bn, size_node, dmax);
            }
        } else if (strcmp(argv[1],"2ani")==0){
            if (bpc){
                pcf_2aniBPC(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bn, size_node, size_box, dmax);
            } else {
                pcf_2ani(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bn, size_node, dmax);
            }
        }
        */

        /* =======================================================================*/
        /* ========================== Free memory ================================*/
        /* =======================================================================*/

        CUCHECK(cudaFree(dnodeD));
        CUCHECK(cudaFree(d_dataD));
        CUCHECK(cudaFree(dpipsD));
        CUCHECK(cudaFree(dnodeR));
        CUCHECK(cudaFree(d_dataR));
        CUCHECK(cudaFree(dpipsR));

        free(data_name);

        if (rand_required)
        {
            free(nonzero_Rnodes);
            for (int i = 0; i < n_randfiles + 1; i++) free(histo_names[i]);
            free(histo_names);
        }

        printf("Program terminated...\n");

    }
    else
    {
        fprintf(stderr, "Invalid <calc_type> option or not enough parameters. \nSee --help for more information. \n");
        exit(1);
    }
    return 0;
}
