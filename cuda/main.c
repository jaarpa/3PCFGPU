/* CUDA check macro */
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

#include "help.h"
#include "create_grid.h"
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
        int npips=0, np = 0, minimum_number_lines;
        char **histo_names;

        //Declare variables for data.
        PointW3D *dataD, *h_ordered_pointsD, *d_ordered_pointsD;
        int32_t *pipsD;
        DNode *hnodeD_s, *dnodeD;
        Node ***hnodeD;
        int nonzero_Dnodes=0, k_element=0, idxD=0, last_pointD = 0, n_pips=0;
        
        //Declare variables for random.
        PointW3D **dataR, *h_ordered_pointsR_s, *d_ordered_pointsR;
        int32_t **pipsR;
        DNode *hnodeR_s, *dnodeR;
        Node ****hnodeR;
        int *nonzero_Rnodes, *acum_nonzero_Rnodes, *idxR, *last_pointR, rnp=0, n_randfiles=1, tot_randnodes=0, n_pipsR=0;
        char **rand_files;

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
                char *directory_path=malloc((9+strlen(rand_name))*sizeof(char));
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

                    rand_files = malloc(n_randfiles * sizeof(char *));
                    CHECKALLOC(rand_files);
                    histo_names = malloc((1 + n_randfiles) * sizeof(char *));
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

                        rand_files[j] = malloc((strlen(rand_name)+strlen(nombre_archivo)+1)*sizeof(char));
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
                rand_files = malloc(n_randfiles * sizeof(char *));
                CHECKALLOC(rand_files);
                rand_files[0] = strdup(rand_name);
                histo_names = malloc((1 + n_randfiles) * sizeof(char *));
                CHECKALLOC(histo_names);
                histo_names[0] = strdup(data_name);
                histo_names[1] = strdup(rand_name);
            }

            dataR = calloc(n_randfiles, sizeof(PointW3D *));
            nonzero_Rnodes = calloc(n_randfiles,sizeof(int));
            idxR = calloc(n_randfiles,sizeof(int));
            last_pointR = calloc(n_randfiles,sizeof(int));
            acum_nonzero_Rnodes = calloc(n_randfiles,sizeof(int));
            if (pip_calculation) pipsR = calloc(n_randfiles, sizeof(int32_t *));

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

        //Make nodes
        if (rand_required){
            hnodeR = calloc(n_randfiles, sizeof(Node ***));
            CHECKALLOC(hnodeR);
            for (int i=0; i<n_randfiles; i++){
                hnodeR[i] = calloc(partitions, sizeof(Node **));
                CHECKALLOC(hnodeR[i]);
                for (int j=0; j<partitions; j++){
                    hnodeR[i][j] = calloc(partitions, sizeof(Node *));
                    CHECKALLOC(hnodeR[i][j]);
                    for (int k=0; k<partitions; k++){
                        hnodeR[i][j][k] = calloc(partitions, sizeof(Node));
                        CHECKALLOC(hnodeR[i][j][k]);
                    }
                }
            }
        }
        
        hnodeD = calloc(partitions, sizeof(Node **));
        CHECKALLOC(hnodeD);
        for (int i=0; i<partitions; i++){
            *(hnodeD+i) = calloc(partitions, sizeof(Node *));
            CHECKALLOC(hnodeD[i]);
            for (int j=0; j<partitions; j++){
                *(*(hnodeD+i)+j) = calloc(partitions, sizeof(Node));
                CHECKALLOC(hnodeD[i][j]);
            }
        }

        //make_nodos(hnodeD, dataD, partitions, size_node, np);
        /*
        if (rand_required){
            for (int i=0; i<n_randfiles; i++){
                make_nodos(hnodeR[i], dataR[i], partitions, size_node, np);
            }
        };

        //Count nonzero data nodes
        for(int row=0; row<partitions; row++){
            for(int col=0; col<partitions; col++){
                for(int mom=0; mom<partitions; mom++){
                    if(hnodeD[row][col][mom].len>0)  nonzero_Dnodes+=1;

                    if (rand_required){
                        for (int i=0; i<n_randfiles; i++){
                            if(hnodeR[i][row][col][mom].len>0) {
                                nonzero_Rnodes[i]+=1;
                                tot_randnodes+=1;
                            }
                            
                        }
                    }

                }
            }
        }

        if (rand_required){
            for (int i=1; i<n_randfiles; i++){
                acum_nonzero_Rnodes[i] = acum_nonzero_Rnodes[i-1] + nonzero_Rnodes[i-1];
            }
        }

        //Deep copy into linear nodes and an ordered elements array
        hnodeD_s = new DNode[nonzero_Dnodes];
        h_ordered_pointsD = new PointW3D[np];
        if (rand_required){
            hnodeR_s = new DNode[tot_randnodes];
            h_ordered_pointsR_s = new PointW3D[n_randfiles*np];
        }

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
                            h_ordered_pointsD[j] = hnodeD[row][col][mom].elements[k_element];
                        }
                        idxD++;
                    }

                    if (rand_required){
                        for (int i=0; i<n_randfiles; i++){
                            if (hnodeR[i][row][col][mom].len>0){
                                hnodeR_s[acum_nonzero_Rnodes[i] + idxR[i]].nodepos = hnodeR[i][row][col][mom].nodepos;
                                hnodeR_s[acum_nonzero_Rnodes[i] + idxR[i]].start = i*np + last_pointR[i];
                                hnodeR_s[acum_nonzero_Rnodes[i] + idxR[i]].len = hnodeR[i][row][col][mom].len;
                                last_pointR[i] = last_pointR[i] + hnodeR[i][row][col][mom].len;
                                hnodeR_s[acum_nonzero_Rnodes[i] + idxR[i]].end = i*np + last_pointR[i];
                                for (int j=hnodeR_s[acum_nonzero_Rnodes[i] + idxR[i]].start; j<hnodeR_s[acum_nonzero_Rnodes[i] + idxR[i]].end; j++){
                                    k_element = j-hnodeR_s[acum_nonzero_Rnodes[i] + idxR[i]].start;
                                    h_ordered_pointsR_s[j] = hnodeR[i][row][col][mom].elements[k_element];
                                }
                                idxR[i]++;
                            }
                        }
                    }

                }
            }
        }

        //Allocate and copy the nodes into device memory
        CUCHECK(cudaMalloc(&dnodeD, nonzero_Dnodes*sizeof(DNode)));
        CUCHECK(cudaMalloc(&d_ordered_pointsD, np*sizeof(PointW3D)));
        CUCHECK(cudaMemcpy(dnodeD, hnodeD_s, nonzero_Dnodes*sizeof(DNode), cudaMemcpyHostToDevice));
        CUCHECK(cudaMemcpy(d_ordered_pointsD, h_ordered_pointsD, np*sizeof(PointW3D), cudaMemcpyHostToDevice));
        if (rand_required){
            CUCHECK(cudaMalloc(&dnodeR, tot_randnodes*sizeof(DNode)));
            CUCHECK(cudaMalloc(&d_ordered_pointsR, n_randfiles*np*sizeof(PointW3D)));
            CUCHECK(cudaMemcpy(dnodeR, hnodeR_s, tot_randnodes*sizeof(DNode), cudaMemcpyHostToDevice));
            CUCHECK(cudaMemcpy(d_ordered_pointsR, h_ordered_pointsR_s, n_randfiles*np*sizeof(PointW3D), cudaMemcpyHostToDevice));
        }
        stop_timmer_host = clock();
        htime = ((float)(stop_timmer_host-start_timmer_host))/CLOCKS_PER_SEC;
        cout << "Succesfully readed the data. All set to compute the histograms in " << htime*1000 << " miliseconds" << endl;
        */

        /* =======================================================================*/
        /* ===================== Free unused host memory =========================*/
        /* =======================================================================*/

        free(data_name);
        free(rand_name);

        for (int i=0; i<partitions; i++){
            for (int j=0; j<partitions; j++){
                free(hnodeD[i][j]);
            }
            free(hnodeD[i]);
        }    
        free(hnodeD);

        free(dataD);
        // free(hnodeD_s);
        // free(h_ordered_pointsD);
        
        if (rand_required)
        {
            for (int i = 0; i < n_randfiles; i++)
            {
                free(rand_files[i]);
                free(dataR[i]);
                for (int j=0; j<partitions; j++){
                    for (int k=0; k<partitions; k++){
                        free(hnodeR[i][j][k]);
                    }
                    free(hnodeR[i][j]);
                }
                free(hnodeR[i]);
            }
            free(rand_files);
            free(dataR);
            free(hnodeR);
            // free(idxR);
            // free(last_pointR);
            // free(hnodeR_s);
            // free(h_ordered_pointsR_s);
        }

        /* =======================================================================*/
        /* ============ Launch the right histogram maker function ================*/
        /* =======================================================================*/

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
        } else if (strcmp(argv[1],"2ani")==0){
            if (bpc){
                pcf_2aniBPC(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, size_box, dmax);
            } else {
                pcf_2ani(histo_names, dnodeD, d_ordered_pointsD, nonzero_Dnodes, dnodeR, d_ordered_pointsR, nonzero_Rnodes, acum_nonzero_Rnodes, n_randfiles, bins, size_node, dmax);
            }
        }
        */

        /* =======================================================================*/
        /* ========================== Free memory ================================*/
        /* =======================================================================*/

        //CUCHECK(cudaFree(dnodeD));
        //CUCHECK(cudaFree(d_ordered_pointsD));

        if (rand_required)
        {
            //CUCHECK(cudaFree(dnodeR));
            //CUCHECK(cudaFree(d_ordered_pointsR));
            //free(nonzero_Rnodes);
            //free(acum_nonzero_Rnodes);
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
