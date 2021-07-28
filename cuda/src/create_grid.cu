#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <time.h>
#include "create_grid.cuh"

/* Complains if it cannot open a file */
#define CHECKOPENFILE(p)  if(p == NULL) {\
    fprintf(stderr, "%s (line %d): Error - unable to open the file \n", __FILE__, __LINE__);\
    exit(1);\
    }\

/* Complains if it cannot allocate the array */
#define CHECKALLOC(p)  if(p == NULL) {\
    fprintf(stderr, "%s (line %d): Error - unable to allocate required memory \n", __FILE__, __LINE__);\
    exit(1);\
}\

#define DATADIR "/home/jaarpa/3PCFGPU/data/"
#define RESULTDIR "/home/jaarpa/3PCFGPU/results/"

//==================== Files reading ================================
void open_files(char *name_file, PointW3D **data, int *pts)
{

    //These will be function variables
    char mypathto_files[] = DATADIR;
    char *full_path;
    full_path = (char *)calloc(strlen(mypathto_files)+strlen(name_file)+1, sizeof(char));

    CHECKALLOC(full_path);

    strcpy(full_path, mypathto_files);
    strcat(full_path, name_file); //Set up the full path

    FILE *file;
    file = fopen(full_path,"r"); //Open the file

    CHECKOPENFILE(file);

    //Read line by line
    (*pts) = 0;
    char *line = NULL;
    size_t len = 0;
    while (getline(&line, &len, file) != -1) (*pts)++;
    
    rewind(file);

    //Allocate memory for data
    (*data) = (PointW3D *)calloc((*pts), sizeof(PointW3D));
    CHECKALLOC(*data);

    //Read line by line again
    line = NULL;
    len = 0;
    char *number;
    int j=0;

    while (getline(&line, &len, file) != -1) {

        number = strtok(line, " ");
        (*data + j)->x = atof(number);

        number = strtok(NULL, " ");
        (*data + j)->y = atof(number);

        number = strtok(NULL, " ");
        (*data + j)->z = atof(number);

        number = strtok(NULL, " ");
        (*data + j)->w = atof(number);

        j++;
    }
    

    fclose(file); //Close the file

    free(full_path);
}

void open_pip_files(int32_t **pips, char *name_file, int np, int *n_pips)
{

    //These will be function variables
    char mypathto_files[] = DATADIR;
    char *full_path;

    full_path = (char *)calloc(strlen(mypathto_files)+strlen(name_file)+1, sizeof(char));
    CHECKALLOC(full_path);
    strcpy(full_path, mypathto_files);
    strcat(full_path, name_file); //Set up the full path
    
    //Changes the extension from .* to .pip
    int last_point = 0, length = strlen(full_path);
    for (int i = 0; i <= length; i++)
    {
        if (full_path[i] == '.') last_point = i;
        if (full_path[i] == '\0' && last_point == 0) last_point = i;
    }
    if (last_point+4 > length)
    {
        char *temp = strdup(full_path);
        free(full_path);
        full_path = (char *)calloc(last_point+4, sizeof(char));
        CHECKALLOC(full_path);
        for (int i = 0; i <= length; i++) full_path[i] = temp[i];
        free(temp);

    }
    full_path[last_point] = '.';
    full_path[last_point+1] = 'p';
    full_path[last_point+2] = 'i';
    full_path[last_point+3] = 'p';
    full_path[last_point+4] = '\0';

    FILE *file;
    file = fopen(full_path,"r"); //Open the file

    CHECKOPENFILE(file);

    //Read line by line
    char *line = NULL;
    char *number;
    int j=0;
    size_t len = 0;
    
    //Get the number of columns
    *n_pips=0;
    if (getline(&line, &len, file) == -1)
    {
        fprintf(stderr, "Recived empty file at %s \n", full_path);
        exit(1);
    }
    number = strtok(line, " ");
    while (number != NULL)
    {
        (*n_pips)++;
        number = strtok(NULL, " ");
    }

    rewind(file);
    *pips = (int32_t*)calloc( np * (*n_pips), sizeof(int32_t) );

    CHECKALLOC(*pips);

    for (int i = 0; i < np; i++)
    {
        if (getline(&line, &len, file) == -1)
        {
            fprintf(stderr, "Number of pips entries and number of coordinated points do not match %s has %i while the coordinates file has %i. \n ", full_path, i, np);
            exit(1);
        }
        number = strtok(line, " ");
        while (number != NULL)
        {
            *(*pips +j) = atoi(number);
            j++;
            number = strtok(NULL, " ");
        }
    }

    fclose(file); //Close the file

    free(full_path);
}

void read_random_files(char ***rand_files, char ***histo_names, int **rnp, PointW3D ***dataR, int *n_randfiles, char *rand_name, int rand_dir)
{
    //Check if a directory of random files was provided to change n_randfiles
    //Instead of rand name should be an array with the name of each rand array or something like that.
    if (rand_dir)
    {
        char data_path[] = DATADIR;
        char *directory_path = (char*)calloc((strlen(data_path)+strlen(rand_name))+1,sizeof(char));
        CHECKALLOC(directory_path);
        strcpy(directory_path, data_path);
        strcat(directory_path, rand_name); //Set up the full path
        DIR *folder = opendir(directory_path);
        
        if(folder != NULL)
        {
            (*n_randfiles) = 0;
            struct dirent *archivo;
            while( (archivo=readdir(folder)) )
            if (strcmp(archivo->d_name,".") != 0 && strcmp(archivo->d_name,"..") != 0 && strcmp(&(archivo->d_name)[strlen((archivo->d_name))-4],".pip") != 0)
            (*n_randfiles)++;
            
            if (!(*n_randfiles))
            {
                fprintf(stderr, "There are no suitable files in %s \n", directory_path);
                exit(1);
            }
            //Reset the folder stream to actually read the files
            rewinddir(folder);
            
            (*rand_files) = (char **)malloc((*n_randfiles) * sizeof(char *));
            CHECKALLOC((*rand_files));
            (*histo_names) = (char **)malloc((1 + (*n_randfiles)) * sizeof(char *));
            CHECKALLOC((*histo_names));
            
            int j = 0;
            char *nombre_archivo;
            while( (archivo=readdir(folder)) )
            {
                nombre_archivo = archivo->d_name;
                if (strcmp(nombre_archivo,".") == 0 || strcmp(nombre_archivo,"..") == 0 || strcmp(&(nombre_archivo)[strlen((nombre_archivo))-4],".pip") == 0)
                    continue;
                (*histo_names)[j+1] = strdup(nombre_archivo);
                
                (*rand_files)[j] = (char*)malloc((strlen(rand_name)+strlen(nombre_archivo)+1)*sizeof(char));
                CHECKALLOC((*rand_files)[j]);
                strcpy((*rand_files)[j], rand_name);
                strcat((*rand_files)[j], nombre_archivo); //Set up the full path
                
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
        (*rand_files) = (char**)malloc((*n_randfiles) * sizeof(char *));
        CHECKALLOC((*rand_files));
        (*rand_files)[0] = strdup(rand_name);
        (*histo_names) = (char**)malloc((1 + (*n_randfiles)) * sizeof(char *));
        CHECKALLOC((*histo_names));
        (*histo_names)[1] = strdup(rand_name);
    }
    
    (*dataR) = (PointW3D**)calloc((*n_randfiles), sizeof(PointW3D *));
    CHECKALLOC((*dataR));
    (*rnp) = (int*)calloc((*n_randfiles), sizeof(int));
    CHECKALLOC((*rnp));

    for (int i=0; i<(*n_randfiles); i++)
        open_files((*rand_files)[i], &(*dataR)[i], &(*rnp)[i]);
}

//=================== Creating the nodes =============================
int create_nodes_wpips(DNode **nod, PointW3D **dat, int32_t **pips, int pips_width, int partitions, float size_node, int np)
{
    int row, col, mom, idx, non_zero_idx, len, non_zero_nodes;
    Node *hnode = (Node *)calloc(partitions*partitions*partitions, sizeof(Node));

    // First allocate memory as an empty node:
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){
                idx = row*partitions*partitions + col*partitions + mom;
                hnode[idx].nodepos.z = ((float)(mom)*(size_node));
                hnode[idx].nodepos.y = ((float)(col)*(size_node));
                hnode[idx].nodepos.x = ((float)(row)*(size_node));
                hnode[idx].len = 0;
                hnode[idx].elements = (PointW3D *)calloc(0,sizeof(PointW3D));
                hnode[idx].pips = (int32_t *)calloc(0,sizeof(int32_t));
            }
        }
    }

    // Classificate the ith elment of the data into a node and add that point to the node with the add function:
    for (int i=0; i<np; i++){
        row = (int)((*dat)[i].x/size_node);
        col = (int)((*dat)[i].y/size_node);
        mom = (int)((*dat)[i].z/size_node);
        idx = row*partitions*partitions + col*partitions + mom;
        len = ++hnode[idx].len;
        hnode[idx].elements = (PointW3D *)realloc(hnode[idx].elements, len*sizeof(PointW3D));
        hnode[idx].elements[len-1] = (*dat)[i];
        hnode[idx].pips = (int32_t *)realloc(hnode[idx].pips, len*pips_width*sizeof(int32_t));
        for (int j = 0; j < pips_width; j++)
            hnode[idx].pips[(len - 1)*pips_width + j] = (*pips)[i * pips_width + j];
    }

    //Counts non zero nodes
    non_zero_nodes = 0;
    for (idx = 0; idx<partitions*partitions*partitions; idx++) 
        if (hnode[idx].len > 0)
            non_zero_nodes++;

    *nod = (DNode *)malloc(non_zero_nodes*sizeof(DNode));
    idx = -1, non_zero_idx = 0, len = 0; //len is no the accumulated length of all the previous idx nodes
    while (len < np)
    {
        idx++;
        if (hnode[idx].len <= 0) continue;
        (*nod)[non_zero_idx].nodepos = hnode[idx].nodepos;
        (*nod)[non_zero_idx].len = hnode[idx].len;
        (*nod)[non_zero_idx].start = len;
        for (int n_pto = 0; n_pto < hnode[idx].len; n_pto++)
        {
            (*dat)[len + n_pto] = hnode[idx].elements[n_pto];
            for (int n_pip = 0; n_pip < pips_width; n_pip++) (*pips)[(len + n_pto)*pips_width + n_pip] = hnode[idx].pips[n_pto*pips_width + n_pip];
        }
        len += hnode[idx].len;
        (*nod)[non_zero_idx].end = len;
        non_zero_idx++;
    }

    for (idx = 0; idx < partitions*partitions*partitions; idx++)
    {
        free(hnode[idx].elements);
        free(hnode[idx].pips);
    }
    free(hnode);

    return non_zero_nodes;    

}

int create_nodes(DNode **nod, PointW3D **dat, int partitions, float size_node, int np)
{
    int row, col, mom, idx, non_zero_idx, len, non_zero_nodes;
    Node *hnode = (Node *)calloc(partitions*partitions*partitions, sizeof(Node));

    // First allocate memory as an empty node:
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){
                idx = row*partitions*partitions + col*partitions + mom;
                hnode[idx].nodepos.z = ((float)(mom)*(size_node));
                hnode[idx].nodepos.y = ((float)(col)*(size_node));
                hnode[idx].nodepos.x = ((float)(row)*(size_node));
                hnode[idx].len = 0;
                hnode[idx].elements = (PointW3D *)calloc(0,sizeof(PointW3D));
                hnode[idx].pips = (int32_t *)calloc(0,sizeof(int32_t));
            }
        }
    }

    // Classificate the ith elment of the data into a node and add that point to the node with the add function:
    for (int i=0; i<np; i++){
        row = (int)((*dat)[i].x/size_node);
        col = (int)((*dat)[i].y/size_node);
        mom = (int)((*dat)[i].z/size_node);
        idx = row*partitions*partitions + col*partitions + mom;
        len = ++hnode[idx].len;
        hnode[idx].elements = (PointW3D*)realloc(hnode[idx].elements, len*sizeof(PointW3D));
        hnode[idx].elements[len-1] = (*dat)[i];
    }

    //Counts non zero nodes
    non_zero_nodes = 0;
    for (idx = 0; idx<partitions*partitions*partitions; idx++) 
        if (hnode[idx].len > 0)
            non_zero_nodes++;

    *nod = (DNode *)malloc(non_zero_nodes*sizeof(DNode));
    idx = -1, non_zero_idx = 0, len = 0; //len is no the accumulated length of all the previous idx nodes
    while (len < np)
    {
        idx++;
        if (hnode[idx].len <= 0) continue;
        (*nod)[non_zero_idx].nodepos = hnode[idx].nodepos;
        (*nod)[non_zero_idx].len = hnode[idx].len;
        (*nod)[non_zero_idx].start = len;
        for (int n_pto = 0; n_pto < hnode[idx].len; n_pto++)
            (*dat)[len + n_pto] = hnode[idx].elements[n_pto];
        len += hnode[idx].len;
        (*nod)[non_zero_idx].end = len;
        non_zero_idx++;
    }

    for (idx = 0; idx < partitions*partitions*partitions; idx++)
    {
        free(hnode[idx].elements);
        free(hnode[idx].pips);
    }
    free(hnode);

    return non_zero_nodes;    

}

//================== Saving the histograms ===========================
void save_histogram1D(char *name, int bns, double *histo, int nhistos)
{
    /* This function saves a 1 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */

    //This creates the full path to where I save the histograms files    char *full_path;
    char mypathto_files[] = RESULTDIR;
    char *full_path = (char *)calloc(strlen(mypathto_files)+strlen(name)+1, sizeof(char));

    CHECKALLOC(full_path);

    strcpy(full_path, mypathto_files);
    strcat(full_path, name); //Set up the full path

    FILE *file;
    file = fopen(full_path,"w"); //Open the file

    CHECKOPENFILE(file);
    
    for (int i = 0; i < bns; i++) fprintf(file,"%.4f ",histo[nhistos*bns + i]);
    //file2 << setprecision(12) << histo[nhistos*bns + i] << endl;

    fclose(file);
    free(full_path);
}

void save_histogram2D(char *name, int bns, double *histo, int nhistos)
{
    /* This function saves a 2 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */
    
    //This creates the full path to where I save the histograms files    char *full_path;
    char mypathto_files[] = RESULTDIR;
    char *full_path = (char *)calloc(strlen(mypathto_files)+strlen(name)+1, sizeof(char));

    CHECKALLOC(full_path);

    strcpy(full_path, mypathto_files);
    strcat(full_path, name); //Set up the full path

    FILE *file;
    file = fopen(full_path,"w"); //Open the file

    CHECKOPENFILE(file);

    int idx;

    for (int i = 0; i < bns; i++){
        for (int j = 0; j < bns; j++){
            idx = nhistos*bns*bns + i*bns + j;
            fprintf(file,"%.4f ",histo[idx]);
            //file << setprecision(12) << histo[idx] << ' ';
        }
        fprintf(file,"\n");
    }
    fclose(file);
    free(full_path);
}

void save_histogram3D(char *name, int bns, double *histo, int nhistos)
{
    /* This function saves a 3 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */
    
    //This creates the full path to where I save the histograms files    char *full_path;
    char mypathto_files[] = RESULTDIR;
    char *full_path = (char *)calloc(strlen(mypathto_files)+strlen(name)+1, sizeof(char));

    CHECKALLOC(full_path);

    strcpy(full_path, mypathto_files);
    strcat(full_path, name); //Set up the full path

    FILE *file;
    file = fopen(full_path,"w"); //Open the file

    CHECKOPENFILE(file);
    
    int idx;

    for (int i = 0; i < bns; i++){
        for (int j = 0; j < bns; j++){
            for (int k = 0; k < bns; k++){
                idx = nhistos*bns*bns*bns + i*bns*bns + j*bns + k;
                fprintf(file,"%.4f ",histo[idx]);
                //file << setprecision(12) << histo[idx] << ' ';
            }
            fprintf(file,"\n");
        }
        fprintf(file,"\n \n");
    }
    fclose(file);
    free(full_path);
}

void save_histogram5D(char *name, int bns, double *histo, int nhistos)
{
    /* This function saves a 5 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */

    //This creates the full path to where I save the histograms files    char *full_path;
    char mypathto_files[] = RESULTDIR;
    char *full_path = (char *)calloc(strlen(mypathto_files)+strlen(name)+1, sizeof(char));

    CHECKALLOC(full_path);

    strcpy(full_path, mypathto_files);
    strcat(full_path, name); //Set up the full path

    FILE *file;
    file = fopen(full_path,"w"); //Open the file

    CHECKOPENFILE(file);

    int idx;

    for (int i = 0; i < bns; i++){
        for (int j = 0; j < bns; j++){
            for (int k = 0; k < bns; k++){
                for (int l = 0; l < bns; l++){
                    for (int m = 0; m < bns; m++){
                        idx = nhistos*bns*bns*bns*bns*bns + i*bns*bns*bns*bns + j*bns*bns*bns + k*bns*bns + l*bns + m;
                        fprintf(file,"%.4f ",histo[idx]);
                        //file << setprecision(12) << histo[idx] << ' ';
                    }
                    fprintf(file,"\n");
                    //file << "\n";
                }
                fprintf(file,"\n");
                //file << "\n";
            }
            fprintf(file,"\n");
            //file << "\n";
        }
        fprintf(file,"\n \n");
        //file << "\n" << endl;
    }
    fclose(file);
    free(full_path);
}
