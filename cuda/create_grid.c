#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "create_grid.h"

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

//==================== Files reading ================================
int get_smallest_file(char **file_names, int n_files)
{
    size_t len = 0;
    int smallest = -1, nlines=0;
    char mypathto_files[] = "../data/";
    char *full_path, *line = NULL;

    for (int i = 0; i < n_files; i++)
    {
        full_path = calloc(strlen(mypathto_files)+strlen(file_names[i])+1, sizeof(char));
        CHECKALLOC(full_path);
        strcpy(full_path, mypathto_files);
        strcat(full_path, file_names[i]); //Set up the full path

        FILE *file;
        file = fopen(full_path,"r"); //Open the file
        while (getline(&line, &len, file) != -1) nlines++;
        if (nlines<smallest || smallest == -1) smallest = nlines;

        free(full_path);
        nlines=0;
    }
    
    return smallest;
}

void open_files(char *name_file, PointW3D **data, int *pts, float *size_box){

    //These will be function variables
    char mypathto_files[] = "../data/";
    char *full_path;
    full_path = calloc(strlen(mypathto_files)+strlen(name_file)+1, sizeof(char));

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
    (*data) = calloc((*pts), sizeof(PointW3D));
    CHECKALLOC(*data);

    //Read line by line again
    line = NULL;
    len = 0;
    char *number;
    int j=0;

    while (getline(&line, &len, file) != -1) {

        number = strtok(line, " ");
        (*data + j)->x = atof(number);
        if ((*data + j)->x > (*size_box)) (*size_box) = (int)(*data + j)->x + 1;

        number = strtok(NULL, " ");
        (*data + j)->y = atof(number);
        if ((*data + j)->y > (*size_box)) (*size_box) = (int)(*data + j)->y + 1;

        number = strtok(NULL, " ");
        (*data + j)->z = atof(number);
        if ((*data + j)->z > (*size_box)) (*size_box) = (int)(*data + j)->z + 1;

        number = strtok(NULL, " ");
        (*data + j)->w = atof(number);
        if ((*data + j)->w > (*size_box)) (*size_box) = (int)(*data + j)->w + 1;

        j++;
    }
    

    fclose(file); //Close the file

    free(full_path);
}

void open_pip_files(int32_t **pips, char *name_file, int np, int *n_pips){

    //These will be function variables
    char mypathto_files[] = "../data/";
    char *full_path;

    full_path = calloc(strlen(mypathto_files)+strlen(name_file)+1, sizeof(char));
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
        full_path = calloc(last_point+4, sizeof(char));
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
    *pips = calloc( np * (*n_pips), sizeof(int32_t) );

    CHECKALLOC(*pips);

    for (int i = 0; i < np; i++)
    {
        if (getline(&line, &len, file) == -1)
        {
            fprintf(stderr, "Number of pips entries and number of coordinated points does not match %s has %i while the coordinates file has %i. \n ", full_path, i, np);
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

//================= Sampling of the data =============================
void random_sample_wpips(PointW3D **data, int32_t **pips, int array_length, int pips_width, int sample_size)
{
    if ((RAND_MAX - array_length)<100) printf("The array length %i is very close to the maximum random int %i. The shuffling might bee poor \n", array_length, RAND_MAX);
    
    srand(time(NULL)); // use current time as seed for random generator
    int i, j, k;
    int32_t *temp_pip = malloc(pips_width*sizeof(int32_t)), *pip_aux = malloc(sample_size*pips_width*sizeof(int32_t));
    CHECKALLOC(temp_pip);
    CHECKALLOC(pip_aux);
    PointW3D temp_point, *points_aux = malloc(sample_size*sizeof(PointW3D));
    CHECKALLOC(points_aux);
    for (i = 0; i < array_length; i++)
    {
        j = i + rand() / (RAND_MAX / (array_length - i)+1);

        for (k = 0; k < pips_width; k++)
        {
            temp_pip[k] = (*pips)[j*pips_width + k];
            (*pips)[j*pips_width + k] = (*pips)[i*pips_width + k];
            (*pips)[i*pips_width + k] = temp_pip[k];
        }

        //temp_pip = (*pips)[j];
        //(*pips)[j] = (*pips)[i];
        //(*pips)[i] = temp_pip;

        temp_point = (*data)[j];
        (*data)[j] = (*data)[i];
        (*data)[i] = temp_point;
    }
    free(temp_pip);

    for (i = 0; i < sample_size; i++)
    {
        for (k = 0; k < pips_width; k++)  pip_aux[i*pips_width + k] = (*pips)[i*pips_width + k];
        //pip_aux[i] = (*pips)[i];
        points_aux[i] = (*data)[i];
    }

    free(*pips);
    *pips = malloc(sample_size*pips_width*sizeof(int32_t));
    CHECKALLOC(*pips);
    memcpy(*pips,pip_aux,sample_size*pips_width*sizeof(int32_t));
    free(pip_aux);

    free(*data);
    *data = malloc(sample_size*sizeof(PointW3D));
    CHECKALLOC(*data);
    memcpy(*data,points_aux,sample_size*sizeof(PointW3D));
    free(points_aux);
}

void random_sample(PointW3D **data, int array_length, int sample_size)
{
    if ((RAND_MAX - array_length)<100) printf("The array length %i is very close to the maximum random int %i. The shuffling might bee poor \n", array_length, RAND_MAX);
    
    srand(time(NULL)); // use current time as seed for random generator
    int j;
    PointW3D temp_point, *points_aux = malloc(sample_size*sizeof(PointW3D));
    CHECKALLOC(points_aux);
    for (int i = 0; i < array_length; i++)
    {
        j = i + rand() / (RAND_MAX / (array_length - i)+1);
        temp_point = (*data)[j];
        (*data)[j] = (*data)[i];
        (*data)[i] = temp_point;
    }

    for (int i = 0; i < sample_size; i++)
    {
        points_aux[i] = (*data)[i];
    }

    free(*data);
    *data = malloc(sample_size*sizeof(PointW3D));
    CHECKALLOC(*data);
    memcpy(*data,points_aux,sample_size*sizeof(PointW3D));
    free(points_aux);
}

//=================== Creating the nodes =============================
//void add(PointW3D *&array, int &lon, float _x, float _y, float _z, float _w){
    /*
    This function manages adding points to an specific Node. It receives the previous array, longitude and point to add
    and updates the previous array and length with the same array with the new point at the end and adds +1 to the length +1

    It manages the memory allocation and free of the previous and new elements.
    */
    /*
    lon++;
    PointW3D *array_aux = new PointW3D[lon];
    for (int i=0; i<lon-1; i++){
        array_aux[i].x = array[i].x;
        array_aux[i].y = array[i].y;
        array_aux[i].z = array[i].z;
        array_aux[i].w = array[i].w;
    }
    delete[] array;
    array = array_aux;
    array[lon-1].x = _x;
    array[lon-1].y = _y;
    array[lon-1].z = _z;
    array[lon-1].w = _w;
} */

//void make_nodos(Node ***nod, PointW3D *dat, int partitions, float size_node, int np){
    /*
    This function classifies the data in the nodes

    Args
    nod: Node 3D array where the data will be classified
    dat: array of PointW3D data to be classified and stored in the nodes
    partitions: number nodes in each direction
    size_node: dimensions of a single node
    np: number of points in the dat array
    */
   /*
    int row, col, mom;

    // First allocate memory as an empty node:
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){
                nod[row][col][mom].nodepos.z = ((float)(mom)*(size_node));
                nod[row][col][mom].nodepos.y = ((float)(col)*(size_node));
                nod[row][col][mom].nodepos.x = ((float)(row)*(size_node));
                nod[row][col][mom].len = 0;
                nod[row][col][mom].elements = new PointW3D[0];
            }
        }
    }

    // Classificate the ith elment of the data into a node and add that point to the node with the add function:
    for (int i=0; i<np; i++){
        row = (int)(dat[i].x/size_node);
        col = (int)(dat[i].y/size_node);
        mom = (int)(dat[i].z/size_node);
        add(nod[row][col][mom].elements, nod[row][col][mom].len, dat[i].x, dat[i].y, dat[i].z, dat[i].w);
    }
} */

//================== Saving the histograms ===========================
void save_histogram1D(char *name, int bns, double *histo, int nhistos){
    /* This function saves a 1 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */

    //This creates the full path to where I save the histograms files    char *full_path;
    char mypathto_files[] = "../results/";
    char *full_path = calloc(strlen(mypathto_files)+strlen(name)+1, sizeof(char));

    CHECKALLOC(full_path);

    strcpy(full_path, mypathto_files);
    strcat(full_path, name); //Set up the full path

    FILE *file;
    file = fopen(full_path,"w"); //Open the file

    CHECKOPENFILE(file);
    
    for (int i = 0; i < bns; i++) fprintf(file,"%.12f",histo[nhistos*bns + i]);
    //file2 << setprecision(12) << histo[nhistos*bns + i] << endl;

    fclose(file);
    free(full_path);
}

void save_histogram2D(char *name, int bns, double *histo, int nhistos){
    /* This function saves a 2 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */
    
    //This creates the full path to where I save the histograms files    char *full_path;
    char mypathto_files[] = "../results/";
    char *full_path = calloc(strlen(mypathto_files)+strlen(name)+1, sizeof(char));

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
            fprintf(file,"%.12f ",histo[idx]);
            //file << setprecision(12) << histo[idx] << ' ';
        }
        fprintf(file,"\n");
    }
    fclose(file);
    free(full_path);
}

void save_histogram3D(char *name, int bns, double *histo, int nhistos){
    /* This function saves a 3 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */
    
    //This creates the full path to where I save the histograms files    char *full_path;
    char mypathto_files[] = "../results/";
    char *full_path = calloc(strlen(mypathto_files)+strlen(name)+1, sizeof(char));

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
                fprintf(file,"%.12f ",histo[idx]);
                //file << setprecision(12) << histo[idx] << ' ';
            }
            fprintf(file,"\n");
        }
        fprintf(file,"\n \n");
    }
    fclose(file);
    free(full_path);
}

void save_histogram5D(char *name, int bns, double *histo, int nhistos){
    /* This function saves a 5 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */

    //This creates the full path to where I save the histograms files    char *full_path;
    char mypathto_files[] = "../results/";
    char *full_path = calloc(strlen(mypathto_files)+strlen(name)+1, sizeof(char));

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
                        fprintf(file,"%.12f ",histo[idx]);
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
