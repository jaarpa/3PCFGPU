#ifndef CREATE_GRID
#define CREATE_GRID

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct
{
    float x;
    float y;
    float z;
} Point3D;

//Point with weight value. Structure
typedef struct
{
    float x;
    float y;
    float z;
    float w;
} PointW3D;

typedef struct
{
    //Position of the node
    Point3D nodepos;
    // Number of points in the node
    int len;
    // Points in the node
    PointW3D *elements;
    //Pips of the node
    int32_t *pips;
} Node;

//Defines the node in the device without using elements to avoid deep copy
typedef struct
{ 
    Point3D nodepos;
    int len;
    // prev element idx
    int start;
    // last element idx [non inclusive]
    int end;
} DNode;

//==================== Files reading ================================

/*
Now data needs to be deallocated with cudaFreeHost. This function opens and reads
a file located at ../data/ + char *name_file, stores the data in the PointW3D 
**data array, stores the number of lines in the file in the int *pts value, 
and the largest component of the points in float *size_box.

It is destructive relative to the data, pts and size_box variables.
params:
    char *name_file: char array with the location of file relative to ../data/
    PointW3D **data
    int *pts
    float *size_box

returns:
    None: PointW3D **data, int *pts, float *size_box will be overwritten

example:
    open_files(rand_files[i], &dataR[i], &rnp, &size_box);
*/
void open_files(PointW3D **data, int *pts, char *name_file);
void open_pip_files(int32_t **pips, int *n_pips, char *name_file, int np);
void read_random_files(char ***rand_files, char ***histo_names, int **rnp, PointW3D ***dataR, int *n_randfiles, char *rand_name, int rand_dir);

//=================== Creating the nodes =============================
/*
This function classifies the data into nodes

Args:
    nod: Node 3D array where the data will be classified
    dat: array of PointW3D data to be classified and stored in the nodes
    partitions: number nodes in each direction
    size_node: dimensions of a single node
    np: number of points in the dat array

Returns:
    None. But creates the dnodes in nod_s_dest, and orders the points in dat and pips

*/
int create_nodes(DNode **nod, PointW3D **dat, int32_t **pips, int pips_width, int partitions, float size_node, int np);

//int create_nodes(DNode **nod, PointW3D **dat, int partitions, float size_node, int np);

//================== Saving the histograms ===========================
void save_histogram1D(char *name, int bns, double *histo, int nhistos);
void save_histogram2D(char *name, int bns, double *histo, int nhistos);
void save_histogram3D(char *name, int bns, double *histo, int nhistos);
void save_histogram5D(char *name, int bns, double *histo, int nhistos);
#ifdef __cplusplus
}
#endif

#endif