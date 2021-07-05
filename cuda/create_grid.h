#ifndef CREATE_GRID
#define CREATE_GRID

#include <stdint.h>

typedef struct {
	float x;
	float y; 
	float z;
} Point3D;

//Point with weight value. Structure
typedef struct {
    float x;
    float y; 
    float z;
    float w;
} PointW3D;

typedef struct {
    Point3D nodepos; //Position of the node
    int len;		// Number of points in the node
    PointW3D *elements;	// Points in the node
    int32_t *pips; //Pips of the node
} Node;

typedef struct { //Defines the node in the device without using elements to avoid deep copy
    Point3D nodepos; //Position of the node
    int len;		// Number of points in the node
    int start; //prev element idx
    int end; //last element idx [non inclusive]
} DNode;

//==================== Files reading ================================

/*
This function receives an array (char **file_names) of (int n_files) char arrays with the location of files relative to ../data/ and reads
them all to find which of them has the minimum number of lines.
params:
    char **file_names: array of char arrays with the location of files relative to ../data/
    int n_files: Number of char arrays in file_names

returns:
    int np: Number of lines that the file with less line has.

example:
    int rnp = get_smallest_file(file_names, n_files);
*/
int get_smallest_file(char **file_names, int n_files);

/*
This function opens and reads a file located at ../data/ + char *name_file, stores the data in the PointW3D **data array,
stores the number of lines in the file in the int *pts value, and the largest component of the points in float *size_box.

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
void open_files(char *name_file, PointW3D **data, int *pts, float *size_box);
void open_pip_files(int32_t **pips, char *name_file, int np, int *n_pips);

//================= Sampling of the data =============================
void random_sample_wpips(PointW3D **data, int32_t **pips, int array_length, int pips_width, int sample_size);
void random_sample(PointW3D **data, int array_length, int sample_size);

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
int create_nodes_wpips(DNode **nod, PointW3D **dat, int32_t **pips, int pips_width, int partitions, float size_node, int np);

int create_nodes(DNode **nod, PointW3D **dat, int partitions, float size_node, int np);

//================== Saving the histograms ===========================
void save_histogram1D(char *name, int bns, double *histo, int nhistos);
void save_histogram2D(char *name, int bns, double *histo, int nhistos);
void save_histogram3D(char *name, int bns, double *histo, int nhistos);
void save_histogram5D(char *name, int bns, double *histo, int nhistos);

#endif