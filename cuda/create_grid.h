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
} Node;

typedef struct { //Defines the node in the device without using elements to avoid deep copy
    Point3D nodepos; //Position of the node
    int len;		// Number of points in the node
    int start; //prev element idx
    int end; //last element idx [non inclusive]
} DNode;

//=================================================================== 
void open_files(char *name_file, PointW3D **data, int *pts, float *size_box);
void open_pip_files(int32_t **pips, char *name_file, int pts, int *n_pips);

//=================================================================== 
//void add(PointW3D *array, int *lon, float _x, float _y, float _z, float _w);

//void make_nodos(Node ***nod, PointW3D *dat, unsigned int partitions, float size_node, unsigned int np);

//====================================================================
void save_histogram2D(char *name, int bns, double *histo, int nhistos);

//====================================================================
void save_histogram3D(char *name, int bns, double *histo, int nhistos);

//====================================================================
void save_histogram5D(char *name, int bns, double *histo, int nhistos);

#endif