#ifndef PCF3ISO_CUH
#define PCF3ISO_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "create_grid.cuh"

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

void pcf_3iso(
    DNode *d_nodeD, PointW3D *d_dataD, int32_t *d_pipsD,
    int nonzero_Dnodes, cudaStream_t streamDD, cudaEvent_t DDcopy_done, 
    DNode **d_nodeR, PointW3D **d_dataR, int32_t **d_pipsR,
    int *nonzero_Rnodes, cudaStream_t *streamRR, cudaEvent_t *RRcopy_done,
    char **histo_names, int n_randfiles, int bins, float size_node, float dmax,
    int pips_width
);

#ifdef __cplusplus
}
#endif
#endif