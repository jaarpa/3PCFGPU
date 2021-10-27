#ifndef BF_PCF2ISO_CUH
#define BF_PCF2ISO_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "create_grid.cuh"

/*
Main function to calculate the isotropic 2 point correlation function using brute force. Saves
three different histograms in the same location of this script with the names
DD.dat DR.dat RR.dat. This program do not consider periodic boundary conditions.
The file must contain 4 columns, the first 3 are the x,y,z coordinates and the
4th the weigh of the measurment.

Args:
dnodeD: (DNode)
d_ordered_pointsD: (PointW3D)
nonzero_Dnodes: (int)
hnodeR_s: (DNode)
h_ordered_pointsR_s: (PointW3D)
nonzero_Rnodes: (int)
n_randfiles: (int)
bn: (int)
dmax: (float)

*/
void pcf_bf_2iso(
    PointW3D *d_dataD, int32_t *d_pipsD, cudaStream_t streamDD, cudaEvent_t DDcopy_done, int np,
    PointW3D **d_dataR, cudaStream_t *streamRR, cudaEvent_t *RRcopy_done, int *rnp,
    char **histo_names, int n_randfiles, int bins, float dmax,
    int pips_width
);

#ifdef __cplusplus
}
#endif
#endif