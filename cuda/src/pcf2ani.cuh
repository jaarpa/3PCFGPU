#ifndef PCF2ANI_CUH
#define PCF2ANI_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "create_grid.cuh"

/*
Main function to calculate the anisotropic 2 point correlation function. Saves three different histograms in the same location of this script
with the names DD.dat DR.dat RR.dat. This program does not consider periodic boundary conditions. The file must contain 4 columns, the first 3 
are the x,y,z coordinates and the 4 the weigh of the measurment.

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
void pcf_2ani(
    DNode *dnodeD, PointW3D *d_dataD, int nonzero_Dnodes,
    DNode **dnodeR, PointW3D **d_dataR, int *nonzero_Rnodes,
    char **histo_names, int n_randfiles, int bn, float size_node, float dmax
);

void pcf_2ani_wpips(char **histo_names, DNode *dnodeD, PointW3D *dataD, int32_t *dpipsD, int nonzero_Dnodes, DNode *dnodeR, PointW3D *dataR, int32_t *dpipsR, int *nonzero_Rnodes, int *acum_nonzero_Rnodes, int n_pips, int n_randfiles, int bn, float size_node, float dmax);


#ifdef __cplusplus
}
#endif
#endif