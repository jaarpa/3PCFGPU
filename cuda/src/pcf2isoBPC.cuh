#ifndef PCF2ISO_BPC_CUH
#define PCF2ISO_BPC_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include "create_grid.cuh"

/*
Main function to calculate the isotropic 2 point correlation function. Saves three different histograms in the same location of this script
with the names DD.dat DR.dat RR.dat. This program does consider periodic boundary conditions. The file must contain 4 columns, the first 3 
are the x,y,z coordinates and the 4 the weigh of the measurment.

Args:
*histo_names: (string)
*dnodeD: (DNode)
*d_ordered_pointsD: (PointW3D)
nonzero_Dnodes: (int)
*dnodeR: (DNode)
*d_ordered_pointsR: (PointW3D)
*nonzero_Rnodes: (int)
*acum_nonzero_Rnodes: (int)
n_randfiles: (int)
bn: (int)
size_node: (float)
dmax: (float)
analytic=true: (bool)

*/
void pcf_2iso_BPC(
    DNode *d_nodeD, PointW3D *d_dataD,
    int nonzero_Dnodes, cudaStream_t streamDD, cudaEvent_t DDcopy_done, 
    DNode **d_nodeR, PointW3D **d_dataR,
    int *nonzero_Rnodes, cudaStream_t *streamRR, cudaEvent_t *RRcopy_done,
    char **histo_names, int n_randfiles, int bins, float size_node, float dmax,
    float size_box
);


/*
Main function to calculate the isotropic 2 point correlation function. Saves three different histograms in the same location of this script
with the names DD.dat DR.dat RR.dat. This program does consider periodic boundary conditions.  The RR and DR are compueted analytically.
The file must contain 4 columns, the first 3 are the x,y,z coordinates and the 4 the weigh of the measurment.

Args:
*histo_names: (string)
*dnodeD: (DNode)
*d_ordered_pointsD: (PointW3D)
nonzero_Dnodes: (int)
*dnodeR: (DNode)
*d_ordered_pointsR: (PointW3D)
*nonzero_Rnodes: (int)
*acum_nonzero_Rnodes: (int)
n_randfiles: (int)
bn: (int)
size_node: (float)
dmax: (float)
analytic=true: (bool)
*/
void pcf_2iso_BPCanalytic(
    DNode *d_nodeD, PointW3D *d_dataD,
    int nonzero_Dnodes, cudaStream_t streamDD,
    int bins, int np, float size_node, float size_box, float dmax,
    char *data_name
);

#ifdef __cplusplus
}
#endif
#endif