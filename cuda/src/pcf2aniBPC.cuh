#ifndef PCF2ANI_BPC_CUH
#define PCF2ANI_BPC_CUH

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
void pcf_2aniBPC(
        char **histo_names, DNode *dnodeD, PointW3D *d_dataD, int nonzero_Dnodes, 
        DNode *dnodeR, PointW3D *d_dataR, int *nonzero_Rnodes, int *acum_nonzero_Rnodes, int n_randfiles, 
        int bins, float size_node, float size_box, float dmax
    );

/*
Kernel function to calculate the pure histograms for the 2 point anisotropic correlation function WITH 
boundary periodic conditions. It stores the counts in the XX histogram.

args:
XX: (double*) The histogram where the distances are counted.
elements: (PointW3D*) Array of the points ordered coherently with the nodes.
nodeD: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node.
nonzero_nodes: (int) Number of nonzero nodes where the points have been classificated.
bn: (int) NUmber of bins in the XY histogram.
dmax: (float) The maximum distance of interest between points.
d_max_node: (float) The maximum internodal distance.
size_box: (float) The size of the box where the points were contained. It is used for the boundary periodic conditions
size_node: (float) Size of the nodes.
*/
__global__ void XX2ani_BPC(
        double *XX, PointW3D *elements, DNode *nodeD, int nonzero_nodes, int bn, 
        float dmax, float d_max_node, float size_box, float size_node, 
        int node_offset, int bn_offset
    );
    
/*
Kernel function to calculate the mixed histograms for the 2 point anisotropic correlation function with 
boundary periodic conditions. It stores the counts in the XY histogram.

args:
XY: (double*) The histogram where the distances are counted.
elementsD: (PointW3D*) Array of the points ordered coherently with the nodes. For the data points.
nodeD: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node. For the data points
nonzero_Dnodes: (int) Number of nonzero nodes where the points have been classificated. For the data points
elementsR: (PointW3D*) Array of the points ordered coherently with the nodes. For the random points.
nodeR: (DNode) Array of DNodes each of which define a node and the elements of element that correspond to that node. For the random points
nonzero_Rnodes: (int) Number of nonzero nodes where the points have been classificated. For the random points
bn: (int) NUmber of bins in the XY histogram.
dmax: (float) The maximum distance of interest between points.
d_max_node: (float) The maximum internodal distance.
size_box: (float) The size of the box where the points were contained. It is used for the boundary periodic conditions
size_node: (float) Size of the nodes.
*/
__global__ void XY2ani_BPC(
    double *XY, PointW3D *elementsD, DNode *nodeD, int nonzero_Dnodes, 
    PointW3D *elementsR,  DNode *nodeR, int nonzero_Rnodes, int bn, 
    float dmax, float d_max_node, float size_box, float size_node, 
    int node_offset, int bn_offset
);

#ifdef __cplusplus
}
#endif
#endif