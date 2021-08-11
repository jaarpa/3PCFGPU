#include <stdint.h>
#include "device_functions.cuh"

__device__ double get_weight(
    int32_t *pipsA, int A_i, int32_t *pipsB, int B_i, int pips_width
)
{
    int i;
    double weight = 0, totbits = pips_width*32;
    for (i = 0; i < pips_width; i++)
        weight += __popc(pipsA[A_i*pips_width + i] & pipsB[B_i*pips_width + i]);

    return weight/totbits;
}

__device__ double get_3d_weight(
    int32_t *pipsA, int A_i, int32_t *pipsB, int B_i,
    int32_t *pipsC, int C_i, int pips_width
)
{
    int i;
    double weight = 0, totbits = pips_width*32;
    for (i = 0; i < pips_width; i++)
        weight += __popc(
            pipsA[A_i*pips_width + i] & pipsB[B_i*pips_width + i]
            & pipsC[C_i*pips_width + i]
        );

    return weight/totbits;
}