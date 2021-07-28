#include <stdint.h>
#include "device_functions.cuh"

__device__ double get_weight(int32_t *a, int ai, int32_t *b, int bi, int l){
    int i;
    double weight = 0, totbits = l*32;
    for (i = 0; i < l; i++){
        weight += __popc(a[ai*l + i] & b[bi*l + i]);
    }
    return weight/totbits;
}

__device__ double get_3d_weight(int32_t *a, int ai, int32_t *b, int bi, int32_t *c, int ci, int l)
{
    int i;
    double weight = 0, totbits = l*32;
    for (i = 0; i < l; i++){
        weight += __popc(a[ai*l + i] & b[bi*l + i] & c[ci*l + i]);
    }
    return weight/totbits;
}