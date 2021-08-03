#ifndef DEVICE_FUNCTIONS_CUH
#define DEVICE_FUNCTIONS_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

__device__ double get_weight(
    int32_t *pipsA, int A_i, int32_t *pipsB, int B_i, int pips_width
);
__device__ double get_3d_weight(
    int32_t *pipsA, int A_i, int32_t *pipsB, int B_i,
    int32_t *pipsC, int C_i, int pips_width
);

#ifdef __cplusplus
}
#endif

#endif