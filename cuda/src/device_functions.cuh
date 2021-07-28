#ifndef DEVICE_FUNCTIONS_CUH
#define DEVICE_FUNCTIONS_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

__device__ double get_weight(int32_t *a, int ai, int32_t *b, int bi, int l);

#ifdef __cplusplus
}
#endif

#endif