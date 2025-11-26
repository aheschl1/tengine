#include "../include/kernels.h"

extern "C" __global__ void add_kernel(const float * a, const float * b, float * c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
