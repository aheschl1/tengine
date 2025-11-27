#include "../include/kernels.h"

template <typename T>
__global__ void unary_kernel(
    T* __restrict__ data, 
    const size_t* __restrict__ offsets, 
    size_t n, 
    uint8_t op, 
    T value
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    const size_t idx = offsets[i];
    const T x = data[idx];

    switch (op) {
        case 0: data[idx] = x + value; break;
        case 1: data[idx] = x - value; break;
        case 2: data[idx] = x * value; break;
    }
}

template <typename T>
void launch_unary_op(T* data, size_t* offsets, size_t n, uint8_t op, T value, unsigned int block_size) {
    // Clamp block size to valid range [32, 1024] and ensure it's a multiple of warp size (32)
    block_size = (block_size < 32) ? 32 : (block_size > 1024) ? 1024 : block_size;
    block_size = (block_size / 32) * 32;  // Round down to nearest multiple of 32
    
    const unsigned int grid = (unsigned int)((n + block_size - 1) / block_size);
    unary_kernel<T><<<grid, block_size>>>(data, offsets, n, op, value);
}

#define DECLARE_UNARY_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_unary_##SUFFIX(TYPE* data, size_t* offsets, size_t n, uint8_t op, TYPE value, unsigned int block_size) { \
        launch_unary_op(data, offsets, n, op, value, block_size); \
    }

DECLARE_UNARY_LAUNCHER(float, f32)
DECLARE_UNARY_LAUNCHER(double, f64)
DECLARE_UNARY_LAUNCHER(uint8_t, u8)
DECLARE_UNARY_LAUNCHER(uint16_t, u16)
DECLARE_UNARY_LAUNCHER(uint32_t, u32)
DECLARE_UNARY_LAUNCHER(uint64_t, u64)
DECLARE_UNARY_LAUNCHER(__uint128_t, u128)
DECLARE_UNARY_LAUNCHER(int8_t, i8)
DECLARE_UNARY_LAUNCHER(int16_t, i16)
DECLARE_UNARY_LAUNCHER(int32_t, i32)
DECLARE_UNARY_LAUNCHER(int64_t, i64)
DECLARE_UNARY_LAUNCHER(__int128_t, i128)

