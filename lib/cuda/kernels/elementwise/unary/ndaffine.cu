#include "../../../include/scalar.h"

template <typename T>
__global__ void negate_nd_affine_kernel(
    T* __restrict__ data,
    size_t offset,
    const ptrdiff_t* __restrict__ stride,
    const size_t* __restrict__ shape,
    size_t rank,
    size_t size
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x
    ) {
        size_t linear = idx;
        ptrdiff_t phys = (ptrdiff_t)offset;

        for (int dim = (int)rank - 1; dim >= 0; --dim) {
            size_t coord = linear % shape[dim];
            linear /= shape[dim];
            phys += (ptrdiff_t)coord * stride[dim];
        }

        size_t final_idx = (size_t)phys;
        data[final_idx] = -data[final_idx];
    }
}

template <typename T>
void launch_negate_nd_affine_op(
    T* data,
    size_t offset,
    const ptrdiff_t* stride,
    const size_t* shape,
    size_t rank,
    size_t size,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((size + block_size - 1) / block_size), 65535u);
    negate_nd_affine_kernel<T><<<grid, block_size>>>(data, offset, stride, shape, rank, size);
}

#define DECLARE_NEGATE_ND_AFFINE_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_negate_nd_affine_##SUFFIX( \
        TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, \
        size_t rank, size_t size, unsigned int block_size \
    ) { \
        launch_negate_nd_affine_op<TYPE>(data, offset, stride, shape, rank, size, block_size); \
    }

DECLARE_NEGATE_ND_AFFINE_LAUNCHER(float,  f32)
DECLARE_NEGATE_ND_AFFINE_LAUNCHER(double, f64)
DECLARE_NEGATE_ND_AFFINE_LAUNCHER(int8_t,  i8)
DECLARE_NEGATE_ND_AFFINE_LAUNCHER(int16_t, i16)
DECLARE_NEGATE_ND_AFFINE_LAUNCHER(int32_t, i32)
DECLARE_NEGATE_ND_AFFINE_LAUNCHER(int64_t, i64)
DECLARE_NEGATE_ND_AFFINE_LAUNCHER(__int128_t, i128)


/*
    RELU
*/

template <typename T>
__global__ void relu_nd_affine_kernel(
    T* __restrict__ data,
    size_t offset,
    const ptrdiff_t* __restrict__ stride,
    const size_t* __restrict__ shape,
    size_t rank,
    size_t size
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x
    ) {
        size_t linear = idx;
        ptrdiff_t phys = (ptrdiff_t)offset;

        for (int dim = (int)rank - 1; dim >= 0; --dim) {
            size_t coord = linear % shape[dim];
            linear /= shape[dim];
            phys += (ptrdiff_t)coord * stride[dim];
        }

        size_t final_idx = (size_t)phys;

        if(data[final_idx] < (T) 0) {
            data[final_idx] = (T) 0;
        }
        // data[final_idx] = -data[final_idx];
    }
}

template <typename T>
void launch_relu_nd_affine_op(
    T* data,
    size_t offset,
    const ptrdiff_t* stride,
    const size_t* shape,
    size_t rank,
    size_t size,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((size + block_size - 1) / block_size), 65535u);
    relu_nd_affine_kernel<T><<<grid, block_size>>>(data, offset, stride, shape, rank, size);
}

#define DECLARE_RELU_ND_AFFINE_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_relu_nd_affine_##SUFFIX( \
        TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, \
        size_t rank, size_t size, unsigned int block_size \
    ) { \
        launch_relu_nd_affine_op<TYPE>(data, offset, stride, shape, rank, size, block_size); \
    }

DECLARE_RELU_ND_AFFINE_LAUNCHER(float,  f32)
DECLARE_RELU_ND_AFFINE_LAUNCHER(double, f64)
DECLARE_RELU_ND_AFFINE_LAUNCHER(int8_t,  i8)
DECLARE_RELU_ND_AFFINE_LAUNCHER(int16_t, i16)
DECLARE_RELU_ND_AFFINE_LAUNCHER(int32_t, i32)
DECLARE_RELU_ND_AFFINE_LAUNCHER(int64_t, i64)
DECLARE_RELU_ND_AFFINE_LAUNCHER(__int128_t, i128)


template <typename T>
__global__ void sigmoid_nd_affine_kernel(
    T* __restrict__ data,
    size_t offset,
    const ptrdiff_t* __restrict__ stride,
    const size_t* __restrict__ shape,
    size_t rank,
    size_t size
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x
    ) {
        size_t linear = idx;
        ptrdiff_t phys = (ptrdiff_t)offset;

        for (int dim = (int)rank - 1; dim >= 0; --dim) {
            size_t coord = linear % shape[dim];
            linear /= shape[dim];
            phys += (ptrdiff_t)coord * stride[dim];
        }

        size_t final_idx = (size_t)phys;

        data[final_idx] = ((T) 1) / (((T) 1) + exp(-data[final_idx]));
        // data[final_idx] = -data[final_idx];
    }
}

template <typename T>
void launch_sigmoid_nd_affine_op(
    T* data,
    size_t offset,
    const ptrdiff_t* stride,
    const size_t* shape,
    size_t rank,
    size_t size,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((size + block_size - 1) / block_size), 65535u);
    sigmoid_nd_affine_kernel<T><<<grid, block_size>>>(data, offset, stride, shape, rank, size);
}

#define DECLARE_SIGMOID_ND_AFFINE_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_sigmoid_nd_affine_##SUFFIX( \
        TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, \
        size_t rank, size_t size, unsigned int block_size \
    ) { \
        launch_sigmoid_nd_affine_op<TYPE>(data, offset, stride, shape, rank, size, block_size); \
    }

DECLARE_SIGMOID_ND_AFFINE_LAUNCHER(float,  f32)
DECLARE_SIGMOID_ND_AFFINE_LAUNCHER(double, f64)




template <typename T>
__global__ void tanh_nd_affine_kernel(
    T* __restrict__ data,
    size_t offset,
    const ptrdiff_t* __restrict__ stride,
    const size_t* __restrict__ shape,
    size_t rank,
    size_t size
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x
    ) {
        size_t linear = idx;
        ptrdiff_t phys = (ptrdiff_t)offset;

        for (int dim = (int)rank - 1; dim >= 0; --dim) {
            size_t coord = linear % shape[dim];
            linear /= shape[dim];
            phys += (ptrdiff_t)coord * stride[dim];
        }

        size_t final_idx = (size_t)phys;

        T a = exp(data[final_idx]);
        T b = exp(-data[final_idx]);

        data[final_idx] = (a - b) / (a + b);
        // data[final_idx] = -data[final_idx];
    }
}

template <typename T>
void launch_tanh_nd_affine_op(
    T* data,
    size_t offset,
    const ptrdiff_t* stride,
    const size_t* shape,
    size_t rank,
    size_t size,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((size + block_size - 1) / block_size), 65535u);
    tanh_nd_affine_kernel<T><<<grid, block_size>>>(data, offset, stride, shape, rank, size);
}

#define DECLARE_TANH_ND_AFFINE_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_tanh_nd_affine_##SUFFIX( \
        TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, \
        size_t rank, size_t size, unsigned int block_size \
    ) { \
        launch_tanh_nd_affine_op<TYPE>(data, offset, stride, shape, rank, size, block_size); \
    }

DECLARE_TANH_ND_AFFINE_LAUNCHER(float,  f32)
DECLARE_TANH_ND_AFFINE_LAUNCHER(double, f64)