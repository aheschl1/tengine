#include "../../../include/scalar.h"

template <typename T>
__global__ void negate_strided_kernel(
    T* __restrict__ data,
    size_t offset,
    ptrdiff_t stride,
    size_t len
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {

        size_t idx = (size_t)((ptrdiff_t)offset + (ptrdiff_t)i * stride);
        data[idx] = -data[idx];
    }
}

template <typename T>
void launch_negate_strided_op(
    T* data,
    size_t offset,
    ptrdiff_t stride,
    size_t len,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    negate_strided_kernel<T><<<grid, block_size>>>(data, offset, stride, len);
}

#define DECLARE_NEGATE_STRIDED_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_negate_strided_##SUFFIX( \
        TYPE* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size \
    ) { \
        launch_negate_strided_op<TYPE>(data, offset, stride, len, block_size); \
    }

template <typename T>
__global__ void relu_strided_kernel(
    T* __restrict__ data,
    size_t offset,
    ptrdiff_t stride,
    size_t len
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {

        size_t idx = (size_t)((ptrdiff_t)offset + (ptrdiff_t)i * stride);
        // data[idx] = -data[idx];
        if (data[idx] < (T)0) {
            data[idx] = (T) 0;
        }
    }
}

template <typename T>
void launch_relu_strided_op(
    T* data,
    size_t offset,
    ptrdiff_t stride,
    size_t len,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    relu_strided_kernel<T><<<grid, block_size>>>(data, offset, stride, len);
}

#define DECLARE_RELU_STRIDED_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_relu_strided_##SUFFIX( \
        TYPE* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size \
    ) { \
        launch_relu_strided_op<TYPE>(data, offset, stride, len, block_size); \
    }


DECLARE_NEGATE_STRIDED_LAUNCHER(float,  f32)
DECLARE_NEGATE_STRIDED_LAUNCHER(double, f64)
DECLARE_NEGATE_STRIDED_LAUNCHER(int8_t,  i8)
DECLARE_NEGATE_STRIDED_LAUNCHER(int16_t, i16)
DECLARE_NEGATE_STRIDED_LAUNCHER(int32_t, i32)
DECLARE_NEGATE_STRIDED_LAUNCHER(int64_t, i64)
DECLARE_NEGATE_STRIDED_LAUNCHER(__int128_t, i128)

DECLARE_RELU_STRIDED_LAUNCHER(float,  f32)
DECLARE_RELU_STRIDED_LAUNCHER(double, f64)
DECLARE_RELU_STRIDED_LAUNCHER(int8_t,  i8)
DECLARE_RELU_STRIDED_LAUNCHER(int16_t, i16)
DECLARE_RELU_STRIDED_LAUNCHER(int32_t, i32)
DECLARE_RELU_STRIDED_LAUNCHER(int64_t, i64)
DECLARE_RELU_STRIDED_LAUNCHER(__int128_t, i128)



template <typename T>
__global__ void sigmoid_strided_kernel(
    T* __restrict__ data,
    size_t offset,
    ptrdiff_t stride,
    size_t len
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {

        size_t idx = (size_t)((ptrdiff_t)offset + (ptrdiff_t)i * stride);
        // data[idx] = -data[idx];
        data[idx] = ((T) 1) / (((T) 1) + exp(-data[idx]));
    }
}

template <typename T>
void launch_sigmoid_strided_op(
    T* data,
    size_t offset,
    ptrdiff_t stride,
    size_t len,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    sigmoid_strided_kernel<T><<<grid, block_size>>>(data, offset, stride, len);
}

#define DECLARE_SIGMOID_STRIDED_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_sigmoid_strided_##SUFFIX( \
        TYPE* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size \
    ) { \
        launch_sigmoid_strided_op<TYPE>(data, offset, stride, len, block_size); \
    }



DECLARE_SIGMOID_STRIDED_LAUNCHER(float,  f32)
DECLARE_SIGMOID_STRIDED_LAUNCHER(double, f64)


template <typename T>
__global__ void tanh_strided_kernel(
    T* __restrict__ data,
    size_t offset,
    ptrdiff_t stride,
    size_t len
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {

        size_t idx = (size_t)((ptrdiff_t)offset + (ptrdiff_t)i * stride);
        // data[idx] = -data[idx];
        T a = exp(data[idx]);
        T b = exp(-data[idx]);

        data[idx] = (a - b) / (a + b);
    }
}

template <typename T>
void launch_tanh_strided_op(
    T* data,
    size_t offset,
    ptrdiff_t stride,
    size_t len,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    tanh_strided_kernel<T><<<grid, block_size>>>(data, offset, stride, len);
}

#define DECLARE_TANH_STRIDED_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_tanh_strided_##SUFFIX( \
        TYPE* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size \
    ) { \
        launch_tanh_strided_op<TYPE>(data, offset, stride, len, block_size); \
    }



DECLARE_TANH_STRIDED_LAUNCHER(float,  f32)
DECLARE_TANH_STRIDED_LAUNCHER(double, f64)