#include "../../../include/scalar.h"


template <typename T>
__global__ void negate_contiguous_kernel(
    T* __restrict__ data,
    size_t start,
    size_t len
) {
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("kernel launched: start=%zu len=%zu\n", start, len);
    //     fflush(stdout);
    // }
  
    // grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {
        
        size_t idx = start + i;
        data[idx] = -data[idx];
        // data[idx] = -data[idx];
    }
}

template <typename T>
void launch_negate_contiguous_op(
    T* data,
    size_t start,
    size_t len,
    unsigned int block_size
) {

    block_size = ALIGN_BLOCK_SIZE(block_size);

    

    // printf("Hi\n");
    // cudaDeviceSynchronize();

    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    negate_contiguous_kernel<T><<<grid, block_size>>>(data, start, len);

    // cudaError_t e2 = cudaGetLastError();
    // fprintf(stderr, "cudaGetLastError after launch: %s\n", cudaGetErrorString(e2));

    // printf("Done grid=%d, block_size=%d\n", grid, block_size);

    

    // cudaPointerAttributes attr;
    // cudaError_t e = cudaPointerGetAttributes(&attr, data);
    // fprintf(stderr, "cudaPointerGetAttributes: %s\n", cudaGetErrorString(e));
    // if (e == cudaSuccess) {
    //     #if CUDART_VERSION >= 10000
    //         fprintf(stderr, "  type=%d (1=host,2=device,3=managed)\n", (int)attr.type);
    //     #else
    //         fprintf(stderr, "  memoryType=%d (1=host,2=device)\n", (int)attr.memoryType);
    //     #endif
    // }

    // cudaDeviceSynchronize();
    // fflush(stdout);
    // fflush(stderr);
}


template <typename T>
__global__ void relu_contiguous_kernel(
    T* __restrict__ data,
    size_t start,
    size_t len
) {
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("kernel launched: start=%zu len=%zu\n", start, len);
    //     fflush(stdout);
    // }
  
    // grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {
        
        size_t idx = start + i;
        if (data[idx] < (T) 0) {
            data[idx] = 0;
        }
        // data[idx] = (T)0;
        // data[idx] = -data[idx];
    }
}

template <typename T>
void launch_relu_contiguous_op(
    T* data,
    size_t start,
    size_t len,
    unsigned int block_size
) {

    block_size = ALIGN_BLOCK_SIZE(block_size);
    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    relu_contiguous_kernel<T><<<grid, block_size>>>(data, start, len);
}

#define DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_negate_contiguous_##SUFFIX( \
        TYPE* data, size_t start, size_t len, unsigned int block_size \
    ) { \
        launch_negate_contiguous_op<TYPE>(data, start, len, block_size); \
    }

#define DECLARE_RELU_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_relu_contiguous_##SUFFIX( \
        TYPE* data, size_t start, size_t len, unsigned int block_size \
    ) { \
        launch_relu_contiguous_op<TYPE>(data, start, len, block_size); \
    }

DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(float,  f32)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(double, f64)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int8_t,  i8)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int16_t, i16)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int32_t, i32)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(int64_t, i64)
DECLARE_NEGATE_CONTIGUOUS_LAUNCHER(__int128_t, i128)

DECLARE_RELU_CONTIGUOUS_LAUNCHER(float,  f32)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(double, f64)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(int8_t,  i8)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(int16_t, i16)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(int32_t, i32)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(int64_t, i64)
DECLARE_RELU_CONTIGUOUS_LAUNCHER(__int128_t, i128)



template <typename T>
__global__ void sigmoid_contiguous_kernel(
    T* __restrict__ data,
    size_t start,
    size_t len
) {
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("kernel launched: start=%zu len=%zu\n", start, len);
    //     fflush(stdout);
    // }
  
    // grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {
        
        size_t idx = start + i;
        data[idx] = ((T) 1) / (((T) 1) + exp(-data[idx]));
        // data[idx] = (T)0;
        // data[idx] = -data[idx];
    }
}

template <typename T>
void launch_sigmoid_contiguous_op(
    T* data,
    size_t start,
    size_t len,
    unsigned int block_size
) {

    block_size = ALIGN_BLOCK_SIZE(block_size);
    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    sigmoid_contiguous_kernel<T><<<grid, block_size>>>(data, start, len);
}

#define DECLARE_SIGMOID_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_sigmoid_contiguous_##SUFFIX( \
        TYPE* data, size_t start, size_t len, unsigned int block_size \
    ) { \
        launch_sigmoid_contiguous_op<TYPE>(data, start, len, block_size); \
    }


DECLARE_SIGMOID_CONTIGUOUS_LAUNCHER(float,  f32)
DECLARE_SIGMOID_CONTIGUOUS_LAUNCHER(double, f64)

template <typename T>
__global__ void tanh_contiguous_kernel(
    T* __restrict__ data,
    size_t start,
    size_t len
) {
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("kernel launched: start=%zu len=%zu\n", start, len);
    //     fflush(stdout);
    // }
  
    // grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {
        
        size_t idx = start + i;

        T a = exp(data[idx]);
        T b = exp(-data[idx]);

        data[idx] = (a - b) / (a + b);

        // data[idx] = ((T) 1) / (((T) 1) + exp(-data[idx]));
        // data[idx] = (T)0;
        // data[idx] = -data[idx];
    }
}

template <typename T>
void launch_tanh_contiguous_op(
    T* data,
    size_t start,
    size_t len,
    unsigned int block_size
) {

    block_size = ALIGN_BLOCK_SIZE(block_size);
    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    tanh_contiguous_kernel<T><<<grid, block_size>>>(data, start, len);
}

#define DECLARE_TANH_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_tanh_contiguous_##SUFFIX( \
        TYPE* data, size_t start, size_t len, unsigned int block_size \
    ) { \
        launch_tanh_contiguous_op<TYPE>(data, start, len, block_size); \
    }


DECLARE_TANH_CONTIGUOUS_LAUNCHER(float,  f32)
DECLARE_TANH_CONTIGUOUS_LAUNCHER(double, f64)
