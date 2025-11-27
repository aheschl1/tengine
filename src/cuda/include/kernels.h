#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
// Generic template launcher with configurable block size
template <typename T>
void launch_unary_op(T* data, size_t* offsets, size_t n, uint8_t op, T value, unsigned int block_size);

extern "C" {
#endif

// C wrappers for FFI - block_size parameter allows tuning from Rust
void launch_unary_f32(float* data, size_t* offsets, size_t n, uint8_t op, float value, unsigned int block_size);
void launch_unary_f64(double* data, size_t* offsets, size_t n, uint8_t op, double value, unsigned int block_size);
void launch_unary_u8(uint8_t* data, size_t* offsets, size_t n, uint8_t op, uint8_t value, unsigned int block_size);
void launch_unary_u16(uint16_t* data, size_t* offsets, size_t n, uint8_t op, uint16_t value, unsigned int block_size);
void launch_unary_u32(uint32_t* data, size_t* offsets, size_t n, uint8_t op, uint32_t value, unsigned int block_size);
void launch_unary_u64(uint64_t* data, size_t* offsets, size_t n, uint8_t op, uint64_t value, unsigned int block_size);
void launch_unary_u128(__uint128_t* data, size_t* offsets, size_t n, uint8_t op, __uint128_t value, unsigned int block_size);
void launch_unary_i8(int8_t* data, size_t* offsets, size_t n, uint8_t op, int8_t value, unsigned int block_size);
void launch_unary_i16(int16_t* data, size_t* offsets, size_t n, uint8_t op, int16_t value, unsigned int block_size);
void launch_unary_i32(int32_t* data, size_t* offsets, size_t n, uint8_t op, int32_t value, unsigned int block_size);
void launch_unary_i64(int64_t* data, size_t* offsets, size_t n, uint8_t op, int64_t value, unsigned int block_size);
void launch_unary_i128(__int128_t* data, size_t* offsets, size_t n, uint8_t op, __int128_t value, unsigned int block_size);

#ifdef __cplusplus
}
#endif
