#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;


#[cfg(feature = "cuda")]
/// Trait for types that can be stored in tensors (with CUDA support).
/// 
/// Implemented for standard numeric types: f32, f64, i8-i128, u8-u128, isize, usize, bool.
pub trait TensorValue: 
    Copy + 
    Default +
    TensorDefault +
    DeviceRepr +
    Send + Sync +
    std::ops::Add<Output = Self> + 
    std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> +
    'static
{
    const DTYPE: crate::core::value::Dtypes;
}

#[cfg(not(feature = "cuda"))]
/// Trait for types that can be stored in tensors.
/// 
/// Implemented for standard numeric types: f32, f64, i8-i128, u8-u128, isize, usize, bool.
pub trait TensorValue: 
    Copy + 
    Default +
    TensorDefault +
    Send + Sync +
    std::ops::Add<Output = Self> + 
    std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> +
    'static
{
    const DTYPE: crate::core::value::Dtypes;
}

/// Provides default constant values for tensor element types.
pub trait TensorDefault {
    /// Returns the zero value.
    fn zero() -> Self;
    
    /// Returns the one value.
    fn one() -> Self;
    
    /// Returns the minimum value.
    fn min() -> Self;
    
    /// Returns the maximum value.
    fn max() -> Self;
}


macro_rules! impl_tensor_values {
    ($(($type:ty, $dtype:expr)),+ $(,)?) => {
        $(
            impl TensorValue for $type {
                const DTYPE: crate::core::value::Dtypes = $dtype;
            }
        )+
    };
}

macro_rules! impl_default {
    ($type:ty, $zero:expr, $one:expr, $min:expr, $max:expr) => {
        impl TensorDefault for $type {
            fn zero() -> Self {$zero}
            fn one() -> Self {$one}
            fn min() -> Self {$min}
            fn max() -> Self {$max}
        }
    };
}

impl_tensor_values!((f32, Dtypes::F32), (f64, Dtypes::F64), (i8, Dtypes::I8), (i16, Dtypes::I16), (i32, Dtypes::I32), (i64, Dtypes::I64), (i128, Dtypes::I128), (isize, Dtypes::I64), (u8, Dtypes::U8), (u16, Dtypes::U16), (u32, Dtypes::U32), (u64, Dtypes::U64), (u128, Dtypes::U128), (usize, Dtypes::U64));

impl_default!(f32, 0.0f32, 1.0f32, f32::MIN, f32::MAX);
impl_default!(f64, 0.0f64, 1.0f64, f64::MIN, f64::MAX);
impl_default!(i8, 0i8, 1i8, i8::MIN, i8::MAX);
impl_default!(i16, 0i16, 1i16, i16::MIN, i16::MAX);
impl_default!(i32, 0i32, 1i32, i32::MIN, i32::MAX);
impl_default!(i64, 0i64, 1i64, i64::MIN, i64::MAX);
impl_default!(i128, 0i128, 1i128, i128::MIN, i128::MAX);
impl_default!(isize, 0isize, 1isize, isize::MIN, isize::MAX);
impl_default!(u8, 0u8, 1u8, u8::MIN, u8::MAX);
impl_default!(u16, 0u16, 1u16, u16::MIN, u16::MAX);
impl_default!(u32, 0u32, 1u32, u32::MIN, u32::MAX);
impl_default!(u64, 0u64, 1u64, u64::MIN, u64::MAX);
impl_default!(u128, 0u128, 1u128, u128::MIN, u128::MAX);
impl_default!(usize, 0usize, 1usize, usize::MIN, usize::MAX);
impl_default!(bool, false, true, false, true);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtypes {
    U8,
    I8,
    U16,
    I16,
    U32,
    U128,
    I32,
    U64,
    I64,
    I128,
    F16,
    F32,
    F64,
}

