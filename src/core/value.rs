
pub trait TensorValue: 
    Copy + 
    Default +
{}

pub trait TensorValueElementwise: 
    TensorValue + 
    std::ops::Add<Output = Self> + 
    std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self>
{}

impl TensorValue for f32 {}
impl TensorValue for f64 {}
impl TensorValue for i8 {}
impl TensorValue for i16 {}
impl TensorValue for i32 {}
impl TensorValue for i64 {}
impl TensorValue for i128 {}
impl TensorValue for isize {}
impl TensorValue for u8 {}
impl TensorValue for u16 {}
impl TensorValue for u32 {}
impl TensorValue for u64 {}
impl TensorValue for u128 {}
impl TensorValue for usize {}
impl TensorValue for bool {}
impl TensorValue for char {}

impl TensorValueElementwise for f32 {}
impl TensorValueElementwise for f64 {}
impl TensorValueElementwise for i8 {}
impl TensorValueElementwise for i16 {}
impl TensorValueElementwise for i32 {}
impl TensorValueElementwise for i64 {}
impl TensorValueElementwise for i128 {}
impl TensorValueElementwise for u8 {}
impl TensorValueElementwise for u16 {}
impl TensorValueElementwise for u32 {}
impl TensorValueElementwise for u64 {}
impl TensorValueElementwise for u128 {}
