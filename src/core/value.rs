
pub trait TensorValue: 
    Copy + 
    Default +
{}

pub trait TensorValueUnary: 
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
impl TensorValue for u8 {}
impl TensorValue for u16 {}
impl TensorValue for u32 {}
impl TensorValue for u64 {}
impl TensorValue for u128 {}
impl TensorValue for bool {}
impl TensorValue for char {}

impl TensorValueUnary for f32 {}
impl TensorValueUnary for f64 {}
impl TensorValueUnary for i8 {}
impl TensorValueUnary for i16 {}
impl TensorValueUnary for i32 {}
impl TensorValueUnary for i64 {}
impl TensorValueUnary for i128 {}
impl TensorValueUnary for u8 {}
impl TensorValueUnary for u16 {}
impl TensorValueUnary for u32 {}
impl TensorValueUnary for u64 {}
impl TensorValueUnary for u128 {}
