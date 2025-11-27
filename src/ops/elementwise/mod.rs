use crate::core::value::TensorValue;

pub mod add;
pub mod sub;
pub mod mul;

pub enum UnaryTensorOp<T> 
where T: TensorValue
{
    Add(T),
    Sub(T),
    Mul(T),
}

impl<T> UnaryTensorOp<T> 
where T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + TensorValue
{
    pub fn apply(&self, x: T) -> T {
        match self {
            UnaryTensorOp::Add(val) => x + *val,
            UnaryTensorOp::Sub(val) => x - *val,
            UnaryTensorOp::Mul(val) => x * *val,
        }
    }
}

#[cfg(feature = "cuda")]
impl<T> UnaryTensorOp<T>
where T: TensorValue
{
    /// Convert operation to op code (0=Add, 1=Sub, 2=Mul)
    pub fn to_op_code(&self) -> u8 {
        match self {
            UnaryTensorOp::Add(_) => 0,
            UnaryTensorOp::Sub(_) => 1,
            UnaryTensorOp::Mul(_) => 2,
        }
    }

    /// Get the operation value
    pub fn value(&self) -> T {
        match self {
            UnaryTensorOp::Add(v) => *v,
            UnaryTensorOp::Sub(v) => *v,
            UnaryTensorOp::Mul(v) => *v,
        }
    }
}
