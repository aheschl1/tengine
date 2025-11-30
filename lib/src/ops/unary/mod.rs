use crate::core::value::TensorValue;

pub mod add;
pub mod sub;
pub mod mul;

pub enum ElementwiseTensorOp<T> 
where T: TensorValue
{
    Add(T),
    Sub(T),
    Mul(T),
}

impl<T> ElementwiseTensorOp<T> 
where T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + TensorValue
{
    #[inline(always)]
    pub fn apply(&self, x: T) -> T {
        match self {
            ElementwiseTensorOp::Add(val) => x + *val,
            ElementwiseTensorOp::Sub(val) => x - *val,
            ElementwiseTensorOp::Mul(val) => x * *val,
        }
    }
}

#[cfg(feature = "cuda")]
impl<T> ElementwiseTensorOp<T>
where T: TensorValue
{
    /// Convert operation to op code (0=Add, 1=Sub, 2=Mul)
    pub fn to_op_code(&self) -> u8 {
        match self {
            ElementwiseTensorOp::Add(_) => 0,
            ElementwiseTensorOp::Sub(_) => 1,
            ElementwiseTensorOp::Mul(_) => 2,
        }
    }

    /// Get the operation value
    pub fn value(&self) -> T {
        match self {
            ElementwiseTensorOp::Add(v) => *v,
            ElementwiseTensorOp::Sub(v) => *v,
            ElementwiseTensorOp::Mul(v) => *v,
        }
    }
}
