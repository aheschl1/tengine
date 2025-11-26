
use crate::core::{primitives::TensorValue, tensor::TensorError};

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_tests;

pub trait Backend<T: TensorValue> {
    type Buf;

    fn from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError>;
    fn alloc(&self, len: usize) -> Result<Self::Buf, TensorError>;
    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), TensorError>;
    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, TensorError>;
    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), TensorError>;
    fn len(&self, buf: &Self::Buf) -> usize;
    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError>;

    fn apply_each<F>(&self, buf: &mut Self::Buf, f: F, offsets: impl Iterator<Item = usize>) -> Result<(), TensorError>
    where
        F: Fn(T) -> T;
    
    fn new() -> Self;
}
