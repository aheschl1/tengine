use std::fmt;

use crate::core::tensor::TensorError;

pub mod cpu;

pub trait Backend<T> {
    type Buf;

    fn from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError>;

    fn alloc(&self, len: usize) -> Result<Self::Buf, TensorError>;

    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), TensorError>;
    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, TensorError>;

    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), TensorError>;

    fn len(&self, buf: &Self::Buf) -> usize;

    fn new() -> Self;
}
