use crate::{backend::Backend, core::tensor::TensorError};

#[derive(Debug, PartialEq, Eq)]
pub struct Cpu;

impl<T: Copy + Default> Backend<T> for Cpu {
    type Buf = Box<[T]>;



    fn alloc(&self, len: usize) -> Result<Box<[T]>, TensorError> {
        Ok(vec![T::default(); len].into())
    }

    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), TensorError> {
        if dst.len() != src.len() {
            return Err(TensorError::SizeMismatch);
        }
        dst.copy_from_slice(src);
        Ok(())
    }

    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, TensorError> {
        Ok(buf.get(offset).ok_or(
            TensorError::IdxOutOfBounds
        )?.clone())
    }

    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), TensorError> {
        match buf.get_mut(offset) {
            Some(slot) => {
                *slot = value;
                Ok(())
            }
            None => Err(TensorError::IdxOutOfBounds),
        }
    }
    
    fn from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError> {
        Ok(src)
    }
    
    fn len(&self, buf: &Self::Buf) -> usize {
        buf.len()
    }
    
    fn new() -> Self {
        Self
    }
}
