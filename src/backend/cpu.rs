use crate::{backend::Backend, core::{primitives::TensorValue, tensor::TensorError}};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Cpu;

impl<T: TensorValue> Backend<T> for Cpu {
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
        Ok(*buf.get(offset).ok_or(
            TensorError::IdxOutOfBounds
        )?)
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
    
    fn apply_each<F>(&self, buf: &mut Self::Buf, f: F, offsets: impl Iterator<Item = usize>) -> Result<(), TensorError>
    where
        F: Fn(T) -> T {
        for offset in offsets {
            let value = self.read(buf, offset)?;
            let new_value = f(value);
            self.write(buf, offset, new_value)?;
        }
        Ok(())
    }
    
    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError> {
        let mut dst = self.alloc(src.len())?;
        dst.copy_from_slice(src);
        Ok(dst)
    }
}
