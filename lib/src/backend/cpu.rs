
use crate::{backend::{Backend, BackendUnaryElementwise}, core::{meta::TensorOffsetIterator, tensor::TensorError, value::{TensorValue, TensorValueElementwise}}, ops::unary::ElementwiseTensorOp};

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
    
    fn alloc_from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError> {
        Ok(src)
    }
    
    fn len(&self, buf: &Self::Buf) -> usize {
        buf.len()
    }
    
    fn new() -> Self {
        Self
    }

    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError> {
        let mut dst = self.alloc(src.len())?;
        dst.copy_from_slice(src);
        Ok(dst)
    }
    
    fn dump(&self, src: &Self::Buf) -> Result<Box<[T]>, TensorError> {
        Ok(src.clone())
    }
}

impl<T: TensorValue + TensorValueElementwise> BackendUnaryElementwise<T> for Cpu {
    fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        for item in bufptr.iter_mut().skip(start).take(len) {
            *item = op.apply(*item);
        }
        Ok(())
    }
    
    fn apply_elementwise_1d_strided(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        let mut idx: isize = offset as isize;
        for _ in 0..len {
            bufptr[idx as usize] = op.apply(bufptr[idx as usize]);
            idx += stride;
        }
        Ok(())
    }
    
    fn apply_elementwise_nd(
        &self,
        buf: &mut Self::Buf,
        op: &ElementwiseTensorOp<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        let mut iterator = TensorOffsetIterator::new(
            shape.to_vec(),
            stride.to_vec(),
            offset,
        );
        for idx in iterator {
            bufptr[idx] = op.apply(bufptr[idx]);
        }
        Ok(())
    }
}

// impl<T> BackendBinaryElementwise<T> for Cpu 
// where T: TensorValue + TensorValueElementwise
// {
//     fn merge(
//         &self, 
//         left: (&Self::Buf, &[crate::core::meta::MemRegion]), 
//         right: (&Self::Buf, &[crate::core::meta::MemRegion]),
//         dst: (&mut Self::Buf, &[crate::core::meta::MemRegion]),
//         op: ElementwiseTensorOp<T>
//     ) -> Result<(), TensorError> {
//         todo!()
//     }
// }