

use crate::{core::{tensor::TensorError, value::{TensorValue, TensorValueElementwise}, MetaTensor, MetaTensorView}, ops::unary::ElementwiseTensorOp};

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_tests;

pub trait Backend<T: TensorValue> {
    type Buf;

    fn alloc_from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError>;
    fn alloc(&self, len: usize) -> Result<Self::Buf, TensorError>;
    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), TensorError>;
    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, TensorError>;
    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), TensorError>;
    fn len(&self, buf: &Self::Buf) -> usize;
    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError>;
    fn dump(&self, src: &Self::Buf) -> Result<Box<[T]>, TensorError>;
    fn new() -> Self;
}

pub trait BackendUnaryElementwise<T: TensorValue + TensorValueElementwise>: Backend<T> {
    fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_elementwise_1d_strided(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_elementwise_nd(
        &self,
        buf: &mut Self::Buf,
        op: &ElementwiseTensorOp<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError>;

    fn apply_elementwise(&self, buf: &mut Self::Buf, op: ElementwiseTensorOp<T>, meta: &MetaTensor) -> Result<(), TensorError>{
        if meta.is_contiguous() {
            return self.apply_elementwise_contiguous(buf, &op, meta.offset, meta.size())
        }

        let non_singleton_dims = meta.non_singleton_dims();
        if non_singleton_dims.len() == 1 {
            // essentially 1D
            // optimial because we can just stride along the single non-singleton dimension
            let (_, dim_size, dim_stride) = non_singleton_dims[0];
            return self.apply_elementwise_1d_strided(
                buf, 
                &op, 
                meta.offset, 
                dim_stride, 
                dim_size
            )
        }

        // worst case because we have to handle full nD striding
        // better than a full gather scatter
        self.apply_elementwise_nd(
            buf,
            &op,
            meta.offset,
            &meta.shape,
            &meta.stride,
        )
    }
}

pub trait BackendBinaryElementwise<T: TensorValue + TensorValueElementwise>: Backend<T> {
    // / Rules of the merge operation:
    // / - left and right regions, when combined, cover the same number of indices
    // / - left and right regions, when combined, cover the same number of indices as the dst regions
    // / - dst addresses do not overlap with left or right addresses
    // / 
    // / In general, broadcasting rukes with regard to shape are:
    // / Traversing from most minor to most major dimension:
    // / - if dimensions are equal, they align
    // / - if one dimension is 1, it is broadcast to the other dimension
    // / - if one dimension is missing, it is treated as 1 and broadcast to the other dimension
    // / 
    // / The memory layouts can be arbitrary, as long as the above rules are followed.
    // fn merge(
    //     &self, 
    //     left: (&Self::Buf, &[MemRegion]), 
    //     right: (&Self::Buf, &[MemRegion]),
    //     dst: (&mut Self::Buf, &[MemRegion]),
    //     op: ElementwiseTensorOp<T>
    // ) -> Result<(), TensorError>;
}