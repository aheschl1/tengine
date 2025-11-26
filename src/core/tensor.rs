
use crate::{backend::{Backend, cpu::Cpu}, core::{CpuTensor, CpuTensorView, Dim, MetaTensor, Shape, Stride, TensorView, TensorViewMut, idx::Idx, primitives::{CpuTensorViewMut, TensorValue}}};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum TensorError {
    #[error("index out of bounds")]
    IdxOutOfBounds,

    #[error("wrong number of dimensions")]
    WrongDims,

    #[error("invalid tensor shape")]
    InvalidShape,

    #[error("invalid dimension")]
    InvalidDim,

    #[error("size mismatch between tensors")]
    SizeMismatch,

    #[error("backend error: {0}")]
    BackendError(String),

    #[cfg(feature = "cuda")]
    #[error("cuda error: {0}")]
    CudaError(String),
}

pub trait AsView<T: TensorValue, B: Backend<T>> {
    /// Returns an immutable view over the tensor data, sharing the same
    /// underlying buffer and metadata (shape/stride/offset) without copying.
    fn view(&self) -> TensorView<'_, T, B>;
}

pub trait AsViewMut<T: TensorValue, B: Backend<T>> : AsView<T, B> {
    /// Returns a mutable view over the tensor data, sharing the same
    /// underlying buffer and metadata (shape/stride/offset) without copying.
    fn view_mut(&mut self) -> TensorViewMut<'_, T, B>;
}

impl<T: TensorValue> AsView<T, Cpu> for CpuTensor<T> {
    fn view(&self) -> CpuTensorView<'_, T> {
        CpuTensorView::from_parts(
            &self.raw, 
            &self.backend, 
            self.meta.clone()
        )
    }
}

impl<T: TensorValue> AsViewMut<T, Cpu> for CpuTensor<T> {
    fn view_mut<'a>(&'a mut self) -> CpuTensorViewMut<'a, T> {
        CpuTensorViewMut::from_parts(
            &mut self.raw, 
            &self.backend, 
            self.meta.clone()
        )
    }
}

#[cfg(feature = "cuda")]
impl<T: TensorValue + cudarc::driver::DeviceRepr> AsView<T, crate::backend::cuda::CudaBackend> for crate::core::primitives::CudaTensor<T> {
    fn view(&self) -> crate::core::primitives::CudaTensorView<'_, T> {
        crate::core::primitives::CudaTensorView::from_parts(
            &self.raw, 
            &self.backend, 
            self.meta.clone()
        )
    }
}

#[cfg(feature = "cuda")]
impl<T: TensorValue + cudarc::driver::DeviceRepr> AsViewMut<T, crate::backend::cuda::CudaBackend> for crate::core::primitives::CudaTensor<T> {
    fn view_mut<'a>(&'a mut self) -> crate::core::primitives::CudaTensorViewMut<'a, T> {
        crate::core::primitives::CudaTensorViewMut::from_parts(
            &mut self.raw, 
            &self.backend, 
            self.meta.clone()
        )
    }
}

impl<T: TensorValue, B: Backend<T>> AsView<T, B> for TensorView<'_, T, B> 
{
    fn view<'a>(&'a self) -> TensorView<'a, T, B> {
        TensorView::from_parts(
            self.raw, 
            self.backend,
            self.meta.clone()
        )
    }
}


pub trait TensorAccess<T: TensorValue, B: Backend<T>>: Sized {
    /// Get element at given index
    fn get<'a, I: Into<Idx<'a>>>(&self, idx: I) -> Result<T, TensorError>;

    fn item(&self) -> Result<T, TensorError> {
        self.get(Idx::Item)
    }
    /// Create a slice/view of the tensor along a specific dimension at a given index
    fn slice<'a>(&'a self, dim: Dim, idx: Dim) -> Result<TensorView<'a, T, B>, TensorError> where Self: Sized;
}

pub trait TensorAccessMut<T: TensorValue, B: Backend<T>>: TensorAccess<T, B> {
    /// Slice mutable tensor to get a mutable view
    fn slice_mut<'a>(&'a mut self, dim: Dim, idx: Dim) -> Result<TensorViewMut<'a, T, B>, TensorError> where Self: Sized;
    /// sets a value at given index
    fn set<'a, I: Into<Idx<'a>>>(&mut self, idx: I, value: T) -> Result<(), TensorError>;
}

impl<T: TensorValue, B: Backend<T>> TensorAccess<T, B> for TensorView<'_, T, B>
where B: Backend<T>
{
    /// Returns a reference to the element at a logical index, converting
    /// coordinates into a buffer position via stride and offset.
    ///
    /// Errors
    /// - `WrongDims` if the index rank doesn't match the tensor rank.
    /// - `IdxOutOfBounds` if the computed buffer index is outside the backing slice.
    fn get<'a, I: Into<Idx<'a>>>(&self, idx: I) -> Result<T, TensorError> {
        let idx = logical_to_buffer_idx(&idx.into(), self.meta.stride(), self.meta.offset())?;
        self.backend.read(self.raw, idx)
    }

    /// Creates a new immutable view by fixing `dim` to `idx`, effectively
    /// removing that dimension and adjusting shape/stride/offset accordingly.
    ///
    /// Errors
    /// - `InvalidDim` if `dim` is out of range.
    /// - `IdxOutOfBounds` if `idx` exceeds the size of `dim`.
    fn slice(&self, dim: Dim, idx: Dim) -> Result<TensorView<'_, T, B>, TensorError> where Self: Sized {
        let (new_shape, new_stride, offset) = compute_sliced_parameters(
            self.meta.shape(), 
            self.meta.stride(), 
            &self.meta.offset(),
            dim, 
            idx
        )?;
        
        let v = TensorView::from_parts(self.raw, self.backend, MetaTensor::new(new_shape, new_stride, offset));
        Ok(v)
    }
}

impl<T: TensorValue, B: Backend<T>> TensorAccess<T, B> for TensorViewMut<'_, T, B>
where B: Backend<T>
{
    /// Returns a reference to the element at a logical index, converting
    /// coordinates into a buffer position via stride and offset.
    ///
    /// Errors
    /// - `WrongDims` if the index rank doesn't match the tensor rank.
    /// - `IdxOutOfBounds` if the computed buffer index is outside the backing slice.
    fn get<'a, I: Into<Idx<'a>>>(&self, idx: I) -> Result<T, TensorError> {
        let idx = logical_to_buffer_idx(&idx.into(), self.meta.stride(), self.meta.offset())?;
        self.backend.read(self.raw, idx)
    }

    /// Creates a new immutable view by fixing `dim` to `idx`, effectively
    /// removing that dimension and adjusting shape/stride/offset accordingly.
    ///
    /// Errors
    /// - `InvalidDim` if `dim` is out of range.
    /// - `IdxOutOfBounds` if `idx` exceeds the size of `dim`.
    fn slice(&self, dim: Dim, idx: Dim) -> Result<TensorView<'_, T, B>, TensorError> where Self: Sized {
        let (new_shape, new_stride, offset) = compute_sliced_parameters(
            self.meta.shape(), 
            self.meta.stride(), 
            &self.meta.offset(),
            dim, 
            idx
        )?;
        
        let v = TensorView::from_parts(self.raw, self.backend, MetaTensor::new(new_shape, new_stride, offset));
        Ok(v)
    }
}


impl<T: TensorValue, B: Backend<T>> TensorAccessMut<T, B> for TensorViewMut<'_, T, B>
{
    /// Creates a new mutable view by fixing `dim` to `idx`, effectively
    /// removing that dimension and adjusting shape/stride/offset accordingly.
    ///
    /// Errors
    /// - `InvalidDim` if `dim` is out of range.
    /// - `IdxOutOfBounds` if `idx` exceeds the size of `dim`.
    fn slice_mut(&mut self, dim: Dim, idx: Dim) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        let (new_shape, new_stride, offset) =
            compute_sliced_parameters(self.meta.shape(), self.meta.stride(), &self.meta.offset(), dim, idx)?;
    
        Ok(TensorViewMut::from_parts(self.raw, self.backend, MetaTensor::new(new_shape, new_stride, offset)))
    }
    
    fn set<'a, I: Into<Idx<'a>>>(&mut self, idx: I, value: T) -> Result<(), TensorError> {
        let idx = idx.into();
        let buf_idx = logical_to_buffer_idx(&idx, self.meta.stride(), self.meta.offset())?;
        self.backend.write(self.raw, buf_idx, value)
    }

}


// impl<'a, T, B, S: Into<Idx<'a>>> Index<S> for TensorViewBase<'a, T, B> 
//     where B: AsRef<[T]> + 'a
// {
//     type Output = T;

//     /// Indexes the view at a logical index and returns a reference.
//     ///
//     /// Panics
//     /// - If the index is out of bounds or has the wrong rank.
//     fn index(&self, index: S) -> &Self::Output {
//         self.get(&index.into()).unwrap()
//     }
// }

// impl<'a, T, S: Into<Idx<'a>>> IndexMut<S> for CpuTensorViewMut<'a, T> {
//     /// Indexes the view at a logical index and returns a mutable reference.
//     ///
//     /// Panics
//     /// - If the index is out of bounds or has the wrong rank.
//     fn index_mut(&mut self, index: S) -> &mut Self::Output {
//         self.get_mut(&index.into()).unwrap()
//     }
// }


/// Converts a logical index (coordinate, single position, or scalar) into a
/// linear buffer index using the provided stride and offset.
///
/// Behavior
/// - `Coord(&[d0, d1, ...])` computes `offset + sum(di*stride[i])`.
/// - `At(i)` is treated as `Coord(&[i])`.
/// - `Item` is only valid for scalars (rank 0).
///
/// Errors
/// - `WrongDims` if index rank differs from stride length, or `Item` is used on non-scalars.
/// - `IdxOutOfBounds` is not checked here (caller validates against buffer length).
fn logical_to_buffer_idx(idx: &Idx, stride: &Stride, offset: usize) -> Result<usize, TensorError> {
    match idx {
        Idx::Coord(idx) => {
            if idx.len() != stride.len() {
                Err(TensorError::WrongDims)
            }else{
                Ok(idx
                    .iter()
                    .zip(stride)
                    .fold(offset, |acc, (a, b)| acc + *a*b))
            }
        },
        Idx::Item => {
            if stride.is_empty() {
                Ok(offset)
            }else{
                Err(TensorError::WrongDims)
            }
        },
        Idx::At(i) => {
            // Single-dimensional index; only valid when there is exactly one dimension
            logical_to_buffer_idx(&Idx::Coord(&[*i]), stride, offset)
        }
    }
}

/// Computes the new shape, stride, and offset for a view obtained by fixing
/// the dimension `dim` to index `idx`.
///
/// Behavior
/// - Removes the selected dimension from `shape` and `stride`.
/// - Advances `offset` by `stride[dim] * idx`.
///
/// Errors
/// - `InvalidDim` if `dim` is out of bounds.
/// - `IdxOutOfBounds` if `idx >= shape[dim]`.
fn compute_sliced_parameters(shape: &Shape, stride: &Stride, offset: &usize, dim: Dim, idx: Dim) -> Result<(Shape, Stride, usize), TensorError> {
    if dim >= shape.len() {
        return Err(TensorError::InvalidDim);
    }
    if idx >= shape[dim] {
        return Err(TensorError::IdxOutOfBounds);
    }
    let mut new_shape = shape.clone();
    new_shape.remove(dim);
    let mut new_stride = stride.clone();
    new_stride.remove(dim);
    let offset = offset + stride[dim] * idx;
    Ok((new_shape, new_stride, offset))
}
