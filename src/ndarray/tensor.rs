use std::{marker::PhantomData, ops::{Index, IndexMut}};

use crate::ndarray::{Dim, Shape, Stride, TensorOwned, TensorView, TensorViewBase, TensorViewMut, idx::Idx};


#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum TensorError {
    IdxOutOfBounds,
    WrongDims,
    InvalidShape,
    InvalidDim
}

pub trait AsView<T> {
    fn view(&self) -> TensorView<'_, T>;
}

pub trait AsViewMut<T> : AsView<T> {
    fn view_mut(&mut self) -> TensorViewMut<'_, T>;
}

impl<T> AsView<T> for TensorOwned<T> {
    fn view(&self) -> TensorView<'_, T> {
        TensorView::from_parts(self.raw.as_ref(), self.shape.clone(), self.stride.clone(), 0)
    }
}

impl<T> AsViewMut<T> for TensorOwned<T> {
    fn view_mut<'a>(&'a mut self) -> TensorViewMut<'a, T> {
        TensorViewMut::from_parts(self.raw.as_mut(), self.shape.clone(), self.stride.clone(), 0)
    }
}

impl<T, B> AsView<T> for TensorViewBase<'_, T, B> 
where B: AsRef<[T]>
{
    fn view<'a>(&'a self) -> TensorView<'a, T> {
        TensorView::from_parts(self.raw.as_ref(), self.shape.clone(), self.stride.clone(), self.offset)
    }
}

impl<T> AsViewMut<T> for TensorViewMut<'_, T> {
    fn view_mut<'a>(&'a mut self) -> TensorViewMut<'a, T> {
        TensorViewMut::from_parts(self.raw.as_mut(), self.shape.clone(), self.stride.clone(), self.offset)
    }
}


pub trait Tensor<T>: Sized {
    /// Get the size of a specific dimension
    fn dim(&self, dim: Dim) -> Dim;
    /// Get the shape of the tensor
    fn shape(&self) -> &Shape;
    /// Get the number of dimensions
    fn dims(&self) -> usize {
        self.stride().len()
    }
    /// Get element at given index
    fn get(&self, idx: &Idx) -> Result<&T, TensorError>;
    /// Check if tensor is a scalar (0-dimensional)
    fn is_scalar(&self) -> bool {
        self.stride().is_empty()
    }

    fn item(&self) -> Result<&T, TensorError> {
        self.get(&Idx::Item)
    }
    /// Get the stride of the tensor
    fn stride(&self) -> &Stride;
    /// Get the total number of elements in the tensor
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
    /// Create a slice/view of the tensor along a specific dimension at a given index
    fn slice<'a>(&'a self, dim: Dim, idx: Dim) -> Result<TensorView<'a, T>, TensorError> where Self: Sized;

    fn is_row(&self) -> bool {
        self.shape().len() == 2 && self.shape()[0] == 1
    }

    fn is_column(&self) -> bool {
        self.shape().len() == 1
    }
}
pub trait TensorMut<T>: Tensor<T> {
    /// Get mutable element at given index
    fn get_mut(&mut self, idx: &Idx) -> Result<&mut T, TensorError>;
    /// Slice mutable tensor to get a mutable view
    fn slice_mut<'a>(&'a mut self, dim: Dim, idx: Dim) -> Result<TensorViewMut<'a, T>, TensorError> where Self: Sized;
    /// sets a value at given index
    fn set(&mut self, idx: &Idx, value: T) -> Result<(), TensorError> {
        let slot = self.get_mut(idx)?;
        *slot = value;
        Ok(())
    }
}

impl<T, B> Tensor<T> for TensorViewBase<'_, T, B>
where B: AsRef<[T]>
{
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn stride(&self) -> &Stride {
        &self.stride
    }

    fn get(&self, idx: &Idx) -> Result<&T, TensorError> {
        let idx = logical_to_buffer_idx(idx, &self.stride, self.offset)?;
        self.raw.as_ref().get(idx).ok_or(TensorError::IdxOutOfBounds)
    }

    fn slice(&self, dim: Dim, idx: Dim) -> Result<TensorView<'_, T>, TensorError> where Self: Sized {
        let (new_shape, new_stride, offset) = compute_sliced_parameters(
            self.shape(), 
            self.stride(), 
            &self.offset,
            dim, 
            idx
        )?;
        
        let v = TensorView::from_parts(self.raw.as_ref(), new_shape, new_stride, offset);
        Ok(v)
    }
    
    fn dim(&self, dim: Dim) -> Dim {
        self.shape()[dim]
    }
}

impl<T> TensorMut<T> for TensorViewMut<'_, T>
{
    fn get_mut(&mut self, idx: &Idx) -> Result<&mut T, TensorError> {
        let idx = logical_to_buffer_idx(idx, &self.stride, self.offset)?;
        self.raw.get_mut(idx).ok_or(TensorError::IdxOutOfBounds)
    }

    fn slice_mut(&mut self, dim: Dim, idx: Dim) -> Result<TensorViewMut<'_, T>, TensorError> {
        let (new_shape, new_stride, offset) =
            compute_sliced_parameters(self.shape(), self.stride(), &self.offset, dim, idx)?;
    
        Ok(TensorViewMut::from_parts(self.raw, new_shape, new_stride, offset))
    }

}

impl<'a, T, B, S: Into<Idx<'a>>> Index<S> for TensorViewBase<'a, T, B> 
    where B: AsRef<[T]> + 'a
{
    type Output = T;

    fn index(&self, index: S) -> &Self::Output {
        self.get(&index.into()).unwrap()
    }
}

impl<'a, T, S: Into<Idx<'a>>> IndexMut<S> for TensorViewMut<'a, T> {
    fn index_mut(&mut self, index: S) -> &mut Self::Output {
        self.get_mut(&index.into()).unwrap()
    }
}


fn logical_to_buffer_idx(idx: &Idx, stride: &Stride, offset: usize) -> Result<usize, TensorError> {
    match idx {
        Idx::Coord(idx) => {
            if idx.len() != stride.len() {
                return Err(TensorError::WrongDims)
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
            logical_to_buffer_idx(&Idx::At(*i), stride, offset)
        }
    }
}

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
