use std::path::Iter;

use crate::{backend::Backend, core::{TensorViewMut, primitives::{TensorBase, TensorValue}}};

use super::primitives::TensorView;

pub type Dim = usize;
pub type Stride = Vec<usize>;
pub type Shape = Vec<Dim>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MetaTensor {
    shape: Shape,
    stride: Stride,
    offset: usize,
}

impl MetaTensor {
    /// Creates tensor metadata with explicit shape, stride and offset.
    pub fn new(shape: Shape, stride: Stride, offset: usize) -> Self {
        Self { shape, stride, offset }
    }

    /// Returns true when the metadata describes a scalar (rank 0).
    pub fn is_scalar(&self) -> bool { self.stride.is_empty() }
    /// Returns true when the metadata describes a 1xN row tensor.
    pub fn is_row(&self) -> bool { self.shape.len() == 2 && self.shape[0] == 1 }
    /// Returns true when the metadata describes a 1-D column tensor.
    pub fn is_column(&self) -> bool { self.shape.len() == 1 }
    /// Number of dimensions (rank).
    pub fn dims(&self) -> usize { self.shape.len() }
    /// Total number of elements (product of all dimensions).
    pub fn size(&self) -> usize { self.shape.iter().product() }
    /// Whether the layout is contiguous in row-major order, allowing strides of
    /// one for non-singleton dims and ignoring dims of size 1.
    pub fn is_contiguous(&self) -> bool { is_contiguous_relaxed(&self.shape, &self.stride) }
    /// Borrow the shape vector.
    pub fn shape(&self) -> &Shape { &self.shape }
    /// Borrow the stride vector.
    pub fn stride(&self) -> &Stride { &self.stride }
    /// Return the starting offset (in elements) into the underlying buffer.
    pub fn offset(&self) -> usize { self.offset }
    /// Returns the size of a single dimension by index.
    pub fn dim(&self, dim: Dim) -> Dim { self.shape[dim] }
    /// returns all offsets in the underlying buffer for this tensor/view.
    pub fn iter_offsets(&self) -> impl Iterator<Item = usize> + '_ {
        let shape = self.shape.clone();
        let stride = self.stride.clone();
        let offset = self.offset;
        TensorOffsetIterator::new(shape, stride, offset)
    }
}

struct TensorOffsetIterator {
    shape: Shape,
    stride: Stride,
    current_indices: Vec<usize>,
    done: bool,
    base_offset: usize,
}

impl TensorOffsetIterator {
    fn new(shape: Shape, stride: Stride, base_offset: usize) -> Self {
        let dims = shape.len();
        Self {
            shape,
            stride,
            current_indices: vec![0; dims],
            done: dims == 0, // done immediately for scalar
            base_offset,
        }
    }
}

impl Iterator for TensorOffsetIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut offset = self.base_offset;
        for (idx, stride) in self.current_indices.iter().zip(self.stride.iter()) {
            offset += idx * stride;
        }

        // Increment indices
        for i in (0..self.current_indices.len()).rev() {
            self.current_indices[i] += 1;
            if self.current_indices[i] < self.shape[i] {
                break;
            } else {
                self.current_indices[i] = 0;
                if i == 0 {
                    self.done = true;
                }
            }
        }

        Some(offset)
    }
}

/// Computes the standard row-major stride for a given shape.
pub fn shape_to_stride(shape: &Shape) -> Stride {
    let mut stride = vec![1; shape.len()];
    for i in (0..shape.len()).rev() {
        if i < shape.len() - 1 {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
    }
    stride
}


/// Checks whether a layout (shape/stride) is contiguous in a relaxed sense:
/// ignores singleton dimensions and accepts empty shapes.
pub(crate) fn is_contiguous_relaxed(shape: &Shape, stride: &Stride) -> bool {
    if shape.is_empty() { return true; }
    if shape.contains(&0) { return true; }
    if shape.len() != stride.len() { return false; }

    let mut expected = 1usize;
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        let s = stride[i];
        if dim != 1 {
            if s != expected { return false; }
            expected = expected.saturating_mul(dim);
        }
    }
    true
}

/// Read-only metadata view for tensors and views.
pub trait MetaTensorView {
    /// Borrow the shape vector.
    fn shape(&self) -> &Shape;
    /// Borrow the stride vector.
    fn stride(&self) -> &Stride;
    /// Starting offset (in elements) into the underlying buffer.
    fn offset(&self) -> usize;
    /// Number of dimensions (rank).
    fn dims(&self) -> usize { self.shape().len() }
    /// Size of one dimension by index.
    fn dim(&self, dim: Dim) -> Dim { self.shape()[dim] }
    /// Total number of elements (product of all dimensions).
    fn size(&self) -> usize { self.shape().iter().product() }
    /// True for rank-0 (scalar) tensors.
    fn is_scalar(&self) -> bool { self.stride().is_empty() }
    /// True for 1xN row tensors.
    fn is_row(&self) -> bool { self.shape().len() == 2 && self.shape()[0] == 1 }
    /// True for 1-D column tensors.
    fn is_column(&self) -> bool { self.shape().len() == 1 }
    /// Whether the layout is contiguous in row-major order under relaxed rules
    /// (ignoring singleton dimensions).
    fn is_contiguous(&self) -> bool { is_contiguous_relaxed(self.shape(), self.stride()) }
    fn iter_offsets(&self) -> impl Iterator<Item = usize> + '_ {
        let shape = self.shape().clone();
        let stride = self.stride().clone();
        let offset = self.offset();
        TensorOffsetIterator::new(shape, stride, offset)
    }
}

impl MetaTensorView for MetaTensor {
    fn shape(&self) -> &Shape { self.shape() }
    fn stride(&self) -> &Stride { self.stride() }
    fn offset(&self) -> usize { self.offset() }
}

impl<B, T: TensorValue> MetaTensorView for TensorBase<B, T> 
where
    B: Backend<T>,
{
    fn shape(&self) -> &Shape { self.meta.shape() }
    fn stride(&self) -> &Stride { self.meta.stride() }
    fn offset(&self) -> usize { self.meta.offset() }
}

impl<T: TensorValue, B> MetaTensorView for TensorView<'_, T, B>
where
    B: Backend<T>,
{
    fn shape(&self) -> &Shape { self.meta.shape() }
    fn stride(&self) -> &Stride { self.meta.stride() }
    fn offset(&self) -> usize { self.meta.offset() }
}

impl <T: TensorValue, B> MetaTensorView for TensorViewMut<'_, T, B>
where
    B: Backend<T>,
{
    fn shape(&self) -> &Shape { self.meta.shape() }
    fn stride(&self) -> &Stride { self.meta.stride() }
    fn offset(&self) -> usize { self.meta.offset() }
}
