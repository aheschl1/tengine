use super::primitives::{TensorOwned, TensorViewBase};

pub type Dim = usize;
pub type Stride = Vec<usize>;
pub type Shape = Vec<Dim>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TensorMeta {
    shape: Shape,
    stride: Stride,
    offset: usize,
}

impl TensorMeta {
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
    if shape.iter().any(|&d| d == 0) { return true; }
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
pub trait TensorMetaView {
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
}

impl TensorMetaView for TensorMeta {
    fn shape(&self) -> &Shape { self.shape() }
    fn stride(&self) -> &Stride { self.stride() }
    fn offset(&self) -> usize { self.offset() }
}

impl<T> TensorMetaView for TensorOwned<T> {
    fn shape(&self) -> &Shape { self.meta.shape() }
    fn stride(&self) -> &Stride { self.meta.stride() }
    fn offset(&self) -> usize { self.meta.offset() }
}

impl<T, B> TensorMetaView for TensorViewBase<'_, T, B>
where
    B: AsRef<[T]>,
{
    fn shape(&self) -> &Shape { self.meta.shape() }
    fn stride(&self) -> &Stride { self.meta.stride() }
    fn offset(&self) -> usize { self.meta.offset() }
}
