use std::marker::PhantomData;

use crate::ndarray::{shape_to_stride, Shape, TensorMeta};
use crate::ndarray::tensor::TensorError;

/// An owned, contiguous tensor stored in row-major order.
///
/// Holds the backing buffer and the associated layout metadata. The `offset`
/// in `meta` is expected to be zero for owned tensors created via
/// `from_buf`/`row`/`column`/`scalar`.
#[derive(Debug, PartialEq, Eq)]
pub struct TensorOwned<T: Sized> {
    pub(crate) raw: Box<[T]>, // row major order
    pub(crate) meta: TensorMeta,
}

/// A non-owning view over tensor data with explicit layout metadata.
///
/// The `B` parameter abstracts over borrowed storage (e.g., `&[T]` or
/// `&mut [T]`), while `meta` carries shape, stride, and offset describing how
/// to interpret the underlying buffer.
pub struct TensorViewBase<'a, T, B>
where
    B: AsRef<[T]> + 'a,
{
    pub(crate) raw: B,
    pub(crate) meta: TensorMeta,
    pub(crate) _l: PhantomData<&'a ()>,
    pub(crate) _t: PhantomData<T>,
}

impl<'a, T, B> TensorViewBase<'a, T, B>
where
    B: AsRef<[T]> + 'a,
{
    /// Builds a tensor view from raw storage and metadata. No copying occurs;
    /// caller guarantees that `meta` correctly describes the layout within `raw`.
    pub(crate) fn from_parts(raw: B, meta: TensorMeta) -> Self {
        Self {
            raw,
            meta,
            _l: PhantomData,
            _t: PhantomData,
        }
    }

    /// Reinterprets this view with a different shape of the same total number
    /// of elements. Stride is recomputed for standard row-major layout; offset
    /// is preserved.
    ///
    /// Errors
    /// - `InvalidShape` if the product of the new shape doesn't match the old size.
    pub(crate) fn view_as(self, shape: Shape) -> Result<Self, TensorError> {
        let new_size: usize = shape.iter().product();
        let old_size = self.meta.shape().iter().product();
        match new_size.cmp(&old_size) {
            std::cmp::Ordering::Equal => {
                let stride = shape_to_stride(&shape);
                let meta = TensorMeta::new(shape, stride, self.meta.offset());
                let tensor = Self::from_parts(self.raw, meta);
                Ok(tensor)
            }
            _ => Err(TensorError::InvalidShape),
        }
    }
}

pub type TensorView<'a, T> = TensorViewBase<'a, T, &'a [T]>;
pub type TensorViewMut<'a, T> = TensorViewBase<'a, T, &'a mut [T]>;

impl<T: Sized> TensorOwned<T> {
    /// Constructs a tensor from a contiguous buffer in row-major order and a
    /// given shape. Validates that `shape` size equals `raw.len()`.
    ///
    /// Errors
    /// - `InvalidShape` if element count doesn't match.
    pub fn from_buf(raw: impl Into<Box<[T]>>, shape: Shape) -> Result<Self, TensorError> {
        let raw = raw.into();
        if shape.iter().fold(1, |p, x| p * x) != raw.len() {
            return Err(TensorError::InvalidShape);
        }
        let stride = shape_to_stride(&shape);
        Ok(Self {
            raw,
            meta: TensorMeta::new(shape, stride, 0),
        })
    }

    /// Creates a rank-0 (scalar) tensor holding `value`.
    pub fn scalar(value: T) -> Self {
        Self::from_buf(vec![value], vec![]).unwrap()
    }

    /// Creates a 1-D column tensor from the provided values.
    pub fn column(column: impl Into<Box<[T]>>) -> Self {
        let column = column.into();
        let shape = vec![column.len()];
        Self::from_buf(column, shape).unwrap()
    }

    /// Creates a 1xN row tensor from the provided values.
    pub fn row(row: impl Into<Box<[T]>>) -> Self {
        let row = row.into();
        let shape = vec![1, row.len()];
        Self::from_buf(row, shape).unwrap()
    }
}
