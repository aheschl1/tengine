use std::ops::{Index, IndexMut};

use crate::utils::{Dim, Shape, Stride, shape_to_stride};


#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum TensorError {
    IdxOutOfBounds,
    WrongDims,
    InvalidShape,
    InvalidDim
}

pub enum Idx {
    Coord(Vec<usize>),
    At(usize),
    Item
}

pub trait ViewableTensor<T: Sized> {
    fn view(&self) -> TensorView<'_, T>;
}
pub trait ViewableTensorMut<T: Sized>: ViewableTensor<T> {
    fn view_mut(&mut self) -> TensorViewMut<'_, T>;
}

impl<T: Sized> ViewableTensor<T> for TensorOwned<T> {
    fn view(&self) -> TensorView<'_, T> {
        TensorView{
            raw: &self.raw,
            stride: self.stride.clone(),
            offset: 0,
            shape: self.shape.clone(),
        }
    }
}
// TODO beware vec clones of Shape and Stride
impl<T: Sized> ViewableTensor<T> for TensorView<'_, T> {
    fn view(&self) -> TensorView<'_, T> {
        TensorView{
            raw: self.raw,
            stride: self.stride.clone(),
            offset: self.offset,
            shape: self.shape.clone(),
        }
    }
}

impl<T: Sized> ViewableTensor<T> for TensorViewMut<'_, T> {
    fn view(&self) -> TensorView<'_, T> {
        TensorView{
            raw: self.raw,
            stride: self.stride.clone(),
            offset: self.offset,
            shape: self.shape.clone(),
        }
    }
}

impl<T: Sized> ViewableTensorMut<T> for TensorViewMut<'_, T> {
    fn view_mut(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut{
            raw: self.raw,
            stride: self.stride.clone(),
            offset: self.offset,
            shape: self.shape.clone(),
        }
    }
}

impl<T: Sized> ViewableTensorMut<T> for TensorOwned<T> {
    fn view_mut(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut{
            raw: &mut self.raw,
            stride: self.stride.clone(),
            offset: 0,
            shape: self.shape.clone(),
        }
    }
}

pub trait Tensor<T>: Sized {
    /// Get the size of a specific dimension
    fn size(&self, dim: Dim) -> Dim;
    /// Get the shape of the tensor
    fn shape(&self) -> Shape;
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
    fn stride(&self) -> Stride;
    /// Get the total number of elements in the tensor
    fn num_elements(&self) -> usize {
        self.shape().iter().fold(1, |p, x| p * x)
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
impl<T, W> Tensor<T> for W 
where W: ViewableTensor<T>
{
    fn shape(&self) -> Shape {
        self.view().shape.clone()
    }

    fn stride(&self) -> Stride {
        self.view().stride.clone()
    }

    fn get(&self, idx: &Idx) -> Result<&T, TensorError> {
        match idx {
            Idx::Coord(idx) => {
                if idx.len() != self.dims() {
                    return Err(TensorError::WrongDims)
                }else{
                    let view = self.view();
                    let bidx = idx
                        .iter()
                        .zip(&view.stride)
                        .fold(view.offset, |acc, (a, b)| acc + *a*b);
                    view.raw.get(bidx).ok_or(TensorError::IdxOutOfBounds)
                }
            },
            Idx::Item => {
                if self.is_scalar() {
                    self.get(&Idx::Coord(vec![]))
                }else{
                    Err(TensorError::WrongDims)
                }
            },
            Idx::At(i) => {
                self.get(&Idx::Coord(vec![*i]))
            }
        }
    }

    fn slice(&self, dim: Dim, idx: Dim) -> Result<TensorView<'_, T>, TensorError> where Self: Sized {
        if dim >= self.dims() {
            return Err(TensorError::InvalidDim);
        }
        if idx >= self.size(dim) {
            return Err(TensorError::IdxOutOfBounds);
        }
        let mut new_shape = self.shape();
        new_shape.remove(dim);
        let mut new_stride = self.stride();
        new_stride.remove(dim);
        let mut v = self.view();
        v.offset = v.offset + v.stride()[dim] * idx;
        v.stride = new_stride.clone();
        v.shape = new_shape.clone();
        Ok(v)
    }
    
    fn size(&self, dim: Dim) -> Dim {
        self.shape()[dim]
    }
}

impl<T, W> TensorMut<T> for W
where W: ViewableTensorMut<T>
{
    fn get_mut(&mut self, idx: &Idx) -> Result<&mut T, TensorError> {
        match idx {
            Idx::Coord(idx) => {
                if idx.len() != self.dims() {
                    return Err(TensorError::WrongDims)
                }else{
                    let view = self.view_mut();
                    let bidx = idx
                        .iter()
                        .zip(&view.stride)
                        .fold(view.offset, |acc, (a, b)| acc + *a*b);
                    view.raw.get_mut(bidx).ok_or(TensorError::IdxOutOfBounds)
                }
            },
            Idx::Item => {
                if self.is_scalar() {
                    self.get_mut(&Idx::Item)
                }else{
                    Err(TensorError::WrongDims)
                }
            },
            Idx::At(i) => {
                self.get_mut(&Idx::Coord(vec![*i]))
            },
        }
    }

    fn slice_mut<'a>(&'a mut self, dim: Dim, idx: Dim) -> Result<TensorViewMut<'a, T>, TensorError> where Self: Sized {
        if dim >= self.dims() {
            return Err(TensorError::InvalidDim);
        }
        if idx >= self.size(dim) {
            return Err(TensorError::IdxOutOfBounds);
        }
        let mut new_shape = self.shape().clone();
        new_shape.remove(dim);
        let mut new_stride = self.stride().clone();
        new_stride.remove(dim);

        let mut v = self.view_mut();
        v.offset = v.offset + v.stride()[dim] * idx;
        v.stride = new_stride.clone();
        v.shape = new_shape.clone();
        Ok(v)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct TensorOwned<T: Sized>{
    raw: Box<[T]>, // row major order
    stride: Stride,
    shape: Shape,
}

#[derive(Debug)]
pub struct TensorView<'a, T: Sized>{
    raw: &'a [T], // row major order
    stride: Stride,
    offset: usize,
    shape: Shape,
}

#[derive(Debug)]
pub struct TensorViewMut<'a, T: Sized>{
    raw: &'a mut [T], // row major order
    stride: Stride,
    offset: usize,
    shape: Shape,
}

impl<T: Sized> TensorOwned<T> {
    pub fn from_buf(raw: impl Into<Box<[T]>>, shape: Shape) -> Result<Self, TensorError>{
        let raw = raw.into();
        if shape.iter().fold(1, |p, x| p*x) != raw.len() {
            return Err(TensorError::InvalidShape);
        }
        Ok(Self{
            raw,
            stride: shape_to_stride(&shape),
            shape,
        })
    }

    pub fn scalar(value: T) -> Self {
        Self{
            raw: vec![value].into(),
            stride: vec![],
            shape: vec![],
        }
    }

    pub fn column(column: impl Into<Box<[T]>>) -> Self {
        let column = column.into();
        Self{
            shape: vec![column.len()],
            raw: column,
            stride: vec![1],
        }
    }

    pub fn row(row: impl Into<Box<[T]>>) -> Self {
        let row = row.into();
        Self{
            shape: vec![1, row.len()],
            stride: vec![row.len(), 1],
            raw: row,
        }
    }

    pub fn empty() -> TensorOwned<()> {
        TensorOwned::<()>{
            raw: vec![].into(),
            stride: vec![],
            shape: vec![],
        }
    }
}


impl<'a, T, S: Into<Vec<usize>>> Index<S> for TensorOwned<T> {
    type Output = T;

    fn index(&self, index: S) -> &Self::Output {
        self.get(&Idx::Coord(index.into())).unwrap()
    }
}

impl<'a, T, S: Into<Vec<usize>>> Index<S> for TensorView<'_, T> {
    type Output = T;

    fn index(&self, index: S) -> &Self::Output {
        self.get(&Idx::Coord(index.into())).unwrap()
    }
}

impl<'a, T, S: Into<Vec<usize>>> Index<S> for TensorViewMut<'_, T> {
    type Output = T;

    fn index(&self, index: S) -> &Self::Output {
        self.get(&Idx::Coord(index.into())).unwrap()
    }
}

impl<'a, T, S: Into<Vec<usize>>> IndexMut<S> for TensorOwned<T> {
    fn index_mut(&mut self, index: S) -> &mut Self::Output {
        self.get_mut(&Idx::Coord(index.into())).unwrap()
    }
}

impl<'a, T, S: Into<Vec<usize>>> IndexMut<S> for TensorViewMut<'_, T> {
    fn index_mut(&mut self, index: S) -> &mut Self::Output {
        self.get_mut(&Idx::Coord(index.into())).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Idx, Shape, Stride, Tensor, TensorMut, TensorOwned, TensorError};

    fn make_tensor<T>(buf: Vec<T>, shape: Shape) -> TensorOwned<T> {
        TensorOwned::from_buf(buf, shape).unwrap()
    }

    #[test]
    fn test_slice_matrix() {
        let buf = vec![
            1, 2, 3, 
            4, 5, 6
        ];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);

        let slice = tensor.slice(0, 0).unwrap(); // slice along rows, should give a view of shape [3]
        assert_eq!(*slice.shape(), vec![3]);
        assert_eq!(*slice.stride(), vec![1]);
        assert_eq!(*index_tensor(Idx::At(0), &slice).unwrap(), 1);
        assert_eq!(*index_tensor(Idx::At(1), &slice).unwrap(), 2);
        assert_eq!(*index_tensor(Idx::At(2), &slice).unwrap(), 3);

        let slice2 = tensor.slice(1, 0).unwrap(); // slice along columns, should give a view of shape [2]
        assert_eq!(*slice2.shape(), vec![2]);
        assert_eq!(*slice2.stride(), vec![3]);
        assert_eq!(*index_tensor(Idx::At(0), &slice2).unwrap(), 1);
        assert_eq!(*index_tensor(Idx::Coord(vec![1]), &slice2).unwrap(), 4);
        assert_eq!(*index_tensor(Idx::At(1), &slice2).unwrap(), 4);
    }

    #[test]
    fn test_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_tensor(buf, shape);

        let slice = tensor.slice(0, 0).unwrap(); // slice along depth, should give a view of shape [2, 2]
        assert_eq!(*slice.shape(), vec![2, 2]);
        assert_eq!(*slice.stride(), vec![2, 1]);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 1);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 2);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 4);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 5);

        // second depth
        let slice_second_depth = tensor.slice(0, 1).unwrap();
        assert_eq!(*slice_second_depth.shape(), vec![2, 2]);
        assert_eq!(*slice_second_depth.stride(), vec![2, 1]);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 0]), &slice_second_depth).unwrap(), 6);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 1]), &slice_second_depth).unwrap(), 7);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 0]), &slice_second_depth).unwrap(), 8);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 1]), &slice_second_depth).unwrap(), 9);

        let slice2 = tensor.slice(1, 0).unwrap(); // slice along row, should give a view of shape [2, 2]
        assert_eq!(*slice2.shape(), vec![2, 2]);
        assert_eq!(*slice2.stride(), vec![4, 1]);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 0]), &slice2).unwrap(), 1);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 1]), &slice2).unwrap(), 2);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 0]), &slice2).unwrap(), 6);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 1]), &slice2).unwrap(), 7);

        // column slice
        let slice3 = tensor.slice(2, 0).unwrap(); // slice along column
        assert_eq!(*slice3.shape(), vec![2, 2]);
        assert_eq!(*slice3.stride(), vec![4, 2]);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 0]), &slice3).unwrap(), 1);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 1]), &slice3).unwrap(), 4);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 0]), &slice3).unwrap(), 6);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 1]), &slice3).unwrap(), 8);
    }

    #[test]
    fn test_slice_of_slice() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);

        let slice = tensor.slice(0, 1).unwrap(); // slice along rows, should give a view of shape [3]
        assert_eq!(*slice.shape(), vec![3]);
        assert_eq!(*index_tensor(Idx::At(0), &slice).unwrap(), 4);
        assert_eq!(*index_tensor(Idx::At(1), &slice).unwrap(), 5);
        assert_eq!(*index_tensor(Idx::At(2), &slice).unwrap(), 6);

        let slice_of_slice = slice.slice(0, 2).unwrap(); // slice along columns, should give a view of shape []
        assert_eq!(*slice_of_slice.shape(), vec![]);
        assert_eq!(*index_tensor(Idx::Coord(vec![]), &slice_of_slice).unwrap(), 6);
    }

    #[test]
    fn slice_of_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_tensor(buf, shape);

        let slice = tensor.slice(0, 1).unwrap(); // slice along depth, should give a view of shape [2, 2]
        assert_eq!(*slice.shape(), vec![2, 2]);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 6);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 7);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 8);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 9);

        let slice_of_slice = slice.slice(1, 0).unwrap(); // slice along row, should give a view of shape [2]
        assert_eq!(*slice_of_slice.shape(), vec![2]);
        assert_eq!(*index_tensor(Idx::At(0), &slice_of_slice).unwrap(), 6);
        assert_eq!(*index_tensor(Idx::At(1), &slice_of_slice).unwrap(), 8);

        // slice of slice of slice
        let slice_of_slice_of_slice = slice_of_slice.slice(0, 1).unwrap(); // slice along column, should give a view of shape []
        assert_eq!(*slice_of_slice_of_slice.shape(), vec![]);
        assert_eq!(*index_tensor(Idx::Item, &slice_of_slice_of_slice).unwrap(), 8);
    }

    #[test]
    fn test_mut_slices() {
        // mut slice from owned tensor
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);
        let mut slice = tensor.slice_mut(0, 1).unwrap(); // slice along rows, should give a view of shape [3]
        assert_eq!(*slice.shape(), vec![3]);
        assert_eq!(*index_tensor(Idx::At(0), &slice).unwrap(), 4);
        assert_eq!(*index_tensor(Idx::At(1), &slice).unwrap(), 5);
        assert_eq!(*index_tensor(Idx::At(2), &slice).unwrap(), 6);
        *slice.get_mut(&Idx::At(1)).unwrap() = 50;
        assert_eq!(*index_tensor(Idx::At(1), &slice).unwrap(), 50);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 1]), &tensor).unwrap(), 50);

        // TODO figure out
        // mut slice of mut slice
        // drops previous mutable borrow
        // let mut slice_of_slice = slice.slice_mut(0, 2).unwrap(); //
        // assert_eq!(*slice_of_slice.shape(), vec![]);
        // assert_eq!(*slice_of_slice.get(&[]).unwrap(), 6);
        // *slice_of_slice.get_mut(&[]).unwrap() = 60;
        // assert_eq!(*slice_of_slice.get(&[]).unwrap(), 60);
        // assert_eq!(*slice.get(&[2]).unwrap(), 60);
        // assert_eq!(*tensor.get(&[1, 2]).unwrap(), 60);
    }

    #[test]
    fn test_column() {
        let tensor = TensorOwned::column(vec![1, 2, 3]);
        assert_eq!(*tensor.shape(), vec![3]);
        assert_eq!(*index_tensor(Idx::At(0), &tensor).unwrap(), 1);
        assert_eq!(*index_tensor(Idx::At(1), &tensor).unwrap(), 2);
        assert_eq!(*index_tensor(Idx::At(2), &tensor).unwrap(), 3);
    }

    #[test]
    fn test_row() {
        let tensor = TensorOwned::row(vec![1, 2, 3]);
        assert_eq!(*tensor.shape(), vec![1, 3]);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 0]), &tensor).unwrap(), 1);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 1]), &tensor).unwrap(), 2);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 2]), &tensor).unwrap(), 3);

        assert_eq!(tensor[vec![0, 1]], 2);
    }

    #[test]
    fn test_empty() {
        let tensor = TensorOwned::<()>::empty();
        assert_eq!(*tensor.shape(), vec![]);
        assert!(tensor.raw.is_empty());
        assert!(tensor.stride.is_empty());
    }

    #[test]
    fn test_scalar() {
        let buf = vec![42];
        let shape = vec![];
        let tensor = make_tensor(buf, shape);

        assert_eq!(*index_tensor(Idx::Item, &tensor).unwrap(), 42);
        assert!(tensor.is_scalar());
        assert_eq!(TensorOwned::scalar(42), tensor);
    }

    #[test]
    fn test_array() {
        let buf = vec![1, 2, 3];
        let shape = vec![3];
        let mut tensor = make_tensor(buf, shape);

        assert_eq!(*index_tensor(Idx::At(0), &tensor).unwrap(), 1);
        assert_eq!(*index_tensor(Idx::At(1), &tensor).unwrap(), 2);
        assert_eq!(*index_tensor(Idx::At(2), &tensor).unwrap(), 3);

        *tensor.get_mut(&Idx::At(1)).unwrap() = 1;
        assert_eq!(*index_tensor(Idx::At(1), &tensor).unwrap(), 1);
    }

    #[test]
    fn test_matrix() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);

        assert_eq!(*index_tensor(Idx::Coord(vec![0, 0]), &tensor).unwrap(), 1);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 1]), &tensor).unwrap(), 2);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 2]), &tensor).unwrap(), 3);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 0]), &tensor).unwrap(), 4);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 1]), &tensor).unwrap(), 5);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 2]), &tensor).unwrap(), 6);

        *tensor.get_mut(&Idx::Coord(vec![1, 2])).unwrap() = 100;
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 2]), &tensor).unwrap(), 100);
    }

    #[test]
    fn test_cube() {
        //
        let buf = vec![
            1, 2,
            4, 5,

            6, 7,
            8, 9
        ];
        let shape = vec![2, 2, 2];
        let mut tensor = make_tensor(buf, shape);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 0, 0]), &tensor).unwrap(), 1); // depth, row, column
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 0, 1]), &tensor).unwrap(), 2);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 1, 0]), &tensor).unwrap(), 4);
        assert_eq!(*index_tensor(Idx::Coord(vec![0, 1, 1]), &tensor).unwrap(), 5);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 0, 0]), &tensor).unwrap(), 6);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 0, 1]), &tensor).unwrap(), 7);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 1, 0]), &tensor).unwrap(), 8);
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 1, 1]), &tensor).unwrap(), 9);

        // modify
        *tensor.get_mut(&Idx::Coord(vec![1, 0, 0])).unwrap() = 67;
        assert_eq!(*index_tensor(Idx::Coord(vec![1, 0, 0]), &tensor).unwrap(), 67);
    }

    fn index_tensor<'a, T: Clone + Eq + std::fmt::Debug>(index: Idx, tensor: &'a impl Tensor<T>) -> Result<&'a T, TensorError> {
        let r: Result<&T, TensorError> = tensor.get(&index);
        let a = match r.as_ref() {
            Ok(v) => Ok(*v),
            Err(e) => return Err(e.clone()),
        }.clone();
        let b = match &index {
            Idx::At(i) => tensor.get(&Idx::Coord(vec![*i])),
            Idx::Coord(idx) => tensor.get(&Idx::Coord(idx.clone())),
            Idx::Item => tensor.item(),
        };
        assert_eq!(a, b);
        r
    }

    #[test]
    fn test_shape_to_stride() {
        let shape = vec![2, 2, 3];
        let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![6, 3, 1]);
    }

    #[test]
    fn test_shape_to_stride_single_dim() {
        let shape = vec![4];
        let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![1]);
    }

    #[test]
    fn test_shape_to_stride_empty() {
        let shape: Shape = vec![];
        let stride: Stride = super::shape_to_stride(&shape);

        assert!(stride.is_empty());
    }

    #[test]
    fn test_shape_to_stride_ones() {
        let shape = vec![1, 1, 1];
        let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![1, 1, 1]);
    }

    #[test]
    fn test_shape_to_stride_mixed() {
        let shape = vec![5, 1, 2];
        let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![2, 2, 1]);
    }

    #[test]
    fn test_shape_to_stride_larger() {
        let shape = vec![3, 4, 5];
        let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![20, 5, 1]);
    }

    #[test]
    fn test_from_buf_error() {
        let buf = vec![1, 2, 3, 4];
        let shape = vec![2, 3];
        assert!(matches!(
            TensorOwned::from_buf(buf, shape),
            Err(super::TensorError::InvalidShape)
        ));
    }

    #[test] 
    fn test_get_errors() {
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(matches!(
            index_tensor(Idx::Coord(vec![0, 0, 0]), &tensor),
            Err(super::TensorError::WrongDims)
        ));
        assert!(matches!(
            index_tensor(Idx::Coord(vec![2, 0]), &tensor),
            Err(super::TensorError::IdxOutOfBounds)
        ));
    }

    #[test]
    fn test_slice_errors() {
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(matches!(
            tensor.slice(2, 0),
            Err(super::TensorError::InvalidDim)
        ));
        assert!(matches!(
            tensor.slice(0, 2),
            Err(super::TensorError::IdxOutOfBounds)
        ));
    }

    #[test]
    fn test_index_and_index_mut() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);

        // Test Index on TensorOwned
        assert_eq!(tensor[vec![0, 1]], 2);
        assert_eq!(tensor[vec![1, 2]], 6);

        // Test IndexMut on TensorOwned
        tensor[vec![1, 1]] = 55;
        assert_eq!(*tensor.get(&Idx::Coord(vec![1, 1])).unwrap(), 55);
        assert_eq!(tensor[vec![1, 1]], 55);

        // Test on a slice (TensorView)
        let view = tensor.slice(0, 1).unwrap(); // second row
        assert_eq!(view[vec![0]], 4);
        assert_eq!(view[vec![1]], 55);
        assert_eq!(view[vec![2]], 6);

        // Test on a mutable slice (TensorViewMut)
        let mut mut_view = tensor.slice_mut(0, 0).unwrap(); // first row
        mut_view[vec![2]] = 33;
        assert_eq!(*mut_view.get(&Idx::Coord(vec![2])).unwrap(), 33);
        assert_eq!(mut_view[vec![2]], 33);

        // Verify original tensor was changed
        assert_eq!(tensor[vec![0, 2]], 33);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds_panic() {
        let tensor = make_tensor(vec![1, 2, 3], vec![3]);
        let _ = tensor[vec![3]];
    }

    #[test]
    #[should_panic]
    fn test_index_wrong_dims_panic() {
        let tensor = make_tensor(vec![1, 2, 3], vec![3]);
        let _ = tensor[vec![0, 0]];
    }

    #[test]
    #[should_panic]
    fn test_index_mut_out_of_bounds_panic() {
        let mut tensor = make_tensor(vec![1, 2, 3], vec![3]);
        tensor[vec![3]] = 4;
    }

    #[test]
    #[should_panic]
    fn test_index_mut_wrong_dims_panic() {
        let mut tensor = make_tensor(vec![1, 2, 3], vec![3]);
        tensor[vec![0, 0]] = 4;
    }
}