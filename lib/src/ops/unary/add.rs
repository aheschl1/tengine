use std::{borrow::Borrow, ops::{Add, AddAssign}};

use crate::{backend::BackendUnaryElementwise, core::{primitives::TensorBase, tensor::{AsTensor, AsViewMut}, value::{TensorValue, TensorValueElementwise}, TensorView, TensorViewMut}, ops::unary::ElementwiseTensorOp};

impl<'a, T, B, O> AddAssign<O> for TensorViewMut<'a, T, B> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendUnaryElementwise<T>,
          O: Borrow<T>
{
    fn add_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            self.raw, 
            ElementwiseTensorOp::Add(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

impl<'a, T, B, O> Add<O> for TensorViewMut<'a, T, B> 
    where T: TensorValueElementwise  + TensorValue,
          B: BackendUnaryElementwise<T>,
          O: Borrow<T>
{
    type Output = TensorBase<B, T>;

    fn add(self, rhs: O) -> Self::Output {
        let mut result = self.owned();
        let mut view = result.view_mut();
        view += rhs;
        result
    }
}

impl<'a, T, B, O> Add<O> for TensorView<'a, T, B> 
    where T: TensorValueElementwise  + TensorValue,
          B: BackendUnaryElementwise<T>,
          O: Borrow<T>
{
    type Output = TensorBase<B, T>;

    fn add(self, rhs: O) -> Self::Output {
        let mut result = self.owned();
        let mut view = result.view_mut();
        view += rhs;
        result
    }
}