use std::{borrow::Borrow, ops::{Add, AddAssign}};

use crate::{backend::{Backend, BackendUnary}, core::{primitives::TensorBase, tensor::{AsTensor, AsViewMut}, value::{TensorValue, TensorValueUnary}, TensorView, TensorViewMut}, ops::elementwise::UnaryTensorOp};

impl<'a, T, B, O> AddAssign<O> for TensorViewMut<'a, T, B> 
    where T: TensorValueUnary + TensorValue,
          B: BackendUnary<T>,
          O: Borrow<T>
{
    fn add_assign(&mut self, rhs: O) {
        self.backend.apply_unary(
            self.raw, 
            UnaryTensorOp::Add(*rhs.borrow()),
            self.meta.offsets()
        ).unwrap();
    }
}

impl<'a, T, B, O> Add<O> for TensorViewMut<'a, T, B> 
    where T: TensorValueUnary  + TensorValue,
          B: BackendUnary<T>,
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
    where T: TensorValueUnary  + TensorValue,
          B: BackendUnary<T>,
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