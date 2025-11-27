use std::{borrow::Borrow, ops::{Mul, MulAssign}};

use crate::{backend::Backend, core::{primitives::{TensorBase, TensorValue}, tensor::{AsTensor, AsViewMut}, TensorView, TensorViewMut}};

impl<'a, T, B, O> MulAssign<O> for TensorViewMut<'a, T, B> 
    where T: std::ops::Mul<Output = T> + TensorValue,
          B: Backend<T>,
          O: Borrow<T>
{
    fn mul_assign(&mut self, rhs: O) {
        self.backend.apply_each(
            self.raw, 
            |x| x * *rhs.borrow(),
            self.meta.iter_offsets()
        ).unwrap();
    }
}

impl<'a, T, B, O> Mul<O> for TensorViewMut<'a, T, B> 
    where T: std::ops::Mul<Output = T> + TensorValue,
          B: Backend<T>,
          O: Borrow<T>
{
    type Output = TensorBase<B, T>;

    fn mul(self, rhs: O) -> Self::Output {
        let mut result = self.owned();
        let mut view = result.view_mut();
        view *= rhs;
        result
    }
}

impl<'a, T, B, O> Mul<O> for TensorView<'a, T, B> 
    where T: std::ops::Mul<Output = T> + TensorValue,
          B: Backend<T>,
          O: Borrow<T>
{
    type Output = TensorBase<B, T>;

    fn mul(self, rhs: O) -> Self::Output {
        let mut result = self.owned();
        let mut view = result.view_mut();
        view *= rhs;
        result
    }
}