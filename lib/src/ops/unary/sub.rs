use std::ops::{Sub, SubAssign};

use crate::{backend::BackendUnaryElementwise, core::{primitives::TensorBase, tensor::{AsTensor, AsViewMut}, value::{TensorValue, TensorValueElementwise}, TensorView, TensorViewMut}, ops::unary::ElementwiseTensorOp};

impl<T, B> SubAssign<T> for TensorViewMut<'_, T, B> 
    where T: TensorValueElementwise+ TensorValue,
          B: BackendUnaryElementwise<T>
{
    fn sub_assign(&mut self, rhs: T) {
        self.backend.apply_elementwise(
            self.raw, 
            ElementwiseTensorOp::Sub(rhs),
            &self.meta
        ).unwrap();
    }
}

impl<T, B> Sub<T> for TensorViewMut<'_, T, B> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendUnaryElementwise<T>
{
    type Output = TensorBase<B, T>;

    fn sub(self, rhs: T) -> Self::Output {
        let mut result = self.owned();
        let mut view = result.view_mut();
        view -= rhs;
        result
    }
}

impl<T, B> Sub<T> for TensorView<'_, T, B> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendUnaryElementwise<T>
{
    type Output = TensorBase<B, T>;

    fn sub(self, rhs: T) -> Self::Output {
        let mut result = self.owned();
        let mut view = result.view_mut();
        view -= rhs;
        result
    }
}
