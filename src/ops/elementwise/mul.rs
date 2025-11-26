use std::ops::MulAssign;

use crate::{backend::Backend, core::{primitives::TensorValue, TensorViewMut}};

impl<'a, T, B> MulAssign<T> for TensorViewMut<'a, T, B> 
    where T: std::ops::Mul<Output = T> + TensorValue,
          B: Backend<T>
{
    fn mul_assign(&mut self, rhs: T) {
        self.backend.apply_each(
            self.raw, 
            |x| x * rhs,
            self.meta.iter_offsets()
        ).unwrap();
    }
}

impl<'a, T, B> MulAssign<&T> for TensorViewMut<'a, T, B> 
    where T: std::ops::Mul<Output = T> + TensorValue,
          B: Backend<T>
{
    fn mul_assign(&mut self, rhs: &T) {
        self.backend.apply_each(
            self.raw, 
            |x| x * *rhs,
            self.meta.iter_offsets()
        ).unwrap();
    }
}
