use std::ops::AddAssign;

use crate::{backend::Backend, core::{primitives::TensorValue, TensorViewMut}};

impl<'a, T, B> AddAssign<T> for TensorViewMut<'a, T, B> 
    where T: std::ops::Add<Output = T> + TensorValue,
          B: Backend<T>
{
    fn add_assign(&mut self, rhs: T) {
        self.backend.apply_each(
            self.raw, 
            |x| x + rhs,
            self.meta.iter_offsets()
        ).unwrap();
    }
}

impl<'a, T, B> AddAssign<&T> for TensorViewMut<'a, T, B> 
    where T: std::ops::Add<Output = T> + TensorValue,
          B: Backend<T>
{
    fn add_assign(&mut self, rhs: &T) {
        self.backend.apply_each(
            self.raw, 
            |x| x + *rhs,
            self.meta.iter_offsets()
        ).unwrap();
    }
}
