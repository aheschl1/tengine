use std::ops::{Sub, SubAssign};

use crate::{backend::Backend, core::{primitives::TensorValue, TensorViewMut}};

impl<'a, T, B> SubAssign<T> for TensorViewMut<'a, T, B> 
    where T: std::ops::Sub<Output = T> + TensorValue,
          B: Backend<T>
{
    fn sub_assign(&mut self, rhs: T) {
        self.backend.apply_each(
            self.raw, 
            |x| x - rhs,
            self.meta.iter_offsets()
        ).unwrap();
    }
}

impl<'a, T, B> SubAssign<&T> for TensorViewMut<'a, T, B> 
    where T: std::ops::Sub<Output = T> + TensorValue,
          B: Backend<T>
{
    fn sub_assign(&mut self, rhs: &T) {
        self.backend.apply_each(
            self.raw, 
            |x| x - *rhs,
            self.meta.iter_offsets()
        ).unwrap();
    }
}
