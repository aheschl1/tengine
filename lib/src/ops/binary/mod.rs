use crate::{backend::Backend, core::{value::TensorValue, TensorView}};



pub trait PointwiseTensorOp<T, B: Backend<T>> 
where T: TensorValue
{
    fn add(&'_ self, other: &TensorView<T, B>) -> TensorView<'_, T, B>;
}

impl <T, B> PointwiseTensorOp<T, B> for TensorView<'_, T, B> 
where T: TensorValue + std::ops::Add<Output = T>,
      B: Backend<T>
{
    fn add(&'_ self, other: &TensorView<T, B>) -> TensorView<'_, T, B> {
        unimplemented!()
    }
}