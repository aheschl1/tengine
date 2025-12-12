
use crate::{backend::{remote::TensorId, Backend}, core::{erased::UntypedTensor, tensor::{AsView, AsViewMut, TensorError}, value::TensorValue, Shape, TensorView, TensorViewMut}};


struct RemoteTensor<B: Backend> {
    pub id: TensorId,
    tensor: dyn UntypedTensor<B>
}

impl<T: TensorValue, B: Backend> AsView<T, B> for RemoteTensor<B> {
    fn view(&self) -> TensorView<'_, T, B> { 
        self.tensor.typed::<T>().expect("Failed to downcast tensor").view() 
    }
    fn view_as(&self, shape: Shape) -> Result<TensorView<'_, T, B>, TensorError> { 
        self.tensor.typed::<T>().expect("Failed to downcast tensor").view_as(shape) 
    }
}

impl<T: TensorValue, B: Backend> AsViewMut<T, B> for RemoteTensor<B> {
    fn view_mut(&'_ mut self) -> TensorViewMut<'_, T, B> { 
        self.tensor.typed_mut::<T>().expect("Failed to downcast tensor").view_mut() 
    }
    fn view_as_mut(&'_ mut self, shape: Shape) -> Result<TensorViewMut<'_, T, B>, TensorError> { 
        self.tensor.typed_mut::<T>().expect("Failed to downcast tensor").view_as_mut(shape) 
    }
}
