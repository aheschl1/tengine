use crate::{backend::{Backend}, core::{primitives::{DeviceType, TensorBase}, tensor::{AsView, AsViewMut}, value::{Dtypes, TensorValue}, MetaTensor, TensorView, TensorViewMut}};

/// Trait for erased tensors, allowing dynamic dispatch on tensor types.
/// Implemented for all `TensorBase<T, B>` where `T: TensorValue` and `B: Backend<T>`.
pub trait UntypedTensor<B: Backend>: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    fn device(&self) -> DeviceType;
    fn dtype(&self) -> Dtypes;
    fn meta(&self) -> &MetaTensor;
}

impl<T, B> UntypedTensor<B> for TensorBase<T, B>
where
    T: crate::core::value::TensorValue + 'static,
    B: crate::backend::Backend + 'static,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn device(&self) -> DeviceType {
        B::device_type()
    }

    fn dtype(&self) -> Dtypes {
        T::DTYPE
    }

    fn meta(&self) -> &MetaTensor {
        &self.meta
    }
}

/// Downcasting methods for `ErasedTensor`.
/// Allows retrieving the concrete tensor type from the erased trait object.
/// Requires knowing the original `T` and `B` types.
impl<B: Backend> dyn UntypedTensor<B> {
    pub fn typed<T>(&self) -> Option<&TensorBase<T, B>>
    where
        T: TensorValue + 'static,
    {
        self.as_any().downcast_ref::<TensorBase<T, B>>()
    }

    pub fn typed_mut<T>(&mut self) -> Option<&mut TensorBase<T, B>>
    where
        T: TensorValue + 'static,
    {
        self.as_any_mut().downcast_mut::<TensorBase<T, B>>()
    }

    pub fn view_typed<T>(&self) -> Option<TensorView<'_, T, B>>
    where
        T: TensorValue + 'static,
    {
        self.typed::<T>().map(|t| t.view())
    }

    pub fn view_typed_mut<T>(&mut self) -> Option<TensorViewMut<'_, T, B>>
    where
        T: TensorValue + 'static,
    {
        self.typed_mut::<T>().map(|t| t.view_mut())
    }
}



#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, core::{erased::UntypedTensor, primitives::DeviceType, tensor::TensorError, value::TensorValue, Shape, Tensor}};

    #[test]
    fn test_erased_tensor_downcast() -> Result<(), TensorError> {
        // cpu, f32 tensor
        let tensor = Tensor::<f32>::zeros((2, 3));
        let erased: Box<dyn UntypedTensor<Cpu>> = Box::new(tensor);
        assert_eq!(erased.device(), DeviceType::Cpu);
        assert_eq!(erased.dtype(), f32::DTYPE);
        let downcasted = erased.typed::<f32>().unwrap();
        assert_eq!(downcasted.meta().shape, Shape::from((2, 3)));
        Ok(())
    }
}