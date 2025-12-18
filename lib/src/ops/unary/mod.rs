use crate::{backend::Backend, core::{primitives::TensorBase, tensor::{AsTensor, AsViewMut}, value::TensorValue}};

pub mod neg;

pub use neg::Negate;

pub trait InplaceUnaryOp<T: TensorValue, B: Backend> {
    fn apply_relu(
        &mut self
    );
    fn apply_sigmoid(
        &mut self
    );
    fn apply_tanh(
        &mut self
    );
}

pub trait UnaryOp<T: TensorValue, B: Backend> {
    fn relu(
        &self,
    ) -> TensorBase<T, B>;
    fn sigmoid(
        &self,
    ) -> TensorBase<T, B>;
    fn tanh(
        &self,
    ) -> TensorBase<T, B>;
}


impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> InplaceUnaryOp<T, B> for V {
    fn apply_relu(
        &mut self
    ) {
        todo!()
    }

    fn apply_sigmoid(
        &mut self
    ) {
        todo!()
    }

    fn apply_tanh(
        &mut self
    ) {
        todo!()
    }
}


impl<T: TensorValue, B: Backend, V: AsTensor<T, B>> UnaryOp<T, B> for V {
    fn relu(
        &self,
    ) -> TensorBase<T, B> {
        let mut result = self.owned();
        result.apply_relu();
        result
    }

    fn sigmoid(
        &self,
    ) -> TensorBase<T, B> {
        let mut result = self.owned();
        result.apply_sigmoid();
        result
    }

    fn tanh(
        &self,
    ) -> TensorBase<T, B> {
        let mut result = self.owned();
        result.apply_tanh();
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{core::Tensor, ops::unary::Negate};

    #[test]
    fn test_negate() {
        let mut tensor = Tensor::<f32>::ones((1, 2));
        tensor.neg_inplace();
        let expected = Tensor::<f32>::from_buf(vec![-1.0, -1.0], (1, 2));
        assert_eq!(tensor, expected.unwrap());

        let tensor2 = -tensor;
        let expected2 = Tensor::<f32>::from_buf(vec![1.0, 1.0], (1, 2));
        assert_eq!(tensor2, expected2.unwrap());
    }
}

#[cfg(test)]
mod remote_tests {
    use std::{sync::OnceLock, thread};

    use crate::{backend::{remote::{client::RemoteBackend, get_backend_default, server::RemoteServer}, Backend}, core::{primitives::{RemoteTensor, TensorBase}, tensor::TensorError, value::TensorValue, MetaTensor, Shape}};

    // Lazy static backend shared across all tests
    static BACKEND: OnceLock<RemoteBackend> = OnceLock::new();
    
    fn get_backend() -> RemoteBackend {

        BACKEND.get_or_init(|| {
            // Start the server
            let mut server = RemoteServer::new("127.0.0.1".parse().unwrap(), 7878);
            thread::spawn(move || {
                let _ = server.serve();
            });
            thread::sleep(std::time::Duration::from_millis(10));

            // Create and connect the backend
            let backend = get_backend_default().unwrap();
            
            backend
        }).clone()
    }
    
    fn make_remote_tensor<T: TensorValue>(buf: Vec<T>, shape: impl Into<Shape>) -> Result<RemoteTensor<T>, TensorError> {
        let shape: Shape = shape.into();
        let buf_len = buf.len();
        let expected_len: usize = shape.iter().product();
        
        if buf_len != expected_len {
            return Err(TensorError::InvalidShape(format!(
                "Element count mismatch: shape implies {} elements, but buffer has {} elements",
                expected_len,
                buf_len
            )));
        }
        
        let backend = get_backend();
        let buffer = backend.alloc_from_slice(buf.into())?;
        let stride = crate::core::shape_to_stride(&shape);
        
        // Clone the backend for this tensor
        let tensor_backend = backend.clone();
        drop(backend); // Release the lock
        
        Ok(TensorBase::from_parts(tensor_backend, buffer, MetaTensor::new(shape, stride, 0)))
    }

    #[test]
    fn test_remote_negate() {
        let tensor: TensorBase<f32, RemoteBackend> = make_remote_tensor(vec![1.0f32, -2.0, 3.0], (3,)).unwrap();
        let negated = -tensor;

        let expected = make_remote_tensor(vec![-1.0f32, 2.0, -3.0], (3,)).unwrap();
        assert_eq!(negated.cpu().unwrap(), expected.cpu().unwrap());
    }

}