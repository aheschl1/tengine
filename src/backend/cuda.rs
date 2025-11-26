use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, DeviceRepr};

use crate::{backend::Backend, core::{primitives::TensorValue, tensor::TensorError}};



pub struct CudaBuf<T: TensorValue> {
    pub(crate) ptr: CudaSlice<T>,
    pub(crate) len: usize,
    _ctx: Arc<CudaContext> // because if the backend is dropped, the memory may not be freed
}

pub struct CudaBackend {
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaBackend {
    fn stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.ctx.default_stream()
    }
}

impl<T: TensorValue + DeviceRepr> Backend<T> for CudaBackend {
    type Buf = CudaBuf<T>;
    
    fn from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, crate::core::tensor::TensorError> {
        let ptr = self.stream()
            .clone_htod(src.as_ref())
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(CudaBuf { 
            ptr, 
            len: src.len(),
            _ctx: self.ctx.clone() 
        })
    }
    
    fn alloc(&self, len: usize) -> Result<Self::Buf, crate::core::tensor::TensorError> {
        let ptr;
        unsafe{
            ptr = self.stream()
                .alloc::<T>(len)
                .map_err(|e| TensorError::CudaError(e.to_string()))?;
        }
        Ok(CudaBuf { 
            ptr, 
            len, 
            _ctx: self.ctx.clone() 
        })
    }
    
    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), crate::core::tensor::TensorError> {
        if src.len() != dst.len {
            return Err(TensorError::CudaError(format!("Source slice length {} does not match destination buffer length {}", src.len(), dst.len)));
        }

        self.stream()
            .memcpy_htod(src.as_ref(), &mut dst.ptr)
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(())
    }
    
    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, crate::core::tensor::TensorError> {
        if offset >= buf.len {
            return Err(TensorError::IdxOutOfBounds);
        }

        let mut host_buf = vec![T::default(); 1];
        self.stream()
            .memcpy_dtoh(&buf.ptr.slice(offset..offset+1), host_buf.as_mut_slice())
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(host_buf[0])
    }
    
    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), crate::core::tensor::TensorError> {
        if offset >= buf.len {
            return Err(TensorError::IdxOutOfBounds);
        }

        self.stream()
            .memcpy_htod(&[value], &mut buf.ptr.slice_mut(offset..offset+1))
            .map_err(|e| TensorError::CudaError(e.to_string()))?;

        Ok(())
    }
    
    fn len(&self, buf: &Self::Buf) -> usize {
        buf.len
    }
    
    fn apply_each<F>(&self, buf: &mut Self::Buf, f: F, offsets: impl Iterator<Item = usize>) -> Result<(), crate::core::tensor::TensorError>
    where
        F: Fn(T) -> T {
        todo!()
    }
    
    fn new() -> Self {
        // TODO multiple devices
        let ctx = CudaContext::new(0).expect("Failed to initialize CUDA context");
        Self { ctx }
    }

    
}