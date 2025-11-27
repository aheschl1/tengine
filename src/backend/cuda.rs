use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DeviceRepr};

use crate::{backend::{Backend, BackendUnary}, core::{tensor::TensorError, value::{TensorValue, TensorValueUnary}}, ops::elementwise::UnaryTensorOp};

// FFI declarations for CUDA kernel launchers
#[cfg(feature = "cuda")]
extern "C" {
    fn launch_unary_f32(data: *mut f32, offsets: *mut usize, n: usize, op: u8, value: f32, block_size: u32);
    fn launch_unary_f64(data: *mut f64, offsets: *mut usize, n: usize, op: u8, value: f64, block_size: u32);
    fn launch_unary_u8(data: *mut u8, offsets: *mut usize, n: usize, op: u8, value: u8, block_size: u32);
    fn launch_unary_u16(data: *mut u16, offsets: *mut usize, n: usize, op: u8, value: u16, block_size: u32);
    fn launch_unary_u32(data: *mut u32, offsets: *mut usize, n: usize, op: u8, value: u32, block_size: u32);
    fn launch_unary_u64(data: *mut u64, offsets: *mut usize, n: usize, op: u8, value: u64, block_size: u32);
    fn launch_unary_u128(data: *mut u128, offsets: *mut usize, n: usize, op: u8, value: u128, block_size: u32);
    fn launch_unary_i8(data: *mut i8, offsets: *mut usize, n: usize, op: u8, value: i8, block_size: u32);
    fn launch_unary_i16(data: *mut i16, offsets: *mut usize, n: usize, op: u8, value: i16, block_size: u32);
    fn launch_unary_i32(data: *mut i32, offsets: *mut usize, n: usize, op: u8, value: i32, block_size: u32);
    fn launch_unary_i64(data: *mut i64, offsets: *mut usize, n: usize, op: u8, value: i64, block_size: u32);
    fn launch_unary_i128(data: *mut i128, offsets: *mut usize, n: usize, op: u8, value: i128, block_size: u32);
}

// Default block size for CUDA kernels - 256 is a good balance for modern GPUs
// Can be tuned based on kernel characteristics and GPU architecture
const DEFAULT_BLOCK_SIZE: u32 = 256;

pub struct CudaBuf<T: TensorValue> {
    pub(crate) ptr: CudaSlice<T>,
    pub(crate) len: usize,
}

pub struct CudaBackend {
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaBackend {
    fn stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.ctx.default_stream()
    }

    pub(crate) fn construct(device: usize) -> Result<Self, TensorError> {
        // TODO multiple devices
        let ctx = CudaContext::new(device).map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(Self { ctx })
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
    
    // fn apply_unary<V: TensorValueUnary>(&self, buf: &mut Self::Buf, op: UnaryTensorOp<V>, offsets: Vec<usize>) -> Result<(), crate::core::tensor::TensorError> {
    //     todo!()
    // }
    
    fn new() -> Self {
        // TODO multiple devices
        Self::construct(0).unwrap()
    }
    
    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError> {
        let mut dst = self.alloc(src.len)?;
        self.stream()
            .memcpy_dtod(&src.ptr, &mut dst.ptr)
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(dst)
    }
    
    fn dump(&self, src: &Self::Buf) -> Result<Box<[T]>, TensorError> {
        let mut host_buf = vec![T::default(); src.len];
        self.stream()
            .memcpy_dtoh(&src.ptr, host_buf.as_mut_slice())
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(host_buf.into_boxed_slice())
    }

    
}

impl<T: TensorValueUnary + DeviceRepr + 'static> BackendUnary<T> for CudaBackend {
    fn apply_unary(&self, buf: &mut Self::Buf, op: UnaryTensorOp<T>, offsets: Vec<usize>) -> Result<(), TensorError> {
        // Upload offsets to device
        let offsets_ptr_device = self.stream()
            .clone_htod(offsets.as_slice())
            .map_err(|e| TensorError::CudaError(e.to_string()))?;

        let op_code = op.to_op_code();
        let value = op.value();
        let n = offsets.len();
        let stream = self.stream();

        macro_rules! launch_unary {
            ($launch_fn:ident, $t:ty) => {{
                // transmute value from T to actual type
                let concrete_value: $t = unsafe { std::mem::transmute_copy(&value) };
                let (data_ptr, _) = buf.ptr.device_ptr(&stream);
                let (offsets_ptr, _) = offsets_ptr_device.device_ptr(&stream);
                
                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        offsets_ptr as *mut usize,
                        n,
                        op_code,
                        concrete_value,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                stream.synchronize()
                    .map_err(|e| TensorError::CudaError(e.to_string()))?;
                Ok(())
            }};
        }

        // Dispatch based on type
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_unary!(launch_unary_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_unary!(launch_unary_f64, f64),
            id if id == std::any::TypeId::of::<u8>() => launch_unary!(launch_unary_u8, u8),
            id if id == std::any::TypeId::of::<u16>() => launch_unary!(launch_unary_u16, u16),
            id if id == std::any::TypeId::of::<u32>() => launch_unary!(launch_unary_u32, u32),
            id if id == std::any::TypeId::of::<u64>() => launch_unary!(launch_unary_u64, u64),
            id if id == std::any::TypeId::of::<u128>() => launch_unary!(launch_unary_u128, u128),
            id if id == std::any::TypeId::of::<i8>() => launch_unary!(launch_unary_i8, i8),
            id if id == std::any::TypeId::of::<i16>() => launch_unary!(launch_unary_i16, i16),
            id if id == std::any::TypeId::of::<i32>() => launch_unary!(launch_unary_i32, i32),
            id if id == std::any::TypeId::of::<i64>() => launch_unary!(launch_unary_i64, i64),
            id if id == std::any::TypeId::of::<i128>() => launch_unary!(launch_unary_i128, i128),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA unary operation".to_string())),
        }
    }
}