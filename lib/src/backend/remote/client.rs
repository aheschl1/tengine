use std::{io::{Read, Write}, net::{IpAddr, TcpStream}, sync::{atomic::AtomicBool, Arc, Mutex}};

use prost::Message;

use crate::{backend::{remote::{protocol, TensorId}, Backend, BackendMatMul}, core::{meta::ContiguityTypes, primitives::DeviceType, tensor::TensorError, value::TensorValue, MetaTensor}};

pub struct RemoteBuf<T: TensorValue> {
    pub id: TensorId,
    dirty: Arc<AtomicBool>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: TensorValue> From<&RemoteBuf<T>> for Option<protocol::RemoteBuf> {
    fn from(buf: &RemoteBuf<T>) -> Self {
        Some(protocol::RemoteBuf {
            tensor_id: buf.id,
            dtype: Into::<protocol::DType>::into(T::DTYPE) as i32,
        })
    }
}

impl<T: TensorValue> From<&mut RemoteBuf<T>> for Option<protocol::RemoteBuf> {
    fn from(buf: &mut RemoteBuf<T>) -> Self {
        Some(protocol::RemoteBuf {
            tensor_id: buf.id,
            dtype: Into::<protocol::DType>::into(T::DTYPE) as i32,
        })
    }
}

pub struct RemoteBackend {
    pub address: IpAddr,
    pub port: u16,
    stream: Option<Mutex<TcpStream>>
}

impl RemoteBackend {
    pub fn new(address: IpAddr, port: u16) -> Self {
        Self {
            address,
            port,
            stream: None,
        }
    }

    pub fn connect(&mut self) -> Result<(), std::io::Error> {
        tracing::debug!("Attempting TCP connection to {}:{}", self.address, self.port);
        let stream = TcpStream::connect((self.address, self.port))?;
        tracing::info!("TCP connection established to {}:{}", self.address, self.port);
        self.stream = Some(Mutex::new(stream));
        Ok(())
    }

    pub fn send_message(&self, message: protocol::BackendRequest) -> Result<protocol::BackendResponse, std::io::Error> {
        tracing::debug!("Sending message: {:?}", message.request);
        let stream = self.stream.as_ref().ok_or(std::io::Error::new(std::io::ErrorKind::NotConnected, "Not connected"))?;
        let bytes = message.encode_to_vec();
        let size = (bytes.len() as u32).to_le_bytes();

        tracing::debug!("Encoded message size: {} bytes", bytes.len());
        let mut stream = stream.lock().unwrap();
        stream.write_all(&size)?;
        stream.write_all(&bytes)?;
        tracing::debug!("Message sent, waiting for response");

        let mut size_buf = [0u8; 4];
        stream.read_exact(&mut size_buf)?;
        let size = u32::from_le_bytes(size_buf) as usize;
        tracing::debug!("Response size: {} bytes", size);
        
        let mut resp_buf = vec![0u8; size];
        stream.read_exact(&mut resp_buf)?;
        tracing::debug!("Response received");

        let response = protocol::BackendResponse::decode(&resp_buf[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        tracing::debug!("Response decoded successfully");
        Ok(response)
    }
}

impl Backend for RemoteBackend {
    type Buf<T: TensorValue> = RemoteBuf<T>;

    fn device_type() -> crate::core::primitives::DeviceType {
        let _message = protocol::DeviceTypeRequest {};
        DeviceType::Remote {
            ip: IpAddr::V4(0.into()),
            port: 0,
            remote_type: Box::new(DeviceType::Cpu),
        }
    }

    fn alloc_from_slice<T: TensorValue>(&self, src: Box<[T]>) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        let vect: Vec<T> = src.into();
        // turn into bytes 
        let vecbytes = unsafe {
            std::slice::from_raw_parts(
                vect.as_ptr() as *const u8,
                vect.len() * std::mem::size_of::<T>(),
            ).to_vec()
        };
        let request = protocol::AllocFromSliceRequest {
            dtype: Into::<protocol::DType>::into(T::DTYPE) as i32,
            data: vecbytes,
        };
        let response = self.send_message(protocol::BackendRequest {
            request: Some(protocol::backend_request::Request::AllocFromSlice(request)),
            request_id: 0,
        }).map_err(|e| crate::core::tensor::TensorError::BackendError(e.to_string()))?;
        match response.response.unwrap() {
            protocol::backend_response::Response::Alloc(alloc_response) => {
                if alloc_response.success {
                    let new_buf = alloc_response.new_buf.ok_or(crate::core::tensor::TensorError::BackendError("Missing buffer in response".to_string()))?;
                    Ok(RemoteBuf {
                        id: new_buf.tensor_id,
                        dirty: Arc::new(AtomicBool::new(false)),
                        _phantom: std::marker::PhantomData,
                    })
                } else {
                    Err(crate::core::tensor::TensorError::BackendError(alloc_response.error))
                }
            },
            _ => Err(crate::core::tensor::TensorError::BackendError("Unexpected response type".to_string())),
        }
    }

    fn alloc<T: TensorValue>(&self, len: usize) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        tracing::debug!("Client alloc: requesting buffer of {} elements of type {:?}", len, T::DTYPE);
        let pdtyppe: protocol::DType = T::DTYPE.into();
        let _request = protocol::AllocRequest {
            dtype: pdtyppe as i32,
            len: len as u64,
        };

        tracing::debug!("Sending alloc request");
        let response = self.send_message(protocol::BackendRequest {
            request: Some(protocol::backend_request::Request::Alloc(_request)),
            request_id: 0,
        }).map_err(|e| crate::core::tensor::TensorError::BackendError(e.to_string()))?;

        tracing::debug!("Received alloc response");
        match response.response.unwrap() {
            protocol::backend_response::Response::Alloc(alloc_response) => {
                if alloc_response.success {
                    let new_buf = alloc_response.new_buf.ok_or(crate::core::tensor::TensorError::BackendError("Missing buffer in response".to_string()))?;
                    tracing::info!("Successfully allocated remote buffer with id={}", new_buf.tensor_id);
                    Ok(RemoteBuf {
                        id: new_buf.tensor_id,
                        dirty: Arc::new(AtomicBool::new(false)),
                        _phantom: std::marker::PhantomData,
                    })
                } else {
                    tracing::error!("Alloc failed: {}", alloc_response.error);
                    Err(crate::core::tensor::TensorError::BackendError(alloc_response.error))
                }
            },
            _ => {
                tracing::error!("Unexpected response type");
                Err(crate::core::tensor::TensorError::BackendError("Unexpected response type".to_string()))
            },
        }
    }

    fn copy_from_slice<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &[T]) -> Result<(), crate::core::tensor::TensorError> {
        let vecbytes = unsafe {
            std::slice::from_raw_parts(
                src.as_ptr() as *const u8,
                src.len() * std::mem::size_of::<T>(),
            ).to_vec()
        };
        let _request = protocol::CopyFromSliceRequest {
            buf: dst.into(),
            data: vecbytes,
        };

        todo!()
    }

    fn read<T: TensorValue>(&self, buf: &Self::Buf<T>, offset: usize) -> Result<T, crate::core::tensor::TensorError> {
        tracing::debug!("Client read: tensor_id={}, offset={}, dtype={:?}", buf.id, offset, T::DTYPE);
        let _request = protocol::ReadRequest {
            buf: buf.into(),
            offset: offset as u64,
        };

        tracing::debug!("Sending read request");
        let response = self.send_message(protocol::BackendRequest {
            request: Some(protocol::backend_request::Request::Read(_request)),
            request_id: 0,
        }).map_err(|e| crate::core::tensor::TensorError::BackendError(e.to_string()))?;

        tracing::debug!("Received read response");
        match response.response.unwrap() {
            protocol::backend_response::Response::Read(read_response) => {
                if read_response.success {
                    let tensor_value = read_response.value.ok_or(
                        crate::core::tensor::TensorError::BackendError("Missing value in response".to_string())
                    )?;
                    
                    // Extract bytes from TensorValue based on dtype
                    let bytes = match tensor_value.value.ok_or(
                        crate::core::tensor::TensorError::BackendError("Missing value field in TensorValue".to_string())
                    )? {
                        protocol::tensor_value::Value::U8Value(b) => b,
                        protocol::tensor_value::Value::U16Value(b) => b,
                        protocol::tensor_value::Value::U32Value(b) => b,
                        protocol::tensor_value::Value::U64Value(b) => b,
                        protocol::tensor_value::Value::U128Value(b) => b,
                        protocol::tensor_value::Value::I8Value(b) => b,
                        protocol::tensor_value::Value::I16Value(b) => b,
                        protocol::tensor_value::Value::I32Value(b) => b,
                        protocol::tensor_value::Value::I64Value(b) => b,
                        protocol::tensor_value::Value::I128Value(b) => b,
                        protocol::tensor_value::Value::F32Value(b) => b,
                        protocol::tensor_value::Value::F64Value(b) => b,
                        protocol::tensor_value::Value::IsizeValue(b) => b,
                        protocol::tensor_value::Value::UsizeValue(b) => b,
                        protocol::tensor_value::Value::BoolValue(b) => b,
                    };

                    // Convert bytes back to T
                    if bytes.len() != std::mem::size_of::<T>() {
                        return Err(crate::core::tensor::TensorError::BackendError(
                            format!("Received {} bytes but expected {} for type {:?}", 
                                bytes.len(), std::mem::size_of::<T>(), T::DTYPE)
                        ));
                    }

                    let value = unsafe {
                        std::ptr::read(bytes.as_ptr() as *const T)
                    };
                    
                    tracing::debug!("Successfully read value at offset {}", offset);
                    Ok(value)
                } else {
                    tracing::error!("Read failed: {}", read_response.error);
                    Err(crate::core::tensor::TensorError::BackendError(read_response.error))
                }
            },
            _ => {
                tracing::error!("Unexpected response type");
                Err(crate::core::tensor::TensorError::BackendError("Unexpected response type".to_string()))
            },
        }
    }

    fn write<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, value: T) -> Result<(), crate::core::tensor::TensorError> {
        let valuebytes = unsafe {
            std::slice::from_raw_parts(
                &value as *const T as *const u8,
                std::mem::size_of::<T>(),
            ).to_vec()
        };
        let dtype: protocol::DType = T::DTYPE.into();
        let _request = protocol::WriteRequest {
            buf: buf.into(),
            offset: offset as u64,
            value: Some(protocol::TensorValue::from_bytes_and_dtype(valuebytes, dtype)),
        };

        self.send_message(protocol::BackendRequest {
            request: Some(protocol::backend_request::Request::Write(_request)),
            request_id: 0,
        }).map_err(|e| crate::core::tensor::TensorError::BackendError(e.to_string()))?;
        Ok(())
    }

    fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize {
        let _request = protocol::LenRequest {
            buf: buf.into(),
        };

        todo!()
    }

    fn copy<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        let _request = protocol::CopyRequest {
            src: src.into(),
        };

        todo!()
    }

    fn dump<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Box<[T]>, crate::core::tensor::TensorError> {
        let _request = protocol::DumpRequest {
            buf: src.into(),
        };

        todo!()
    }

    fn new() -> Self {
        todo!()
    }

    fn apply_elementwise_contiguous<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (crate::ops::base::OpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), crate::core::tensor::TensorError> {
        let valuebytes = unsafe {
            std::slice::from_raw_parts(
                &op.1 as *const T as *const u8,
                std::mem::size_of::<T>(),
            ).to_vec()
        };
        let dtype: protocol::DType = T::DTYPE.into();
        let _request = protocol::ApplyElementwiseContiguousRequest {
            buf: buf.into(),
            op: Some(protocol::ElementwiseOp {
                op_type: Into::<protocol::OpType>::into(op.0) as i32,
                value: Some(protocol::TensorValue::from_bytes_and_dtype(valuebytes, dtype)),
            }),
            start: start as u64,
            len: len as u64,
        };

        // TODO: Send request and parse response
        todo!()
    }

    fn apply_elementwise_1d_strided<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (crate::ops::base::OpType, T), 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), crate::core::tensor::TensorError> {
        let valuebytes = unsafe {
            std::slice::from_raw_parts(
                &op.1 as *const T as *const u8,
                std::mem::size_of::<T>(),
            ).to_vec()
        };
        let dtype: protocol::DType = T::DTYPE.into();
        let _request = protocol::ApplyElementwise1dStridedRequest {
            buf: buf.into(),
            op: Some(protocol::ElementwiseOp {
                op_type: Into::<protocol::OpType>::into(op.0) as i32,
                value: Some(protocol::TensorValue::from_bytes_and_dtype(valuebytes, dtype)),
            }),
            offset: offset as u64,
            stride: stride as i64,
            len: len as u64,
        };

        // TODO: Send request and parse response
        todo!()
    }

    fn apply_elementwise_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        op: (crate::ops::base::OpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), crate::core::tensor::TensorError> {
        let valuebytes = unsafe {
            std::slice::from_raw_parts(
                &op.1 as *const T as *const u8,
                std::mem::size_of::<T>(),
            ).to_vec()
        };
        let dtype: protocol::DType = T::DTYPE.into();
        let _request = protocol::ApplyElementwiseNdRequest {
            buf: buf.into(),
            op: Some(protocol::ElementwiseOp {
                op_type: Into::<protocol::OpType>::into(op.0) as i32,
                value: Some(protocol::TensorValue::from_bytes_and_dtype(valuebytes, dtype)),
            }),
            offset: offset as u64,
            shape: shape.iter().map(|&s| s as u64).collect(),
            stride: stride.iter().map(|&s| s as i64).collect(),
        };

        // TODO: Send request and parse response
        todo!()
    }

    unsafe fn broadcast<T: TensorValue>(
        &self, 
        left: (*const Self::Buf<T>, &crate::core::MetaTensor), 
        right: (*const Self::Buf<T>, &crate::core::MetaTensor),
        dst: (*mut Self::Buf<T>, &crate::core::MetaTensor),
        op: crate::ops::base::OpType
    ) -> Result<(), crate::core::tensor::TensorError> {
        let _request = protocol::BroadcastRequest {
            left: Some(protocol::RemoteBufWithMeta {
                buf: (&*left.0).into(),
                meta: Some(protocol::MetaTensor {
                    offset: left.1.offset as u64,
                    shape: left.1.shape.iter().map(|&s| s as u64).collect(),
                    strides: left.1.strides.iter().map(|&s| s as i64).collect(),
                }),
            }),
            right: Some(protocol::RemoteBufWithMeta {
                buf: (&*right.0).into(),
                meta: Some(protocol::MetaTensor {
                    offset: right.1.offset as u64,
                    shape: right.1.shape.iter().map(|&s| s as u64).collect(),
                    strides: right.1.strides.iter().map(|&s| s as i64).collect(),
                }),
            }),
            dst: Some(protocol::RemoteBufWithMeta {
                buf: (&*dst.0).into(),
                meta: Some(protocol::MetaTensor {
                    offset: dst.1.offset as u64,
                    shape: dst.1.shape.iter().map(|&s| s as u64).collect(),
                    strides: dst.1.strides.iter().map(|&s| s as i64).collect(),
                }),
            }),
            op: Into::<protocol::OpType>::into(op) as i32,
        };

        // TODO: Send request and parse response
        todo!()
    }
}


impl<T: TensorValue> BackendMatMul<T> for RemoteBackend {
    fn matmul(
        &self,
        lhs: (&Self::Buf<T>, &MetaTensor),
        rhs: (&Self::Buf<T>, &MetaTensor),
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        contiguity: ContiguityTypes,
    ) -> Result<Self::Buf<T>, TensorError> {
        let _request = protocol::MatMulRequest {
            lhs: Some(protocol::RemoteBufWithMeta {
                buf: lhs.0.into(),
                meta: Some(protocol::MetaTensor {
                    offset: lhs.1.offset as u64,
                    shape: lhs.1.shape.iter().map(|&s| s as u64).collect(),
                    strides: lhs.1.strides.iter().map(|&s| s as i64).collect(),
                }),
            }),
            rhs: Some(protocol::RemoteBufWithMeta {
                buf: rhs.0.into(),
                meta: Some(protocol::MetaTensor {
                    offset: rhs.1.offset as u64,
                    shape: rhs.1.shape.iter().map(|&s| s as u64).collect(),
                    strides: rhs.1.strides.iter().map(|&s| s as i64).collect(),
                }),
            }),
            b: b as u64,
            m: m as u64,
            k: k as u64,
            n: n as u64,
            contiguity: Into::<protocol::ContiguityType>::into(contiguity) as i32,
        };

        // TODO: Send request and parse response to get new RemoteBuf
        todo!()
    }
}
