
use core::panic;
use std::{collections::HashMap, io::{Read, Write}, net::{IpAddr, TcpListener, TcpStream}, sync::{atomic::AtomicU32, Arc}, thread};

use crate::{backend::{cpu::Cpu, remote::{protocol::{backend_request::Request, AllocResponse}, TensorId}, Backend}, core::{tensor::{AsView, AsViewMut, TensorError}, untyped::UntypedTensor, value::{self, DType, TensorValue}, Shape, TensorView, TensorViewMut }};

#[cfg(feature = "cuda")]
use crate::backend::cuda::Cuda;

use prost::Message;
use super::protocol;


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

pub(crate) struct BackendServer {
    address: IpAddr,
    port: u16,
    handles: HashMap<std::net::SocketAddr, thread::JoinHandle<()>>,
}

impl BackendServer {
    pub(crate) fn new(address: IpAddr, port: u16) -> Self {
        Self { address, port, handles: HashMap::new() }
    }

    pub(crate) fn serve(&mut self) -> Result<(), std::io::Error> {
        tracing::info!("Starting BackendServer on {}:{}", self.address, self.port);
        let listener = TcpListener::bind((self.address, self.port))?;
        tracing::info!("Server listening on {}:{}", self.address, self.port);
        
        for stream in listener.incoming() {
            let stream = stream?;
            let peer_addr = stream.peer_addr()?;
            tracing::info!("New connection from {}", peer_addr);
            
            let mut handle = ConnectionHandle::new(stream);
            let handle_thread = thread::spawn(move || {
                tracing::debug!("Connection handler thread started for {}", peer_addr);
                handle.serve();
                tracing::debug!("Connection handler thread ending for {}", peer_addr);
            });
            self.handles.insert(peer_addr, handle_thread);
        }
        tracing::info!("Server shutting down");
        Ok(())
    }
}


struct ConnectionHandle {
    cpu: Cpu,
    stream: TcpStream,
    next_tensor_id: Arc<AtomicU32>,
    cpu_buffers_u8: HashMap<TensorId, <Cpu as Backend>::Buf<u8>>,
    #[cfg(feature = "cuda")]
    tensors_cuda: HashMap<TensorId, Box<dyn UntypedTensor<Cuda>>>,
}

impl ConnectionHandle {
    fn new(stream: TcpStream) -> Self {
        Self {
            cpu: Cpu::new(),
            stream,
            next_tensor_id: Arc::new(AtomicU32::new(0)),
            cpu_buffers_u8: HashMap::new(),
            #[cfg(feature = "cuda")]
            tensors_cuda: HashMap::new(),
        }
    }

    fn serve (&mut self) {
        tracing::debug!("ConnectionHandle serve loop starting");
        loop {
            tracing::debug!("Waiting for message size header");
            let mut num_bytes = [0u8; 4];
            self.stream.read_exact(&mut num_bytes).expect("Add better failure handling");
            let msg_len = u32::from_le_bytes(num_bytes) as usize;
            tracing::debug!("Received message size: {} bytes", msg_len);

            let mut buf = vec![0u8; msg_len];
            self.stream.read_exact(&mut buf).expect("Add better failure handling");
            tracing::debug!("Received message data");

            let request = match protocol::BackendRequest::decode(&buf[..]) {
                Ok(req) => {
                    tracing::debug!("Decoded request: {:?}", req.request);
                    req
                },
                Err(e) => {
                    tracing::error!("Failed to decode protobuf message: {}", e);
                    continue;
                }
            };

            tracing::debug!("Processing request");
            let response = self.handle_request(request);
            tracing::debug!("Request handled, encoding response");
            
            let mut response_buf = Vec::new();
            response.encode(&mut response_buf).expect("Failed to encode response");
            let response_len = response_buf.len() as u32;
            
            tracing::debug!("Sending response of {} bytes", response_len);
            self.stream.write_all(&response_len.to_le_bytes()).expect("Failed to send response length");
            self.stream.write_all(&response_buf).expect("Failed to send response");
            tracing::debug!("Response sent successfully");

        }
    }

    fn handle_request(&mut self, request: protocol::BackendRequest) -> protocol::BackendResponse {
        tracing::debug!("Handling request type: {:?}", request.request);
        let response = match request.request.unwrap() {
            Request::AllocFromSlice(req) => {
                let id = self.next_tensor_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tracing::debug!("AllocFromSlice: assigned tensor_id={}, dtype={:?}, data_len={}", id, req.dtype, req.data.len());
                let dtype: DType = req.dtype.into();
                let buf = match dtype {
                    DType::U8 => {
                        let data: Vec<u8> = req.data;
                        tracing::debug!("Allocating {} u8 elements on CPU backend", data.len());
                        let buf = self.cpu.alloc_from_slice(data.into_boxed_slice()).unwrap();
                        self.cpu_buffers_u8.insert(id, buf);
                        tracing::debug!("Buffer allocated and stored successfully");
                        protocol::RemoteBuf {
                            tensor_id: id,
                            dtype: protocol::DType::U8 as i32,
                        }
                    },
                    _ => panic!("Unsupported dtype"),
                };
                protocol::backend_response::Response::Alloc(AllocResponse {
                    success: true,
                    new_buf: Some(buf),
                    error: "".to_string(),
                })
            }
            Request::Alloc(req) => {
                let id = self.next_tensor_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tracing::debug!("Alloc: assigned tensor_id={}, dtype={:?}, len={}", id, req.dtype, req.len);
                let dtype: DType = req.dtype.into();
                let buf = match dtype {
                    DType::U8 => {
                        tracing::debug!("Allocating {} u8 elements on CPU backend", req.len);
                        let buf = self.cpu.alloc(req.len as usize).unwrap();
                        self.cpu_buffers_u8.insert(id, buf);
                        tracing::debug!("Buffer allocated and stored successfully");
                        protocol::RemoteBuf {
                            tensor_id: id,
                            dtype: protocol::DType::U8 as i32,
                        }
                    },
                    _ => panic!("Unsupported dtype"),
                };
                protocol::backend_response::Response::Alloc(AllocResponse {
                    success: true,
                    new_buf: Some(buf),
                    error: "".to_string(),
                })
            },
            Request::Write(protocol::WriteRequest {buf : Some(inner), offset, value} ) => {
                let buf = inner;
                let dtype: DType = buf.dtype.into();
                assert!(buf.dtype == protocol::DType::U8 as i32);
                let b = self.cpu_buffers_u8.get_mut(&buf.tensor_id).expect("Buffer not found");
                let value_bytes = match value.unwrap().value.unwrap() {
                    protocol::tensor_value::Value::U8Value(bytes) => {
                        bytes
                    },
                    _ => panic!("Unsupported value type"),
                };
                let value = u8::from_le_bytes(value_bytes.try_into().unwrap());
                self.cpu.write(b, offset as usize, value).unwrap();
                protocol::backend_response::Response::Operation(protocol::OperationResponse { success: true, error: "".into() })
            },
            Request::Read(protocol::ReadRequest { buf: Some(inner), offset }) => {
                tracing::debug!("Read: tensor_id={}, offset={}", inner.tensor_id, offset);
                let dtype: DType = inner.dtype.into();
                
                let tensor_value = match dtype {
                    DType::U8 => {
                        let b = self.cpu_buffers_u8.get(&inner.tensor_id).expect("Buffer not found");
                        let value = self.cpu.read(b, offset as usize).unwrap();
                        tracing::debug!("Read u8 value: {} at offset {}", value, offset);
                        protocol::TensorValue {
                            value: Some(protocol::tensor_value::Value::U8Value(value.to_le_bytes().to_vec())),
                        }
                    },
                    _ => panic!("Unsupported dtype for read: {:?}", dtype),
                };
                
                protocol::backend_response::Response::Read(protocol::ReadResponse {
                    success: true,
                    value: Some(tensor_value),
                    error: "".to_string(),
                })
            },
            _ => {
                panic!()
            }
        };
        protocol::BackendResponse {
            response: Some(response),
            request_id: request.request_id,
        }
    }
}