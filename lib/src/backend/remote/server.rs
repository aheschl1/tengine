use std::{collections::HashMap, io::{Read, Write}, net::IpAddr, sync::{atomic::AtomicU32, Arc, RwLock}, thread};

use crate::{backend::{cpu::Cpu, cuda::Cuda, remote::{client::RemoteBuf, protocol::{Messages, Request, Response, Slice, TypelessBuf}}, Backend}, core::{primitives::DeviceType, tensor::TensorError, value::DType}};



struct RemoteServer {
    address: IpAddr,
    port: u16
}

// this is pure evil
struct BufferCollection<B:Backend> {
    u8_buffers: HashMap<u32, B::Buf<u8>>,
    u16_buffers: HashMap<u32, B::Buf<u16>>,
    u32_buffers: HashMap<u32, B::Buf<u32>>,
    u64_buffers: HashMap<u32, B::Buf<u64>>,
    u128_buffers: HashMap<u32, B::Buf<u128>>,
    usize_buffers: HashMap<u32, B::Buf<usize>>,
    i8_buffers: HashMap<u32, B::Buf<i8>>,
    i16_buffers: HashMap<u32, B::Buf<i16>>,
    i32_buffers: HashMap<u32, B::Buf<i32>>,
    i64_buffers: HashMap<u32, B::Buf<i64>>,
    i128_buffers: HashMap<u32, B::Buf<i128>>,
    isize_buffers: HashMap<u32, B::Buf<isize>>,
    f32_buffers: HashMap<u32, B::Buf<f32>>,
    f64_buffers: HashMap<u32, B::Buf<f64>>,
}

impl<B: Backend> Default for BufferCollection<B> {
    fn default() -> Self {
        Self {
            u8_buffers: HashMap::new(),
            u16_buffers: HashMap::new(),
            u32_buffers: HashMap::new(),
            u64_buffers: HashMap::new(),
            u128_buffers: HashMap::new(),
            usize_buffers: HashMap::new(),
            i8_buffers: HashMap::new(),
            i16_buffers: HashMap::new(),
            i32_buffers: HashMap::new(),
            i64_buffers: HashMap::new(),
            i128_buffers: HashMap::new(),
            isize_buffers: HashMap::new(),
            f32_buffers: HashMap::new(),
            f64_buffers: HashMap::new(),
        }
    }
}

#[derive(Clone)]
struct ClientConnection {
    #[cfg(feature = "cuda")]
    cuda_buffers: Arc<RwLock<BufferCollection<Cuda>>>,
    cpu_buffers: Arc<RwLock<BufferCollection<Cpu>>>,
    cpu: Cpu,
    next_buffer_id: Arc<AtomicU32>,
    #[cfg(feature = "cuda")]
    cuda: Cuda,
}

impl RemoteServer {
    pub fn new(address: IpAddr, port: u16) -> Self {
        Self {
            address,
            port,
        }
    }

    pub fn serve(&mut self) -> std::io::Result<()> {
        let listener = std::net::TcpListener::bind((self.address, self.port))?;
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let connection = ClientConnection { 
                        cpu_buffers: Arc::new(RwLock::new(BufferCollection::default())),
                        cuda_buffers: Arc::new(RwLock::new(BufferCollection::default())),
                        cpu: Cpu::new(),
                        next_buffer_id: Arc::new(AtomicU32::new(0)),
                        #[cfg(feature = "cuda")]
                        cuda: Cuda::new(),
                    };
                    // launch a new thread 
                    std::thread::spawn(move || {
                        handle_connection(connection, stream);
                    });
                }
                Err(e) => {
                    eprintln!("Connection failed: {}", e);
                }
            }
        }
        Ok(())
    }
}


#[inline(always)]
fn select_buffer(_connection: &ClientConnection) -> DeviceType {
    // For simplicity, we always use CPU buffers in this example.
    DeviceType::Cpu
}

macro_rules! alloc_from_slice_for_dtype {
    ($slice:expr, $connection:expr, $dtype_variant:ident, $rust_type:ty, $buffer_field:ident) => {{
        let boxed_slice = $slice.to_boxed_slice::<$rust_type>()?;
        let device_type = select_buffer($connection);
        let buffer = match device_type {
            DeviceType::Cpu => {
                let buf = $connection.cpu.alloc_from_slice(boxed_slice)?;
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                $connection.cpu_buffers.write().unwrap().$buffer_field.insert(buffer_id, buf);
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let buf = $connection.cuda.alloc_from_slice(boxed_slice)?;
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                $connection.cuda_buffers.write().unwrap().$buffer_field.insert(buffer_id, buf);
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        };
        TypelessBuf::from(buffer)
    }};
}

macro_rules! alloc_for_dtype {
    ($len:expr, $connection:expr, $dtype_variant:ident, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        let buffer = match device_type {
            DeviceType::Cpu => {
                let buf = $connection.cpu.alloc::<$rust_type>($len)?;
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                $connection.cpu_buffers.write().unwrap().$buffer_field.insert(buffer_id, buf);
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let buf = $connection.cuda.alloc::<$rust_type>($len)?;
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                $connection.cuda_buffers.write().unwrap().$buffer_field.insert(buffer_id, buf);
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        };
        TypelessBuf::from(buffer)
    }};
}

macro_rules! copy_from_slice_for_dtype {
    ($dst_id:expr, $src_slice:expr, $connection:expr, $rust_type:ty, $buffer_field:ident) => {{
        let boxed_slice = $src_slice.to_boxed_slice::<$rust_type>()?;
        let src_slice_ref: &[$rust_type] = &boxed_slice;
        let device_type = select_buffer($connection);
        match device_type {
            DeviceType::Cpu => {
                let mut buffers = $connection.cpu_buffers.write().unwrap();
                let dst_buf = buffers.$buffer_field
                    .get_mut(&$dst_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $dst_id)))?;
                $connection.cpu.copy_from_slice(dst_buf, src_slice_ref)?;
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let mut buffers = $connection.cuda_buffers.write().unwrap();
                let dst_buf = buffers.$buffer_field
                    .get_mut(&$dst_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $dst_id)))?;
                $connection.cuda.copy_from_slice(dst_buf, src_slice_ref)?;
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        }
        Ok(())
    }};
}

#[inline(always)]
fn handle_alloc_from_slice(
    slice: Slice,
    connection: &ClientConnection,
) -> Result<TypelessBuf, TensorError> {
    let remote_buf = match slice.dtype {
        DType::U8 => alloc_from_slice_for_dtype!(slice, connection, U8, u8, u8_buffers),
        DType::U16 => alloc_from_slice_for_dtype!(slice, connection, U16, u16, u16_buffers),
        DType::U32 => alloc_from_slice_for_dtype!(slice, connection, U32, u32, u32_buffers),
        DType::U64 => alloc_from_slice_for_dtype!(slice, connection, U64, u64, u64_buffers),
        DType::U128 => alloc_from_slice_for_dtype!(slice, connection, U128, u128, u128_buffers),
        DType::I8 => alloc_from_slice_for_dtype!(slice, connection, I8, i8, i8_buffers),
        DType::I16 => alloc_from_slice_for_dtype!(slice, connection, I16, i16, i16_buffers),
        DType::I32 => alloc_from_slice_for_dtype!(slice, connection, I32, i32, i32_buffers),
        DType::I64 => alloc_from_slice_for_dtype!(slice, connection, I64, i64, i64_buffers),
        DType::I128 => alloc_from_slice_for_dtype!(slice, connection, I128, i128, i128_buffers),
        DType::F32 => alloc_from_slice_for_dtype!(slice, connection, F32, f32, f32_buffers),
        DType::F64 => alloc_from_slice_for_dtype!(slice, connection, F64, f64, f64_buffers),
    };
    Ok(remote_buf)
}

#[inline(always)]
fn handle_alloc(
    len: usize,
    dtype: DType,
    connection: &ClientConnection,
) -> Result<TypelessBuf, TensorError> {
    let remote_buf = match dtype {
        DType::U8 => alloc_for_dtype!(len, connection, U8, u8, u8_buffers),
        DType::U16 => alloc_for_dtype!(len, connection, U16, u16, u16_buffers),
        DType::U32 => alloc_for_dtype!(len, connection, U32, u32, u32_buffers),
        DType::U64 => alloc_for_dtype!(len, connection, U64, u64, u64_buffers),
        DType::U128 => alloc_for_dtype!(len, connection, U128, u128, u128_buffers),
        DType::I8 => alloc_for_dtype!(len, connection, I8, i8, i8_buffers),
        DType::I16 => alloc_for_dtype!(len, connection, I16, i16, i16_buffers),
        DType::I32 => alloc_for_dtype!(len, connection, I32, i32, i32_buffers),
        DType::I64 => alloc_for_dtype!(len, connection, I64, i64, i64_buffers),
        DType::I128 => alloc_for_dtype!(len, connection, I128, i128, i128_buffers),
        DType::F32 => alloc_for_dtype!(len, connection, F32, f32, f32_buffers),
        DType::F64 => alloc_for_dtype!(len, connection, F64, f64, f64_buffers),
    };
    Ok(remote_buf)
}

#[inline(always)]
fn handle_copy_from_slice(
    dst: TypelessBuf,
    src: Slice,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match dst.dtype {
        DType::U8 => copy_from_slice_for_dtype!(dst.id, src, connection, u8, u8_buffers),
        DType::U16 => copy_from_slice_for_dtype!(dst.id, src, connection, u16, u16_buffers),
        DType::U32 => copy_from_slice_for_dtype!(dst.id, src, connection, u32, u32_buffers),
        DType::U64 => copy_from_slice_for_dtype!(dst.id, src, connection, u64, u64_buffers),
        DType::U128 => copy_from_slice_for_dtype!(dst.id, src, connection, u128, u128_buffers),
        DType::I8 => copy_from_slice_for_dtype!(dst.id, src, connection, i8, i8_buffers),
        DType::I16 => copy_from_slice_for_dtype!(dst.id, src, connection, i16, i16_buffers),
        DType::I32 => copy_from_slice_for_dtype!(dst.id, src, connection, i32, i32_buffers),
        DType::I64 => copy_from_slice_for_dtype!(dst.id, src, connection, i64, i64_buffers),
        DType::I128 => copy_from_slice_for_dtype!(dst.id, src, connection, i128, i128_buffers),
        DType::F32 => copy_from_slice_for_dtype!(dst.id, src, connection, f32, f32_buffers),
        DType::F64 => copy_from_slice_for_dtype!(dst.id, src, connection, f64, f64_buffers),
    }
}

#[inline(always)]
fn handle_request(
    request: Request, 
    connection: &ClientConnection,
    stream: &mut std::net::TcpStream,
) -> Result<(), TensorError> {
    let task_id = request.task_id;
    match request.message {
        Messages::AllocFromSlice { slice } => {
            let remote_buf = handle_alloc_from_slice(slice, connection)?;
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::AllocFromSliceResponse { buf: Ok(remote_buf) },
            };
            send_response(stream, &response)?;
            Ok(())
        },
        Messages::Alloc { len, dtype } => {
            let remote_buf = handle_alloc(len, dtype, connection)?;
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::AllocResponse { buf: Ok(remote_buf) },
            };
            send_response(stream, &response)?;
            Ok(())
        },
        Messages::CopyFromSlice { dst, src } => {
            // Send initial acknowledgment
            let ack_response = Response {
                asynchronous: true,
                complete: false,
                task_id,
                message: Messages::CopyFromSliceResponse { result: Ok(()) },
            };
            send_response(stream, &ack_response)?;
            
            // Clone what we need for the background thread
            let connection_clone = connection.clone();
            let stream_clone = stream.try_clone()
                .map_err(|e| TensorError::RemoteError(format!("Failed to clone stream: {}", e)))?;
            
            // Spawn background task
            thread::spawn(move || {
                let result = handle_copy_from_slice(dst, src, &connection_clone);
                let completion_response = Response {
                    asynchronous: true,
                    complete: true,
                    task_id,
                    message: Messages::CopyFromSliceResponse { result },
                };
                let _ = send_response(&mut stream_clone.try_clone().unwrap(), &completion_response);
            });
            
            Ok(())
        },
        _ => {
            Err(TensorError::RemoteError("Unsupported operation".into()))
        }
    }
}

fn send_response(stream: &mut std::net::TcpStream, response: &Response) -> Result<(), TensorError> {
    let serialized = response.serialize()
        .map_err(|e| TensorError::RemoteError(format!("Failed to serialize response: {}", e)))?;
    let n = serialized.len() as u32;
    stream.write_all(&n.to_le_bytes())
        .map_err(|e| TensorError::RemoteError(format!("Failed to write response length: {}", e)))?;
    stream.write_all(&serialized)
        .map_err(|e| TensorError::RemoteError(format!("Failed to write response: {}", e)))?;
    Ok(())
}

fn handle_connection(connection: ClientConnection, mut stream: std::net::TcpStream) {
    // Handle communication with the client
    let mut n_buffer = [0u8; 4];
    loop {
        // Read from and write to connection.stream
        match stream.read_exact(&mut n_buffer) {
            Ok(_) => {
                let n = u32::from_le_bytes(n_buffer) as usize;
                let mut data_buffer = vec![0u8; n];
                match stream.read_exact(&mut data_buffer) {
                    Ok(_) => {
                        let request = Request::deserialize(&data_buffer).expect("Failed to deserialize request");
                        if let Err(e) = handle_request(request, &connection, &mut stream) {
                            eprintln!("Failed to handle request: {}", e);
                            // TODO: Send error response back to client
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to read data: {}", e);
                        break;
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to read size: {}", e);
                break;
            }
        }
    }
}