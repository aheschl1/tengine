use std::{collections::HashMap, fmt::Debug, io::{Read, Write}, net::IpAddr, sync::{atomic::{AtomicBool, AtomicU32, Ordering}, Arc, Condvar, Mutex, RwLock}};

use crate::{backend::{remote::{get_backend_default, protocol::{Request, Response, Slice, TypelessBuf, Value}}, Backend, BackendMatMul}, core::{primitives::DeviceType, primops::{Exp, InvExp}, tensor::TensorError, value::{DType, TensorValue}}};
use flume;


#[derive(Debug, Clone)]
pub struct RemoteBuf<T: TensorValue> {
    pub(crate) id: u32,
    pub(crate) dtype: DType,
    pub (crate) _marker: std::marker::PhantomData<T>,
}

impl<T: TensorValue> RemoteBuf<T> {
    #[inline(always)]
    fn to_typeless(&self) -> TypelessBuf {
        TypelessBuf {
            id: self.id,
            dtype: self.dtype,
        }
    }

    #[inline(always)]
    pub(crate) fn from_typeless(buf: TypelessBuf) -> Self {
        Self {
            id: buf.id,
            dtype: buf.dtype,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: TensorValue> From<&mut RemoteBuf<T>> for TypelessBuf {
    fn from(buf: &mut RemoteBuf<T>) -> Self {
        buf.to_typeless()
    }
}

impl<T: TensorValue> From<&RemoteBuf<T>> for TypelessBuf {
    fn from(buf: &RemoteBuf<T>) -> Self {
        buf.to_typeless()
    }
}

impl<T: TensorValue> From<*const RemoteBuf<T>> for TypelessBuf {
    fn from(buf: *const RemoteBuf<T>) -> Self {
        unsafe { (&*buf).to_typeless() }
    }
}

impl<T: TensorValue> From<*mut RemoteBuf<T>> for TypelessBuf {
    fn from(buf: *mut RemoteBuf<T>) -> Self {
        unsafe { (&*buf).to_typeless() }
    }
}

#[derive(Debug)]
struct PendingHandler {
    count: Arc<AtomicU32>,
    cv: Condvar,
    mutex: Mutex<()>,
}

impl PendingHandler {
    fn inc(&self) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }
    fn dec(&self) {
        let prev = self.count.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            self.cv.notify_all();
        }
    }
    fn sync(&self) {
        while self.count.load(Ordering::SeqCst) > 0 {
            let lock = self.mutex.lock().unwrap();
            let _unused = self.cv.wait(lock).unwrap();
        }
    }
}

#[derive(Clone)]
pub struct RemoteBackend {
    remote_addr: IpAddr,
    remote_port: u16,
    message_id: Arc<AtomicU32>,
    pending: Arc<PendingHandler>,
    messages_outgoing_sender: flume::Sender<Request>,
    messages_outgoing_receiver: flume::Receiver<Request>,
    pending_response: Arc<RwLock<HashMap<u32, flume::Sender<rpc_proc::RpcMessages>>>>,
    poisoned: Arc<AtomicBool>,
}
    
impl Debug for RemoteBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteBackend")
            .field("remote_addr", &self.remote_addr)
            .field("remote_port", &self.remote_port)
            .field("message_id", &self.message_id)
            .field("pending", &self.pending)
            .field("pending_response", &self.pending_response)
            .finish()
    }
}

impl RemoteBackend {
    pub fn new_with_address(remote_addr: IpAddr, remote_port: u16) -> Result<Self, std::io::Error> {
        let pending = PendingHandler {
            count: Arc::new(AtomicU32::new(0)),
            cv: Condvar::new(),
            mutex: Mutex::new(()),
        };
        let (sender, receiver) = flume::unbounded();
        let res = Self {
            remote_addr,
            remote_port,
            pending: Arc::new(pending),
            message_id: Arc::new(AtomicU32::new(0)),
            messages_outgoing_sender: sender,
            messages_outgoing_receiver: receiver,
            pending_response: Arc::new(RwLock::new(HashMap::new())),
            poisoned: Arc::new(AtomicBool::new(false)),
        };
        Ok(res)
    }

    fn poison(&self) {
        self.poisoned.store(true, Ordering::SeqCst);
    }

    pub fn sync(&self) {
        self.pending.sync();
    }

    #[inline(always)]
    fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst)
    }

    fn send_message(&self, msg: rpc_proc::RpcMessages) -> flume::Receiver<rpc_proc::RpcMessages>{
        if self.is_poisoned() {
            panic!("Attempted to send message on poisoned RemoteBackend. Reasons for poison:
            1. An asynchronous operation reported an error from the remote backend.
            2. The RemoteBackend is in an inconsistent state and can no longer process messages safely.");
        }
        self.pending.inc();
        let (sender, receiver) = flume::bounded(1);
        let mid = self.next_message_id();
        {
            let mut pending = self.pending_response.write().unwrap();
            pending.insert(mid, sender);
        }
        let req = Request {
            task_id: mid,
            message: msg,
        };
        self.messages_outgoing_sender.send(req).expect("Failed to send message to outgoing channel");
        receiver
    }

    pub(crate) fn connect(&mut self) -> Result<(), std::io::Error> {
        let stream = std::net::TcpStream::connect((self.remote_addr, self.remote_port))?;
        stream.set_nodelay(true)?;

        let read_stream = stream.try_clone()?;
        let write_stream = stream;

        let remote = self.clone();
        std::thread::spawn(move || {
            drain_outgoing(remote, write_stream);
        });
        let remote = self.clone();
        std::thread::spawn(move || {
            read_incoming(remote, read_stream);
        });
        Ok(())
    }

    pub fn address(&self) -> (IpAddr, u16) {
        (self.remote_addr, self.remote_port)
    }

    #[inline(always)]
    fn next_message_id(&self) -> u32 {
        self.message_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

}

macro_rules! send_recv {
    ($self:expr, $message:expr, $response_pattern:pat => $result:expr) => {{
        let receiver = $self.send_message($message);
        let response = receiver.recv()
            .map_err(|_| TensorError::BackendError("Failed to receive response".to_string()))?;
        match response {
            $response_pattern => $result,
            _ => Err(TensorError::BackendError("Unexpected response type".to_string())),
        }
    }};
}

#[rpc_proc::routines(
    Box<[T]> = Slice,
    &T = Slice,
    Self::Buf<T> = TypelessBuf,
    &Self::Buf<T> = TypelessBuf,
    &mut Self::Buf<T> = TypelessBuf,
    T = Value,
    &[T] = Slice,
    Result<Self::Buf<T>, crate::core::tensor::TensorError> = Result<TypelessBuf, TensorError>,
    Result<T, crate::core::tensor::TensorError> = Result<Value, TensorError>,
    Result<Box<[T]>, crate::core::tensor::TensorError> = Result<Slice, crate::core::tensor::TensorError>,
    (*const Self::Buf<T>, &crate::core::MetaTensor) = (TypelessBuf, crate::core::MetaTensor),
    (*mut Self::Buf<T>, &crate::core::MetaTensor) = (TypelessBuf, crate::core::MetaTensor),
    (crate::ops::base::BinaryOpType, T) = (crate::ops::base::BinaryOpType, Value),
    &[usize] = Vec<usize>,
    &[isize] = Vec<isize>,
)]
impl Backend for RemoteBackend {
    type Buf<T: TensorValue> = RemoteBuf<T>;

    #[rpc(skip)]
    fn new() -> Self {
        get_backend_default().expect("No default remote backend available")
    }
    
    #[rpc(skip)]
    fn device_type() -> crate::core::primitives::DeviceType {
        // let message = Messages::DeviceType;
        // let receiver = self.send_message(message);
        // let response = receiver.recv().unwrap();
        // match response {
        //     Messages::DeviceTypeResponse { device_type } => device_type,
        //     _ => panic!("Unexpected response type"),
        // }
        DeviceType::Remote {
            ip: "127.0.0.1".parse().unwrap(),
            port: 7878,
            remote_type: DeviceType::Cpu.into(),
        }
    }

    fn alloc_from_slice<T: TensorValue>(&self, src: Box<[T]>) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        let message = Messages::AllocFromSlice {
            slice: src.into(),
        };
        send_recv!(self, message, Messages::AllocFromSliceResponse { buf } => {
            buf?.into()
        })
    }

    #[rpc(extra(dtype: DType = T::DTYPE))]
    fn alloc<T: TensorValue>(&self, len: usize) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        let message = Messages::Alloc {
            len: len.into(),
            dtype: T::DTYPE,
        };
        send_recv!(self, message, Messages::AllocResponse { buf } => {
            buf?.into()
        })
    }

    fn copy_from_slice<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &[T]) -> Result<(), crate::core::tensor::TensorError> {
        self.pending.sync();
        let message = Messages::CopyFromSlice {
            dst: dst.into(),
            src: src.into(),
        };
        send_recv!(self, message, Messages::CopyFromSliceResponse { result } => {result?.into()})
    }

    fn read<T: TensorValue>(&self, buf: &Self::Buf<T>, offset: usize) -> Result<T, crate::core::tensor::TensorError> {
        self.pending.sync();
        let message = Messages::Read {
            buf: buf.into(),
            offset: offset.into(),
        };
        send_recv!(self, message, Messages::ReadResponse { value } => {
            value?.into()
        })
    }

    fn write<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, value: T) -> Result<(), crate::core::tensor::TensorError> {
        self.pending.sync();
        let message = Messages::Write {
            buf: buf.into(),
            offset: offset.into(),
            value: value.into(),
        };
        send_recv!(self, message, Messages::WriteResponse { result } => {result?.into()})
    }

    fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize {
        let message = Messages::Len {
            buf: buf.into(),
        };
        let receiver = self.send_message(message);
        match receiver.recv() {
            Ok(Messages::LenResponse { len }) => len,
            _ => panic!("Failed to get buffer length or unexpected response"),
        }
    }

    fn copy<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        self.pending.sync();
        let message = Messages::Copy {
            src: src.into(),
        };
        send_recv!(self, message, Messages::CopyResponse { buf } => {
            buf?.into()
        })
    }

    fn dump<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Box<[T]>, crate::core::tensor::TensorError> {
        self.pending.sync();
        let message = Messages::Dump {
            src: src.into(),
        };
        send_recv!(self, message, Messages::DumpResponse { data } => {
            data?.into()
        })
    }

    fn broadcast<T: TensorValue>(
        &self,
        left: (*const Self::Buf<T>, &crate::core::MetaTensor), 
        right: (*const Self::Buf<T>, &crate::core::MetaTensor),
        dst: (*mut Self::Buf<T>, &crate::core::MetaTensor),
        op: crate::ops::base::BinaryOpType
    ) -> Result<(), crate::core::tensor::TensorError> {
        let message = Messages::Broadcast {
            left: (left.0.into(), left.1.into()),
            right: (right.0.into(), right.1.into()),
            dst: (dst.0.into(), dst.1.into()),
            op: op.into(),
        };
        send_recv!(self, message, Messages::BroadcastResponse { result } => {result?.into()})
    }

    fn apply_elementwise_binary_contiguous<T: TensorValue>(
        &self, 
        buf: &mut Self::Buf<T>, 
        op: (crate::ops::base::BinaryOpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), crate::core::tensor::TensorError> {
        let message = Messages::ApplyElementwiseBinaryContiguous {
            buf: buf.into(),
            op: (op.0.into(), op.1.into()),
            start: start.into(),
            len: len.into(),
        };
        send_recv!(self, message, Messages::ApplyElementwiseBinaryContiguousResponse { result } => {result?.into()})
    }

    fn apply_elementwise_binary_1d_strided<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (crate::ops::base::BinaryOpType, T), 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), crate::core::tensor::TensorError> {
        let message = Messages::ApplyElementwiseBinary1DStrided {
            buf: buf.into(),
            op: (op.0.into(), op.1.into()),
            offset: offset.into(),
            stride: stride.into(),
            len: len.into(),
        };
        send_recv!(self, message, Messages::ApplyElementwiseBinary1DStridedResponse { result } => {result?.into()})
    }
    
    fn apply_elementwise_binary_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        op: (crate::ops::base::BinaryOpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        let message = Messages::ApplyElementwiseBinaryND {
            buf: buf.into(),
            op: (op.0.into(), op.1.into()),
            offset: offset.into(),
            shape: shape.into(),
            stride: stride.into(),
        };
        send_recv!(self, message, Messages::ApplyElementwiseBinaryNDResponse { result } => {result?.into()})
    }
    
    fn apply_neg_contiguous<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        // async jon
        let message: Messages = Messages::ApplyNegContiguous {
            buf: buf.into(),
            start: start.into(),
            len: len.into(),
        };
        send_recv!(self, message, Messages::ApplyNegContiguousResponse { result } => result)
    }
    
    fn apply_neg_1d_strided<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {
        let message = Messages::ApplyNeg1DStrided {
            buf: buf.into(),
            offset: offset.into(),
            stride: stride.into(),
            len: len.into(),
        };
        send_recv!(self, message, Messages::ApplyNeg1DStridedResponse { result } => {result?.into()})
    }
    
    fn apply_neg_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        let message = Messages::ApplyNegND {
            buf: buf.into(),
            offset: offset.into(),
            shape: shape.into(),
            stride: stride.into(),
        };
        send_recv!(self, message, Messages::ApplyNegNDResponse { result } => {result?.into()})
    }
    
    fn apply_relu_nd<T:TensorValue>(&self,buf: &mut Self::Buf<T>,offset:usize,shape: &[usize],stride: &[isize],) -> Result<(),TensorError>  {
        todo!()
    }
    
    fn apply_relu_1d_strided<T:TensorValue>(&self,buf: &mut Self::Buf<T>,offset:usize,stride:isize,len:usize) -> Result<(),TensorError>  {
        todo!()
    }
    
    fn apply_relu_contiguous<T:TensorValue>(&self,buf: &mut Self::Buf<T>,start:usize,len:usize) -> Result<(),TensorError>  {
        todo!()
    }
    
    fn apply_sigmoid_nd<T:TensorValue>(&self,buf: &mut Self::Buf<T>,offset:usize,shape: &[usize],stride: &[isize],) -> Result<(),TensorError>where T:InvExp {
        todo!()
    }
    
    fn apply_sigmoid_1d_strided<T:TensorValue>(&self,buf: &mut Self::Buf<T>,offset:usize,stride:isize,len:usize) -> Result<(),TensorError>where T:InvExp {
        todo!()
    }
    
    fn apply_sigmoid_contiguous<T:TensorValue>(&self,buf: &mut Self::Buf<T>,start:usize,len:usize) -> Result<(),TensorError>where T:InvExp {
        todo!()
    }
    
    fn apply_tanh_nd<T:TensorValue>(&self,buf: &mut Self::Buf<T>,offset:usize,shape: &[usize],stride: &[isize],) -> Result<(),TensorError>where T:Exp {
        todo!()
    }
    
    fn apply_tanh_1d_strided<T:TensorValue>(&self,buf: &mut Self::Buf<T>,offset:usize,stride:isize,len:usize) -> Result<(),TensorError>where T:Exp {
        todo!()
    }
    
    fn apply_tanh_contiguous<T:TensorValue>(&self,buf: &mut Self::Buf<T>,start:usize,len:usize) -> Result<(),TensorError>where T:Exp {
        todo!()
    }
}

impl<T: TensorValue> BackendMatMul<T> for RemoteBackend {
    fn matmul(
        &self,
        lhs: (&Self::Buf<T>, &crate::core::MetaTensor),
        rhs: (&Self::Buf<T>, &crate::core::MetaTensor),
        dst: &mut Self::Buf<T>,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        contiguity: crate::core::meta::ContiguityTypes,
    ) -> Result<(), TensorError> {
        let message: Messages = Messages::Matmul {
            lhs: (lhs.0.to_typeless(), lhs.1.clone()),
            rhs: (rhs.0.to_typeless(), rhs.1.clone()),
            dst: dst.to_typeless(),
            b,
            m,
            k,
            n,
            contiguity,
        };
        send_recv!(self, message, Messages::MatmulResponse { result } => result)
    }
}

#[inline]
fn read_response(stream: &mut std::net::TcpStream, len_buf: &mut  [u8; 4]) -> Result<Response, Box<bincode::ErrorKind>> {
    stream.read_exact(len_buf).unwrap();
    let msg_len = u32::from_le_bytes(*len_buf) as usize;
    let mut msg_buf = vec![0u8; msg_len];
    stream.read_exact(&mut msg_buf).unwrap();
    Response::deserialize(&msg_buf)
}

#[inline]
fn send_message_to_channel(remote: &RemoteBackend, msg: Response) {
    let task_id = msg.task_id;
    let sender = {
        let mut pending = remote.pending_response.write().unwrap();
        pending.remove(&task_id)
    };
    if let Some(sender) = sender {
        sender.send(msg.message).unwrap();
    }
}

// todo, make this async

/// Thread function to drain outgoing messages and send them over the TCP stream
/// # Arguments
/// * `remote` - The RemoteBackend instance
/// * `stream` - The TCP stream to send messages over
fn drain_outgoing(remote: RemoteBackend, mut stream: std::net::TcpStream) {
    let receiver = remote.messages_outgoing_receiver.clone();
    loop {        
        if let Ok(req) = receiver.recv() {
            if remote.is_poisoned() {
                break;
            }
            let serialized = req.serialize().unwrap();
            let n = serialized.len();
            let n_bytes = (n as u32).to_le_bytes();
            stream.write_all(&n_bytes).unwrap();
            stream.write_all(&serialized).unwrap();
        } else {
            // Channel closed, exit thread
            break;
        }
    }
}

/// Thread function to read incoming messages from the TCP stream
/// 
/// Upon receiving a message, if it is marked as not asynchronous, it sends the message to the waiting channel
/// Otherwise, it handles asynchronous messages accordingly. If the asynchronous message is complete and does not
/// indicate an error, it simply decrements the pending count. If it indicates an error, it poisons the RemoteBackend to prevent further operations.
/// If the asynchronous message is not complete, it sends a follow-up message to the waiting channel, and does not decrement the pending count.
/// 
/// # Arguments
/// * `remote` - The RemoteBackend instance
/// * `stream` - The TCP stream to read messages from
fn read_incoming(remote: RemoteBackend, mut stream: std::net::TcpStream) {
    let mut len_buf = [0u8; 4];
    loop {
        let msg = read_response(&mut stream, &mut len_buf).unwrap();
        if !msg.asynchronous {
            debug_assert!(msg.complete);
            send_message_to_channel(&remote, msg);
            remote.pending.dec();
        }else if msg.complete {
            // no need to send follow up, just decrement pending. because, there was an incomplete one before
            // that sent the message. async followup is just a backend notification.
            // nobody waits on follow up of async
            if let Some(e) = msg.error {
                remote.poison();
                panic!("Inconsistent state detected. Received error in async message: {:?}", e);
            } 
            remote.pending.dec();
        }else{
            //send initial follow up to receiver, do not decrement pending yet
            send_message_to_channel(&remote, msg);
        }
    }
}
