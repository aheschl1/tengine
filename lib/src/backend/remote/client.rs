use std::{net::IpAddr, sync::{atomic::AtomicBool, Arc}};

use crate::{backend::{remote::TensorId, Backend}, core::value::TensorValue};


struct RemoteCtx {
    pub address: IpAddr,
    pub port: u16
}

pub struct RemoteBuf {
    pub id: TensorId,
    dirty: Arc<AtomicBool>,
}

pub struct RemoteBackend {
    remote: RemoteCtx,
}

// impl Backend for RemoteBackend {
//     type Buf<T: TensorValue> = RemoteBuf;
// }