// Include the generated protobuf code
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/tensors.backend.rs"));
}

pub use proto::*;


impl From<crate::core::value::DType> for DType {
    fn from(dtype: crate::core::value::DType) -> Self {
        match dtype {
            crate::core::value::DType::U8 => DType::U8,
            crate::core::value::DType::I8 => DType::I8,
            crate::core::value::DType::U16 => DType::U16,
            crate::core::value::DType::I16 => DType::I16,
            crate::core::value::DType::U32 => DType::U32,
            crate::core::value::DType::U128 => DType::U128,
            crate::core::value::DType::I32 => DType::I32,
            crate::core::value::DType::U64 => DType::U64,
            crate::core::value::DType::I64 => DType::I64,
            crate::core::value::DType::I128 => DType::I128,
            crate::core::value::DType::F32 => DType::F32,
            crate::core::value::DType::F64 => DType::F64,
        }
    }
}

impl From<DType> for crate::core::value::DType {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::U8 => crate::core::value::DType::U8,
            DType::I8 => crate::core::value::DType::I8,
            DType::U16 => crate::core::value::DType::U16,
            DType::I16 => crate::core::value::DType::I16,
            DType::U32 => crate::core::value::DType::U32,
            DType::U128 => crate::core::value::DType::U128,
            DType::I32 => crate::core::value::DType::I32,
            DType::U64 => crate::core::value::DType::U64,
            DType::I64 => crate::core::value::DType::I64,
            DType::I128 => crate::core::value::DType::I128,
            DType::F32 => crate::core::value::DType::F32,
            DType::F64 => crate::core::value::DType::F64,
        }
    }
}

impl From<i32> for crate::core::value::DType {
    fn from(dtype: i32) -> Self {
        DType::try_from(dtype).unwrap().into()
    }
}


impl From<crate::ops::base::OpType> for OpType {
    fn from(op: crate::ops::base::OpType) -> Self {
        match op {
            crate::ops::base::OpType::Add => OpType::Add,
            crate::ops::base::OpType::Sub => OpType::Sub,
            crate::ops::base::OpType::Mul => OpType::Mul,
        }
    }
}

impl From<OpType> for crate::ops::base::OpType {
    fn from(op: OpType) -> Self {
        match op {
            OpType::Add => crate::ops::base::OpType::Add,
            OpType::Sub => crate::ops::base::OpType::Sub,
            OpType::Mul => crate::ops::base::OpType::Mul,
        }
    }
}

impl From<crate::core::meta::ContiguityTypes> for ContiguityType {
    fn from(contiguity: crate::core::meta::ContiguityTypes) -> Self {
        match contiguity {
            crate::core::meta::ContiguityTypes::RowMajor => ContiguityType::RowMajor,
            crate::core::meta::ContiguityTypes::ColumnMajor => ContiguityType::ColumnMajor,
            crate::core::meta::ContiguityTypes::None => ContiguityType::None,
        }
    }
}

impl From<ContiguityType> for crate::core::meta::ContiguityTypes {
    fn from(contiguity: ContiguityType) -> Self {
        match contiguity {
            ContiguityType::RowMajor => crate::core::meta::ContiguityTypes::RowMajor,
            ContiguityType::ColumnMajor => crate::core::meta::ContiguityTypes::ColumnMajor,
            ContiguityType::None => crate::core::meta::ContiguityTypes::None,
        }
    }
}

impl From<crate::core::primitives::DeviceType> for DeviceType {
    fn from(device: crate::core::primitives::DeviceType) -> Self {
        match device {
            crate::core::primitives::DeviceType::Cpu => DeviceType::Cpu,
            #[cfg(feature = "cuda")]
            crate::core::primitives::DeviceType::Cuda(_) => DeviceType::Cuda,
            #[cfg(feature = "remote")]
            crate::core::primitives::DeviceType::Remote { .. } => DeviceType::Remote,
        }
    }
}

impl From<crate::core::primitives::DeviceType> for Device {
    fn from(device: crate::core::primitives::DeviceType) -> Self {
        match device {
            crate::core::primitives::DeviceType::Cpu => Device {
                device: Some(proto::device::Device::Cpu(CpuDevice {})),
            },
            #[cfg(feature = "cuda")]
            crate::core::primitives::DeviceType::Cuda(index) => Device {
                device: Some(proto::device::Device::Cuda(CudaDevice {
                    device_index: index as u32,
                })),
            },
            #[cfg(feature = "remote")]
            crate::core::primitives::DeviceType::Remote { ip, port, remote_type } => Device {
                device: Some(proto::device::Device::Remote(Box::new(RemoteDeviceInfo {
                    ip: ip.to_string(),
                    port: port as u32,
                    remote_type: Some(Box::new((*remote_type).into())),
                }))),
            },
        }
    }
}

impl TryFrom<Device> for crate::core::primitives::DeviceType {
    type Error = &'static str;

    fn try_from(device: Device) -> Result<Self, Self::Error> {
        match device.device {
            Some(proto::device::Device::Cpu(_)) => Ok(crate::core::primitives::DeviceType::Cpu),
            #[cfg(feature = "cuda")]
            Some(proto::device::Device::Cuda(cuda)) => {
                Ok(crate::core::primitives::DeviceType::Cuda(cuda.device_index as usize))
            },
            #[cfg(feature = "remote")]
            Some(proto::device::Device::Remote(remote_info)) => {
                use std::net::IpAddr;
                let ip: IpAddr = remote_info.ip.parse()
                    .map_err(|_| "Invalid IP address")?;
                let remote_type = remote_info.remote_type
                    .ok_or("Missing remote_type")?;
                let remote_device = (*remote_type).try_into()?;
                Ok(crate::core::primitives::DeviceType::Remote {
                    ip,
                    port: remote_info.port as u16,
                    remote_type: Box::new(remote_device),
                })
            },
            None => Err("Device oneof is not set"),
            #[cfg(not(feature = "cuda"))]
            Some(proto::device::Device::Cuda(_)) => Err("CUDA feature not enabled"),
            #[cfg(not(feature = "remote"))]
            Some(proto::device::Device::Remote(_)) => Err("Remote feature not enabled"),
        }
    }
}

/// Convert bytes and dtype to the appropriate TensorValue variant
impl proto::TensorValue {
    pub fn from_bytes_and_dtype(bytes: Vec<u8>, dtype: DType) -> Self {
        use proto::tensor_value::Value;
        
        let value = match dtype {
            DType::F32 => Value::F32Value(bytes),
            DType::F64 => Value::F64Value(bytes),
            DType::I8 => Value::I8Value(bytes),
            DType::I16 => Value::I16Value(bytes),
            DType::I32 => Value::I32Value(bytes),
            DType::I64 => Value::I64Value(bytes),
            DType::I128 => Value::I128Value(bytes),
            DType::U8 => Value::U8Value(bytes),
            DType::U16 => Value::U16Value(bytes),
            DType::U32 => Value::U32Value(bytes),
            DType::U64 => Value::U64Value(bytes),
            DType::U128 => Value::U128Value(bytes),
        };
        
        TensorValue {
            value: Some(value),
        }
    }
}
