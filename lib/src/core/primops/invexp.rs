use std::ops::Div;

use crate::core::value::{TensorValue, types::boolean};



pub trait InvExp: TensorValue {
    fn apply_invexp(&self) -> Self;
}



impl InvExp for f32 {
    fn apply_invexp(&self) -> Self {
        (-*self).exp()
    }
}

impl InvExp for f64 {
    fn apply_invexp(&self) -> Self {
        (-*self).exp()
    }
}
