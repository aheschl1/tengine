use std::{collections::HashMap, hash::Hash};

use rand::distr::uniform;
use slotmap::new_key_type;

use crate::{backend::Backend, core::{primitives::TensorBase, tensor::{RandomTensor, TensorError}, value::TensorValue, MetaTensorView}};
use crate::ops::linalg::MatMul;

#[derive(thiserror::Error)]
#[derive(Debug)]
pub enum GraphError {
    #[error("Invalid input provided {0}")]
    InvalidInput(String),
    #[error("Invalid input provided {0}")]
    ComputationError(String),
}


new_key_type! {
    pub struct TensorId;
}

pub trait Node<T: TensorValue, B: Backend> {
    fn id(&self) -> usize;
    fn weights(&self) -> &HashMap<String, TensorBase<T, B>>;
    fn apply(&self, inputs: Vec<TensorBase<T, B>>) -> Result<Vec<TensorBase<T, B>>, GraphError>;
}

pub struct Group<T: TensorValue, B: Backend> {
    nodes: slotmap::SlotMap<TensorId, Box<dyn Node<T, B>>>,
    children: HashMap<TensorId, Vec<TensorId>>,
    roots: Vec<TensorId>,
}

impl<T: TensorValue, B: Backend> Group<T, B> {
    pub fn new() -> Self {
        Self {
            nodes: slotmap::SlotMap::with_key(),
            children: HashMap::new(),
            roots: Vec::new(),
        }
    }

    pub fn add_node() {
        todo!()
    }


}

impl<T: TensorValue, B: Backend> Node<T, B> for Group<T, B> {
    fn id(&self) -> usize {
        todo!()
    }

    fn weights(&self) -> &HashMap<String, TensorBase<T, B>> {
        todo!()
    }

    fn apply(&self, inputs: Vec<TensorBase<T, B>>) -> Result<Vec<TensorBase<T, B>>, GraphError> {
        todo!()
    }
}


/// linear layer operates over a tensor of size [B, in_features] and produces a tensor of size [B, out_features]
struct Linear<T: TensorValue, B: Backend> {
    in_features: usize,
    out_features: usize,
    weights: HashMap<String, TensorBase<T, B>>,
}

impl <T: TensorValue + uniform::SampleUniform, B: Backend> Linear<T, B> {
    pub fn new(in_features: usize, out_features: usize) -> Result<Self, TensorError> {
        Self::new_random(in_features, out_features)
    }

    pub fn new_random(in_features: usize, out_features: usize) -> Result<Self, TensorError> {
        let weight = TensorBase::<T, B>::uniform((in_features, out_features))?;
        let bias = TensorBase::<T, B>::uniform((out_features,))?;

        let mut weights = HashMap::new();
        weights.insert("weight".to_string(), weight);
        weights.insert("bias".to_string(), bias);

        Ok(Self {
            in_features,
            out_features,
            weights,
        })
    }
}

impl<T: TensorValue, B: Backend> Node<T, B> for Linear<T, B> {
    fn id(&self) -> usize {
        todo!()
    }

    fn weights(&self) -> &HashMap<String, TensorBase<T, B>> {
        &self.weights
    }

    fn apply(&self, inputs: Vec<TensorBase<T, B>>) -> Result<Vec<TensorBase<T, B>>, GraphError> {
        if inputs.len() != 1 {
            return Err(GraphError::InvalidInput(
                "Linear layer expects a single input tensor".to_string(),
            ));
        }

        let input = &inputs[0];

        if input.size() != 2 || input.shape()[1] != self.in_features {
            return Err(GraphError::InvalidInput(
                format!("Input tensor must have shape [B, {}]", self.in_features),
            ));
        }

        let weight = self.weights.get("weight").ok_or_else(|| {
            GraphError::ComputationError("Weight tensor not found".to_string())
        })?;
        let bias = self.weights.get("bias").ok_or_else(|| {
            GraphError::ComputationError("Bias tensor not found".to_string())
        })?;

        let output = input.matmul(weight)?.add(bias)?;

        Ok(vec![output])
        
    }
}