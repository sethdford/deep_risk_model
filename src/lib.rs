#![recursion_limit = "4096"]

pub mod config;
pub mod error;
pub mod gru;
pub mod gat;
pub mod model;
pub mod risk_model;
pub mod types;

pub use crate::{
    config::ModelConfig,
    error::ModelError,
    model::DeepRiskModel,
    risk_model::RiskModel,
    types::MarketData,
};

pub mod array_utils {
    use ndarray::Array;
    use num_traits::Float;

    pub fn array_to_vec<T, D>(array: &Array<T, D>) -> Vec<T>
    where
        T: Clone,
        D: ndarray::Dimension,
    {
        array.iter().cloned().collect()
    }

    pub fn vec_to_array<T, D>(vec: Vec<T>, shape: D) -> Array<T, D>
    where
        T: Clone,
        D: ndarray::Dimension,
    {
        Array::from_shape_vec(shape, vec).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Float;
    // ... existing code ...
}