#![recursion_limit = "4096"]

use ndarray::{Array1, Array2, ArrayD, IxDyn};

pub mod model;
pub mod gru;
pub mod gat;
pub mod error;
pub mod config;
pub mod mcp_client;
pub mod risk_model;
pub mod utils;
pub mod types;

pub use model::DeepRiskModel;
pub use error::ModelError;
pub use config::ModelConfig;
pub use risk_model::{RiskModel, RiskFactors};
pub use mcp_client::{MCPClient, MCPConfig};
pub use gru::GRUModule;
pub use gat::GATModule;
pub use types::MarketData;

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