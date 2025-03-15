use ndarray::{Array2, Array3, ArrayD};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Feature tensor of shape [num_stocks, seq_len, num_features]
    pub features: ArrayD<f32>,
    /// Returns tensor of shape [num_stocks]
    pub returns: ArrayD<f32>,
}

impl MarketData {
    pub fn new(features: ArrayD<f32>, returns: ArrayD<f32>) -> Self {
        Self { features, returns }
    }

    pub fn features(&self) -> &ArrayD<f32> {
        &self.features
    }

    pub fn returns(&self) -> &ArrayD<f32> {
        &self.returns
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactors {
    pub factors: ArrayD<f32>,
    pub covariance: Option<ArrayD<f32>>,
}

impl RiskFactors {
    pub fn new(factors: ArrayD<f32>) -> Self {
        Self { factors, covariance: None }
    }

    pub fn factors(&self) -> &ArrayD<f32> {
        &self.factors
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPConfig {
    api_key: String,
    base_url: String,
}

impl MCPConfig {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self { api_key, base_url }
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    num_layers: usize,
}

impl ModelConfig {
    pub fn new(hidden_size: usize, num_heads: usize, head_dim: usize, num_layers: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            head_dim,
            num_layers,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
} 