#![recursion_limit = "256"]

use std::sync::Arc;
use parking_lot::RwLock;
use ndarray::{Array2, Array3, ArrayD, Ix3};
use crate::gru::GRUModule;
use crate::gat::GATModule;
use crate::error::ModelError;
use crate::config::ModelConfig;
use crate::risk_model::{RiskModel, RiskFactors};
use crate::types::MarketData;
use async_trait::async_trait;
use anyhow::Result;
use tokio::sync::RwLock as TokioRwLock;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::error::Error;

/// Deep Risk Model that combines GRU and GAT modules for risk factor generation
pub struct DeepRiskModel {
    gru: GRUModule,
    gat: GATModule,
}

impl DeepRiskModel {
    /// Creates a new DeepRiskModel instance
    pub fn new(config: &ModelConfig) -> Result<Self, ModelError> {
        let gru = GRUModule::new(
            config.input_size as usize,
            config.hidden_size as usize,
            config.num_layers as usize
        ).map_err(|e| ModelError::InvalidDimension(e.to_string()))?;
        
        let gat = GATModule::new(
            config.hidden_size as usize,
            config.num_heads as usize
        )?;
        
        Ok(Self { gru, gat })
    }
}

#[async_trait]
impl RiskModel for DeepRiskModel {
    async fn train(&mut self, _data: &MarketData) -> Result<(), ModelError> {
        // Training implementation
        Ok(())
    }

    async fn generate_factors(&self, market_data: &MarketData) -> Result<RiskFactors, ModelError> {
        let features = market_data.features();
        let features_f32 = features.mapv(|x| x as f32);
        let shape = features_f32.shape();
        let (dim0, dim1) = (shape[0], shape[1]);
        let features_3d = features_f32.into_shape((1, dim0, dim1))?;
        
        let gru_output = self.gru.forward(&features_3d, None)?;
        let gat_output = self.gat.forward(&gru_output)?;
        
        // Convert GAT output to ArrayD<f64>
        let factors = gat_output.mapv(|x| x as f64).into_dyn();
        let covariance = self.estimate_covariance(market_data).await?;
        
        Ok(RiskFactors { factors, covariance })
    }

    async fn estimate_covariance(&self, market_data: &MarketData) -> Result<Array2<f64>, ModelError> {
        let n = market_data.features().shape()[1];
        Ok(Array2::eye(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;

    #[tokio::test]
    async fn test_model_workflow() -> Result<(), ModelError> {
        let input_dim = 10;
        let hidden_dim = 10;  // Match input dimension for compatibility
        let num_layers = 2;
        let config = ModelConfig {
            input_size: input_dim as i64,
            hidden_size: hidden_dim as i64,
            num_heads: 4,
            head_dim: 8,
            num_layers: num_layers as i64,
            output_size: 3,
        };
        let mut model = DeepRiskModel::new(&config)?;
        
        // Create test data
        let n_stocks = 100;
        let seq_len = 50;
        let features = Array3::random((n_stocks, seq_len, input_dim), Normal::new(0.0, 1.0).unwrap()).into_dyn();
        let returns = Array::random(n_stocks, Normal::new(0.0, 1.0).unwrap()).into_dyn();
        let market_data = MarketData::new(features, returns);
        
        // Test factor generation
        let factors = model.generate_factors(&market_data).await?;
        assert_eq!(factors.factors.shape()[0], n_stocks);
        
        // Test covariance estimation
        let cov = model.estimate_covariance(&market_data).await?;
        assert_eq!(cov.shape()[0], n_stocks);
        assert_eq!(cov.shape()[1], n_stocks);
        
        Ok(())
    }
}