#![recursion_limit = "256"]

use std::sync::Arc;
use parking_lot::RwLock;
use ndarray::{Array2, Array3, ArrayD, Ix3, ArrayView3};
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

/// A deep learning-based risk model that combines GRU and GAT modules for risk factor generation.
///
/// # Architecture
///
/// The model consists of two main components:
/// 1. A Gated Recurrent Unit (GRU) for processing temporal features
/// 2. A Graph Attention Network (GAT) for capturing cross-sectional relationships
///
/// # Features
///
/// - Temporal feature extraction using GRU
/// - Cross-sectional feature extraction using GAT
/// - Automatic risk factor generation
/// - Covariance matrix estimation
///
/// # Example
///
/// ```rust
/// use deep_risk_model::{DeepRiskModel, ModelConfig, MarketData};
///
/// #[tokio::main]
/// async fn main() -> Result<(), ModelError> {
///     let config = ModelConfig::default();
///     let mut model = DeepRiskModel::new(&config)?;
///     
///     // Train model with market data
///     model.train(&market_data).await?;
///     
///     // Generate risk factors
///     let factors = model.generate_factors(&market_data).await?;
///     
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct DeepRiskModel {
    /// GRU module for temporal feature processing
    gru: GRUModule,
    /// GAT module for cross-sectional feature processing
    gat: GATModule,
    config: ModelConfig,
}

impl DeepRiskModel {
    /// Creates a new DeepRiskModel instance with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the model
    ///
    /// # Returns
    ///
    /// * `Result<Self, ModelError>` - A new model instance or an error
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = ModelConfig {
    ///     input_size: 64,
    ///     hidden_size: 64,
    ///     num_heads: 4,
    ///     head_dim: 16,
    ///     num_layers: 2,
    ///     output_size: 3,
    /// };
    /// let model = DeepRiskModel::new(&config)?;
    /// ```
    pub fn new(config: &ModelConfig) -> Result<Self, ModelError> {
        let gru = GRUModule::new(
            config.input_size as usize,
            config.hidden_size as usize,
            config.num_layers as usize,
        )?;
        
        let gat = GATModule::new(
            config.hidden_size as usize,
            config.num_heads as usize,
        )?;
        
        Ok(Self {
            gru,
            gat,
            config: config.clone(),
        })
    }
}

#[async_trait]
impl RiskModel for DeepRiskModel {
    /// Trains the model on historical market data.
    ///
    /// # Arguments
    ///
    /// * `data` - Historical market data for training
    ///
    /// # Returns
    ///
    /// * `Result<(), ModelError>` - Success or error
    ///
    /// # Example
    ///
    /// ```rust
    /// model.train(&market_data).await?;
    /// ```
    async fn train(&mut self, data: &MarketData) -> Result<(), ModelError> {
        // Convert dynamic array to Array3 view
        let features_shape = data.features.shape();
        let temporal_features_view = data.features.view()
            .into_shape((features_shape[0], features_shape[1], features_shape[2]))?
            .into_dimensionality::<ndarray::Ix3>()?;
        
        // Process temporal features with GRU
        let mut temporal_features = temporal_features_view.to_owned();
        for _ in 0..self.config.num_layers as usize {
            temporal_features = self.gru.forward(&temporal_features, None)?;
        }
        
        // Process cross-sectional features with GAT
        let _cross_features = self.gat.forward(&temporal_features)?;
        
        Ok(())
    }

    /// Generates risk factors from market data.
    ///
    /// This method processes the input market data through both the GRU and GAT
    /// modules to generate risk factors and estimate their covariance matrix.
    ///
    /// # Arguments
    ///
    /// * `market_data` - Market data to generate factors from
    ///
    /// # Returns
    ///
    /// * `Result<RiskFactors, ModelError>` - Generated risk factors or error
    ///
    /// # Example
    ///
    /// ```rust
    /// let factors = model.generate_factors(&market_data).await?;
    /// println!("Factor shape: {:?}", factors.factors.shape());
    /// ```
    async fn generate_factors(&self, data: &MarketData) -> Result<RiskFactors, ModelError> {
        // Convert dynamic array to Array3 view
        let features_shape = data.features.shape();
        let temporal_features_view = data.features.view()
            .into_shape((features_shape[0], features_shape[1], features_shape[2]))?
            .into_dimensionality::<ndarray::Ix3>()?;
        
        // Process temporal features with GRU
        let mut temporal_features = temporal_features_view.to_owned();
        for _ in 0..self.config.num_layers as usize {
            temporal_features = self.gru.forward(&temporal_features, None)?;
        }
        
        // Process cross-sectional features with GAT
        let cross_features = self.gat.forward(&temporal_features)?;
        
        // Extract risk factors from processed features
        let factors = cross_features.into_dyn();
        
        // Compute factor covariance matrix
        let covariance = self.estimate_covariance(data).await?;
        
        Ok(RiskFactors {
            factors,
            covariance,
        })
    }

    /// Estimates the covariance matrix from market data.
    ///
    /// # Arguments
    ///
    /// * `market_data` - Market data to estimate covariance from
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f32>, ModelError>` - Estimated covariance matrix or error
    ///
    /// # Example
    ///
    /// ```rust
    /// let covariance = model.estimate_covariance(&market_data).await?;
    /// println!("Covariance shape: {:?}", covariance.shape());
    /// ```
    async fn estimate_covariance(&self, data: &MarketData) -> Result<Array2<f32>, ModelError> {
        let n_stocks = data.features.shape()[1];  // Changed from [0] to [1]
        let mut covariance = Array2::<f32>::zeros((n_stocks, n_stocks));
        
        // Generate factors
        let factors = self.generate_factors(data).await?.factors;
        let factors_shape = factors.shape();
        
        // Take the last time step for covariance computation
        let last_factors = factors.slice(ndarray::s![factors_shape[0]-1, .., ..]);
        
        // Compute factor covariance
        for i in 0..n_stocks {
            for j in 0..n_stocks {
                let cov = last_factors.slice(ndarray::s![i, ..])
                    .dot(&last_factors.slice(ndarray::s![j, ..]));
                covariance[[i, j]] = cov;
            }
        }
        
        Ok(covariance)
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
        let hidden_dim = 64;  // Make it divisible by num_heads
        let num_layers = 2;
        let config = ModelConfig {
            input_size: input_dim as i64,
            hidden_size: hidden_dim as i64,
            num_heads: 4,
            head_dim: 16,
            num_layers: num_layers as i64,
            output_size: 3,
        };
        let mut model = DeepRiskModel::new(&config)?;
        
        // Create test data
        let n_stocks = 100;
        let seq_len = 50;
        let features = Array::random((seq_len, n_stocks, input_dim), Normal::new(0.0, 1.0).unwrap()).into_dyn();
        let returns = Array::random((seq_len, n_stocks, 1), Normal::new(0.0, 1.0).unwrap()).into_dyn();
        
        let market_data = MarketData {
            features,
            returns,
        };
        
        // Test training
        model.train(&market_data).await?;
        
        // Test factor generation
        let factors = model.generate_factors(&market_data).await?;
        assert_eq!(factors.factors.shape()[0], seq_len);
        assert_eq!(factors.factors.shape()[1], n_stocks);
        
        // Test covariance estimation
        let covariance = model.estimate_covariance(&market_data).await?;
        assert_eq!(covariance.shape(), &[n_stocks, n_stocks]);
        
        Ok(())
    }
}