use ndarray::{Array2, ArrayD};
use async_trait::async_trait;
use crate::{ModelError, MarketData};
use serde::Serialize;

/// Learned risk factors and their properties.
/// 
/// This struct represents the output of the risk model, containing
/// both the learned risk factors and their covariance matrix.
#[derive(Debug, Clone, Serialize)]
pub struct RiskFactors {
    /// Risk factor values (N x K matrix where K is number of factors)
    pub factors: ArrayD<f64>,
    /// Factor returns (regression coefficients)
    pub covariance: Array2<f64>,
}

/// Core trait for risk factor models.
/// 
/// This trait defines the interface that all risk factor models must implement.
/// It provides methods for training the model and generating risk factors.
#[async_trait]
pub trait RiskModel: Send + Sync {
    /// Train the model on historical data.
    /// 
    /// # Arguments
    /// * `data` - Historical market data for training
    /// 
    /// # Returns
    /// * `Result<(), ModelError>` - Success or error
    async fn train(&mut self, data: &MarketData) -> Result<(), ModelError>;
    
    /// Generate risk factors for given market data.
    /// 
    /// # Arguments
    /// * `data` - Market data to generate factors for
    /// 
    /// # Returns
    /// * `Result<RiskFactors, ModelError>` - Generated risk factors or error
    async fn generate_factors(&self, data: &MarketData) -> Result<RiskFactors, ModelError>;
    
    /// Estimate covariance matrix using generated risk factors.
    /// 
    /// # Arguments
    /// * `data` - Market data to use for estimation
    /// 
    /// # Returns
    /// * `Result<Array2<f64>, ModelError>` - Estimated covariance matrix or error
    async fn estimate_covariance(&self, data: &MarketData) -> Result<Array2<f64>, ModelError>;
} 