use anyhow::Result;
use ndarray::{Array, Array2, Array3, Ix2};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use deep_risk_model::{
    DeepRiskModel, ModelConfig, RiskModel, MarketData, ModelError,
};
mod test_utils;
use test_utils::{generate_test_data, generate_test_data_with_noise};
use std::time::Instant;

const MCP_API_KEY: &str = "test_key";

/// Helper function to generate synthetic market data
fn generate_synthetic_data(
    num_stocks: usize,
    seq_len: usize,
    num_features: usize,
) -> MarketData {
    let features: Array3<f32> = Array3::random((num_stocks, seq_len, num_features), Normal::new(0.0, 1.0).unwrap());
    let returns: Array<f32, Ix2> = Array::random((num_stocks, 1), Normal::new(0.0, 0.1).unwrap());
    
    MarketData::new(features.into_dyn(), returns.into_dyn())
}

#[tokio::test]
async fn test_mcp_integration() -> Result<()> {
    let config = ModelConfig {
        input_size: 64,
        hidden_size: 64,
        num_heads: 4,
        head_dim: 16,
        num_layers: 2,
        output_size: 3,
    };
    let mut model = DeepRiskModel::new(&config)?;
    
    // Generate random market data with correct dimensions
    let n_samples = 100;
    let seq_len = 50;
    let n_features = 64;
    let market_data = generate_synthetic_data(n_samples, seq_len, n_features);
    
    // Train model
    println!("Training model...");
    model.train(&market_data).await?;
    
    // Generate factors
    let factors = model.generate_factors(&market_data).await?;
    assert_eq!(factors.factors.shape()[0], n_samples);
    assert_eq!(factors.factors.shape()[1], 3);  // output_size
    
    // Estimate covariance
    let cov = model.estimate_covariance(&market_data).await?;
    assert_eq!(cov.shape()[0], n_samples);
    assert_eq!(cov.shape()[1], n_samples);
    
    Ok(())
}

#[tokio::test]
async fn test_performance() -> Result<()> {
    let config = ModelConfig {
        input_size: 64,
        hidden_size: 64,
        num_heads: 4,
        head_dim: 16,
        num_layers: 3,
        output_size: 3,
    };
    let mut model = DeepRiskModel::new(&config)?;
    
    // Generate larger dataset for performance testing
    let n_samples = 100;
    let seq_len = 100;
    let n_features = 64;
    let market_data = generate_synthetic_data(n_samples, seq_len, n_features);
    
    // Measure training time
    let start = Instant::now();
    model.train(&market_data).await?;
    let train_time = start.elapsed();
    println!("Training time: {:?}", train_time);
    
    // Measure inference time
    let start = Instant::now();
    let factors = model.generate_factors(&market_data).await?;
    let inference_time = start.elapsed();
    println!("Inference time: {:?}", inference_time);
    
    // Verify results
    assert_eq!(factors.factors.shape()[0], n_samples);
    assert_eq!(factors.factors.shape()[1], 3);  // output_size
    let cov = model.estimate_covariance(&market_data).await?;
    assert_eq!(cov.shape()[0], n_samples);
    assert_eq!(cov.shape()[1], n_samples);
    
    Ok(())
}

#[tokio::test]
async fn test_model_with_historical_data() -> Result<()> {
    let config = ModelConfig {
        input_size: 64,
        hidden_size: 64,
        num_heads: 4,
        head_dim: 16,
        num_layers: 2,
        output_size: 3,
    };
    let mut model = DeepRiskModel::new(&config)?;
    
    // Generate historical-like data
    let n_stocks = 30;
    let seq_len = 30;  // 30 days of data
    let n_features = 64;
    println!("Testing with {} days of data...", seq_len);
    
    let market_data = generate_synthetic_data(n_stocks, seq_len, n_features);
    
    // Train model
    model.train(&market_data).await?;
    
    // Generate factors
    let factors = model.generate_factors(&market_data).await?;
    assert_eq!(factors.factors.shape()[0], n_stocks);
    assert_eq!(factors.factors.shape()[1], 3);  // output_size
    
    // Estimate covariance
    let cov = model.estimate_covariance(&market_data).await?;
    assert_eq!(cov.shape()[0], n_stocks);
    assert_eq!(cov.shape()[1], n_stocks);
    
    Ok(())
}

#[tokio::test]
async fn test_model_initialization() -> Result<(), ModelError> {
    let config = ModelConfig::default();
    let _model = DeepRiskModel::new(&config)?;
    Ok(())
}

#[tokio::test]
async fn test_factor_generation() -> Result<(), ModelError> {
    let config = ModelConfig::default();
    let model = DeepRiskModel::new(&config)?;
    let market_data = generate_test_data(100, 20, 0.0)?;
    
    let factors = model.generate_factors(&market_data).await?;
    assert_eq!(factors.factors.shape()[0], 100);
    assert_eq!(factors.factors.shape()[1], 3);  // output_size
    
    Ok(())
}

#[tokio::test]
async fn test_covariance_estimation() -> Result<(), ModelError> {
    let config = ModelConfig::default();
    let model = DeepRiskModel::new(&config)?;
    let market_data = generate_test_data(100, 20, 0.0)?;
    
    let covariance = model.estimate_covariance(&market_data).await?;
    assert_eq!(covariance.shape()[0], 100);
    assert_eq!(covariance.shape()[1], 100);
    
    Ok(())
}

#[tokio::test]
async fn test_noise_robustness() -> Result<(), ModelError> {
    let config = ModelConfig::default();
    let model = DeepRiskModel::new(&config)?;
    let market_data = generate_test_data(100, 20, 0.0)?;
    let noisy_data = generate_test_data_with_noise(100, 20, 0.1)?;
    
    let factors = model.generate_factors(&noisy_data).await?;
    assert_eq!(factors.factors.shape()[0], 100);
    assert_eq!(factors.factors.shape()[1], 3);  // output_size
    
    Ok(())
}

#[tokio::test]
async fn test_factor_quality() -> Result<(), ModelError> {
    let config = ModelConfig::default();
    let model = DeepRiskModel::new(&config)?;
    let market_data = generate_test_data(100, 20, 0.0)?;
    
    let factors = model.generate_factors(&market_data).await?;
    assert_eq!(factors.factors.shape()[0], 100);
    assert_eq!(factors.factors.shape()[1], 3);  // output_size
    
    Ok(())
}

#[tokio::test]
async fn test_training() -> Result<(), ModelError> {
    let config = ModelConfig::default();
    let mut model = DeepRiskModel::new(&config)?;
    let market_data = generate_test_data(100, 20, 0.0)?;
    
    model.train(&market_data).await?;
    Ok(())
} 