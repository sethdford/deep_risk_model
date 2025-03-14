use anyhow::Result;
use deep_risk_model::{
    DeepRiskModel, ModelConfig, RiskModel, MarketData, ModelError,
};
use ndarray::{Array, ArrayD};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use std::time::Instant;

/// Helper function to generate synthetic market data
fn generate_synthetic_data(
    num_stocks: usize,
    seq_len: usize,
    num_features: usize,
) -> MarketData {
    let features: ArrayD<f32> = Array::random((num_stocks, seq_len, num_features), Normal::new(0.0, 1.0).unwrap()).into_dyn();
    let returns: ArrayD<f32> = Array::random((num_stocks, 1), Normal::new(0.0, 0.1).unwrap()).into_dyn();
    
    MarketData::new(features, returns)
}

#[tokio::test]
async fn test_model_workflow() -> Result<()> {
    // Initialize model configuration
    let config = ModelConfig {
        input_size: 64,
        hidden_size: 64,
        num_heads: 4,
        head_dim: 16,
        num_layers: 2,
        output_size: 3,
    };
    
    // Create model instance
    let mut model = DeepRiskModel::new(&config)?;
    
    // Generate synthetic market data
    let n_samples = 100;
    let seq_len = 50;
    let n_features = 64;
    let features: ArrayD<f32> = Array::random((n_samples, seq_len, n_features), Normal::new(0.0, 1.0).unwrap()).into_dyn();
    let returns: ArrayD<f32> = Array::random((n_samples, 1), Normal::new(0.0, 0.1).unwrap()).into_dyn();
    let market_data = MarketData::new(features, returns);
    
    // Train model
    println!("Training model...");
    let start = Instant::now();
    model.train(&market_data).await?;
    let train_time = start.elapsed();
    println!("Training completed in {:?}", train_time);
    
    // Generate risk factors
    println!("Generating risk factors...");
    let start = Instant::now();
    let factors = model.generate_factors(&market_data).await?;
    let factor_time = start.elapsed();
    println!("Factor generation completed in {:?}", factor_time);
    println!("Generated factors shape: {:?}", factors.factors.shape());
    
    // Estimate covariance matrix
    println!("Estimating covariance matrix...");
    let start = Instant::now();
    let covariance = model.estimate_covariance(&market_data).await?;
    let cov_time = start.elapsed();
    println!("Covariance estimation completed in {:?}", cov_time);
    println!("Covariance matrix shape: {:?}", covariance.shape());
    
    Ok(())
}

#[tokio::test]
async fn test_model_with_noise() -> Result<()> {
    // Initialize model configuration
    let config = ModelConfig {
        input_size: 64,
        hidden_size: 64,
        num_heads: 4,
        head_dim: 16,
        num_layers: 2,
        output_size: 3,
    };
    
    // Create model instance
    let mut model = DeepRiskModel::new(&config)?;
    
    // Generate synthetic market data with noise
    let n_samples = 100;
    let seq_len = 50;
    let n_features = 64;
    let features: ArrayD<f32> = Array::random((n_samples, seq_len, n_features), Normal::new(0.0, 1.0).unwrap()).into_dyn();
    let noise: ArrayD<f32> = Array::random((n_samples, seq_len, n_features), Normal::new(0.0, 0.1).unwrap()).into_dyn();
    let returns: ArrayD<f32> = Array::random((n_samples, 1), Normal::new(0.0, 0.1).unwrap()).into_dyn();
    let market_data = MarketData::new((features + noise).into_dyn(), returns);
    
    // Train model
    println!("Training model with noisy data...");
    let start = Instant::now();
    model.train(&market_data).await?;
    let train_time = start.elapsed();
    println!("Training completed in {:?}", train_time);
    
    // Generate risk factors
    println!("Generating risk factors...");
    let start = Instant::now();
    let factors = model.generate_factors(&market_data).await?;
    let factor_time = start.elapsed();
    println!("Factor generation completed in {:?}", factor_time);
    println!("Generated factors shape: {:?}", factors.factors.shape());
    
    Ok(())
}

#[tokio::test]
async fn test_model_performance() -> Result<()> {
    // Initialize model configuration
    let config = ModelConfig {
        input_size: 64,
        hidden_size: 64,
        num_heads: 4,
        head_dim: 16,
        num_layers: 2,
        output_size: 3,
    };
    
    // Create model instance
    let mut model = DeepRiskModel::new(&config)?;
    
    // Generate larger dataset for performance testing
    let n_samples = 1000;
    let seq_len = 100;
    let n_features = 64;
    let features: ArrayD<f32> = Array::random((n_samples, seq_len, n_features), Normal::new(0.0, 1.0).unwrap()).into_dyn();
    let returns: ArrayD<f32> = Array::random((n_samples, 1), Normal::new(0.0, 0.1).unwrap()).into_dyn();
    let market_data = MarketData::new(features, returns);
    
    // Train model
    println!("Training model with large dataset...");
    let start = Instant::now();
    model.train(&market_data).await?;
    let train_time = start.elapsed();
    println!("Training completed in {:?}", train_time);
    
    // Generate risk factors
    println!("Generating risk factors...");
    let start = Instant::now();
    let factors = model.generate_factors(&market_data).await?;
    let factor_time = start.elapsed();
    println!("Factor generation completed in {:?}", factor_time);
    println!("Generated factors shape: {:?}", factors.factors.shape());
    
    Ok(())
}

#[tokio::test]
async fn test_model_robustness() -> Result<()> {
    // Initialize model with smaller dimensions
    let config = ModelConfig {
        input_size: 8,
        hidden_size: 8,
        num_heads: 2,
        head_dim: 4,
        num_layers: 2,
        output_size: 8,
    };
    let mut model = DeepRiskModel::new(&config)?;
    
    // Create test data with fewer stocks
    let n_stocks = 30;
    let seq_len = 50;
    let n_features = 8;
    let market_data = generate_synthetic_data(n_stocks, seq_len, n_features);
    
    // Test training
    model.train(&market_data).await?;
    
    // Test factor generation
    let factors = model.generate_factors(&market_data).await?;
    assert_eq!(factors.factors.shape()[0], n_stocks);
    
    // Test covariance estimation
    let cov = model.estimate_covariance(&market_data).await?;
    assert_eq!(cov.shape(), &[n_stocks, n_stocks]);
    
    Ok(())
} 