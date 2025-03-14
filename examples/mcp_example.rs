use anyhow::Result;
use chrono::{Duration, Utc};
use deep_risk_model::{
    DeepRiskModel, MCPClient, ModelConfig, RiskModel, MarketData, ModelError,
};
use dotenv::dotenv;
use std::env;
use tracing::{info, warn};
use ndarray::{Array, ArrayD};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use std::time::Instant;

/// Example demonstrating the usage of the Deep Risk Model with MCP integration
#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file
    dotenv().ok();

    // Initialize logging
    tracing_subscriber::fmt::init();

    // Initialize MCP client with configuration from environment
    let client = MCPClient::new()?;
    if env::var("MCP_API_KEY").unwrap_or_default().is_empty() {
        warn!("No API key found in environment variables");
    }

    // Define symbols to fetch
    let symbols = vec![
        "AAPL".to_string(),
        "MSFT".to_string(),
        "GOOGL".to_string(),
        "AMZN".to_string(),
        "META".to_string(),
    ];

    // Fetch market data for the last 30 days
    let end_date = Utc::now();
    let start_date = end_date - Duration::days(30);

    info!("Fetching market data from MCP server...");
    let market_data = client.fetch_market_data().await?;
    info!("Fetched market data with shape: {:?}", market_data.features.shape());

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
    let market_data = deep_risk_model::MarketData::new(features, returns);
    
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