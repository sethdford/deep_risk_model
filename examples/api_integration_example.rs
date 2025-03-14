use anyhow::Result;
use chrono::{DateTime, Utc};
use deep_risk_model::{
    DeepRiskModel, MCPClient, ModelConfig, RiskModel,
    mcp_client::MarketData,
    RiskFactors,
    ModelError,
};
use dotenv::dotenv;
use ndarray::{Array2, ArrayD, IxDyn};
use ndarray_linalg::Eigh;
use reqwest::Client;
use serde_json::json;
use std::env;
use std::time::Instant;
use tokio::time::sleep;
use tracing::{info, warn};
use serde::{Deserialize, Serialize};
use ndarray_rand::{RandomExt, rand_distr::Normal};

/// Configuration for API testing
#[derive(Debug, Clone)]
struct APIConfig {
    /// API endpoint URL
    endpoint: String,
    /// Number of requests for performance testing
    num_requests: usize,
}

impl Default for APIConfig {
    fn default() -> Self {
        Self {
            endpoint: env::var("API_ENDPOINT").unwrap_or_else(|_| "http://localhost:3000".to_string()),
            num_requests: 10,
        }
    }
}

/// Fetches market data from MCP server
async fn fetch_market_data(client: &MCPClient) -> Result<MarketData> {
    info!("Fetching market data from MCP server...");
    let market_data = client.fetch_market_data().await?;
    info!("Fetched market data with shape: {:?}", market_data.features.shape());
    Ok(market_data)
}

/// Converts an ArrayD to a Vec<Vec<f64>> for JSON serialization
fn array_to_vec(array: &ArrayD<f64>) -> Vec<Vec<f64>> {
    let shape = array.shape();
    let mut result = Vec::with_capacity(shape[0]);
    
    for i in 0..shape[0] {
        let mut row = Vec::with_capacity(shape[1]);
        for j in 0..shape[1] {
            row.push(array[[i, j]]);
        }
        result.push(row);
    }
    
    result
}

/// Tests API endpoints with real market data
async fn test_with_real_data(config: &APIConfig, market_data: &MarketData) -> Result<()> {
    let client = Client::new();
    
    // Test health check
    info!("Testing health check endpoint...");
    let response = client.get(&format!("{}/health", config.endpoint))
        .send()
        .await?;
    info!("Health check status: {}", response.status());
    info!("Response: {}", response.text().await?);
    
    // Test factor generation
    info!("Testing generate factors endpoint with real data...");
    let payload = json!({
        "features": array_to_vec(&market_data.features),
        "returns": array_to_vec(&market_data.returns)
    });
    
    let response = client.post(&format!("{}/factors", config.endpoint))
        .json(&payload)
        .send()
        .await?;
    
    info!("Factor generation status: {}", response.status());
    if response.status().is_success() {
        let factors: serde_json::Value = response.json().await?;
        info!("Generated factors: {:?}", factors["factors"].as_array().map(|f| f.len()));
        info!("Covariance matrix shape: {:?}", factors["covariance"].as_array().map(|c| c.len()));
    } else {
        warn!("Error: {}", response.text().await?);
    }
    
    // Test covariance estimation
    info!("Testing estimate covariance endpoint with real data...");
    let response = client.post(&format!("{}/covariance", config.endpoint))
        .json(&payload)
        .send()
        .await?;
    
    info!("Covariance estimation status: {}", response.status());
    if response.status().is_success() {
        let covariance: Vec<Vec<f64>> = response.json().await?;
        let covariance = Array2::from_shape_vec((covariance.len(), covariance[0].len()), 
            covariance.into_iter().flatten().collect())?;
        
        // Verify covariance matrix properties
        let is_symmetric = covariance.t().to_owned() == covariance;
        let eigenvalues = covariance.eigh(ndarray_linalg::UPLO::Upper)?;
        let is_positive_definite = eigenvalues.0.iter().all(|&x| x > 0.0);
        
        info!("Covariance matrix is symmetric: {}", is_symmetric);
        info!("Covariance matrix is positive definite: {}", is_positive_definite);
        info!("Eigenvalues: {:?}", eigenvalues.0);
    } else {
        warn!("Error: {}", response.text().await?);
    }
    
    Ok(())
}

/// Tests API performance with real data
async fn test_performance(config: &APIConfig, market_data: &MarketData) -> Result<()> {
    info!("Testing API performance with {} requests...", config.num_requests);
    let client = Client::new();
    
    let payload = json!({
        "features": array_to_vec(&market_data.features),
        "returns": array_to_vec(&market_data.returns)
    });
    
    let mut response_times = Vec::with_capacity(config.num_requests);
    
    for i in 0..config.num_requests {
        let start_time = Instant::now();
        
        let response = client.post(&format!("{}/factors", config.endpoint))
            .json(&payload)
            .send()
            .await?;
        
        let duration = start_time.elapsed().as_secs_f64();
        response_times.push(duration);
        
        info!("Request {}/{}: {:.2} seconds", i + 1, config.num_requests, duration);
        
        if !response.status().is_success() {
            warn!("Error in request {}: {}", i + 1, response.text().await?);
        }
        
        // Small delay between requests
        sleep(tokio::time::Duration::from_millis(100)).await;
    }
    
    // Calculate statistics
    let avg_time = response_times.iter().sum::<f64>() / response_times.len() as f64;
    let std_time = (response_times.iter()
        .map(|&x| (x - avg_time).powi(2))
        .sum::<f64>() / response_times.len() as f64)
        .sqrt();
    let min_time = response_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_time = response_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    info!("Performance Statistics:");
    info!("Average response time: {:.2} seconds", avg_time);
    info!("Standard deviation: {:.2} seconds", std_time);
    info!("Minimum response time: {:.2} seconds", min_time);
    info!("Maximum response time: {:.2} seconds", max_time);
    
    Ok(())
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: String,
    version: String,
}

#[derive(Serialize)]
struct FactorRequest {
    features: ArrayD<f32>,
    returns: ArrayD<f32>,
}

#[derive(Deserialize)]
struct FactorResponse {
    factors: RiskFactors,
    covariance: ArrayD<f32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables
    dotenv().ok();
    
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Initialize configuration
    let config = APIConfig::default();
    
    // Initialize MCP client
    let client = MCPClient::new()?;
    if env::var("MCP_API_KEY").unwrap_or_default().is_empty() {
        warn!("No API key found in environment variables");
    }
    
    // Fetch market data
    let market_data = fetch_market_data(&client).await?;
    
    // Test health check endpoint
    info!("Testing health check endpoint...");
    let health_response: HealthResponse = reqwest::get(&format!("{}/health", config.endpoint))
        .await?
        .json()
        .await?;
    info!("Health check response: {:?}", health_response);

    // Test factor generation
    info!("Testing factor generation...");
    let factor_request = FactorRequest {
        features: market_data.features.clone(),
        returns: market_data.returns.clone(),
    };
    let factors: RiskFactors = reqwest::Client::new()
        .post(&format!("{}/factors", config.endpoint))
        .json(&factor_request)
        .send()
        .await?
        .json()
        .await?;
    info!("Generated factors with shape: {:?}", factors.factors.shape());

    // Test covariance estimation
    info!("Testing covariance estimation...");
    let covariance_request = FactorRequest {
        features: market_data.features.clone(),
        returns: market_data.returns.clone(),
    };
    let covariance: Array2<f32> = reqwest::Client::new()
        .post(&format!("{}/covariance", config.endpoint))
        .json(&covariance_request)
        .send()
        .await?
        .json()
        .await?;
    info!("Estimated covariance with shape: {:?}", covariance.shape());
    
    // Run tests
    test_with_real_data(&config, &market_data).await?;
    test_performance(&config, &market_data).await?;
    
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
    let market_data = MarketData::new(features.clone(), returns.clone());
    
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
    
    // Prepare API request
    let request = FactorRequest {
        features: market_data.features.clone(),
        returns: market_data.returns.clone(),
    };
    let client = reqwest::Client::new();
    
    // Send request to API
    println!("Sending request to API...");
    let start = Instant::now();
    let response = client
        .post("http://localhost:8080/factors")
        .json(&request)
        .send()
        .await?;
    let api_time = start.elapsed();
    println!("API request completed in {:?}", api_time);
    
    // Parse response
    let factor_response: FactorResponse = response.json().await?;
    println!("Received factors shape: {:?}", factor_response.factors.factors.shape());
    println!("Received covariance shape: {:?}", factor_response.covariance.shape());
    
    Ok(())
} 