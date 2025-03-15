use deep_risk_model::{
    DeepRiskModel,
    ModelConfig,
    RiskModel,
    MarketData,
};
use ndarray::{Array, Array3, ArrayD, IxDyn, Array2};
use serde::{Deserialize, Serialize};
use tokio;
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
struct APIConfig {
    input_size: i64,
    hidden_size: i64,
    num_heads: i64,
    head_dim: i64,
    num_layers: i64,
    output_size: i64,
}

impl Default for APIConfig {
    fn default() -> Self {
        Self {
            input_size: 32,
            hidden_size: 64,
            num_heads: 4,
            head_dim: 16,
            num_layers: 2,
            output_size: 16,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Request {
    features: Vec<Vec<f32>>,
    returns: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Response {
    factors: Vec<Vec<f32>>,
    covariance: Vec<Vec<f32>>,
}

fn array_to_vec(array: &ArrayD<f32>) -> Vec<Vec<f32>> {
    let shape = array.shape();
    let mut result = Vec::new();
    
    match shape.len() {
        2 => {
            // Handle 2D array directly
            for i in 0..shape[0] {
                let mut row = Vec::new();
                for j in 0..shape[1] {
                    row.push(array[[i, j]]);
                }
                result.push(row);
            }
        },
        3 => {
            // For 3D array, take the last 2D slice
            let last_slice_idx = shape[0] - 1;
            for i in 0..shape[1] {
                let mut row = Vec::new();
                for j in 0..shape[2] {
                    row.push(array[[last_slice_idx, i, j]]);
                }
                result.push(row);
            }
        },
        _ => panic!("Unsupported array dimension"),
    }
    
    result
}

fn array2_to_vec(array: &Array2<f32>) -> Vec<Vec<f32>> {
    let (rows, cols) = array.dim();
    let mut result = Vec::new();
    for i in 0..rows {
        let mut row = Vec::new();
        for j in 0..cols {
            row.push(array[[i, j]]);
        }
        result.push(row);
    }
    result
}

async fn test_with_real_data(config: &ModelConfig, market_data: &MarketData) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = DeepRiskModel::new(config)?;
    model.train(market_data).await?;
    let factors = model.generate_factors(market_data).await?;
    let covariance = model.estimate_covariance(market_data).await?;
    
    println!("Generated factors shape: {:?}", factors.factors.shape());
    println!("Covariance matrix shape: {:?}", covariance.shape());
    
    Ok(())
}

async fn test_performance(config: &ModelConfig, market_data: &MarketData) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = DeepRiskModel::new(config)?;
    
    // Train the model
    let start = Instant::now();
    model.train(market_data).await?;
    let training_time = start.elapsed();
    println!("Training time: {:?}", training_time);
    
    // Generate factors
    let start = Instant::now();
    let factors = model.generate_factors(market_data).await?;
    let factor_time = start.elapsed();
    println!("Factor generation time: {:?}", factor_time);
    println!("Generated factors shape: {:?}", factors.factors.shape());
    
    // Estimate covariance
    let start = Instant::now();
    let covariance = model.estimate_covariance(market_data).await?;
    let covariance_time = start.elapsed();
    println!("Covariance estimation time: {:?}", covariance_time);
    println!("Covariance matrix shape: {:?}", covariance.shape());
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize configuration
    let api_config = APIConfig::default();
    
    // Convert APIConfig to ModelConfig
    let model_config = ModelConfig {
        input_size: api_config.input_size,
        hidden_size: api_config.hidden_size,
        num_heads: api_config.num_heads,
        head_dim: api_config.head_dim,
        num_layers: api_config.num_layers,
        output_size: api_config.output_size,
    };
    
    // Generate synthetic market data
    let num_stocks: usize = 100;
    let seq_len: usize = 252;
    let num_features: usize = api_config.input_size as usize;
    
    let features = Array::zeros((seq_len, num_stocks, num_features)).into_dyn();
    let returns = Array::zeros((seq_len, num_stocks, 1)).into_dyn();
    
    let market_data = MarketData {
        features,
        returns,
    };
    
    // Run tests
    test_with_real_data(&model_config, &market_data).await?;
    test_performance(&model_config, &market_data).await?;
    
    // Create a request with the last time step
    let request = Request {
        features: array_to_vec(&market_data.features),
        returns: array_to_vec(&market_data.returns),
    };
    
    // Process the request
    let mut model = DeepRiskModel::new(&model_config)?;
    model.train(&market_data).await?;
    let factors = model.generate_factors(&market_data).await?;
    let covariance = model.estimate_covariance(&market_data).await?;
    
    // Create response
    let response = Response {
        factors: array_to_vec(&factors.factors),
        covariance: array2_to_vec(&covariance),
    };
    
    println!("Response: {:?}", response);
    
    Ok(())
}