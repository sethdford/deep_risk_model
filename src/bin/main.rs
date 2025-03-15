use ndarray::{Array, Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use deep_risk_model::{
    config::ModelConfig,
    model::DeepRiskModel,
    MarketData,
    risk_model::RiskFactors,
    RiskModel,
};
use std::error::Error;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let config = ModelConfig {
        input_size: 10,
        hidden_size: 64,
        num_heads: 4,
        head_dim: 8,
        num_layers: 2,
        output_size: 3,
    };

    // Initialize model with proper type casting
    let mut model = DeepRiskModel::new(&config)?;

    // Generate some sample data
    let n_stocks = 100;
    let seq_len = 50;
    let n_features = 10;
    let n_periods = 100;

    let mut data = Vec::with_capacity(n_periods);
    for _ in 0..n_periods {
        let features = Array3::random((n_stocks, seq_len, n_features), Normal::new(0.0, 1.0).unwrap()).into_dyn();
        let returns = Array::random(n_stocks, Normal::new(0.0, 1.0).unwrap()).into_dyn();
        data.push(MarketData::new(features, returns));
    }

    // Train the model
    for batch in data.iter() {
        model.train(batch).await?;
    }

    // Generate risk factors
    let factors = model.generate_factors(&data[data.len() - 1]).await?;
    println!("Generated risk factors: {:?}", factors);

    // Estimate covariance
    let covariance = model.estimate_covariance(&data[data.len() - 1]).await?;
    println!("Estimated covariance matrix shape: {:?}", covariance.shape());

    Ok(())
} 