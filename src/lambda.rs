use lambda_http::lambda_runtime::{run, service_fn, Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use deep_risk_model::{DeepRiskModel, MarketData, ModelConfig, RiskModel};
use aws_sdk_s3::Client as S3Client;
use aws_config::load_defaults;
use aws_config::BehaviorVersion;
use tracing;

#[derive(Deserialize)]
struct Request {
    features: Vec<f32>,
    returns: Vec<f32>,
}

#[derive(Serialize)]
struct Response {
    factors: Vec<f64>,
    covariance: Vec<Vec<f64>>,
}

async fn function_handler(event: LambdaEvent<Request>) -> Result<Response, Error> {
    let (event, _context) = event.into_parts();
    
    // Initialize the model
    let config = ModelConfig::default();
    let model = DeepRiskModel::new(&config).map_err(|e| Error::from(format!("Failed to initialize model: {}", e)))?;
    
    // Create market data from the request
    let features = ndarray::Array::from_vec(event.features).into_dyn();
    let returns = ndarray::Array::from_vec(event.returns).into_dyn();
    let market_data = MarketData::new(features, returns);
    
    // Generate risk factors
    let risk_factors = model.generate_factors(&market_data)
        .await
        .map_err(|e| Error::from(format!("Failed to generate factors: {}", e)))?;
    
    // Convert risk factors to response format
    let response = Response {
        factors: risk_factors.factors.into_raw_vec().into_iter().map(|x| x as f64).collect(),
        covariance: risk_factors.covariance
            .outer_iter()
            .map(|row| row.iter().map(|&x| x as f64).collect())
            .collect(),
    };
    
    Ok(response)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .init();

    // Initialize AWS SDK
    let config = load_defaults(BehaviorVersion::latest()).await;
    let _s3_client = S3Client::new(&config);

    run(service_fn(function_handler)).await
}