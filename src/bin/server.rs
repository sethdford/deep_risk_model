use actix_web::{web, App, HttpResponse, HttpServer};
use deep_risk_model::{
    DeepRiskModel,
    MarketData,
    risk_model::RiskFactors,
    RiskModel,
    ModelConfig,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};
use tracing_subscriber;

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

#[derive(Debug, Deserialize)]
struct FactorRequest {
    market_data: MarketData,
}

#[derive(Debug, Deserialize)]
struct CovarianceRequest {
    factors: Vec<Vec<f32>>,
}

async fn health_check() -> HttpResponse {
    let response = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };
    HttpResponse::Ok().json(response)
}

async fn generate_factors(
    model: web::Data<Arc<RwLock<DeepRiskModel>>>,
    request: web::Json<FactorRequest>,
) -> HttpResponse {
    info!("Received request to generate factors");
    let model = model.read().await;
    
    match model.generate_factors(&request.market_data).await {
        Ok(factors) => {
            info!("Successfully generated factors");
            HttpResponse::Ok().json(factors)
        }
        Err(e) => {
            warn!("Failed to generate factors: {}", e);
            HttpResponse::InternalServerError().body(e.to_string())
        }
    }
}

async fn estimate_covariance(
    model: web::Data<Arc<RwLock<DeepRiskModel>>>,
    request: web::Json<CovarianceRequest>,
) -> HttpResponse {
    info!("Received request to estimate covariance");
    
    let factors = match Array2::from_shape_vec(
        (request.factors.len(), request.factors[0].len()),
        request.factors.iter().flatten().cloned().collect(),
    ) {
        Ok(f) => f,
        Err(e) => {
            warn!("Failed to reshape factors: {}", e);
            return HttpResponse::BadRequest().body(format!("Invalid factors shape: {}", e));
        }
    };

    let model = model.read().await;
    let market_data = MarketData::new(factors.into_dyn(), Array2::zeros((0, 0)).into_dyn());
    
    match model.estimate_covariance(&market_data).await {
        Ok(covariance) => {
            info!("Successfully estimated covariance");
            HttpResponse::Ok().json(covariance)
        }
        Err(e) => {
            warn!("Failed to estimate covariance: {}", e);
            HttpResponse::InternalServerError().body(e.to_string())
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let config = ModelConfig {
        input_size: 64,
        hidden_size: 128,
        num_heads: 4,
        head_dim: 8,
        num_layers: 2,
        output_size: 3,
    };
    
    let model = Arc::new(RwLock::new(DeepRiskModel::new(&config).unwrap()));
    
    info!("Starting server on http://127.0.0.1:8080");
    
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(Arc::clone(&model)))
            .route("/health", web::get().to(health_check))
            .route("/factors", web::post().to(generate_factors))
            .route("/covariance", web::post().to(estimate_covariance))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
} 