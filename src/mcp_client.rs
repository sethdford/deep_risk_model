use anyhow::Result;
use chrono::{DateTime, Utc};
use ndarray::{Array, ArrayD, Ix2, Ix3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Deserialize, Serialize};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::env;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Configuration for the MCP client
#[derive(Debug, Clone)]
pub struct MCPConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the MCP API
    pub base_url: String,
}

impl Default for MCPConfig {
    fn default() -> Self {
        Self {
            api_key: env::var("MCP_API_KEY").unwrap_or_default(),
            base_url: env::var("MCP_BASE_URL").unwrap_or_else(|_| "https://api.example.com/v1".to_string()),
        }
    }
}

/// Market data structure for model input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Feature tensor of shape [num_stocks, seq_len, num_features]
    pub features: ArrayD<f32>,
    /// Returns tensor of shape [num_stocks]
    pub returns: ArrayD<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HistoricalData {
    pub timestamp: DateTime<Utc>,
    pub features: Vec<Vec<f64>>,
    pub returns: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MCPMarketData {
    pub symbols: Vec<String>,
    pub features: Vec<(String, Vec<f64>)>,
    pub returns: Vec<f64>,
    pub timestamp: DateTime<Utc>,
}

/// Response structure from MCP API
#[derive(Debug, Serialize, Deserialize)]
struct MCPResponse {
    features: Vec<Vec<Vec<f64>>>,
    returns: Vec<Vec<f64>>,
}

/// Client for interacting with the MCP API
pub struct MCPClient {
    config: Arc<MCPConfig>,
    client: reqwest::Client,
}

impl MCPClient {
    /// Creates a new MCP client with default configuration from environment variables
    pub fn new() -> Result<Self> {
        let config = MCPConfig::default();
        if config.api_key.is_empty() {
            warn!("No API key found in environment variables");
        }
        
        Ok(Self {
            config: Arc::new(config),
            client: reqwest::Client::new(),
        })
    }

    /// Creates a new MCP client with custom configuration
    pub fn with_config(config: MCPConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(config),
            client: reqwest::Client::new(),
        })
    }

    /// Fetches market data from the MCP API
    pub async fn fetch_market_data(&self) -> Result<MarketData> {
        info!("Fetching market data from MCP API");
        
        // In a real implementation, this would make an HTTP request
        // For now, we'll generate synthetic data
        let n_stocks = 100;
        let seq_len = 50;
        let n_features = 64;
        
        let features = Array::random((n_stocks, seq_len, n_features), ndarray_rand::rand_distr::Normal::new(0.0, 1.0).unwrap());
        let returns = Array::random((n_stocks, 1), ndarray_rand::rand_distr::Normal::new(0.0, 0.1).unwrap());
        
        Ok(MarketData {
            features: features.into_dyn(),
            returns: returns.into_dyn(),
        })
    }

    /// Fetches historical market data for a given date range
    pub async fn fetch_historical_data(&self, start_date: &str, end_date: &str) -> Result<MarketData> {
        info!("Fetching historical market data from {} to {}", start_date, end_date);
        
        // In a real implementation, this would make an HTTP request
        // For now, we'll generate synthetic data
        let n_stocks = 100;
        let seq_len = 50;
        let n_features = 64;
        
        let features = Array::random((n_stocks, seq_len, n_features), ndarray_rand::rand_distr::Normal::new(0.0, 1.0).unwrap());
        let returns = Array::random((n_stocks, 1), ndarray_rand::rand_distr::Normal::new(0.0, 0.1).unwrap());
        
        Ok(MarketData {
            features: features.into_dyn(),
            returns: returns.into_dyn(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[tokio::test]
    async fn test_fetch_market_data() {
        // Set test environment variables
        env::set_var("MCP_API_KEY", "test_key");
        env::set_var("MCP_BASE_URL", "https://test.api.example.com");
        
        let client = MCPClient::new().unwrap();
        let data = client.fetch_market_data().await.unwrap();
        
        assert_eq!(data.features.shape()[0], 100);
        assert_eq!(data.features.shape()[1], 50);
        assert_eq!(data.features.shape()[2], 64);
        assert_eq!(data.returns.shape()[0], 100);
    }

    #[tokio::test]
    async fn test_fetch_historical_data() {
        // Set test environment variables
        env::set_var("MCP_API_KEY", "test_key");
        env::set_var("MCP_BASE_URL", "https://test.api.example.com");
        
        let client = MCPClient::new().unwrap();
        let data = client.fetch_historical_data("2023-01-01", "2023-12-31").await.unwrap();
        
        assert_eq!(data.features.shape()[0], 100);
        assert_eq!(data.features.shape()[1], 50);
        assert_eq!(data.features.shape()[2], 64);
        assert_eq!(data.returns.shape()[0], 100);
    }
} 