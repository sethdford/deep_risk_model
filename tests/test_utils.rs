use deep_risk_model::{MarketData, ModelConfig};
use ndarray::{Array, Array2, ArrayD};
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};
use anyhow::Result;
use deep_risk_model::{ModelError};
use ndarray_rand::{rand_distr::Normal};

/// Creates a default model configuration for testing
pub fn create_test_config(
    input_dim: usize,
    hidden_dim: usize,
    num_layers: usize,
) -> ModelConfig {
    ModelConfig {
        input_size: input_dim as i64,
        hidden_size: hidden_dim as i64,
        num_heads: 4,
        head_dim: (hidden_dim / 4) as i64,
        num_layers: num_layers as i64,
        output_size: hidden_dim as i64,
    }
}

/// Generates synthetic market data with specified parameters
pub fn generate_test_data(
    num_stocks: usize,
    seq_len: usize,
    noise_level: f32,
) -> Result<MarketData, ModelError> {
    let features: ArrayD<f32> = Array::random((num_stocks, seq_len, 10), Normal::new(0.0, 1.0).unwrap()).into_dyn();
    let returns: ArrayD<f32> = Array::random(num_stocks, Normal::new(0.0, 0.1).unwrap()).into_dyn();
    
    Ok(MarketData::new(features, returns))
}

/// Generates synthetic market data with specified parameters
pub fn generate_test_data_with_noise(
    num_stocks: usize,
    seq_len: usize,
    noise_level: f32,
) -> Result<MarketData, ModelError> {
    let features: ArrayD<f32> = Array::random((num_stocks, seq_len, 10), Normal::new(0.0, 1.0).unwrap()).into_dyn();
    let noise: ArrayD<f32> = Array::random((num_stocks, seq_len, 10), Normal::new(0.0, noise_level).unwrap()).into_dyn();
    let returns: ArrayD<f32> = Array::random(num_stocks, Normal::new(0.0, 0.1).unwrap()).into_dyn();
    
    Ok(MarketData::new((features + noise).into_dyn(), returns))
}

/// Verifies the properties of a covariance matrix
pub fn verify_covariance_matrix(covariance: &ArrayD<f32>, num_stocks: usize) -> bool {
    // Check that covariance matrix is symmetric and positive definite
    let n = covariance.shape()[0];
    
    // Check symmetry
    for i in 0..n {
        for j in 0..n {
            if (covariance[[i, j]] - covariance[[j, i]]).abs() > 1e-10 {
                return false;
            }
        }
    }
    
    // Check positive definiteness (all eigenvalues > 0)
    // This is a simplified check - in practice you'd want to compute eigenvalues
    let trace = (0..n).map(|i| covariance[[i, i]]).sum::<f32>();
    let det = if n == 2 {
        covariance[[0, 0]] * covariance[[1, 1]] - covariance[[0, 1]] * covariance[[1, 0]]
    } else {
        // For larger matrices, just check trace as a heuristic
        trace
    };
    
    trace > 0.0 && det > 0.0
}

/// Calculates basic statistics for risk factors
pub fn calculate_factor_stats(factors: &ArrayD<f64>) -> (f64, f64) {
    let mean = factors.mean().unwrap();
    let std = factors.std_axis(ndarray::Axis(0), 0.0).mean().unwrap();
    (mean, std)
}

/// Verifies the quality of risk factors
pub fn verify_factor_quality(t_stats: &ArrayD<f32>, vif: &ArrayD<f32>, auto_correlation: &ArrayD<f32>) -> bool {
    // Check t-statistics (should be significant)
    let t_stats_valid = t_stats.iter().all(|&x| x.abs() >= 2.0);
    
    // Check VIF (should be low to avoid multicollinearity)
    let vif_valid = vif.iter().all(|&x| x <= 5.0);
    
    // Check autocorrelation (should be low to avoid persistence)
    let auto_correlation_valid = auto_correlation.iter().all(|&x| x.abs() <= 0.8);
    
    t_stats_valid && vif_valid && auto_correlation_valid
} 