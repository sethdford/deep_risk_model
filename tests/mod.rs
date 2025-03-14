mod e2e_tests;
mod test_utils;

pub use test_utils::*;

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use deep_risk_model::{
        DeepRiskModel, ModelConfig, RiskModel, MarketData,
    };
    use ndarray::{Array, ArrayD};
    use ndarray_rand::{RandomExt, rand_distr::Normal, rand_distr::StandardNormal};
    use super::test_utils::{verify_covariance_matrix, verify_factor_quality};

    #[tokio::test]
    async fn test_model_initialization() -> Result<()> {
        let config = ModelConfig {
            input_size: 64,
            hidden_size: 64,
            num_heads: 4,
            head_dim: 16,
            num_layers: 2,
            output_size: 3,
        };
        
        let _model = DeepRiskModel::new(&config)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_market_data_generation() -> Result<()> {
        let n_stocks = 100;
        let seq_len = 50;
        let n_features = 64;
        
        let features: ArrayD<f32> = Array::random((n_stocks, seq_len, n_features), Normal::new(0.0, 1.0).unwrap()).into_dyn();
        let returns: ArrayD<f32> = Array::random((n_stocks, 1), Normal::new(0.0, 0.1).unwrap()).into_dyn();
        let market_data = MarketData::new(features, returns);
        
        assert_eq!(market_data.features().shape(), &[n_stocks, seq_len, n_features]);
        assert_eq!(market_data.returns().shape(), &[n_stocks, 1]);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_factor_generation() -> Result<()> {
        let config = ModelConfig {
            input_size: 64,
            hidden_size: 64,
            num_heads: 4,
            head_dim: 16,
            num_layers: 2,
            output_size: 3,
        };
        
        let mut model = DeepRiskModel::new(&config)?;
        
        let n_stocks = 100;
        let seq_len = 50;
        let n_features = 64;
        
        let features: ArrayD<f32> = Array::random((n_stocks, seq_len, n_features), Normal::new(0.0, 1.0).unwrap()).into_dyn();
        let returns: ArrayD<f32> = Array::random((n_stocks, 1), Normal::new(0.0, 0.1).unwrap()).into_dyn();
        let market_data = MarketData::new(features, returns);
        
        model.train(&market_data).await?;
        let factors = model.generate_factors(&market_data).await?;
        
        assert_eq!(factors.factors.shape()[0], n_stocks);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_covariance_estimation() -> Result<()> {
        let config = ModelConfig {
            input_size: 64,
            hidden_size: 64,
            num_heads: 4,
            head_dim: 16,
            num_layers: 2,
            output_size: 3,
        };
        
        let mut model = DeepRiskModel::new(&config)?;
        
        let n_stocks = 100;
        let seq_len = 50;
        let n_features = 64;
        
        let features: ArrayD<f32> = Array::random((n_stocks, seq_len, n_features), Normal::new(0.0, 1.0).unwrap()).into_dyn();
        let returns: ArrayD<f32> = Array::random((n_stocks, 1), Normal::new(0.0, 0.1).unwrap()).into_dyn();
        let market_data = MarketData::new(features, returns);
        
        model.train(&market_data).await?;
        let covariance = model.estimate_covariance(&market_data).await?;
        
        assert_eq!(covariance.shape(), &[n_stocks, n_stocks]);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_covariance_verification() -> anyhow::Result<()> {
        let num_stocks = 30;
        let covariance = Array::random((num_stocks, num_stocks), StandardNormal);
        let covariance = covariance.dot(&covariance.t()); // Make it positive definite
        
        assert!(verify_covariance_matrix(&covariance.into_dyn(), num_stocks));
        Ok(())
    }

    #[tokio::test]
    async fn test_factor_quality_verification() -> anyhow::Result<()> {
        let num_factors = 5;
        
        // Generate t-stats with absolute values >= 2.0
        let t_stats = Array::from_vec(vec![2.5, -2.8, 3.2, -3.5, 4.0]);
        
        // Generate VIF values <= 5.0
        let vif = Array::from_vec(vec![1.2, 2.3, 3.1, 4.2, 4.8]);
        
        // Generate autocorrelation values with absolute values <= 0.8
        let auto_correlation = Array::from_vec(vec![0.3, -0.5, 0.7, -0.6, 0.4]);
        
        assert!(verify_factor_quality(
            &t_stats.into_dyn(),
            &vif.into_dyn(),
            &auto_correlation.into_dyn()
        ));
        Ok(())
    }
}

// Re-export e2e tests
pub use e2e_tests::*; 