# Deep Risk Model

A deep learning-based risk model for financial markets, implemented in Rust. This project combines Graph Attention Networks (GAT) and Gated Recurrent Units (GRU) to generate risk factors and estimate covariance matrices from market data.

## Features

- Deep learning-based risk factor generation
- Graph Attention Network for cross-sectional feature extraction
- Gated Recurrent Unit for temporal feature processing
- Covariance matrix estimation
- REST API server for model inference
- AWS Lambda integration
- Python bindings via PyO3
- Comprehensive test suite

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
deep_risk_model = "0.1.0"
```

## Quick Start

```rust
use deep_risk_model::{DeepRiskModel, ModelConfig, RiskModel, MarketData};
use ndarray::{Array, ArrayD};
use ndarray_rand::{RandomExt, rand_distr::Normal};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
    let features: ArrayD<f32> = Array::random(
        (n_samples, seq_len, n_features),
        Normal::new(0.0, 1.0).unwrap()
    ).into_dyn();
    let returns: ArrayD<f32> = Array::random(
        (n_samples, 1),
        Normal::new(0.0, 0.1).unwrap()
    ).into_dyn();
    let market_data = MarketData::new(features, returns);
    
    // Train model
    model.train(&market_data).await?;
    
    // Generate risk factors
    let factors = model.generate_factors(&market_data).await?;
    println!("Generated factors shape: {:?}", factors.factors.shape());
    
    // Estimate covariance matrix
    let covariance = model.estimate_covariance(&market_data).await?;
    println!("Covariance matrix shape: {:?}", covariance.shape());
    
    Ok(())
}
```

## API Server

Start the API server:

```bash
cargo run --bin server
```

The server provides the following endpoints:

- `GET /health` - Health check endpoint
- `POST /factors` - Generate risk factors from market data
- `POST /covariance` - Estimate covariance matrix from factors

## AWS Lambda Integration

The project includes AWS Lambda integration for serverless deployment:

```bash
cargo lambda build
```

## Python Bindings

Install the Python package:

```bash
pip install deep-risk-model
```

Usage in Python:

```python
from deep_risk_model import DeepRiskModel, ModelConfig
import numpy as np

# Initialize model
config = ModelConfig(
    input_size=64,
    hidden_size=64,
    num_heads=4,
    head_dim=16,
    num_layers=2,
    output_size=3
)
model = DeepRiskModel(config)

# Generate synthetic data
n_samples = 100
seq_len = 50
n_features = 64
features = np.random.normal(0, 1, (n_samples, seq_len, n_features))
returns = np.random.normal(0, 0.1, (n_samples, 1))

# Train and generate factors
model.train(features, returns)
factors = model.generate_factors(features)
covariance = model.estimate_covariance(features)
```

## Testing

Run the test suite:

```bash
cargo test
```

For more detailed test output:

```bash
cargo test -- --nocapture
```

## Documentation

Generate documentation:

```bash
cargo doc --no-deps --open
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ndarray](https://github.com/rust-ndarray/ndarray) for efficient array operations
- [tokio](https://github.com/tokio-rs/tokio) for async runtime
- [serde](https://github.com/serde-rs/serde) for serialization 