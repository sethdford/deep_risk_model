use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array3, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use crate::{DeepRiskModel, ModelConfig, MarketData};
use std::time::Instant;
use tokio::runtime::Runtime;

/// Performance metrics for model operations
#[derive(Debug, Default)]
pub struct ModelMetrics {
    /// Factor generation latency in microseconds
    pub factor_gen_latency_us: Vec<u64>,
    /// Training latency in microseconds
    pub training_latency_us: Vec<u64>,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Number of successful operations
    pub success_count: usize,
    /// Number of failed operations
    pub error_count: usize,
}

impl ModelMetrics {
    /// Creates a new ModelMetrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a latency measurement for factor generation
    pub fn record_factor_gen_latency(&mut self, latency_us: u64) {
        self.factor_gen_latency_us.push(latency_us);
    }

    /// Records a latency measurement for training
    pub fn record_training_latency(&mut self, latency_us: u64) {
        self.training_latency_us.push(latency_us);
    }

    /// Calculates average factor generation latency
    pub fn avg_factor_gen_latency(&self) -> Option<f64> {
        if self.factor_gen_latency_us.is_empty() {
            None
        } else {
            Some(self.factor_gen_latency_us.iter().sum::<u64>() as f64 
                 / self.factor_gen_latency_us.len() as f64)
        }
    }

    /// Calculates average training latency
    pub fn avg_training_latency(&self) -> Option<f64> {
        if self.training_latency_us.is_empty() {
            None
        } else {
            Some(self.training_latency_us.iter().sum::<u64>() as f64 
                 / self.training_latency_us.len() as f64)
        }
    }

    /// Records a successful operation
    pub fn record_success(&mut self) {
        self.success_count += 1;
    }

    /// Records a failed operation
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Calculates success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.error_count;
        if total == 0 {
            0.0
        } else {
            self.success_count as f64 / total as f64
        }
    }
}

/// Runs performance benchmarks for the model
pub fn run_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let config = ModelConfig {
        input_size: 64,
        hidden_size: 128,
        num_heads: 4,
        head_dim: 32,
        num_layers: 2,
        output_size: 3,
    };

    let model = DeepRiskModel::new(&config).unwrap();
    
    // Create test data
    let batch_size = 100;
    let seq_len = 50;
    let features = Array3::random(
        (batch_size, seq_len, config.input_size as usize),
        Normal::new(0.0, 1.0).unwrap()
    ).into_dyn();
    
    let returns = Array2::random(
        (batch_size, 1),
        Normal::new(0.0, 1.0).unwrap()
    ).into_dyn();

    let market_data = MarketData::new(features, returns);

    // Benchmark factor generation
    c.bench_function("generate_factors", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(model.generate_factors(&market_data).await.unwrap());
            });
        });
    });

    // Benchmark covariance estimation
    c.bench_function("estimate_covariance", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(model.estimate_covariance(&market_data).await.unwrap());
            });
        });
    });
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches); 