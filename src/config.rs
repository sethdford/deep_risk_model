/// Configuration for the DeepRiskModel.
/// 
/// This struct contains all the hyperparameters needed to initialize
/// and configure the DeepRiskModel.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Dimension of input features
    pub input_size: i64,
    /// Dimension of hidden layers
    pub hidden_size: i64,
    /// Number of attention heads in GAT
    pub num_heads: i64,
    /// Dimension of each attention head
    pub head_dim: i64,
    /// Number of layers in the model
    pub num_layers: i64,
    /// Dimension of output features
    pub output_size: i64,
}

impl Default for ModelConfig {
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